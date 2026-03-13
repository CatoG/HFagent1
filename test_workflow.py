"""
Tests for the multi-role workflow improvements.

Tests cover:
1. Output format detection
2. Brevity detection
3. Structured QA parsing (JSON and legacy)
4. Role relevance selection (with task-category awareness)
5. Targeted revision identification
6. Internal noise stripping
7. Final answer compression
8. Task classification
9. Evidence system (EvidenceResult, adapters, claim detection)
10. Planner state (failure tracking, escalation, state serialisation)
11. End-to-end workflow scenarios
"""

import json
import unittest

from workflow_helpers import (
    WorkflowConfig,
    detect_output_format,
    detect_brevity_requirement,
    classify_task,
    task_needs_evidence,
    QAIssue,
    QAResult,
    parse_structured_qa,
    select_relevant_roles,
    identify_revision_targets,
    strip_internal_noise,
    compress_final_answer,
    PlannerState,
    FailureRecord,
    get_synthesizer_format_instruction,
    get_qa_format_instruction,
    validate_output_format,
    format_violations_instruction,
    parse_task_assumptions,
    format_assumptions_for_prompt,
    StructuredContribution,
    parse_structured_contribution,
    format_contributions_for_synthesizer,
    format_contributions_for_qa,
    parse_used_contributions,
    check_expert_influence,
)
from evidence import (
    EvidenceItem,
    EvidenceResult,
    gather_evidence,
    extract_search_queries,
    detect_unsupported_claims,
    format_evidence_for_prompt,
    format_evidence_for_qa,
    ResearchToolAdapter,
)


# ============================================================
# Test: Output Format Detection
# ============================================================

class TestOutputFormatDetection(unittest.TestCase):

    def test_single_choice_pick_one(self):
        self.assertEqual(
            detect_output_format("agree if dinner should be meat-rich, veggie, vegan or fastfood. you have to agree on one choice."),
            "single_choice"
        )

    def test_single_choice_choose_one(self):
        self.assertEqual(detect_output_format("choose one color: red, blue, green"), "single_choice")

    def test_short_answer(self):
        self.assertEqual(
            detect_output_format("give me a short answer: is rust faster than python?"),
            "short_answer"
        )

    def test_short_answer_briefly(self):
        self.assertEqual(detect_output_format("briefly explain gravity"), "short_answer")

    def test_table(self):
        self.assertEqual(
            detect_output_format("make a table comparing React, Vue and Svelte"),
            "table"
        )

    def test_code(self):
        self.assertEqual(
            detect_output_format("write Python code to parse a CSV file"),
            "code"
        )

    def test_code_implement(self):
        self.assertEqual(
            detect_output_format("implement a function to sort a list"),
            "code"
        )

    def test_report(self):
        self.assertEqual(detect_output_format("write a detailed analysis of market trends"), "report")

    def test_bullet_list(self):
        self.assertEqual(detect_output_format("list the top 5 programming languages"), "bullet_list")

    def test_paragraph(self):
        self.assertEqual(detect_output_format("explain how photosynthesis works"), "paragraph")

    def test_other(self):
        self.assertEqual(detect_output_format("hello"), "other")


# ============================================================
# Test: Brevity Detection
# ============================================================

class TestBrevityDetection(unittest.TestCase):

    def test_minimal_pick_one(self):
        self.assertEqual(
            detect_brevity_requirement("pick one: A, B, or C"),
            "minimal"
        )

    def test_minimal_yes_or_no(self):
        self.assertEqual(detect_brevity_requirement("yes or no: is it raining?"), "minimal")

    def test_short(self):
        self.assertEqual(detect_brevity_requirement("give a short answer about AI"), "short")

    def test_short_concise(self):
        self.assertEqual(detect_brevity_requirement("concisely explain quantum computing"), "short")

    def test_verbose(self):
        self.assertEqual(detect_brevity_requirement("write a detailed report on climate change"), "verbose")

    def test_normal(self):
        self.assertEqual(detect_brevity_requirement("what is the capital of France?"), "normal")


# ============================================================
# Test: Structured QA Parsing
# ============================================================

class TestStructuredQAParsing(unittest.TestCase):

    def test_parse_json_pass(self):
        qa_text = json.dumps({
            "status": "PASS",
            "reason": "All checks passed",
            "issues": [],
            "correction_instruction": ""
        })
        result = parse_structured_qa(qa_text)
        self.assertTrue(result.passed)
        self.assertEqual(result.status, "PASS")
        self.assertEqual(len(result.issues), 0)

    def test_parse_json_fail(self):
        qa_text = json.dumps({
            "status": "FAIL",
            "reason": "Output too verbose",
            "issues": [
                {
                    "type": "brevity",
                    "message": "Answer is 500 words when short was requested",
                    "owner": "Synthesizer"
                }
            ],
            "correction_instruction": "Shorten to 2-3 sentences"
        })
        result = parse_structured_qa(qa_text)
        self.assertFalse(result.passed)
        self.assertEqual(result.status, "FAIL")
        self.assertEqual(len(result.issues), 1)
        self.assertEqual(result.issues[0].type, "brevity")
        self.assertEqual(result.issues[0].owner, "Synthesizer")
        self.assertEqual(result.correction_instruction, "Shorten to 2-3 sentences")

    def test_parse_json_multiple_issues(self):
        qa_text = json.dumps({
            "status": "FAIL",
            "reason": "Multiple problems",
            "issues": [
                {"type": "format", "message": "Not a table", "owner": "Synthesizer"},
                {"type": "constraint", "message": "Missing Vue data", "owner": "Research Analyst"},
            ],
            "correction_instruction": "Reformat as table and add Vue data"
        })
        result = parse_structured_qa(qa_text)
        self.assertEqual(len(result.issues), 2)
        owners = result.owners()
        self.assertIn("Synthesizer", owners)
        self.assertIn("Research Analyst", owners)

    def test_parse_legacy_pass(self):
        qa_text = (
            "REQUIREMENTS CHECKED:\n- All met\n\n"
            "ISSUES FOUND:\nNone\n\n"
            "RESULT: PASS\n\n"
            "RECOMMENDED FIXES:\nNone"
        )
        result = parse_structured_qa(qa_text)
        self.assertTrue(result.passed)

    def test_parse_legacy_fail(self):
        qa_text = (
            "REQUIREMENTS CHECKED:\n- Format not met\n\n"
            "ISSUES FOUND:\nOutput is too long\n\n"
            "ROLE-SPECIFIC FEEDBACK:\n"
            "• Creative Expert: Too verbose, needs trimming\n"
            "• Technical Expert: Satisfactory\n\n"
            "RESULT: FAIL\n\n"
            "RECOMMENDED FIXES:\nShorten the output"
        )
        result = parse_structured_qa(qa_text)
        self.assertFalse(result.passed)
        self.assertEqual(len(result.issues), 1)
        self.assertEqual(result.issues[0].owner, "Creative Expert")

    def test_parse_json_embedded_in_text(self):
        qa_text = (
            'Here is my assessment:\n'
            '{"status": "FAIL", "reason": "Wrong format", '
            '"issues": [{"type": "format", "message": "Should be a table", "owner": "Synthesizer"}], '
            '"correction_instruction": "Use markdown table"}\n'
            'End of assessment.'
        )
        result = parse_structured_qa(qa_text)
        self.assertFalse(result.passed)
        self.assertEqual(result.issues[0].type, "format")

    def test_to_dict(self):
        result = QAResult(
            status="FAIL",
            reason="test",
            issues=[QAIssue(type="format", message="bad", owner="Synthesizer")],
            correction_instruction="fix it"
        )
        d = result.to_dict()
        self.assertEqual(d["status"], "FAIL")
        self.assertEqual(len(d["issues"]), 1)


# ============================================================
# Test: Role Selection
# ============================================================

class TestRoleSelection(unittest.TestCase):

    def setUp(self):
        self.all_roles = [
            "creative", "technical", "research", "security", "data_analyst",
            "labour_union_rep", "ux_designer", "lawyer",
            "mad_professor", "accountant", "artist", "lazy_slacker",
            "black_metal_fundamentalist", "doris", "chairman_of_board", "maga_appointee",
        ]

    def test_code_question_selects_technical(self):
        config = WorkflowConfig(strict_mode=True, allow_persona_roles=False, max_specialists_per_task=3)
        roles = select_relevant_roles("write Python code to parse a CSV", self.all_roles, config)
        self.assertIn("technical", roles)
        self.assertLessEqual(len(roles), 3)
        # Should not include persona roles
        for r in roles:
            self.assertNotIn(r, config.PERSONA_ROLE_KEYS)

    def test_research_question_selects_research(self):
        config = WorkflowConfig(strict_mode=True, allow_persona_roles=False, max_specialists_per_task=3)
        roles = select_relevant_roles("research the history of AI", self.all_roles, config)
        self.assertIn("research", roles)

    def test_security_question(self):
        config = WorkflowConfig(strict_mode=True, allow_persona_roles=False, max_specialists_per_task=3)
        roles = select_relevant_roles("check for security vulnerabilities in my API", self.all_roles, config)
        self.assertIn("security", roles)

    def test_persona_roles_excluded_by_default(self):
        config = WorkflowConfig(strict_mode=True, allow_persona_roles=False, max_specialists_per_task=8)
        roles = select_relevant_roles("tell me something crazy and radical", self.all_roles, config)
        for r in roles:
            self.assertNotIn(r, config.PERSONA_ROLE_KEYS)

    def test_persona_roles_included_when_allowed(self):
        config = WorkflowConfig(strict_mode=True, allow_persona_roles=True, max_specialists_per_task=8)
        roles = select_relevant_roles("give me the cheapest budget approach", self.all_roles, config)
        self.assertIn("accountant", roles)

    def test_max_specialists_respected(self):
        config = WorkflowConfig(strict_mode=False, allow_persona_roles=True, max_specialists_per_task=2)
        roles = select_relevant_roles("everything about everything", self.all_roles, config)
        self.assertLessEqual(len(roles), 2)

    def test_at_least_one_role_selected(self):
        config = WorkflowConfig(strict_mode=True, allow_persona_roles=False, max_specialists_per_task=3)
        roles = select_relevant_roles("blah blah random", self.all_roles, config)
        self.assertGreaterEqual(len(roles), 1)

    def test_dinner_question_minimal_roles(self):
        """Test 1 from requirements: trivial preference question should select few roles."""
        config = WorkflowConfig(strict_mode=True, allow_persona_roles=False, max_specialists_per_task=3)
        roles = select_relevant_roles(
            "agree if dinner should be meat-rich, veggie, vegan or fastfood. you have to agree on one choice.",
            self.all_roles, config
        )
        self.assertLessEqual(len(roles), 3)
        for r in roles:
            self.assertNotIn(r, config.PERSONA_ROLE_KEYS)

    def test_ux_question(self):
        config = WorkflowConfig(strict_mode=True, allow_persona_roles=False, max_specialists_per_task=3)
        roles = select_relevant_roles("improve the user experience of my login page", self.all_roles, config)
        self.assertIn("ux_designer", roles)


# ============================================================
# Test: Targeted Revision Identification
# ============================================================

class TestTargetedRevisions(unittest.TestCase):

    def setUp(self):
        self.role_label_to_key = {
            "Creative Expert": "creative",
            "Technical Expert": "technical",
            "Research Analyst": "research",
            "Security Reviewer": "security",
            "Data Analyst": "data_analyst",
            "Synthesizer": "synthesizer",
            "Planner": "planner",
            "UX Designer": "ux_designer",
        }

    def test_format_issue_targets_synthesizer(self):
        qa = QAResult(
            status="FAIL", reason="Wrong format",
            issues=[QAIssue(type="format", message="Not a table", owner="Synthesizer")]
        )
        targets = identify_revision_targets(qa, self.role_label_to_key)
        self.assertIn("synthesizer", targets)

    def test_brevity_issue_targets_synthesizer(self):
        qa = QAResult(
            status="FAIL", reason="Too verbose",
            issues=[QAIssue(type="brevity", message="Too long", owner="Synthesizer")]
        )
        targets = identify_revision_targets(qa, self.role_label_to_key)
        self.assertIn("synthesizer", targets)

    def test_specialist_issue_targets_specialist(self):
        qa = QAResult(
            status="FAIL", reason="Missing data",
            issues=[QAIssue(type="constraint", message="No Vue comparison", owner="Research Analyst")]
        )
        targets = identify_revision_targets(qa, self.role_label_to_key)
        self.assertIn("research", targets)

    def test_multiple_issues_multiple_targets(self):
        qa = QAResult(
            status="FAIL", reason="Multiple issues",
            issues=[
                QAIssue(type="format", message="Not a table", owner="Synthesizer"),
                QAIssue(type="other", message="Wrong data", owner="Technical Expert"),
            ]
        )
        targets = identify_revision_targets(qa, self.role_label_to_key)
        self.assertIn("synthesizer", targets)
        self.assertIn("technical", targets)

    def test_no_issues_default_to_synthesizer(self):
        qa = QAResult(status="FAIL", reason="Unknown issue", issues=[])
        targets = identify_revision_targets(qa, self.role_label_to_key)
        self.assertIn("synthesizer", targets)


# ============================================================
# Test: Internal Noise Stripping
# ============================================================

class TestNoiseStripping(unittest.TestCase):

    def test_strip_task_breakdown(self):
        text = "TASK BREAKDOWN:\n- Step 1\n- Step 2\n\nThe answer is 42."
        result = strip_internal_noise(text)
        self.assertIn("42", result)
        self.assertNotIn("TASK BREAKDOWN", result)

    def test_strip_perspectives_summary(self):
        text = "PERSPECTIVES SUMMARY:\n• Role A — point\n\nCOMMON GROUND:\nAll agree\n\nThe real answer is: veggie."
        result = strip_internal_noise(text)
        self.assertIn("veggie", result)
        self.assertNotIn("PERSPECTIVES SUMMARY", result)
        self.assertNotIn("COMMON GROUND", result)

    def test_clean_text_unchanged(self):
        text = "Veggie — it accommodates the widest range of dietary needs."
        result = strip_internal_noise(text)
        self.assertEqual(result, text)

    def test_strip_qa_notes(self):
        text = "RESULT: PASS\nRECOMMENDED FIXES:\nNone\n\nThe answer is correct."
        result = strip_internal_noise(text)
        self.assertNotIn("RESULT:", result)
        self.assertNotIn("RECOMMENDED FIXES:", result)


# ============================================================
# Test: Final Answer Compression
# ============================================================

class TestFinalAnswerCompression(unittest.TestCase):

    def test_single_choice_compression(self):
        draft = (
            "PERSPECTIVES SUMMARY:\n• Expert A — veggie\n\n"
            "COMMON GROUND:\nAll agree on veggie\n\n"
            "UNIFIED RECOMMENDATION:\nVeggie"
        )
        result = compress_final_answer(draft, "single_choice", "minimal", "pick one")
        self.assertIn("Veggie", result)
        # Should be short
        self.assertLess(len(result), 200)

    def test_short_answer_stays_short(self):
        draft = "Yes, Rust is generally faster than Python because it compiles to native code."
        result = compress_final_answer(draft, "short_answer", "short", "is rust faster?")
        self.assertEqual(result, draft)

    def test_noise_stripped_from_any_format(self):
        draft = "TASK BREAKDOWN:\n- analysis\n\nThe answer is 42."
        result = compress_final_answer(draft, "other", "normal", "what is the answer?")
        self.assertNotIn("TASK BREAKDOWN", result)
        self.assertIn("42", result)


# ============================================================
# Test: PlannerState
# ============================================================

class TestPlannerState(unittest.TestCase):

    def test_record_event(self):
        ps = PlannerState(user_request="test")
        ps.record_event("init", "started")
        self.assertEqual(len(ps.history), 1)
        self.assertEqual(ps.history[0]["type"], "init")

    def test_context_string(self):
        ps = PlannerState(
            user_request="test",
            output_format="table",
            brevity_requirement="short",
            selected_roles=["technical", "research"],
        )
        ctx = ps.to_context_string()
        self.assertIn("table", ctx)
        self.assertIn("short", ctx)
        self.assertIn("technical", ctx)

    def test_context_string_with_qa_fail(self):
        ps = PlannerState(user_request="test")
        ps.qa_result = QAResult(
            status="FAIL", reason="Too verbose",
            correction_instruction="Be shorter"
        )
        ctx = ps.to_context_string()
        self.assertIn("FAIL", ctx)
        self.assertIn("Be shorter", ctx)


# ============================================================
# Test: Format-specific Instructions
# ============================================================

class TestFormatInstructions(unittest.TestCase):

    def test_single_choice_synthesizer(self):
        inst = get_synthesizer_format_instruction("single_choice", "minimal")
        self.assertIn("ONE SINGLE CHOICE", inst)
        self.assertIn("BREVITY", inst)

    def test_code_synthesizer(self):
        inst = get_synthesizer_format_instruction("code", "normal")
        self.assertIn("CODE", inst)

    def test_table_synthesizer(self):
        inst = get_synthesizer_format_instruction("table", "normal")
        self.assertIn("TABLE", inst)

    def test_qa_format_single_choice(self):
        inst = get_qa_format_instruction("single_choice", "minimal")
        self.assertIn("FAIL", inst)

    def test_qa_format_table(self):
        inst = get_qa_format_instruction("table", "normal")
        self.assertIn("FAIL", inst)
        self.assertIn("table", inst.lower())


# ============================================================
# Test: End-to-End Workflow Scenarios (Mocked LLM)
# ============================================================

class TestWorkflowScenarios(unittest.TestCase):
    """Test that the workflow control flow behaves correctly.

    These tests verify:
    - Format detection is correct for each scenario
    - Role selection uses minimal roles
    - QA-binding prevents approving FAIL results
    - Targeted revisions work
    """

    def test_dinner_choice_format_detection(self):
        """Test 1: Dinner choice should detect single_choice format with minimal brevity."""
        msg = "agree if dinner should be meat-rich, veggie, vegan or fastfood. you have to agree on one choice."
        fmt = detect_output_format(msg)
        brevity = detect_brevity_requirement(msg)
        self.assertEqual(fmt, "single_choice")
        self.assertEqual(brevity, "minimal")

    def test_rust_python_format_detection(self):
        """Test 2: Short answer question should detect short_answer format."""
        msg = "give me a short answer: is rust faster than python?"
        fmt = detect_output_format(msg)
        brevity = detect_brevity_requirement(msg)
        self.assertEqual(fmt, "short_answer")
        self.assertEqual(brevity, "short")

    def test_table_format_detection(self):
        """Test 3: Table comparison should detect table format."""
        msg = "make a table comparing React, Vue and Svelte"
        fmt = detect_output_format(msg)
        self.assertEqual(fmt, "table")

    def test_code_format_detection(self):
        """Test 4: Code request should detect code format."""
        msg = "write Python code to parse a CSV file"
        fmt = detect_output_format(msg)
        self.assertEqual(fmt, "code")

    def test_qa_binding_blocks_approval(self):
        """Verify that QA FAIL prevents approval at the code level."""
        qa_fail = QAResult(
            status="FAIL", reason="Too verbose",
            issues=[QAIssue(type="brevity", message="Answer too long", owner="Synthesizer")],
            correction_instruction="Shorten it"
        )
        self.assertFalse(qa_fail.passed)

        qa_pass = QAResult(status="PASS", reason="All good")
        self.assertTrue(qa_pass.passed)

    def test_targeted_revision_for_format_issue(self):
        """Format issues should target only the synthesizer, not rerun all specialists."""
        role_map = {
            "Synthesizer": "synthesizer",
            "Technical Expert": "technical",
            "Research Analyst": "research",
        }
        qa = QAResult(
            status="FAIL", reason="Output format wrong",
            issues=[QAIssue(type="format", message="Should be a table", owner="Synthesizer")]
        )
        targets = identify_revision_targets(qa, role_map)
        self.assertEqual(targets, ["synthesizer"])

    def test_targeted_revision_for_specialist_issue(self):
        """Specialist-owned issues should target that specialist specifically."""
        role_map = {
            "Synthesizer": "synthesizer",
            "Research Analyst": "research",
        }
        qa = QAResult(
            status="FAIL", reason="Missing data",
            issues=[QAIssue(type="constraint", message="No comparison for Vue", owner="Research Analyst")]
        )
        targets = identify_revision_targets(qa, role_map)
        self.assertIn("research", targets)


# ============================================================
# Test: Task Classification
# ============================================================

class TestTaskClassification(unittest.TestCase):

    def test_factual_question(self):
        self.assertEqual(classify_task("What is the capital of France?"), "factual_question")

    def test_factual_who(self):
        self.assertEqual(classify_task("Who is the CEO of Microsoft?"), "factual_question")

    def test_coding_task(self):
        self.assertEqual(classify_task("Write Python code to parse a CSV and count rows"), "coding_task")

    def test_coding_task_implement(self):
        self.assertEqual(classify_task("Implement a binary search in Rust"), "coding_task")

    def test_creative_writing(self):
        self.assertEqual(classify_task("Write a short poem about rain"), "creative_writing")

    def test_comparison(self):
        self.assertEqual(classify_task("Compare two approaches to urban planning"), "comparison")

    def test_comparison_vs(self):
        self.assertEqual(classify_task("Python vs Rust for web servers"), "comparison")

    def test_summarization(self):
        self.assertEqual(classify_task("Summarize the key findings of the report"), "summarization")

    def test_analysis(self):
        self.assertEqual(classify_task("Evaluate the effectiveness of remote work policies"), "analysis")

    def test_planning(self):
        self.assertEqual(classify_task("Create a roadmap for our product launch"), "planning")

    def test_opinion_discussion(self):
        self.assertEqual(
            classify_task("Discuss the role of black metal music in modern culture from 2 different perspectives"),
            "opinion_discussion",
        )

    def test_other_fallback(self):
        self.assertEqual(classify_task("hello"), "other")

    def test_needs_evidence_factual(self):
        self.assertTrue(task_needs_evidence("factual_question"))

    def test_needs_evidence_comparison(self):
        self.assertTrue(task_needs_evidence("comparison"))

    def test_needs_evidence_analysis(self):
        self.assertTrue(task_needs_evidence("analysis"))

    def test_needs_evidence_summarization(self):
        self.assertTrue(task_needs_evidence("summarization"))

    def test_no_evidence_coding(self):
        self.assertFalse(task_needs_evidence("coding_task"))

    def test_no_evidence_creative(self):
        self.assertFalse(task_needs_evidence("creative_writing"))

    def test_no_evidence_opinion(self):
        self.assertFalse(task_needs_evidence("opinion_discussion"))

    def test_no_evidence_other(self):
        self.assertFalse(task_needs_evidence("other"))


# ============================================================
# Test: Evidence System
# ============================================================

class _MockAdapter(ResearchToolAdapter):
    """Test adapter that returns canned results."""

    def __init__(self, items: list = None, fail: bool = False):
        self._items = items or []
        self._fail = fail

    @property
    def name(self) -> str:
        return "Mock"

    @property
    def source_type(self) -> str:
        return "mock"

    def search(self, query: str) -> list:
        if self._fail:
            raise RuntimeError("mock failure")
        return self._items


class TestEvidenceSystem(unittest.TestCase):

    # --- EvidenceItem ---

    def test_evidence_item_to_dict(self):
        item = EvidenceItem(title="T", source="web", snippet="S", url="http://x")
        d = item.to_dict()
        self.assertEqual(d["title"], "T")
        self.assertEqual(d["url"], "http://x")

    # --- EvidenceResult ---

    def test_evidence_result_empty(self):
        er = EvidenceResult(query="q")
        self.assertFalse(er.has_evidence)
        self.assertEqual(er.confidence, "low")

    def test_evidence_result_has_evidence(self):
        er = EvidenceResult(query="q", results=[
            EvidenceItem(title="T", source="web", snippet="S"),
        ])
        self.assertTrue(er.has_evidence)

    def test_evidence_result_to_dict(self):
        er = EvidenceResult(query="q", results=[
            EvidenceItem(title="T", source="web", snippet="S"),
        ], confidence="medium")
        d = er.to_dict()
        self.assertEqual(d["query"], "q")
        self.assertEqual(len(d["results"]), 1)
        self.assertEqual(d["confidence"], "medium")

    def test_evidence_result_merge(self):
        er1 = EvidenceResult(query="q1", results=[
            EvidenceItem(title="A", source="web", snippet="a"),
        ], confidence="low")
        er2 = EvidenceResult(query="q2", results=[
            EvidenceItem(title="B", source="wiki", snippet="b"),
        ], confidence="high")
        er1.merge(er2)
        self.assertEqual(len(er1.results), 2)
        self.assertEqual(er1.confidence, "high")

    def test_evidence_result_merge_medium_beats_low(self):
        er1 = EvidenceResult(query="q1", confidence="low")
        er2 = EvidenceResult(query="q2", confidence="medium")
        er1.merge(er2)
        self.assertEqual(er1.confidence, "medium")

    # --- gather_evidence ---

    def test_gather_evidence_basic(self):
        items = [
            EvidenceItem(title="R1", source="mock", snippet="data1"),
            EvidenceItem(title="R2", source="mock", snippet="data2"),
        ]
        adapter = _MockAdapter(items=items)
        result = gather_evidence(["test query"], [adapter])
        self.assertEqual(len(result.results), 2)
        self.assertIn(result.confidence, ("medium", "high"))

    def test_gather_evidence_multi_adapter_high_confidence(self):
        web_items = [
            EvidenceItem(title="W1", source="web", snippet="w"),
            EvidenceItem(title="W2", source="web", snippet="w"),
        ]
        wiki_items = [
            EvidenceItem(title="K1", source="wiki", snippet="k"),
        ]
        result = gather_evidence(
            ["test"],
            [_MockAdapter(items=web_items), _MockAdapter(items=wiki_items)],
        )
        self.assertEqual(result.confidence, "high")
        self.assertEqual(len(result.results), 3)

    def test_gather_evidence_empty(self):
        result = gather_evidence(["test"], [_MockAdapter(items=[])])
        self.assertFalse(result.has_evidence)
        self.assertEqual(result.confidence, "low")

    def test_gather_evidence_adapter_failure_graceful(self):
        result = gather_evidence(["test"], [_MockAdapter(fail=True)])
        self.assertFalse(result.has_evidence)
        self.assertEqual(result.confidence, "low")

    # --- extract_search_queries ---

    def test_extract_queries_basic(self):
        queries = extract_search_queries("What are the biggest AI news stories this week?")
        self.assertTrue(len(queries) >= 1)
        self.assertIn("biggest", queries[0].lower())

    def test_extract_queries_with_plan(self):
        plan = "KEY FINDINGS:\n- Large language models are evolving rapidly"
        queries = extract_search_queries("AI news", plan)
        self.assertTrue(len(queries) >= 2)

    def test_extract_queries_max_three(self):
        plan = "KEY FINDINGS:\n- Point one\n- Point two\n- Point three"
        queries = extract_search_queries("query", plan)
        self.assertLessEqual(len(queries), 3)

    # --- detect_unsupported_claims ---

    def test_detect_unsupported_no_claims(self):
        evidence = EvidenceResult(query="q", results=[
            EvidenceItem(title="T", source="web", snippet="general info"),
        ])
        claims = detect_unsupported_claims("This is a normal sentence.", evidence)
        self.assertEqual(len(claims), 0)

    def test_detect_unsupported_fake_citation(self):
        evidence = EvidenceResult(query="q", results=[
            EvidenceItem(title="T", source="web", snippet="some info about cats"),
        ])
        text = 'According to Dr. Smith, published in 2023, the Feline Research Institute found a 15% increase.'
        claims = detect_unsupported_claims(text, evidence)
        # Should detect at least one unsupported claim
        self.assertTrue(len(claims) >= 1)

    # --- format_evidence_for_prompt ---

    def test_format_evidence_no_results(self):
        evidence = EvidenceResult(query="q")
        formatted = format_evidence_for_prompt(evidence)
        self.assertIn("No evidence", formatted)

    def test_format_evidence_with_results(self):
        evidence = EvidenceResult(query="q", results=[
            EvidenceItem(title="Title1", source="web", snippet="Snippet1"),
        ], confidence="medium")
        formatted = format_evidence_for_prompt(evidence)
        self.assertIn("RETRIEVED EVIDENCE", formatted)
        self.assertIn("Title1", formatted)
        self.assertIn("medium", formatted)
        self.assertIn("RULE", formatted)

    # --- format_evidence_for_qa ---

    def test_format_evidence_qa_no_results(self):
        evidence = EvidenceResult(query="q")
        formatted = format_evidence_for_qa(evidence)
        self.assertIn("EVIDENCE VALIDATION", formatted)
        self.assertIn("FAIL", formatted)

    def test_format_evidence_qa_with_results(self):
        evidence = EvidenceResult(query="q", results=[
            EvidenceItem(title="T", source="web", snippet="S"),
        ], confidence="high")
        formatted = format_evidence_for_qa(evidence)
        self.assertIn("1 items retrieved", formatted)
        self.assertIn("high", formatted)


# ============================================================
# Test: Failure Tracking & Escalation
# ============================================================

class TestFailureTracking(unittest.TestCase):

    def test_failure_record_to_dict(self):
        fr = FailureRecord(revision=0, owner="research", issue_type="accuracy",
                           message="Wrong fact", correction="Fix it")
        d = fr.to_dict()
        self.assertEqual(d["revision"], 0)
        self.assertEqual(d["owner"], "research")
        self.assertEqual(d["issue_type"], "accuracy")

    def test_record_failure_from_qa(self):
        ps = PlannerState(user_request="test")
        qa = QAResult(
            status="FAIL", reason="Issues found",
            correction_instruction="Correct it",
            issues=[
                QAIssue(type="accuracy", owner="research",
                        message="Wrong data"),
            ],
        )
        ps.record_failure(qa)
        self.assertEqual(len(ps.failure_history), 1)
        self.assertEqual(ps.failure_history[0].owner, "research")

    def test_has_repeated_failure(self):
        ps = PlannerState(user_request="test")
        # First failure at revision 0
        qa = QAResult(
            status="FAIL", reason="bad",
            correction_instruction="fix",
            issues=[
                QAIssue(type="accuracy", owner="research",
                        message="err"),
            ],
        )
        ps.record_failure(qa)
        # Advance revision
        ps.revision_count = 1
        # Same failure should be detected
        self.assertTrue(ps.has_repeated_failure("research", "accuracy"))

    def test_no_repeated_failure(self):
        ps = PlannerState(user_request="test")
        self.assertFalse(ps.has_repeated_failure("research", "accuracy"))

    def test_get_repeat_failures(self):
        ps = PlannerState(user_request="test")
        # Add same failure type twice
        qa = QAResult(
            status="FAIL", reason="bad",
            correction_instruction="fix",
            issues=[
                QAIssue(type="accuracy", owner="research",
                        message="err"),
            ],
        )
        ps.record_failure(qa)
        ps.revision_count = 1
        ps.record_failure(qa)
        repeats = ps.get_repeat_failures()
        self.assertIn(("research", "accuracy"), repeats)

    def test_escalation_none(self):
        ps = PlannerState(user_request="test")
        self.assertEqual(ps.get_escalation_strategy(), "none")

    def test_escalation_suppress_role(self):
        ps = PlannerState(user_request="test")
        qa = QAResult(
            status="FAIL", reason="bad",
            correction_instruction="fix",
            issues=[
                QAIssue(type="accuracy", owner="research",
                        message="err"),
            ],
        )
        ps.record_failure(qa)
        ps.revision_count = 1
        ps.record_failure(qa)
        self.assertEqual(ps.get_escalation_strategy(), "suppress_role")

    def test_escalation_rewrite_from_state(self):
        ps = PlannerState(user_request="test")
        qa = QAResult(
            status="FAIL", reason="bad",
            correction_instruction="fix format",
            issues=[
                QAIssue(type="format", owner="synthesizer",
                        message="wrong format"),
            ],
        )
        ps.record_failure(qa)
        ps.revision_count = 1
        ps.record_failure(qa)
        self.assertEqual(ps.get_escalation_strategy(), "rewrite_from_state")

    def test_get_roles_to_suppress(self):
        ps = PlannerState(user_request="test")
        qa = QAResult(
            status="FAIL", reason="bad",
            correction_instruction="fix",
            issues=[
                QAIssue(type="accuracy", owner="creative",
                        message="err"),
            ],
        )
        ps.record_failure(qa)
        ps.revision_count = 1
        ps.record_failure(qa)
        suppressed = ps.get_roles_to_suppress()
        self.assertIn("creative", suppressed)
        self.assertNotIn("synthesizer", suppressed)


# ============================================================
# Test: Extended PlannerState
# ============================================================

class TestPlannerStateExtended(unittest.TestCase):

    def test_to_state_dict(self):
        ps = PlannerState(
            user_request="test req",
            task_category="factual_question",
            selected_roles=["research", "technical"],
        )
        ps.specialist_outputs["research"] = "some output"
        d = ps.to_state_dict()
        self.assertEqual(d["task_category"], "factual_question")
        self.assertEqual(d["specialist_outputs"]["research"], "some output")
        self.assertIn("selected_roles", d)

    def test_to_state_dict_truncates_draft(self):
        ps = PlannerState(user_request="test")
        ps.current_draft = "X" * 1000
        d = ps.to_state_dict()
        self.assertEqual(len(d["current_draft"]), 500)

    def test_evidence_field(self):
        ps = PlannerState(user_request="test")
        er = EvidenceResult(query="q", results=[
            EvidenceItem(title="T", source="web", snippet="S"),
        ])
        ps.evidence = er.to_dict()
        self.assertIsNotNone(ps.evidence)
        self.assertEqual(ps.evidence["query"], "q")

    def test_context_string_includes_category(self):
        ps = PlannerState(
            user_request="test",
            task_category="comparison",
        )
        ctx = ps.to_context_string()
        self.assertIn("comparison", ctx)

    def test_context_string_includes_failures(self):
        ps = PlannerState(user_request="test")
        qa = QAResult(
            status="FAIL", reason="bad",
            correction_instruction="fix",
            issues=[
                QAIssue(type="accuracy", owner="research",
                        message="err"),
            ],
        )
        ps.record_failure(qa)
        ctx = ps.to_context_string()
        self.assertIn("failure", ctx.lower())

    def test_context_string_includes_escalation(self):
        ps = PlannerState(user_request="test")
        qa = QAResult(
            status="FAIL", reason="bad",
            correction_instruction="fix",
            issues=[
                QAIssue(type="accuracy", owner="research",
                        message="err"),
            ],
        )
        ps.record_failure(qa)
        ps.revision_count = 1
        ps.record_failure(qa)
        ctx = ps.to_context_string()
        self.assertIn("suppress_role", ctx)

    def test_final_answer_tracking(self):
        ps = PlannerState(user_request="test")
        ps.final_answer = "The answer is 42."
        self.assertEqual(ps.final_answer, "The answer is 42.")
        d = ps.to_state_dict()
        self.assertEqual(d["final_answer"], "The answer is 42.")


# ============================================================
# Test: Scenario - Role Selection with Task Categories
# ============================================================

# ============================================================
# Test: Output Format Validation
# ============================================================

class TestFormatValidation(unittest.TestCase):

    def test_paragraph_with_bullets_fails(self):
        text = "This is a paragraph.\n- bullet one\n- bullet two"
        violations = validate_output_format(text, "paragraph", "normal")
        self.assertTrue(any("bullet" in v.lower() for v in violations))

    def test_paragraph_with_headings_fails(self):
        text = "## Heading\nSome paragraph text."
        violations = validate_output_format(text, "paragraph", "normal")
        self.assertTrue(any("heading" in v.lower() for v in violations))

    def test_paragraph_with_table_fails(self):
        text = "Some text.\n| A | B |\n|---|---|\n| 1 | 2 |"
        violations = validate_output_format(text, "paragraph", "normal")
        self.assertTrue(any("table" in v.lower() for v in violations))

    def test_paragraph_clean_passes(self):
        text = "This is a clean paragraph without any lists or headings."
        violations = validate_output_format(text, "paragraph", "normal")
        self.assertEqual(violations, [])

    def test_code_without_code_fails(self):
        text = "Here is an explanation about coding but no actual code."
        violations = validate_output_format(text, "code", "normal")
        self.assertTrue(any("code" in v.lower() for v in violations))

    def test_code_with_block_passes(self):
        text = "```python\nprint('hello')\n```"
        violations = validate_output_format(text, "code", "normal")
        self.assertEqual(violations, [])

    def test_code_with_recognisable_code_passes(self):
        text = "def hello():\n    return 'world'"
        violations = validate_output_format(text, "code", "normal")
        self.assertEqual(violations, [])

    def test_table_without_table_fails(self):
        text = "Just a paragraph about tables."
        violations = validate_output_format(text, "table", "normal")
        self.assertTrue(any("table" in v.lower() for v in violations))

    def test_table_with_table_passes(self):
        text = "| Name | Value |\n|------|-------|\n| A | 1 |"
        violations = validate_output_format(text, "table", "normal")
        self.assertEqual(violations, [])

    def test_single_choice_too_many_lines_fails(self):
        text = "\n".join(f"Line {i}" for i in range(10))
        violations = validate_output_format(text, "single_choice", "normal")
        self.assertTrue(any("single choice" in v.lower() for v in violations))

    def test_single_choice_short_passes(self):
        text = "Vegan is the best choice."
        violations = validate_output_format(text, "single_choice", "normal")
        self.assertEqual(violations, [])

    def test_minimal_brevity_too_long(self):
        text = "\n".join(f"Line {i}" for i in range(12))
        violations = validate_output_format(text, "paragraph", "minimal")
        self.assertTrue(any("minimal" in v.lower() for v in violations))

    def test_short_brevity_too_long(self):
        text = "\n".join(f"Line {i}" for i in range(25))
        violations = validate_output_format(text, "paragraph", "short")
        self.assertTrue(any("short" in v.lower() for v in violations))

    def test_normal_brevity_no_length_check(self):
        text = "\n".join(f"Line {i}" for i in range(50))
        violations = validate_output_format(text, "paragraph", "normal")
        self.assertEqual(violations, [])

    def test_empty_output(self):
        violations = validate_output_format("", "paragraph", "normal")
        self.assertTrue(any("empty" in v.lower() for v in violations))


class TestFormatViolationsInstruction(unittest.TestCase):

    def test_produces_instruction(self):
        violations = ["Output has bullets.", "Too many lines."]
        result = format_violations_instruction(violations)
        self.assertIn("FORMAT VIOLATIONS", result)
        self.assertIn("Output has bullets.", result)
        self.assertIn("Too many lines.", result)
        self.assertIn("Rewrite", result)

    def test_empty_violations(self):
        result = format_violations_instruction([])
        self.assertIn("FORMAT VIOLATIONS", result)


# ============================================================
# Test: Task Assumptions Parsing
# ============================================================

class TestTaskAssumptions(unittest.TestCase):

    def test_parse_assumptions_basic(self):
        plan = (
            "TASK ASSUMPTIONS:\n"
            "- cost_model: per-unit pricing\n"
            "- coverage_rate: 95%\n"
            "- time_frame: 2024 Q4\n"
            "TASK BREAKDOWN:\n"
            "1. Do the thing"
        )
        result = parse_task_assumptions(plan)
        self.assertEqual(result["cost_model"], "per-unit pricing")
        self.assertEqual(result["coverage_rate"], "95%")
        self.assertEqual(result["time_frame"], "2024 Q4")

    def test_parse_assumptions_missing_section(self):
        plan = "TASK BREAKDOWN:\n1. Do the thing"
        result = parse_task_assumptions(plan)
        self.assertEqual(result, {})

    def test_parse_assumptions_multiple_headers(self):
        plan = (
            "TASK ASSUMPTIONS:\n"
            "units: metric\n"
            "scope: global\n"
            "ROLE TO CALL:\n"
            "Technical Specialist"
        )
        result = parse_task_assumptions(plan)
        self.assertEqual(result["units"], "metric")
        self.assertEqual(result["scope"], "global")
        self.assertNotIn("technical_specialist", result)

    def test_parse_assumptions_normalises_keys(self):
        plan = "TASK ASSUMPTIONS:\nCost Model: expensive\n"
        result = parse_task_assumptions(plan)
        self.assertIn("cost_model", result)

    def test_format_assumptions_empty(self):
        result = format_assumptions_for_prompt({})
        self.assertEqual(result, "")

    def test_format_assumptions_nonempty(self):
        result = format_assumptions_for_prompt({"units": "metric", "scope": "global"})
        self.assertIn("SHARED TASK ASSUMPTIONS", result)
        self.assertIn("units: metric", result)
        self.assertIn("scope: global", result)
        self.assertIn("do NOT invent your own", result)


# ============================================================
# Test: PlannerState Assumptions & Revision Instruction
# ============================================================

class TestPlannerStateNewFields(unittest.TestCase):

    def test_task_assumptions_in_state_dict(self):
        ps = PlannerState(user_request="test")
        ps.task_assumptions = {"units": "metric", "scope": "global"}
        d = ps.to_state_dict()
        self.assertEqual(d["task_assumptions"], {"units": "metric", "scope": "global"})

    def test_revision_instruction_in_state_dict(self):
        ps = PlannerState(user_request="test")
        ps.revision_instruction = "Fix the table format."
        d = ps.to_state_dict()
        self.assertEqual(d["revision_instruction"], "Fix the table format.")

    def test_task_assumptions_in_context_string(self):
        ps = PlannerState(user_request="test")
        ps.task_assumptions = {"rate": "5%"}
        ctx = ps.to_context_string()
        self.assertIn("rate: 5%", ctx)
        self.assertIn("Shared assumptions", ctx)

    def test_revision_instruction_in_context_string(self):
        ps = PlannerState(user_request="test")
        ps.revision_instruction = "Shorten the output."
        ctx = ps.to_context_string()
        self.assertIn("Shorten the output.", ctx)

    def test_empty_assumptions_not_in_context(self):
        ps = PlannerState(user_request="test")
        ctx = ps.to_context_string()
        self.assertNotIn("Shared assumptions", ctx)


# ============================================================
# Test: Task-Aware Scenarios
# ============================================================

class TestTaskAwareScenarios(unittest.TestCase):
    """End-to-end scenario tests validating the 4 user-specified cases."""

    def _all_roles(self):
        return list(WorkflowConfig.CORE_ROLE_KEYS) + list(WorkflowConfig.PERSONA_ROLE_KEYS)

    # Scenario 1: Black metal discussion
    def test_black_metal_discussion(self):
        req = "discuss the role of black metal music in modern culture from 2 different perspectives"
        cat = classify_task(req)
        self.assertEqual(cat, "opinion_discussion")
        # Opinion discussion does not need evidence
        self.assertFalse(task_needs_evidence(cat))
        # Strict mode, no personas
        config = WorkflowConfig(strict_mode=True, allow_persona_roles=False)
        roles = select_relevant_roles(req, self._all_roles(), config, task_category=cat)
        # Should not include persona roles
        for persona in WorkflowConfig.PERSONA_ROLE_KEYS:
            self.assertNotIn(persona, roles)
        # Should have minimal roles (<=3)
        self.assertLessEqual(len(roles), config.max_specialists_per_task)

    # Scenario 2: AI news stories
    def test_ai_news_stories(self):
        req = "what are the three biggest AI news stories this week?"
        cat = classify_task(req)
        self.assertEqual(cat, "factual_question")
        # Factual question needs evidence
        self.assertTrue(task_needs_evidence(cat))
        # Research should be auto-included
        config = WorkflowConfig(always_include_research_for_factual_tasks=True)
        roles = select_relevant_roles(req, self._all_roles(), config, task_category=cat)
        self.assertIn("research", roles)

    # Scenario 3: Python CSV code
    def test_python_csv_code(self):
        req = "write Python code to parse a CSV and count rows"
        cat = classify_task(req)
        self.assertEqual(cat, "coding_task")
        # Coding task does not need evidence
        self.assertFalse(task_needs_evidence(cat))
        # Technical should be preferred
        config = WorkflowConfig(strict_mode=True)
        roles = select_relevant_roles(req, self._all_roles(), config, task_category=cat)
        self.assertIn("technical", roles)
        # Output format should be code
        fmt = detect_output_format(req)
        self.assertEqual(fmt, "code")

    # Scenario 4: Urban planning comparison
    def test_urban_planning_comparison(self):
        req = "compare two approaches to urban planning in one short paragraph"
        cat = classify_task(req)
        self.assertEqual(cat, "comparison")
        # Comparison needs evidence
        self.assertTrue(task_needs_evidence(cat))
        # Brevity should be short
        brevity = detect_brevity_requirement(req)
        self.assertIn(brevity, ("short", "minimal"))
        # Should have minimal roles
        config = WorkflowConfig(strict_mode=True, max_specialists_per_task=3)
        roles = select_relevant_roles(req, self._all_roles(), config, task_category=cat)
        self.assertLessEqual(len(roles), 3)


# ============================================================
# Structured Contribution Tests
# ============================================================

class TestStructuredContribution(unittest.TestCase):
    """Tests for StructuredContribution dataclass and parse_structured_contribution."""

    def test_parse_json_block(self):
        """JSON block in specialist output is parsed correctly."""
        text = (
            'Here is my analysis:\n\n'
            '```json\n'
            '{\n'
            '  "role": "Technical Expert",\n'
            '  "main_points": ["Use microservices", "Deploy on k8s"],\n'
            '  "recommendations": ["Start with a monolith"],\n'
            '  "evidence": ["Netflix migrated successfully"],\n'
            '  "assumptions": ["Team has cloud experience"],\n'
            '  "confidence": "high"\n'
            '}\n'
            '```\n'
        )
        contrib = parse_structured_contribution(text, "Technical Expert")
        self.assertEqual(contrib.role, "Technical Expert")
        self.assertEqual(len(contrib.main_points), 2)
        self.assertIn("Use microservices", contrib.main_points)
        self.assertEqual(contrib.recommendations, ["Start with a monolith"])
        self.assertEqual(contrib.confidence, "high")
        self.assertTrue(contrib.has_substance())

    def test_parse_bare_json(self):
        """Bare JSON object (no fences) is parsed."""
        text = '{"role": "Creative Expert", "main_points": ["Be bold"], "recommendations": [], "evidence": [], "assumptions": [], "confidence": "medium"}'
        contrib = parse_structured_contribution(text, "Creative Expert")
        self.assertEqual(contrib.main_points, ["Be bold"])
        self.assertEqual(contrib.confidence, "medium")

    def test_parse_fallback_heuristic(self):
        """When no JSON is present, heuristic extraction from section headers works."""
        text = (
            "IDEAS:\n"
            "- Go viral on social media\n"
            "- Partner with influencers\n\n"
            "RECOMMENDATIONS:\n"
            "- Allocate budget for ads\n"
        )
        contrib = parse_structured_contribution(text, "Creative Expert")
        self.assertEqual(contrib.role, "Creative Expert")
        # Should have extracted something via heuristic
        self.assertTrue(len(contrib.main_points) > 0 or len(contrib.recommendations) > 0)

    def test_parse_malformed_json(self):
        """Malformed JSON falls back to heuristic without raising."""
        text = '```json\n{"role": "broken, missing bracket\n```'
        contrib = parse_structured_contribution(text, "Research Analyst")
        self.assertEqual(contrib.role, "Research Analyst")
        self.assertEqual(contrib.raw_output, text)
        # Should not raise — just return empty contribution

    def test_has_substance_empty(self):
        """Empty contribution reports no substance."""
        contrib = StructuredContribution(role="Test")
        self.assertFalse(contrib.has_substance())

    def test_to_dict(self):
        """to_dict serializes correctly."""
        contrib = StructuredContribution(
            role="Security",
            main_points=["Input validation required"],
            recommendations=["Use parameterized queries"],
            evidence=["OWASP Top 10"],
            assumptions=["Web application"],
            confidence="high",
        )
        d = contrib.to_dict()
        self.assertEqual(d["role"], "Security")
        self.assertEqual(len(d["main_points"]), 1)
        self.assertEqual(d["confidence"], "high")
        self.assertNotIn("raw_output", d)


class TestFormatContributions(unittest.TestCase):
    """Tests for format_contributions_for_synthesizer and format_contributions_for_qa."""

    def _make_contributions(self):
        return {
            "creative": StructuredContribution(
                role="Creative Expert",
                main_points=["Bold campaign", "Use humor"],
                recommendations=["A/B test messaging"],
                confidence="high",
            ),
            "technical": StructuredContribution(
                role="Technical Expert",
                main_points=["Use React"],
                recommendations=["Add caching"],
                evidence=["React has 200k+ stars"],
                confidence="medium",
            ),
        }

    def test_format_for_synthesizer(self):
        contribs = self._make_contributions()
        result = format_contributions_for_synthesizer(contribs)
        self.assertIn("STRUCTURED EXPERT CONTRIBUTIONS", result)
        self.assertIn("Creative Expert", result)
        self.assertIn("Technical Expert", result)
        self.assertIn("[0] Bold campaign", result)
        self.assertIn("[0] Use React", result)
        self.assertIn("confidence: high", result)

    def test_format_for_synthesizer_empty(self):
        self.assertEqual(format_contributions_for_synthesizer({}), "")

    def test_format_for_qa_used(self):
        contribs = self._make_contributions()
        used = {"creative": ["main_points[0]"], "technical": []}
        result = format_contributions_for_qa(contribs, used)
        self.assertIn("[USED]", result)
        self.assertIn("[NOT USED]", result)
        self.assertIn("EXPERT CONTRIBUTION TRACEABILITY", result)

    def test_format_for_qa_unused(self):
        contribs = self._make_contributions()
        result = format_contributions_for_qa(contribs, {})
        self.assertIn("[NOT USED]", result)
        # All should be NOT USED
        self.assertNotIn("[USED]:", result)


class TestParseUsedContributions(unittest.TestCase):
    """Tests for parse_used_contributions."""

    def test_parse_json_block(self):
        text = (
            "Here is the final answer.\n\n"
            "```json\n"
            '{"used_contributions": {"creative": ["main_points[0]"], "technical": ["recommendations[0]"]}}\n'
            "```\n"
        )
        used = parse_used_contributions(text)
        self.assertIn("creative", used)
        self.assertEqual(used["creative"], ["main_points[0]"])
        self.assertEqual(used["technical"], ["recommendations[0]"])

    def test_parse_used_contributions_section(self):
        text = (
            "Great answer here.\n\n"
            'USED_CONTRIBUTIONS: {"creative": ["main_points[0]", "main_points[1]"]}\n'
        )
        used = parse_used_contributions(text)
        self.assertIn("creative", used)
        self.assertEqual(len(used["creative"]), 2)

    def test_parse_empty(self):
        used = parse_used_contributions("No contributions block here.")
        self.assertEqual(used, {})


class TestCheckExpertInfluence(unittest.TestCase):
    """Tests for check_expert_influence."""

    def _make_contributions(self):
        return {
            "creative": StructuredContribution(
                role="Creative Expert",
                main_points=["Use guerrilla marketing tactics"],
                recommendations=["Target social media"],
                confidence="high",
            ),
            "technical": StructuredContribution(
                role="Technical Expert",
                main_points=["Implement REST API with caching"],
                recommendations=["Use Redis for sessions"],
                confidence="medium",
            ),
        }

    def test_no_contributions_used(self):
        contribs = self._make_contributions()
        issues = check_expert_influence(contribs, {}, "Some generic answer.")
        self.assertTrue(len(issues) > 0)
        self.assertTrue(any("not materially" in i.lower() or "none were used" in i.lower() for i in issues))

    def test_adequate_influence(self):
        contribs = self._make_contributions()
        used = {
            "creative": ["main_points[0]"],
            "technical": ["main_points[0]"],
        }
        # Answer includes expert vocabulary
        answer = "We recommend guerrilla marketing tactics and implementing a REST API with caching."
        issues = check_expert_influence(contribs, used, answer)
        self.assertEqual(issues, [])

    def test_missing_expert(self):
        contribs = self._make_contributions()
        used = {"creative": ["main_points[0]"]}  # technical not used
        answer = "Use guerrilla marketing tactics for the campaign."
        issues = check_expert_influence(contribs, used, answer)
        # Should flag that technical expert was not used
        self.assertTrue(any("Technical Expert" in i for i in issues))

    def test_empty_contributions(self):
        issues = check_expert_influence({}, {}, "Any answer")
        self.assertEqual(issues, [])


class TestNorwegianPromptScenario(unittest.TestCase):
    """Test the Norwegian prompt scenario requested by the user.

    Prompt: "hva er klokken nå, og når bør jeg legge meg om jeg er en black metal fan?"
    This should classify appropriately, select black_metal_fundamentalist, and produce
    structured contributions.
    """

    def test_classification(self):
        req = "hva er klokken nå, og når bør jeg legge meg om jeg er en black metal fan?"
        cat = classify_task(req)
        # Should be classified as general or creative (it's a lifestyle question)
        self.assertIn(cat, ("general", "creative", "factual", "opinion", "other"))

    def test_role_selection_includes_black_metal(self):
        req = "hva er klokken nå, og når bør jeg legge meg om jeg er en black metal fan?"
        all_roles = [
            "creative", "technical", "research", "security", "data_analyst",
            "mad_professor", "accountant", "artist", "lazy_slacker",
            "black_metal_fundamentalist", "labour_union_rep", "ux_designer",
            "doris", "chairman_of_board", "maga_appointee", "lawyer",
        ]
        config = WorkflowConfig(strict_mode=True, allow_persona_roles=True, max_specialists_per_task=5)
        cat = classify_task(req)
        roles = select_relevant_roles(req, all_roles, config, task_category=cat)
        self.assertIn("black_metal_fundamentalist", roles,
                       "black_metal_fundamentalist should be selected for a prompt mentioning 'black metal fan'")

    def test_structured_contribution_parsing_from_black_metal_output(self):
        """Simulate black metal specialist output and verify structured contribution parsing."""
        output = (
            "KVLT VERDICT:\n"
            "The true kvltist sleeps when the moon commands. Bedtime is for posers "
            "who follow society's weak schedules.\n\n"
            "THE GRIM TRUTH:\n"
            "Time is an illusion created by the false light of day.\n\n"
            '```json\n'
            '{\n'
            '  "role": "Black Metal Fundamentalist",\n'
            '  "main_points": [\n'
            '    "True kvltists sleep only when the moon commands",\n'
            '    "Bedtime schedules are for posers and conformists"\n'
            '  ],\n'
            '  "recommendations": [\n'
            '    "Sleep at dawn, rise at dusk — embrace the nocturnal path"\n'
            '  ],\n'
            '  "evidence": [\n'
            '    "Norwegian black metal musicians are known for nocturnal lifestyles"\n'
            '  ],\n'
            '  "assumptions": [\n'
            '    "The user seeks the true kvlt path, not mainstream advice"\n'
            '  ],\n'
            '  "confidence": "high"\n'
            '}\n'
            '```\n'
        )
        contrib = parse_structured_contribution(output, "Black Metal Fundamentalist")
        self.assertEqual(contrib.role, "Black Metal Fundamentalist")
        self.assertEqual(len(contrib.main_points), 2)
        self.assertIn("kvltists", contrib.main_points[0].lower())
        self.assertEqual(len(contrib.recommendations), 1)
        self.assertTrue(contrib.has_substance())
        self.assertEqual(contrib.confidence, "high")


if __name__ == "__main__":
    unittest.main()
