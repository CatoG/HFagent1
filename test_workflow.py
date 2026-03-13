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


if __name__ == "__main__":
    unittest.main()
