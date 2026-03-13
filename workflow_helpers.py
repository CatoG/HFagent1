"""
Workflow helper functions for the multi-role AI workflow.

Contains:
- WorkflowConfig: configuration flags (strict_mode, allow_persona_roles, etc.)
- Output format intent detection
- Structured QA result parsing
- Role relevance metadata and selection
- Targeted revision logic
- Final answer compression / noise stripping
- PlannerState management
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Workflow Configuration
# ============================================================

@dataclass
class WorkflowConfig:
    """Runtime config flags for the multi-role workflow."""
    allow_persona_roles: bool = False
    max_specialists_per_task: int = 3
    strict_mode: bool = True
    always_include_qa: bool = True
    always_include_research_for_factual_tasks: bool = True
    require_evidence_for_factual_claims: bool = True

    # Internal: which role keys are persona/gimmick roles
    PERSONA_ROLE_KEYS: tuple = (
        "mad_professor", "accountant", "artist", "lazy_slacker",
        "black_metal_fundamentalist", "doris", "maga_appointee",
        "chairman_of_board",
    )

    # Internal: core professional role keys
    CORE_ROLE_KEYS: tuple = (
        "creative", "technical", "research", "security",
        "data_analyst", "labour_union_rep", "ux_designer", "lawyer",
    )


DEFAULT_CONFIG = WorkflowConfig()


# ============================================================
# Task Classification
# ============================================================

TASK_CATEGORIES = [
    "factual_question",
    "comparison",
    "coding_task",
    "creative_writing",
    "opinion_discussion",
    "summarization",
    "analysis",
    "planning",
]

_TASK_CATEGORY_PATTERNS = [
    ("coding_task", [
        r"\bwrite\s+(python|code|javascript|typescript|rust|java|c\+\+|go|bash|sql)\b",
        r"\bcode\s+(to|for|that)\b", r"\bimplement\b",
        r"\bscript\s+(to|for|that)\b", r"\bparse\s+a?\s*\w+\s+(file|data)\b",
        r"\bdebug\b", r"\brefactor\b", r"\bfix\s+(the|this|my)\s+(code|bug)\b",
    ]),
    ("creative_writing", [
        r"\bwrite\s+a\b.*\b(poem|story|essay|blog|article|song|haiku)\b",
        r"\bcreative\s+writing\b", r"\bbrainstorm\b", r"\bimagine\b",
        r"\bfiction\b", r"\bnarrative\b",
    ]),
    ("factual_question", [
        r"\bwhat\s+(is|are|was|were)\b", r"\bwho\s+(is|was|are|were)\b",
        r"\bwhen\s+(did|was|is)\b", r"\bwhere\s+(is|was|are)\b",
        r"\bhow\s+many\b", r"\bhow\s+much\b",
        r"\bnews\b", r"\brecent\b", r"\blatest\b", r"\bcurrent\b",
        r"\bfact\b", r"\btrue\s+or\s+false\b",
    ]),
    ("comparison", [
        r"\bcompar(e|ison|ing)\b", r"\bvs\.?\b", r"\bversus\b",
        r"\bdifference\s+between\b", r"\bbetter\s+than\b",
        r"\bwhich\s+is\s+(better|faster|cheaper)\b",
        r"\bpros?\s+and\s+cons?\b", r"\btrade[\s-]?offs?\b",
    ]),
    ("summarization", [
        r"\bsummar(y|ize|ise)\b", r"\btl;?dr\b", r"\bsynopsis\b",
        r"\boverview\b", r"\brecap\b",
    ]),
    ("analysis", [
        r"\banaly(sis|se|ze)\b", r"\bevaluat(e|ion)\b",
        r"\bassess(ment)?\b", r"\breview\b",
        r"\bexamin(e|ation)\b", r"\binvestigat(e|ion)\b",
    ]),
    ("planning", [
        r"\bplan\b", r"\bstrateg(y|ic)\b", r"\broadmap\b",
        r"\baction\s+items?\b", r"\bsteps?\s+to\b",
    ]),
    ("opinion_discussion", [
        r"\bdiscuss\b", r"\bopinion\b", r"\bperspective\b",
        r"\bpoint\s+of\s+view\b", r"\bargue\b", r"\bdebate\b",
        r"\brole\s+of\b",
    ]),
]


def classify_task(user_request: str) -> str:
    """Classify the user's request into a task category.

    Returns one of: factual_question, comparison, coding_task, creative_writing,
    opinion_discussion, summarization, analysis, planning, other.
    """
    lower = user_request.lower()
    best_category = "other"
    best_score = 0
    for category, patterns in _TASK_CATEGORY_PATTERNS:
        score = 0
        for pat in patterns:
            if re.search(pat, lower):
                score += 1
        if score > best_score:
            best_score = score
            best_category = category
    return best_category


def task_needs_evidence(task_category: str) -> bool:
    """Whether this task category benefits from tool-backed evidence retrieval."""
    return task_category in ("factual_question", "comparison", "analysis", "summarization")


# ============================================================
# Output Format Detection
# ============================================================

# Ordered list of (format_name, patterns) — first match wins
_FORMAT_PATTERNS = [
    ("single_choice", [
        r"\bpick\s+one\b", r"\bchoose\s+one\b", r"\bagree\s+on\s+one\b",
        r"\bselect\s+one\b", r"\bjust\s+one\b", r"\bone\s+choice\b",
        r"\bwhich\s+one\b", r"\bone\s+word\b",
    ]),
    ("code", [
        r"\bwrite\s+(python|code|javascript|typescript|rust|java|c\+\+|go|bash|sql)\b",
        r"\bcode\s+(to|for|that)\b", r"\bimplement\b.*\b(function|class|method|script)\b",
        r"\bgive\s+me\s+(the\s+)?code\b", r"\bscript\s+(to|for|that)\b",
    ]),
    ("table", [
        r"\bmake\s+a\s+table\b", r"\bcreate\s+a\s+table\b", r"\btable\s+comparing\b",
        r"\bcomparison\s+table\b", r"\btabular\b", r"\bin\s+table\s+form\b",
    ]),
    ("bullet_list", [
        r"\bbullet\s*(ed)?\s*(point|list)\b", r"\blist\s+(the|all|some)\b",
        r"\bgive\s+me\s+a\s+list\b",
    ]),
    ("short_answer", [
        r"\bshort\s+answer\b", r"\bbrief(ly)?\b", r"\bconcise(ly)?\b",
        r"\bin\s+one\s+sentence\b", r"\byes\s+or\s+no\b", r"\bquick\s+answer\b",
        r"\banswer\s+briefly\b", r"\bkeep\s+it\s+short\b",
    ]),
    ("report", [
        r"\breport\b", r"\banalysis\b", r"\bin[\s-]depth\b", r"\bdetailed\b",
        r"\bcomprehensive\b", r"\btrade[\s-]?offs?\b", r"\bpros?\s+and\s+cons?\b",
    ]),
    ("paragraph", [
        r"\bexplain\b", r"\bdescribe\b", r"\bparagraph\b",
    ]),
]


def detect_output_format(user_request: str) -> str:
    """Classify the expected output format from the user's request text.

    Returns one of: single_choice, short_answer, paragraph, bullet_list,
    table, report, code, other.
    """
    lower = user_request.lower()
    for fmt, patterns in _FORMAT_PATTERNS:
        for pat in patterns:
            if re.search(pat, lower):
                return fmt
    return "other"


def detect_brevity_requirement(user_request: str) -> str:
    """Detect how brief the answer should be.

    Returns: 'minimal', 'short', 'normal', or 'verbose'.
    """
    lower = user_request.lower()

    minimal_signals = [
        r"\bjust\s+(one|the)\b", r"\bone\s+word\b", r"\byes\s+or\s+no\b",
        r"\bpick\s+one\b", r"\bchoose\s+one\b", r"\bagree\s+on\s+one\b",
    ]
    for pat in minimal_signals:
        if re.search(pat, lower):
            return "minimal"

    short_signals = [
        r"\bshort\b", r"\bbrief(ly)?\b", r"\bconcise(ly)?\b",
        r"\bquick\b", r"\bsimple\b", r"\bkeep\s+it\s+short\b",
    ]
    for pat in short_signals:
        if re.search(pat, lower):
            return "short"

    verbose_signals = [
        r"\bdetailed\b", r"\bin[\s-]depth\b", r"\bcomprehensive\b",
        r"\bthorough(ly)?\b", r"\bfull\s+report\b",
    ]
    for pat in verbose_signals:
        if re.search(pat, lower):
            return "verbose"

    return "normal"


# ============================================================
# Structured QA Result
# ============================================================

@dataclass
class QAIssue:
    type: str       # format | brevity | constraint | consistency | directness | other
    message: str
    owner: str      # "Synthesizer" | "Planner" | specialist role display name


@dataclass
class QAResult:
    status: str                          # "PASS" | "FAIL"
    reason: str = ""
    issues: List[QAIssue] = field(default_factory=list)
    correction_instruction: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def owners(self) -> List[str]:
        """Return unique owner labels from issues."""
        return list(dict.fromkeys(issue.owner for issue in self.issues))

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "issues": [
                {"type": i.type, "message": i.message, "owner": i.owner}
                for i in self.issues
            ],
            "correction_instruction": self.correction_instruction,
        }


def parse_structured_qa(qa_text: str) -> QAResult:
    """Parse QA output into a structured QAResult.

    Tries JSON first (if QA produced structured output),
    then falls back to the legacy text format.
    """
    # Try to extract JSON from the QA output
    json_match = re.search(r'\{[\s\S]*"status"\s*:', qa_text)
    if json_match:
        # Find the matching closing brace
        start = json_match.start()
        brace_count = 0
        end = start
        for i, ch in enumerate(qa_text[start:], start):
            if ch == '{':
                brace_count += 1
            elif ch == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break
        try:
            data = json.loads(qa_text[start:end])
            issues = []
            for item in data.get("issues", []):
                issues.append(QAIssue(
                    type=item.get("type", "other"),
                    message=item.get("message", ""),
                    owner=item.get("owner", "Synthesizer"),
                ))
            return QAResult(
                status=data.get("status", "FAIL"),
                reason=data.get("reason", ""),
                issues=issues,
                correction_instruction=data.get("correction_instruction", ""),
            )
        except (json.JSONDecodeError, KeyError):
            pass

    # Fallback: parse from legacy text format
    status = "FAIL"
    lower = qa_text.lower()
    if "result: pass" in lower:
        status = "PASS"

    reason = ""
    if "ISSUES FOUND:" in qa_text:
        section = qa_text.split("ISSUES FOUND:", 1)[1]
        for header in ("ROLE-SPECIFIC FEEDBACK:", "RESULT:", "RECOMMENDED FIXES:"):
            if header in section:
                section = section.split(header, 1)[0]
                break
        reason = section.strip()

    correction = ""
    if "RECOMMENDED FIXES:" in qa_text:
        correction = qa_text.split("RECOMMENDED FIXES:", 1)[1].strip()

    # Build issues from role-specific feedback section
    issues = []
    if "ROLE-SPECIFIC FEEDBACK:" in qa_text:
        fb_section = qa_text.split("ROLE-SPECIFIC FEEDBACK:", 1)[1]
        for header in ("RESULT:", "RECOMMENDED FIXES:"):
            if header in fb_section:
                fb_section = fb_section.split(header, 1)[0]
                break
        for line in fb_section.strip().splitlines():
            line = line.strip().lstrip("•-* ")
            if ":" not in line:
                continue
            role_label, _, feedback = line.partition(":")
            feedback = feedback.strip()
            if feedback and feedback.lower() not in ("satisfactory", "n/a", "none"):
                issues.append(QAIssue(
                    type="other",
                    message=feedback,
                    owner=role_label.strip(),
                ))

    # If no role-specific issues but we have a general reason, attribute to Synthesizer
    if not issues and reason and reason.lower() not in ("none", "n/a", "none."):
        issues.append(QAIssue(type="other", message=reason, owner="Synthesizer"))

    return QAResult(
        status=status,
        reason=reason,
        issues=issues,
        correction_instruction=correction,
    )


# ============================================================
# Role Relevance Metadata and Selection
# ============================================================

# Each role has keywords/domains that indicate when it's relevant
ROLE_RELEVANCE: Dict[str, Dict[str, Any]] = {
    "creative": {
        "keywords": ["brainstorm", "ideas", "creative", "naming", "slogan", "marketing",
                      "framing", "wording", "concept", "design", "brand"],
        "domains": ["marketing", "content", "writing", "communication"],
        "description": "Ideas, framing, wording, brainstorming",
        "role_type": "creative",
        "task_types": ["creative_writing", "opinion_discussion"],
    },
    "technical": {
        "keywords": ["code", "implement", "build", "architecture", "api", "database",
                      "debug", "software", "programming", "algorithm", "system", "deploy",
                      "python", "javascript", "rust", "java", "react", "vue", "svelte",
                      "framework", "library", "performance", "faster"],
        "domains": ["engineering", "development", "devops", "infrastructure"],
        "description": "Code, architecture, implementation, technical solutions",
        "role_type": "factual",
        "task_types": ["coding_task", "analysis"],
    },
    "research": {
        "keywords": ["research", "study", "evidence", "literature", "paper", "facts",
                      "history", "compare", "comparison", "analysis", "data", "statistics",
                      "science", "scientific", "information"],
        "domains": ["academia", "science", "fact-finding"],
        "description": "Information gathering, literature review, fact-finding",
        "role_type": "factual",
        "task_types": ["factual_question", "comparison", "analysis", "summarization"],
    },
    "security": {
        "keywords": ["security", "vulnerability", "attack", "encryption", "auth",
                      "password", "exploit", "firewall", "compliance", "gdpr", "privacy"],
        "domains": ["cybersecurity", "infosec", "compliance"],
        "description": "Security analysis, vulnerability checks, best practices",
        "role_type": "safety",
        "task_types": ["analysis"],
    },
    "data_analyst": {
        "keywords": ["data", "analytics", "statistics", "pattern", "trend", "metric",
                      "dashboard", "visualization", "dataset", "csv", "spreadsheet"],
        "domains": ["analytics", "business intelligence"],
        "description": "Data analysis, statistics, pattern recognition, insights",
        "role_type": "analytical",
        "task_types": ["analysis", "comparison", "factual_question"],
    },
    "labour_union_rep": {
        "keywords": ["worker", "wages", "union", "labor", "labour", "employment",
                      "rights", "workplace", "collective", "bargaining", "fair"],
        "domains": ["labor relations", "HR", "workplace policy"],
        "description": "Worker rights, fair wages, job security",
        "role_type": "analytical",
        "task_types": ["opinion_discussion", "analysis"],
    },
    "ux_designer": {
        "keywords": ["user", "usability", "accessibility", "interface", "ux", "ui",
                      "design", "wireframe", "prototype", "user experience", "user-friendly"],
        "domains": ["design", "product", "UX"],
        "description": "User needs, usability, accessibility",
        "role_type": "analytical",
        "task_types": ["analysis", "planning"],
    },
    "lawyer": {
        "keywords": ["legal", "law", "contract", "liability", "compliance", "regulation",
                      "patent", "copyright", "trademark", "lawsuit", "litigation"],
        "domains": ["law", "compliance", "governance"],
        "description": "Legal compliance, liability, contracts, risk management",
        "role_type": "analytical",
        "task_types": ["analysis"],
    },
    # Persona roles — only active when allow_persona_roles is True
    "mad_professor": {
        "keywords": ["crazy", "radical", "hypothesis", "experiment", "breakthrough"],
        "domains": ["speculation"],
        "description": "Radical scientific hypotheses, extreme speculation",
        "role_type": "persona",
        "task_types": [],
        "is_persona": True,
    },
    "accountant": {
        "keywords": ["cost", "budget", "expense", "cheap", "price", "financial"],
        "domains": ["finance"],
        "description": "Cost scrutiny, budget optimization",
        "role_type": "persona",
        "task_types": [],
        "is_persona": True,
    },
    "artist": {
        "keywords": ["art", "inspiration", "vision", "aesthetic", "beauty"],
        "domains": ["art"],
        "description": "Unhinged creative vision, cosmic vibes",
        "role_type": "persona",
        "task_types": ["creative_writing"],
        "is_persona": True,
    },
    "lazy_slacker": {
        "keywords": ["lazy", "shortcut", "easy", "simple", "quick"],
        "domains": [],
        "description": "Minimum viable effort, shortcuts",
        "role_type": "persona",
        "task_types": [],
        "is_persona": True,
    },
    "black_metal_fundamentalist": {
        "keywords": ["metal", "kvlt", "underground", "nihilism"],
        "domains": [],
        "description": "Nihilistic kvlt critique",
        "role_type": "persona",
        "task_types": [],
        "is_persona": True,
    },
    "doris": {
        "keywords": [],
        "domains": [],
        "description": "Well-meaning but clueless observations",
        "role_type": "persona",
        "task_types": [],
        "is_persona": True,
    },
    "chairman_of_board": {
        "keywords": ["shareholder", "board", "governance", "strategic", "corporate"],
        "domains": ["corporate governance"],
        "description": "Corporate governance, shareholder value",
        "role_type": "persona",
        "task_types": [],
        "is_persona": True,
    },
    "maga_appointee": {
        "keywords": ["america", "patriot", "deregulation"],
        "domains": [],
        "description": "America First perspective",
        "role_type": "persona",
        "task_types": [],
        "is_persona": True,
    },
}


def select_relevant_roles(
    user_request: str,
    active_role_keys: List[str],
    config: WorkflowConfig,
    task_category: str = "other",
) -> List[str]:
    """Select only the most relevant specialist roles for a given request.

    Scores each active role by keyword match frequency and task-category affinity,
    filters persona roles based on config, and returns at most
    config.max_specialists_per_task roles.

    If config.always_include_research_for_factual_tasks is True and the task
    is factual, the research role is always included.
    """
    lower = user_request.lower()
    scored: List[Tuple[int, str]] = []

    for role_key in active_role_keys:
        meta = ROLE_RELEVANCE.get(role_key)
        if not meta:
            continue

        # Skip persona roles unless config allows them
        if meta.get("is_persona") and not config.allow_persona_roles:
            continue

        score = 0
        for kw in meta.get("keywords", []):
            if kw.lower() in lower:
                score += 1

        # Task-category affinity bonus
        role_tasks = meta.get("task_types", [])
        if task_category in role_tasks:
            score += 2

        scored.append((score, role_key))

    # Sort by score descending; keep deterministic order for ties
    scored.sort(key=lambda x: (-x[0], active_role_keys.index(x[1])))

    # Always include at least one role
    selected = []
    for score, role_key in scored:
        if len(selected) >= config.max_specialists_per_task:
            break
        # In strict mode, only include roles with score > 0 (except if we have none)
        if config.strict_mode and score == 0 and selected:
            continue
        selected.append(role_key)

    # Ensure at least one specialist is selected
    if not selected and scored:
        selected.append(scored[0][1])

    # Fallback: if no roles matched at all, use the first available core role
    if not selected:
        for rk in active_role_keys:
            meta = ROLE_RELEVANCE.get(rk, {})
            if not meta.get("is_persona"):
                selected.append(rk)
                break

    # Auto-include research for factual tasks
    if (config.always_include_research_for_factual_tasks
            and task_needs_evidence(task_category)
            and "research" in active_role_keys
            and "research" not in selected):
        selected.append("research")

    return selected


# ============================================================
# Targeted Revision Logic
# ============================================================

def identify_revision_targets(
    qa_result: QAResult,
    role_label_to_key: Dict[str, str],
) -> List[str]:
    """Given a QAResult, return the list of role keys that need rerunning.

    Rules:
    - Format/brevity issues → Synthesizer only (returned as "synthesizer")
    - Issues owned by a specific specialist → that specialist key
    - Issues owned by Planner → "planner"
    - If no clear owner → "synthesizer" (default)
    """
    targets = []
    for issue in qa_result.issues:
        owner = issue.owner.strip()

        if owner.lower() in ("synthesizer", "synthesis"):
            if "synthesizer" not in targets:
                targets.append("synthesizer")
        elif owner.lower() == "planner":
            if "planner" not in targets:
                targets.append("planner")
        else:
            # Try to resolve the owner label to a role key
            key = role_label_to_key.get(owner)
            if key and key not in targets:
                targets.append(key)
            elif "synthesizer" not in targets:
                # Unrecognised owner — attribute to synthesizer
                targets.append("synthesizer")

    # Format/brevity issues → always include synthesizer
    for issue in qa_result.issues:
        if issue.type in ("format", "brevity", "directness") and "synthesizer" not in targets:
            targets.append("synthesizer")

    if not targets:
        targets.append("synthesizer")

    return targets


# ============================================================
# Final Answer Compression / Noise Stripping
# ============================================================

# Sections that are internal workflow noise and should never appear in final output
_INTERNAL_NOISE_HEADERS = [
    "TASK BREAKDOWN:", "ROLE TO CALL:", "SUCCESS CRITERIA:",
    "GUIDANCE FOR SPECIALIST:", "PERSPECTIVES SUMMARY:",
    "COMMON GROUND:", "TENSIONS AND TRADE-OFFS:",
    "REQUIREMENTS CHECKED:", "ISSUES FOUND:",
    "ROLE-SPECIFIC FEEDBACK:", "RESULT:", "RECOMMENDED FIXES:",
    "DECISION: APPROVED", "DECISION: REVISE",
    "REVISED INSTRUCTIONS:", "REVISION",
    "SOURCES CONSULTED:", "SECURITY ANALYSIS:",
    "VULNERABILITIES FOUND:", "COST ANALYSIS:",
    "COST-CUTTING MEASURES:", "CHEAPEST VIABLE APPROACH:",
    "KVLT VERDICT:", "WHAT THE MAINSTREAM GETS WRONG:",
    "COSMIC VISION:", "DO WE EVEN NEED TO DO THIS:",
    "WORKER IMPACT:", "UNION CONCERNS:",
    "BOARD PERSPECTIVE:", "STRATEGIC CONCERNS:",
    "AMERICA FIRST ANALYSIS:", "DEEP STATE CONCERNS:",
    "LEGAL ANALYSIS:", "LIABILITIES AND RISKS:",
    "WHAT DORIS THINKS IS HAPPENING:",
]


def strip_internal_noise(text: str) -> str:
    """Remove internal workflow headers/sections from text intended for the user."""
    lines = text.split("\n")
    cleaned = []
    skip_until_next = False

    for line in lines:
        stripped = line.strip()
        # Check if this line is an internal header
        is_noise = False
        for header in _INTERNAL_NOISE_HEADERS:
            if stripped.startswith(header):
                is_noise = True
                skip_until_next = True
                break

        if is_noise:
            continue

        # If we were skipping noise, stop when we hit a non-empty non-header line
        # that looks like actual content (not a sub-bullet of the skipped section)
        if skip_until_next:
            if stripped == "":
                continue
            # New section header that is NOT noise means we stop skipping
            if stripped.endswith(":") and stripped == stripped.upper():
                skip_until_next = False
            elif not stripped.startswith("•") and not stripped.startswith("-") and not stripped.startswith("*"):
                skip_until_next = False

        if not skip_until_next:
            cleaned.append(line)

    result = "\n".join(cleaned).strip()
    return result if result else text


def compress_final_answer(
    draft: str,
    output_format: str,
    brevity: str,
    user_request: str,
) -> str:
    """Apply rule-based compression to the final answer.

    This does NOT call the LLM — it applies deterministic rules to trim
    the answer. The LLM-based compression happens in the synthesizer.
    """
    # Strip internal noise
    answer = strip_internal_noise(draft)

    # For single_choice: try to extract just the choice
    if output_format == "single_choice" and brevity == "minimal":
        # Look for a UNIFIED RECOMMENDATION section or similar
        for marker in ("UNIFIED RECOMMENDATION:", "RECOMMENDED DRAFT:", "FINAL ANSWER:"):
            if marker in answer:
                answer = answer.split(marker, 1)[1].strip()
                # Take only the first paragraph
                paragraphs = answer.split("\n\n")
                if paragraphs:
                    answer = paragraphs[0].strip()
                break

    # For short_answer: limit length
    if output_format == "short_answer" or brevity in ("minimal", "short"):
        # If the answer is very long relative to what was requested, truncate sensibly
        if len(answer) > 500 and brevity == "minimal":
            # Take the first meaningful paragraph
            paragraphs = [p.strip() for p in answer.split("\n\n") if p.strip()]
            if paragraphs:
                answer = paragraphs[0]

    return answer


# ============================================================
# Planner State
# ============================================================

@dataclass
class FailureRecord:
    """Record of a single failure in the workflow."""
    revision: int
    owner: str          # role key or "synthesizer"
    issue_type: str     # from QAIssue.type
    message: str
    correction: str

    def to_dict(self) -> dict:
        return {
            "revision": self.revision,
            "owner": self.owner,
            "issue_type": self.issue_type,
            "message": self.message,
            "correction": self.correction,
        }


@dataclass
class PlannerState:
    """Persistent state object that tracks the planner's decisions through revisions.

    This is the central working memory for the workflow.
    All stages read from and write to this shared state.
    """
    user_request: str = ""
    task_summary: str = ""
    task_category: str = "other"
    success_criteria: List[str] = field(default_factory=list)
    output_format: str = "other"
    brevity_requirement: str = "normal"
    selected_roles: List[str] = field(default_factory=list)
    specialist_outputs: Dict[str, str] = field(default_factory=dict)
    evidence: Optional[Dict] = None  # serialised EvidenceResult
    current_draft: str = ""
    qa_result: Optional[QAResult] = None
    revision_count: int = 0
    max_revisions: int = 3
    failure_history: List[FailureRecord] = field(default_factory=list)
    history: List[Dict[str, str]] = field(default_factory=list)
    final_answer: str = ""

    def record_event(self, event_type: str, detail: str):
        self.history.append({"type": event_type, "detail": detail[:500]})

    def record_failure(self, qa_result: QAResult):
        """Record QA failures into the failure history."""
        for issue in qa_result.issues:
            self.failure_history.append(FailureRecord(
                revision=self.revision_count,
                owner=issue.owner,
                issue_type=issue.type,
                message=issue.message[:200],
                correction=qa_result.correction_instruction[:200],
            ))

    def has_repeated_failure(self, owner: str, issue_type: str) -> bool:
        """Check if the same owner+issue_type has failed in a previous revision."""
        past = [
            f for f in self.failure_history
            if f.owner == owner and f.issue_type == issue_type
               and f.revision < self.revision_count
        ]
        return len(past) >= 1

    def get_repeat_failures(self) -> List[Tuple[str, str]]:
        """Return (owner, issue_type) pairs that have failed more than once."""
        counts: Dict[Tuple[str, str], int] = {}
        for f in self.failure_history:
            key = (f.owner, f.issue_type)
            counts[key] = counts.get(key, 0) + 1
        return [k for k, v in counts.items() if v >= 2]

    def get_escalation_strategy(self) -> str:
        """Determine escalation strategy when failures repeat.

        Returns:
            'narrow_scope' — reduce role count and simplify
            'rewrite_from_state' — synthesizer should rewrite from state, not reuse draft
            'suppress_role' — a specific role keeps introducing unsupported content
            'none' — no escalation needed
        """
        repeats = self.get_repeat_failures()
        if not repeats:
            return "none"

        synth_repeats = [(o, t) for o, t in repeats if o.lower() in ("synthesizer", "synthesis")]
        role_repeats = [(o, t) for o, t in repeats if o.lower() not in ("synthesizer", "synthesis", "planner")]

        if synth_repeats:
            return "rewrite_from_state"
        if role_repeats:
            return "suppress_role"
        return "narrow_scope"

    def get_roles_to_suppress(self) -> List[str]:
        """Return role owners that keep introducing repeated failures."""
        repeats = self.get_repeat_failures()
        return list({owner for owner, _ in repeats
                     if owner.lower() not in ("synthesizer", "synthesis", "planner")})

    def to_context_string(self) -> str:
        """Produce a compact summary string for inclusion in LLM prompts."""
        lines = [
            f"Task category: {self.task_category}",
            f"Output format required: {self.output_format}",
            f"Brevity requirement: {self.brevity_requirement}",
            f"Revision: {self.revision_count}/{self.max_revisions}",
            f"Selected roles: {', '.join(self.selected_roles)}",
        ]
        if self.success_criteria:
            lines.append(f"Success criteria: {'; '.join(self.success_criteria)}")
        if self.evidence:
            conf = self.evidence.get("confidence", "unknown")
            n_items = len(self.evidence.get("results", []))
            lines.append(f"Evidence: {n_items} items (confidence: {conf})")
        if self.qa_result and not self.qa_result.passed:
            lines.append(f"QA status: FAIL — {self.qa_result.reason}")
            if self.qa_result.correction_instruction:
                lines.append(f"Correction needed: {self.qa_result.correction_instruction}")
        if self.failure_history:
            lines.append(f"Previous failures: {len(self.failure_history)}")
            strategy = self.get_escalation_strategy()
            if strategy != "none":
                lines.append(f"Escalation strategy: {strategy}")
        return "\n".join(lines)

    def to_state_dict(self) -> dict:
        """Serialise the full state to a dictionary."""
        return {
            "user_request": self.user_request,
            "task_summary": self.task_summary,
            "task_category": self.task_category,
            "success_criteria": self.success_criteria,
            "output_format": self.output_format,
            "brevity_requirement": self.brevity_requirement,
            "selected_roles": self.selected_roles,
            "specialist_outputs": self.specialist_outputs,
            "evidence": self.evidence,
            "current_draft": self.current_draft[:500],
            "revision_count": self.revision_count,
            "max_revisions": self.max_revisions,
            "failure_history": [f.to_dict() for f in self.failure_history],
            "final_answer": self.final_answer[:500] if self.final_answer else "",
        }


# ============================================================
# Format-specific Synthesizer Instructions
# ============================================================

def get_synthesizer_format_instruction(output_format: str, brevity: str) -> str:
    """Return format-specific instructions to append to the synthesizer prompt."""
    instructions = {
        "single_choice": (
            "CRITICAL FORMAT RULE: The user wants ONE SINGLE CHOICE.\n"
            "Output ONLY the chosen option and at most one short justification sentence.\n"
            "Do NOT include perspectives summary, common ground, trade-offs, or any multi-section structure.\n"
            "Example: 'Veggie — it accommodates the widest range of dietary needs.'"
        ),
        "short_answer": (
            "CRITICAL FORMAT RULE: The user wants a SHORT, DIRECT answer.\n"
            "Output 1-3 sentences maximum. No sections, no headers, no perspectives summary.\n"
            "Answer the question directly and stop."
        ),
        "code": (
            "CRITICAL FORMAT RULE: The user wants CODE output.\n"
            "Output the code directly. Only include a brief explanation if explicitly requested.\n"
            "Do NOT include perspectives summary, trade-offs, or multi-section structure."
        ),
        "table": (
            "CRITICAL FORMAT RULE: The user wants a TABLE.\n"
            "Output a properly formatted markdown table.\n"
            "Do NOT include perspectives summary or prose-only answers."
        ),
        "bullet_list": (
            "CRITICAL FORMAT RULE: The user wants a BULLET LIST.\n"
            "Output a clean bullet list. Do NOT wrap it in prose or add unnecessary sections."
        ),
        "paragraph": (
            "Output a clear, well-structured paragraph. Keep it focused and avoid unnecessary sections."
        ),
        "report": (
            "The user wants a detailed report. You may include sections like summary, "
            "trade-offs, and perspectives, but keep each section concise."
        ),
    }

    base = instructions.get(output_format, "Match the output format to what the user requested.")

    if brevity == "minimal":
        base += "\nBREVITY: Absolute minimum. Fewer words is better."
    elif brevity == "short":
        base += "\nBREVITY: Keep it concise. No unnecessary elaboration."

    return base


def get_qa_format_instruction(output_format: str, brevity: str) -> str:
    """Return format-specific validation rules for the QA prompt."""
    rules = []
    if output_format == "single_choice":
        rules.append("FAIL if the output contains more than one choice or a long multi-section answer.")
        rules.append("FAIL if the output includes perspectives summary, common ground, or trade-offs.")
    elif output_format == "short_answer":
        rules.append("FAIL if the output is longer than 3-4 sentences.")
        rules.append("FAIL if the output includes unnecessary sections or headers.")
    elif output_format == "code":
        rules.append("FAIL if the output is mostly prose with no code.")
    elif output_format == "table":
        rules.append("FAIL if the output does not contain a markdown table.")
    if brevity in ("minimal", "short"):
        rules.append("FAIL if the output is excessively verbose for a brevity requirement.")
    return "\n".join(rules) if rules else ""
