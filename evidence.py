"""
Tool-backed fact verification module for the multi-role AI workflow.

Provides:
- EvidenceItem / EvidenceResult: structured evidence objects
- ResearchToolAdapter: abstract interface for search tools
- WebSearchAdapter, WikipediaAdapter, ArxivAdapter: concrete implementations
- gather_evidence(): orchestrates multi-tool evidence retrieval
- check_claims_against_evidence(): validates factual claims
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ============================================================
# Structured Evidence Objects
# ============================================================

@dataclass
class EvidenceItem:
    """A single piece of retrieved evidence."""
    title: str
    source: str          # "wikipedia" | "web_search" | "arxiv" | "internal"
    snippet: str
    url: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "source": self.source,
            "snippet": self.snippet,
            "url": self.url,
        }


@dataclass
class EvidenceResult:
    """Aggregated evidence for a research query."""
    query: str
    results: List[EvidenceItem] = field(default_factory=list)
    summary: str = ""
    confidence: str = "low"  # "high" | "medium" | "low"

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
            "confidence": self.confidence,
        }

    @property
    def has_evidence(self) -> bool:
        return len(self.results) > 0

    def merge(self, other: "EvidenceResult"):
        """Merge another EvidenceResult into this one."""
        self.results.extend(other.results)
        if other.confidence == "high" or (other.confidence == "medium" and self.confidence == "low"):
            self.confidence = other.confidence


# ============================================================
# Research Tool Adapter Interface
# ============================================================

class ResearchToolAdapter(ABC):
    """Abstract interface for a tool that retrieves evidence."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this tool."""

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Source identifier for EvidenceItem.source."""

    @abstractmethod
    def search(self, query: str) -> List[EvidenceItem]:
        """Execute a search and return evidence items."""


class WebSearchAdapter(ResearchToolAdapter):
    """Adapter wrapping a DuckDuckGo-style web search function."""

    def __init__(self, search_fn: Callable[[str], str]):
        self._search_fn = search_fn

    @property
    def name(self) -> str:
        return "Web Search"

    @property
    def source_type(self) -> str:
        return "web_search"

    def search(self, query: str) -> List[EvidenceItem]:
        try:
            raw = self._search_fn(query)
        except Exception:
            return []
        if not raw or "unavailable" in raw.lower():
            return []
        return self._parse_ddg_results(raw)

    @staticmethod
    def _parse_ddg_results(text: str) -> List[EvidenceItem]:
        """Parse DuckDuckGo search text into evidence items."""
        items = []
        # DDG results typically come as a continuous text blob
        # Split by common separators
        chunks = re.split(r'\n(?=[A-Z])', text)
        if len(chunks) <= 1:
            # Single blob — treat whole thing as one result
            items.append(EvidenceItem(
                title="Web search result",
                source="web_search",
                snippet=text[:500],
            ))
        else:
            for chunk in chunks[:5]:
                chunk = chunk.strip()
                if not chunk:
                    continue
                # Try to extract a title from the first line
                lines = chunk.split("\n", 1)
                title = lines[0][:100] if lines else "Web result"
                snippet = lines[1][:300] if len(lines) > 1 else chunk[:300]
                items.append(EvidenceItem(
                    title=title,
                    source="web_search",
                    snippet=snippet,
                ))
        return items


class WikipediaAdapter(ResearchToolAdapter):
    """Adapter wrapping a Wikipedia search function."""

    def __init__(self, search_fn: Callable[[str], str]):
        self._search_fn = search_fn

    @property
    def name(self) -> str:
        return "Wikipedia"

    @property
    def source_type(self) -> str:
        return "wikipedia"

    def search(self, query: str) -> List[EvidenceItem]:
        try:
            raw = self._search_fn(query)
        except Exception:
            return []
        if not raw:
            return []
        # Wikipedia wrapper returns article content
        return [EvidenceItem(
            title=f"Wikipedia: {query}",
            source="wikipedia",
            snippet=raw[:600],
        )]


class ArxivAdapter(ResearchToolAdapter):
    """Adapter wrapping an arXiv search function."""

    def __init__(self, search_fn: Callable[[str], str]):
        self._search_fn = search_fn

    @property
    def name(self) -> str:
        return "arXiv"

    @property
    def source_type(self) -> str:
        return "arxiv"

    def search(self, query: str) -> List[EvidenceItem]:
        try:
            raw = self._search_fn(query)
        except Exception:
            return []
        if not raw:
            return []
        items = []
        # arXiv results typically list papers with titles and summaries
        papers = re.split(r'(?:Published:|Title:)', raw)
        for paper in papers[:3]:
            paper = paper.strip()
            if not paper:
                continue
            items.append(EvidenceItem(
                title=paper[:100],
                source="arxiv",
                snippet=paper[:400],
            ))
        if not items:
            items.append(EvidenceItem(
                title=f"arXiv: {query}",
                source="arxiv",
                snippet=raw[:500],
            ))
        return items


# ============================================================
# Evidence Gathering Orchestrator
# ============================================================

def gather_evidence(
    queries: List[str],
    adapters: List[ResearchToolAdapter],
    max_results_per_query: int = 5,
) -> EvidenceResult:
    """Run queries across all available adapters and aggregate results.

    Args:
        queries: Search queries to run.
        adapters: Available research tool adapters.
        max_results_per_query: Max evidence items to keep per query.

    Returns:
        Aggregated EvidenceResult with confidence assessment.
    """
    combined = EvidenceResult(query="; ".join(queries))

    for query in queries:
        for adapter in adapters:
            try:
                items = adapter.search(query)
            except Exception:
                continue
            for item in items[:max_results_per_query]:
                combined.results.append(item)

    # Assess confidence based on evidence quantity and diversity
    sources = {r.source for r in combined.results}
    if len(combined.results) >= 3 and len(sources) >= 2:
        combined.confidence = "high"
    elif len(combined.results) >= 1:
        combined.confidence = "medium"
    else:
        combined.confidence = "low"

    return combined


def extract_search_queries(user_request: str, plan_text: str = "") -> List[str]:
    """Extract meaningful search queries from the user request and plan.

    Returns a list of 1-3 focused queries for evidence retrieval.
    """
    queries = []

    # The main user request is always a query (cleaned up)
    cleaned = re.sub(r'\b(please|could you|can you|i want|give me|write)\b', '', user_request, flags=re.IGNORECASE)
    cleaned = cleaned.strip()
    if cleaned:
        queries.append(cleaned[:200])

    # Extract key noun phrases from the plan for additional queries
    if plan_text:
        # Look for KEY FINDINGS or TASK BREAKDOWN sections
        for section in ("KEY FINDINGS:", "TASK BREAKDOWN:"):
            if section in plan_text:
                content = plan_text.split(section, 1)[1]
                # Take the first substantial line
                for line in content.split("\n"):
                    line = line.strip().lstrip("•-*1234567890. ")
                    if len(line) > 20 and line not in queries:
                        queries.append(line[:200])
                        break

    return queries[:3]


# ============================================================
# Evidence-backed Claim Checking
# ============================================================

_SPECIFIC_CLAIM_PATTERNS = [
    r'(?:according to|cited in|published in|reported by)\s+["\']?[A-Z]',
    r'\b(?:study|research|paper|article|report)\s+(?:by|from|in)\s+[A-Z]',
    r'\b(?:in|published)\s+\d{4}\b',
    r'\b(?:Dr\.|Prof\.|Professor)\s+[A-Z][a-z]+',
    r'"[^"]{10,}"',  # Quoted text that might be a fake citation
    r'\b\d+(?:\.\d+)?%\s+(?:of|increase|decrease|growth|decline)',  # Specific statistics
]


def detect_unsupported_claims(text: str, evidence: EvidenceResult) -> List[str]:
    """Detect specific factual claims in text that aren't backed by evidence.

    Returns a list of potentially unsupported claim excerpts.
    """
    unsupported = []
    evidence_text = " ".join(
        f"{r.title} {r.snippet}" for r in evidence.results
    ).lower()

    for pattern in _SPECIFIC_CLAIM_PATTERNS:
        for match in re.finditer(pattern, text):
            claim_context = text[max(0, match.start() - 30):match.end() + 30]
            # Check if any key terms from the claim appear in evidence
            claim_words = set(re.findall(r'\b[A-Z][a-z]{3,}\b', claim_context))
            if claim_words:
                matched = sum(1 for w in claim_words if w.lower() in evidence_text)
                if matched < len(claim_words) * 0.3:
                    unsupported.append(claim_context.strip())

    return unsupported[:5]  # Limit to 5 most concerning


def format_evidence_for_prompt(evidence: EvidenceResult) -> str:
    """Format evidence into a concise string for injection into LLM prompts."""
    if not evidence.has_evidence:
        return "No evidence was retrieved. Avoid specific claims, citations, or examples."

    lines = [f"RETRIEVED EVIDENCE (confidence: {evidence.confidence}):"]
    for i, item in enumerate(evidence.results[:8], 1):
        lines.append(f"  [{i}] ({item.source}) {item.title}")
        if item.snippet:
            lines.append(f"      {item.snippet[:200]}")
    lines.append("")
    lines.append(
        "RULE: Only cite facts, examples, or statistics that appear in the evidence above. "
        "If evidence is insufficient, give a general answer without fabricated specifics."
    )
    return "\n".join(lines)


def format_evidence_for_qa(evidence: EvidenceResult) -> str:
    """Format evidence context for QA validation."""
    if not evidence.has_evidence:
        return (
            "EVIDENCE VALIDATION: No evidence was retrieved.\n"
            "FAIL any answer that includes specific factual claims, named examples, "
            "citations, case studies, or statistics not backed by retrieved evidence.\n"
            "General reasoning and widely-known facts are acceptable."
        )

    lines = [
        f"EVIDENCE VALIDATION: {len(evidence.results)} items retrieved "
        f"(confidence: {evidence.confidence}).",
        "Evidence sources: " + ", ".join(
            sorted({r.source for r in evidence.results})
        ),
        "RULE: FAIL if the answer includes specific claims, names, dates, or statistics "
        "not supported by the retrieved evidence. General knowledge is acceptable.",
    ]
    return "\n".join(lines)
