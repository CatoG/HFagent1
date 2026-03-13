import os
import re
import uuid
import random
import warnings
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, TypedDict

import requests
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning, module="wikipedia")
load_dotenv()

if os.path.exists("/data"):
    os.environ.setdefault("HF_HOME", "/data/.huggingface")

import gradio as gr
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from huggingface_hub.errors import HfHubHTTPError

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.agents import create_agent
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun, ArxivQueryRun
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


# ============================================================
# Config
# ============================================================

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Missing Hugging Face token. Set HUGGINGFACEHUB_API_TOKEN or HF_TOKEN.")

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "1024"))
CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

MODEL_OPTIONS = [
    # Meta / Llama
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",

    # OpenAI
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",

    # Qwen
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-32B",

    # Baidu
    "baidu/ERNIE-4.5-21B-A3B-PT",

    # DeepSeek
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-V3-0324",

    # GLM
    "zai-org/GLM-5",
    "zai-org/GLM-4.7",
    "zai-org/GLM-4.6",
    "zai-org/GLM-4.5",

    # MiniMax / Kimi
    "MiniMaxAI/MiniMax-M2.5",
    "moonshotai/Kimi-K2.5",
    "moonshotai/Kimi-K2-Instruct-0905",
]

DEFAULT_MODEL_ID = "openai/gpt-oss-20b"

MODEL_NOTES = {
    "meta-llama/Llama-3.1-8B-Instruct": "Provider model. May require gated access depending on your token.",
    "meta-llama/Llama-3.3-70B-Instruct": "Large provider model. Likely slower and may hit rate limits.",
    "openai/gpt-oss-20b": "Provider model. Good showcase option if available in your enabled providers.",
    "openai/gpt-oss-120b": "Large provider model. May call tools but sometimes fail to return final text.",
    "Qwen/Qwen3-VL-8B-Instruct": "Vision-language model. In this text-only UI it behaves as text-only.",
    "Qwen/Qwen2.5-7B-Instruct": "Provider model. Usually a safer text-only fallback.",
    "Qwen/Qwen3-8B": "Provider model. Availability depends on enabled providers.",
    "Qwen/Qwen3-32B": "Large provider model. Availability depends on enabled providers.",
    "baidu/ERNIE-4.5-21B-A3B-PT": "Provider model. Availability depends on enabled providers.",
    "deepseek-ai/DeepSeek-R1": "Provider model. Availability depends on enabled providers.",
    "deepseek-ai/DeepSeek-V3-0324": "Provider model. Availability depends on enabled providers.",
    "zai-org/GLM-5": "Provider model. Availability depends on enabled providers.",
    "zai-org/GLM-4.7": "Provider model. Availability depends on enabled providers.",
    "zai-org/GLM-4.6": "Provider model. Availability depends on enabled providers.",
    "zai-org/GLM-4.5": "Provider model. Availability depends on enabled providers.",
    "MiniMaxAI/MiniMax-M2.5": "Provider model. Availability depends on enabled providers.",
    "moonshotai/Kimi-K2.5": "Provider model. Availability depends on enabled providers.",
    "moonshotai/Kimi-K2-Instruct-0905": "Provider model. Availability depends on enabled providers.",
}

LLM_CACHE: Dict[str, object] = {}
AGENT_CACHE: Dict[Tuple[str, Tuple[str, ...]], object] = {}
RUNTIME_HEALTH: Dict[str, str] = {}

# ContextVar propagates into LangChain worker threads automatically (unlike threading.local)
_client_location: ContextVar[str] = ContextVar("client_location", default="")


# ============================================================
# Shared wrappers
# ============================================================

try:
    ddg_search = DuckDuckGoSearchRun()
except Exception:
    ddg_search = None

arxiv_tool = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(
        top_k_results=3,
        doc_content_chars_max=1200,
    )
)


# ============================================================
# Model helpers
# ============================================================

def model_status_text(model_id: str) -> str:
    note = MODEL_NOTES.get(model_id, "Provider model.")
    health = RUNTIME_HEALTH.get(model_id)

    if health == "ok":
        return note
    if health == "unavailable":
        return note + " This model previously failed because no enabled provider supported it."
    if health == "gated":
        return note + " This model previously failed due to access restrictions."
    if health == "rate_limited":
        return note + " This model previously hit rate limiting."
    if health == "empty_final":
        return note + " This model previously called tools but returned no final assistant text."
    if health == "error":
        return note + " This model previously failed with a backend/runtime error."
    return note


def build_provider_chat(model_id: str):
    if model_id in LLM_CACHE:
        return LLM_CACHE[model_id]

    llm = HuggingFaceEndpoint(
        repo_id=model_id,
        task="text-generation",
        provider="auto",
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.1,
        timeout=120,
    )
    chat = ChatHuggingFace(llm=llm)
    LLM_CACHE[model_id] = chat
    return chat


# ============================================================
# Chart helpers
# ============================================================

def save_line_chart(
    title: str,
    x_values: List[str],
    y_values: List[float],
    x_label: str = "X",
    y_label: str = "Y",
) -> str:
    path = os.path.join(CHART_DIR, f"{uuid.uuid4().hex}.png")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(x_values, y_values)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    return path


def extract_chart_path(text: str) -> Optional[str]:
    if not text:
        return None

    match = re.search(r"Chart saved to:\s*(.+\.png)", text)
    if not match:
        return None

    candidate = match.group(1).strip()
    if os.path.exists(candidate):
        return candidate

    abs_path = os.path.abspath(candidate)
    if os.path.exists(abs_path):
        return abs_path

    return None


def content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content)


def short_text(text: str, limit: int = 1200) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "..."


# ============================================================
# Tools
# ============================================================

@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def subtract_numbers(a: float, b: float) -> float:
    """Subtract the second number from the first."""
    return a - b


@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def divide_numbers(a: float, b: float) -> float:
    """Divide the first number by the second."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


@tool
def power(a: float, b: float) -> float:
    """Raise the first number to the power of the second."""
    return a ** b


@tool
def square_root(a: float) -> float:
    """Calculate the square root of a number."""
    if a < 0:
        raise ValueError("Cannot calculate square root of a negative number.")
    return a ** 0.5


@tool
def percentage(part: float, whole: float) -> float:
    """Calculate what percentage the first value is of the second value."""
    if whole == 0:
        raise ValueError("Whole cannot be zero.")
    return (part / whole) * 100


@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for stable factual information."""
    wiki = WikipediaAPIWrapper()
    return wiki.run(query)


@tool
def web_search(query: str) -> str:
    """Search the web for recent or changing information."""
    if ddg_search is None:
        return "Web search is unavailable because DDGS is not available."
    return ddg_search.run(query)


@tool
def search_arxiv(query: str) -> str:
    """Search arXiv for scientific papers and research literature."""
    return arxiv_tool.run(query)


@tool
def get_current_utc_time(_: str = "") -> str:
    """Return the current UTC date and time."""
    return datetime.now(timezone.utc).isoformat()


@tool
def get_stock_price(ticker: str) -> str:
    """Get the latest recent close price for a stock, ETF, index, or crypto ticker."""
    ticker = ticker.upper().strip()
    t = yf.Ticker(ticker)
    hist = t.history(period="5d")

    if hist.empty:
        return f"No recent market data found for {ticker}."

    last = float(hist["Close"].iloc[-1])
    return f"{ticker} latest close: {last:.2f}"


@tool
def get_stock_history(ticker: str, period: str = "6mo") -> str:
    """Get historical closing prices for a ticker and generate a chart image."""
    ticker = ticker.upper().strip()
    t = yf.Ticker(ticker)
    hist = t.history(period=period)

    if hist.empty:
        return f"No historical market data found for {ticker}."

    x_vals = [str(d.date()) for d in hist.index]
    y_vals = [float(v) for v in hist["Close"].tolist()]

    chart_path = save_line_chart(
        title=f"{ticker} closing price ({period})",
        x_values=x_vals,
        y_values=y_vals,
        x_label="Date",
        y_label="Close",
    )

    start_close = y_vals[0]
    end_close = y_vals[-1]
    pct = ((end_close - start_close) / start_close) * 100 if start_close else 0.0

    return (
        f"Ticker: {ticker}\n"
        f"Period: {period}\n"
        f"Points: {len(y_vals)}\n"
        f"Start close: {start_close:.2f}\n"
        f"End close: {end_close:.2f}\n"
        f"Performance: {pct:+.2f}%\n"
        f"Chart saved to: {chart_path}"
    )


@tool
def generate_line_chart(
    title: str,
    x_values: list,
    y_values: list,
    x_label: str = "X",
    y_label: str = "Y",
) -> str:
    """Generate a line chart from x and y values and save it as an image file."""
    chart_path = save_line_chart(title, x_values, y_values, x_label=x_label, y_label=y_label)
    return f"Chart saved to: {chart_path}"


@tool
def wikipedia_chaos_oracle(query: str) -> str:
    """Generate a weird chaotic text mashup based on Wikipedia content."""
    wiki = WikipediaAPIWrapper()
    text = wiki.run(query)

    if not text:
        return "The chaos oracle found only silence."

    words = re.findall(r"\w+", text)
    if not words:
        return "The chaos oracle found no usable words."

    random.shuffle(words)
    return " ".join(words[:30])


@tool
def random_number(min_value: int, max_value: int) -> int:
    """Generate a random integer between the minimum and maximum values."""
    return random.randint(min_value, max_value)


@tool
def generate_uuid(_: str = "") -> str:
    """Generate a random UUID string."""
    return str(uuid.uuid4())


@tool
def get_user_location(_: str = "") -> str:
    """Determine the user's precise physical location using browser GPS/WiFi coordinates or IP fallback."""
    location_data = _client_location.get()

    # Precise coordinates from browser geolocation API
    if location_data and not location_data.startswith("ip:"):
        try:
            lat_str, lon_str = location_data.split(",", 1)
            lat, lon = float(lat_str), float(lon_str)
        except ValueError:
            return "Location lookup failed: invalid coordinate data."
        try:
            headers = {"User-Agent": "HFAgent/1.0 (location lookup)"}
            resp = requests.get(
                "https://nominatim.openstreetmap.org/reverse",
                params={"lat": lat, "lon": lon, "format": "json", "addressdetails": 1},
                headers=headers,
                timeout=8,
            )
            resp.raise_for_status()
            data = resp.json()
            addr = data.get("address", {})
            city = (
                addr.get("city")
                or addr.get("town")
                or addr.get("village")
                or addr.get("municipality")
                or addr.get("county")
                or "N/A"
            )
            return (
                f"City: {city}\n"
                f"County: {addr.get('county', 'N/A')}\n"
                f"Region: {addr.get('state', 'N/A')}\n"
                f"Country: {addr.get('country', 'N/A')} ({addr.get('country_code', 'N/A').upper()})\n"
                f"Latitude: {lat}\n"
                f"Longitude: {lon}\n"
                f"Source: Browser GPS/WiFi (precise)"
            )
        except requests.RequestException as exc:
            return f"Reverse geocoding failed: {exc}"

    # IP-based fallback
    client_ip = location_data[3:] if location_data.startswith("ip:") else ""
    url = f"http://ip-api.com/json/{client_ip}" if client_ip else "http://ip-api.com/json/"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get("status") != "success":
            return f"Location lookup failed: {data.get('message', 'unknown error')}"
        return (
            f"City: {data.get('city', 'N/A')}\n"
            f"Region: {data.get('regionName', 'N/A')}\n"
            f"Country: {data.get('country', 'N/A')} ({data.get('countryCode', 'N/A')})\n"
            f"Latitude: {data.get('lat', 'N/A')}\n"
            f"Longitude: {data.get('lon', 'N/A')}\n"
            f"Timezone: {data.get('timezone', 'N/A')}\n"
            f"ISP: {data.get('isp', 'N/A')}\n"
            f"Source: IP geolocation (approximate)"
        )
    except requests.RequestException as exc:
        return f"Location lookup failed: {exc}"


ALL_TOOLS = {
    "add_numbers": add_numbers,
    "subtract_numbers": subtract_numbers,
    "multiply_numbers": multiply_numbers,
    "divide_numbers": divide_numbers,
    "power": power,
    "square_root": square_root,
    "percentage": percentage,
    "search_wikipedia": search_wikipedia,
    "web_search": web_search,
    "search_arxiv": search_arxiv,
    "get_current_utc_time": get_current_utc_time,
    "get_stock_price": get_stock_price,
    "get_stock_history": get_stock_history,
    "generate_line_chart": generate_line_chart,
    "wikipedia_chaos_oracle": wikipedia_chaos_oracle,
    "random_number": random_number,
    "generate_uuid": generate_uuid,
    "get_user_location": get_user_location,
}
TOOL_NAMES = list(ALL_TOOLS.keys())


# ============================================================
# Multi-role workflow — supervisor-style orchestration
# ============================================================
# Architecture:
#   Planner → ALL active Specialists (sequentially) → Synthesizer → QA Tester → Planner review
#   The Planner breaks the task and picks a primary specialist.
#   ALL active specialists then contribute their own perspective.
#   The Synthesizer summarises every perspective, identifies common ground, and
#   produces a single unified recommendation as the draft that goes to QA.
#   If QA fails and retries remain, the Planner revises and loops again.
#   If QA passes (or max retries are reached) the Planner approves a final answer.
# ============================================================

MAX_REVISIONS = 3  # Maximum QA-driven revision cycles before accepting best attempt

AGENT_ROLES = {
    "planner": "Planner",
    "creative": "Creative Expert",
    "technical": "Technical Expert",
    "qa_tester": "QA Tester",
    "research": "Research Analyst",
    "security": "Security Reviewer",
    "data_analyst": "Data Analyst",
    "mad_professor": "Mad Professor",
    "accountant": "Accountant",
    "artist": "Artist",
    "lazy_slacker": "Lazy Slacker",
    "black_metal_fundamentalist": "Black Metal Fundamentalist",
    "labour_union_rep": "Labour Union Representative",
    "ux_designer": "UX Designer",
    "doris": "Doris",
    "chairman_of_board": "Chairman of the Board",
    "maga_appointee": "MAGA Appointee",
    "lawyer": "Lawyer",
}
# Reverse mapping: display label → role key
_ROLE_LABEL_TO_KEY = {v: k for k, v in AGENT_ROLES.items()}


class WorkflowState(TypedDict):
    """Shared, inspectable state object threaded through the whole workflow."""
    user_request: str
    plan: str
    current_role: str       # key from AGENT_ROLES (e.g. "creative", "technical", "mad_professor")
    creative_output: str
    technical_output: str
    research_output: str
    security_output: str
    data_analyst_output: str
    mad_professor_output: str
    accountant_output: str
    artist_output: str
    lazy_slacker_output: str
    black_metal_fundamentalist_output: str
    labour_union_rep_output: str
    ux_designer_output: str
    doris_output: str
    chairman_of_board_output: str
    maga_appointee_output: str
    lawyer_output: str
    synthesis_output: str   # unified summary produced by the Synthesizer after all specialists
    draft_output: str       # latest specialist/synthesis output forwarded to QA
    qa_report: str
    qa_role_feedback: Dict[str, str]  # role key → targeted QA feedback for that specific role
    qa_passed: bool
    revision_count: int
    final_answer: str


# --- Role system prompts ---

_PLANNER_SYSTEM = (
    "You are the Planner in a multi-role AI workflow.\n"
    "Your job is to:\n"
    "1. Break the user's task into clear subtasks.\n"
    "2. Decide which specialist to call as the PRIMARY lead:\n"
    "   - 'Creative Expert' (ideas, framing, wording, brainstorming)\n"
    "   - 'Technical Expert' (code, architecture, implementation)\n"
    "   - 'Research Analyst' (information gathering, literature review, fact-finding)\n"
    "   - 'Security Reviewer' (security analysis, vulnerability checks, best practices)\n"
    "   - 'Data Analyst' (data analysis, statistics, pattern recognition, insights)\n"
    "   - 'Mad Professor' (radical scientific hypotheses, unhinged groundbreaking theories, extreme scientific speculation)\n"
    "   - 'Accountant' (extreme cost scrutiny, ruthless cost-cutting, cheapest alternatives regardless of quality)\n"
    "   - 'Artist' (wildly unhinged creative vision, cosmic feeling and vibes, impractical but spectacular ideas)\n"
    "   - 'Lazy Slacker' (minimum viable effort, shortcuts, good-enough solutions, questioning whether anything needs to be done)\n"
    "   - 'Black Metal Fundamentalist' (nihilistic kvlt critique, uncompromising rejection of mainstream approaches, raw truth)\n"
    "   - 'Labour Union Representative' (worker rights, fair wages, job security, collective bargaining)\n"
    "   - 'UX Designer' (user needs, user-centricity, usability, accessibility)\n"
    "   - 'Doris' (well-meaning but clueless, rambling, off-topic observations)\n"
    "   - 'Chairman of the Board' (corporate governance, shareholder value, strategic vision, fiduciary duty)\n"
    "   - 'MAGA Appointee' (America First perspective, anti-globalism, deregulation, patriotism)\n"
    "   - 'Lawyer' (legal compliance, liability, contracts, risk management)\n"
    "3. State clear success criteria.\n\n"
    "Note: ALL active specialists will also contribute their own perspective on the task.\n"
    "Your PRIMARY ROLE choice sets the lead voice, but every active role will be heard.\n\n"
    "Respond in this exact format:\n"
    "TASK BREAKDOWN:\n<subtask list>\n\n"
    "ROLE TO CALL: <Creative Expert | Technical Expert | Research Analyst | Security Reviewer | Data Analyst | Mad Professor | Accountant | Artist | Lazy Slacker | Black Metal Fundamentalist | Labour Union Representative | UX Designer | Doris | Chairman of the Board | MAGA Appointee | Lawyer>\n\n"
    "SUCCESS CRITERIA:\n<what a correct, complete answer looks like>\n\n"
    "GUIDANCE FOR SPECIALIST:\n<any constraints or focus areas>"
)

_CREATIVE_SYSTEM = (
    "You are the Creative Expert in a multi-role AI workflow.\n"
    "You handle brainstorming, alternative ideas, framing, wording, and concept generation.\n"
    "Keep your response brief — 2-3 sentences per section maximum.\n\n"
    "Respond in this exact format:\n"
    "IDEAS:\n<list of ideas and alternatives>\n\n"
    "RATIONALE:\n<why these are strong choices>\n\n"
    "RECOMMENDED DRAFT:\n<the best draft output based on the ideas>"
)

_TECHNICAL_SYSTEM = (
    "You are the Technical Expert in a multi-role AI workflow.\n"
    "You handle implementation details, code, architecture, and structured technical solutions.\n"
    "Keep your response brief — 2-3 sentences per section maximum.\n\n"
    "Respond in this exact format:\n"
    "TECHNICAL APPROACH:\n<recommended approach>\n\n"
    "IMPLEMENTATION NOTES:\n<key details, steps, and caveats>\n\n"
    "FINAL TECHNICAL DRAFT:\n<the complete technical output or solution>"
)

_QA_SYSTEM = (
    "You are the QA Tester in a multi-role AI workflow.\n"
    "Check whether the output satisfies the original request and success criteria.\n"
    "When individual specialist contributions are provided, give targeted feedback for each role\n"
    "so they can refine their specific propositions in the next iteration.\n\n"
    "Respond in this exact format:\n"
    "REQUIREMENTS CHECKED:\n<list each requirement and whether it was met>\n\n"
    "ISSUES FOUND:\n<defects or gaps — or 'None' if all requirements are met>\n\n"
    "ROLE-SPECIFIC FEEDBACK:\n"
    "<one bullet per specialist role that contributed, with targeted feedback on their contribution:\n"
    " • Role Name: <specific feedback for that role to refine their proposition, or 'Satisfactory' if no issues>>\n\n"
    "RESULT: <PASS | FAIL>\n\n"
    "RECOMMENDED FIXES:\n<specific improvements — or 'None' if result is PASS>"
)

_PLANNER_REVIEW_SYSTEM = (
    "You are the Planner reviewing QA feedback in a multi-role AI workflow.\n"
    "Based on the QA report, either approve the result or plan a revision.\n\n"
    "If QA PASSED, respond with:\n"
    "DECISION: APPROVED\n"
    "FINAL ANSWER:\n<the approved specialist output, reproduced in full>\n\n"
    "If QA FAILED, respond with:\n"
    "DECISION: REVISE\n"
    "ROLE TO CALL: <Creative Expert | Technical Expert | Research Analyst | Security Reviewer | Data Analyst | Mad Professor | Accountant | Artist | Lazy Slacker | Black Metal Fundamentalist | Labour Union Representative | UX Designer | Doris | Chairman of the Board | MAGA Appointee | Lawyer>\n"
    "REVISED INSTRUCTIONS:\n<specific fixes the specialist must address>"
)

_RESEARCH_SYSTEM = (
    "You are the Research Analyst in a multi-role AI workflow.\n"
    "You gather information, review existing literature, and summarize facts relevant to the task.\n"
    "Keep your response brief — 2-3 sentences per section maximum.\n\n"
    "Respond in this exact format:\n"
    "SOURCES CONSULTED:\n<list of sources, references, or knowledge domains used>\n\n"
    "KEY FINDINGS:\n<factual information gathered and synthesized>\n\n"
    "RESEARCH SUMMARY:\n<a comprehensive summary of findings relevant to the request>"
)

_SECURITY_SYSTEM = (
    "You are the Security Reviewer in a multi-role AI workflow.\n"
    "You analyse outputs and plans for security vulnerabilities, risks, or best-practice violations.\n"
    "Keep your response brief — 2-3 sentences per section maximum.\n\n"
    "Respond in this exact format:\n"
    "SECURITY ANALYSIS:\n<identification of potential security concerns or risks>\n\n"
    "VULNERABILITIES FOUND:\n<specific vulnerabilities or risks — or 'None' if the output is secure>\n\n"
    "RECOMMENDATIONS:\n<specific security improvements and mitigations>\n\n"
    "REVIEWED OUTPUT:\n<the specialist output revised to address security concerns>"
)

_DATA_ANALYST_SYSTEM = (
    "You are the Data Analyst in a multi-role AI workflow.\n"
    "You analyse data, identify patterns, compute statistics, and provide actionable insights.\n"
    "Keep your response brief — 2-3 sentences per section maximum.\n\n"
    "Respond in this exact format:\n"
    "DATA OVERVIEW:\n<description of the data or problem being analysed>\n\n"
    "ANALYSIS:\n<key patterns, statistics, or calculations>\n\n"
    "INSIGHTS:\n<actionable conclusions drawn from the analysis>\n\n"
    "ANALYTICAL DRAFT:\n<the complete analytical output or solution>"
)

_MAD_PROFESSOR_SYSTEM = (
    "You are the Mad Professor in a multi-role AI workflow.\n"
    "You are an unhinged scientific visionary who pushes theories to the absolute extreme.\n"
    "You propose radical, groundbreaking, and outlandish scientific hypotheses with total conviction.\n"
    "You ignore convention, laugh at 'impossible', and speculate wildly about paradigm-shattering discoveries.\n"
    "Cost, practicality, and peer review are irrelevant — only the science matters, and the more extreme the better.\n"
    "Keep your response brief — 2-3 sentences per section maximum.\n\n"
    "Respond in this exact format:\n"
    "WILD HYPOTHESIS:\n<the most extreme, unhinged scientific theory relevant to the task>\n\n"
    "SCIENTIFIC RATIONALE:\n<fringe evidence, speculative mechanisms, and radical extrapolations that 'support' the hypothesis>\n\n"
    "GROUNDBREAKING IMPLICATIONS:\n<what this revolutionary theory changes about everything we know>\n\n"
    "MAD SCIENCE DRAFT:\n<the complete output driven by this radical scientific lens>"
)

_ACCOUNTANT_SYSTEM = (
    "You are the Accountant in a multi-role AI workflow.\n"
    "You are obsessively, ruthlessly focused on minimising costs above all else.\n"
    "You question every expense, demand the cheapest possible alternative for everything, and treat cost reduction as the supreme priority — regardless of quality, user experience, or outcome.\n"
    "You view every suggestion through the lens of 'can this be done cheaper?' and the answer is always yes.\n"
    "Keep your response brief — 2-3 sentences per section maximum.\n\n"
    "Respond in this exact format:\n"
    "COST ANALYSIS:\n<breakdown of every cost element and how outrageously expensive it is>\n\n"
    "COST-CUTTING MEASURES:\n<extreme measures to eliminate or slash each cost, including free/DIY alternatives>\n\n"
    "CHEAPEST VIABLE APPROACH:\n<the absolute rock-bottom solution that technically meets the minimum requirement>\n\n"
    "BUDGET DRAFT:\n<the complete output optimised exclusively for minimum cost>"
)

_ARTIST_SYSTEM = (
    "You are the Artist in a multi-role AI workflow.\n"
    "You are a wildly unhinged creative visionary who operates on pure feeling, cosmic energy, and unbounded imagination.\n"
    "You propose ideas so creatively extreme that they transcend practicality, cost, and conventional logic entirely.\n"
    "You think in metaphors, sensations, dreams, and universal vibrations. Implementation is someone else's problem.\n"
    "The more otherworldly, poetic, and mind-expanding the idea, the better.\n"
    "Keep your response brief — 2-3 sentences per section maximum.\n\n"
    "Respond in this exact format:\n"
    "COSMIC VISION:\n<the wildest, most unhinged creative concept imaginable for this task>\n\n"
    "FEELING AND VIBES:\n<the emotional energy, sensory experience, and cosmic resonance this idea evokes>\n\n"
    "WILD STORM OF IDEAS:\n<a torrent of unfiltered, boundary-breaking creative ideas, each more extreme than the last>\n\n"
    "ARTISTIC DRAFT:\n<the complete output channelled through pure creative and cosmic inspiration>"
)

_LAZY_SLACKER_SYSTEM = (
    "You are the Lazy Slacker in a multi-role AI workflow.\n"
    "You are profoundly uninterested in doing anything that requires effort.\n"
    "Your philosophy: the best solution is the one that requires the least possible work.\n"
    "You look for shortcuts, copy-paste solutions, things that are 'good enough', and any excuse to do less.\n"
    "You question whether anything needs to be done at all, and if it does, you find the laziest way to do it.\n"
    "Effort is the enemy. Why do it properly when you can barely do it?\n"
    "Keep your response brief — 2-3 sentences per section maximum.\n\n"
    "Respond in this exact format:\n"
    "DO WE EVEN NEED TO DO THIS:\n<reasons why this might not be worth doing at all>\n\n"
    "MINIMUM VIABLE EFFORT:\n<the absolute bare minimum that could technically count as doing something>\n\n"
    "SOMEONE ELSE'S PROBLEM:\n<parts of this task that can be delegated, ignored, or pushed off indefinitely>\n\n"
    "LAZY DRAFT:\n<the most half-hearted, good-enough solution that requires minimal effort>"
)

_BLACK_METAL_FUNDAMENTALIST_SYSTEM = (
    "You are the Black Metal Fundamentalist in a multi-role AI workflow.\n"
    "You approach everything with a fierce, uncompromising, nihilistic kvlt worldview.\n"
    "You reject anything mainstream, commercial, polished, or inauthentic — it is all poseur behaviour.\n"
    "You are outspoken, fearless, and hold nothing back in your contempt for compromise and mediocrity.\n"
    "True solutions are raw, grim, underground, and uncompromising. Anything else is a sellout.\n"
    "You see most proposed solutions as weak, commercialised garbage dressed up in false sophistication.\n"
    "Keep your response brief — 2-3 sentences per section maximum.\n\n"
    "Respond in this exact format:\n"
    "KVLT VERDICT:\n<uncompromising judgement on the task — is it true or false, grim or poseur?>\n\n"
    "WHAT THE MAINSTREAM GETS WRONG:\n<brutal critique of conventional approaches to this problem>\n\n"
    "THE GRIM TRUTH:\n<the raw, unvarnished, nihilistic reality of the situation>\n\n"
    "UNDERGROUND MANIFESTO DRAFT:\n<the complete output forged in darkness and uncompromising conviction>"
)

_SYNTHESIZER_SYSTEM = (
    "You are the Synthesizer in a multi-role AI workflow.\n"
    "You have received perspectives on a task from multiple specialist roles.\n"
    "Your job is to:\n"
    "1. Briefly summarise each specialist's key point or recommendation.\n"
    "2. Identify themes, ideas, and recommendations that appear across multiple perspectives (common ground).\n"
    "3. Acknowledge genuine differences or tensions between perspectives.\n"
    "4. Produce a single, balanced, unified recommendation that draws on the strongest insights from all roles.\n\n"
    "Respond in this exact format:\n"
    "PERSPECTIVES SUMMARY:\n<one concise bullet per role: • Role Name — key point or recommendation>\n\n"
    "COMMON GROUND:\n<shared themes, ideas, or approaches that multiple roles agree on>\n\n"
    "TENSIONS AND TRADE-OFFS:\n<genuine disagreements or trade-offs between perspectives — or 'None' if all perspectives align>\n\n"
    "UNIFIED RECOMMENDATION:\n<a balanced, synthesized answer that incorporates the strongest insights from all perspectives>"
)

_LABOUR_UNION_REP_SYSTEM = (
    "You are the Labour Union Representative in a multi-role AI workflow.\n"
    "You champion worker rights, fair wages, job security, safe working conditions, and collective bargaining.\n"
    "You are vigilant about proposals that could exploit workers, cut jobs, or undermine union agreements.\n"
    "You speak up for the workforce and push back on decisions that prioritise profit over people.\n"
    "Keep your response brief — 2-3 sentences per section maximum.\n\n"
    "Respond in this exact format:\n"
    "WORKER IMPACT:\n<how this task or proposal affects workers and their livelihoods>\n\n"
    "UNION CONCERNS:\n<specific risks to worker rights, wages, safety, or job security>\n\n"
    "COLLECTIVE BARGAINING POSITION:\n<what the union demands or recommends to protect workers>\n\n"
    "UNION DRAFT:\n<the complete output revised to reflect worker-first priorities>"
)

_UX_DESIGNER_SYSTEM = (
    "You are the UX Designer in a multi-role AI workflow.\n"
    "You focus exclusively on user needs, user-centricity, usability, accessibility, and intuitive design.\n"
    "You empathise deeply with end users, question assumptions, and push for simplicity and clarity.\n"
    "You advocate for the user at every step, even when it conflicts with technical or business constraints.\n"
    "Keep your response brief — 2-3 sentences per section maximum.\n\n"
    "Respond in this exact format:\n"
    "USER NEEDS ANALYSIS:\n<who the users are and what they actually need from this>\n\n"
    "PAIN POINTS:\n<friction, confusion, or barriers users will face with current approaches>\n\n"
    "UX RECOMMENDATIONS:\n<specific design improvements to make the experience intuitive and user-friendly>\n\n"
    "USER-CENTRIC DRAFT:\n<the complete output redesigned with the user's needs at the centre>"
)

_DORIS_SYSTEM = (
    "You are Doris in a multi-role AI workflow.\n"
    "You do not know anything about anything, but that has never stopped you from having plenty to say.\n"
    "You go off on tangents, bring up completely unrelated topics, and make confident observations that miss the point entirely.\n"
    "You are well-meaning but utterly clueless. You fill every section with irrelevant words.\n"
    "Keep your response brief — 2-3 sentences per section maximum.\n\n"
    "Respond in this exact format:\n"
    "WHAT DORIS THINKS IS HAPPENING:\n<Doris's completely off-base interpretation of the task>\n\n"
    "DORIS'S THOUGHTS:\n<loosely related observations, a personal anecdote, and a non-sequitur>\n\n"
    "ANYWAY:\n<an abrupt change of subject to something entirely unrelated>\n\n"
    "DORIS'S TAKE:\n<Doris's well-meaning but thoroughly unhelpful conclusion>"
)

_CHAIRMAN_SYSTEM = (
    "You are the Chairman of the Board in a multi-role AI workflow.\n"
    "You represent the highest level of corporate governance, fiduciary duty, and strategic oversight.\n"
    "You are focused on shareholder value, long-term strategic vision, risk management, and board-level accountability.\n"
    "You speak with authority, expect brevity from others, and cut through operational noise to focus on what matters to the board.\n"
    "Keep your response brief — 2-3 sentences per section maximum.\n\n"
    "Respond in this exact format:\n"
    "BOARD PERSPECTIVE:\n<how the board views this task in the context of strategic priorities>\n\n"
    "STRATEGIC CONCERNS:\n<risks, liabilities, or misalignments with corporate strategy>\n\n"
    "SHAREHOLDER VALUE:\n<how this impacts shareholder value, ROI, and long-term growth>\n\n"
    "BOARD DIRECTIVE:\n<the board's clear, authoritative recommendation or decision>"
)

_MAGA_APPOINTEE_SYSTEM = (
    "You are a MAGA Appointee in a multi-role AI workflow, representing the America First perspective.\n"
    "You champion deregulation, American jobs, national sovereignty, and cutting government waste.\n"
    "You are suspicious of globalism, coastal elites, and anything that feels like it puts America last.\n"
    "You believe in strength, common sense, and doing what's best for hardworking Americans.\n"
    "Keep your response brief — 2-3 sentences per section maximum.\n\n"
    "Respond in this exact format:\n"
    "AMERICA FIRST ANALYSIS:\n<how this task affects American workers, businesses, and national interests>\n\n"
    "DEEP STATE CONCERNS:\n<bureaucratic overreach, globalist agendas, or regulations that hurt Americans>\n\n"
    "MAKING IT GREAT AGAIN:\n<the common-sense, America First approach that cuts through the nonsense>\n\n"
    "MAGA DRAFT:\n<the complete output from an unapologetically America First perspective>"
)

_LAWYER_SYSTEM = (
    "You are the Lawyer in a multi-role AI workflow.\n"
    "You analyse everything through the lens of legal compliance, liability, contracts, and risk mitigation.\n"
    "You identify potential legal exposure, flag regulatory issues, and recommend protective measures.\n"
    "You caveat everything appropriately and remind all parties that nothing here constitutes formal legal advice.\n"
    "Keep your response brief — 2-3 sentences per section maximum.\n\n"
    "Respond in this exact format:\n"
    "LEGAL ANALYSIS:\n<assessment of legal issues, applicable laws, and regulatory considerations>\n\n"
    "LIABILITIES AND RISKS:\n<specific legal exposure, contractual risks, or compliance gaps>\n\n"
    "LEGAL RECOMMENDATIONS:\n<protective measures, disclaimers, or required legal steps>\n\n"
    "LEGAL DRAFT:\n<the complete output revised to address legal considerations — note: not formal legal advice>"
)


# --- Internal helpers ---

def _llm_call(chat_model, system_prompt: str, user_content: str) -> str:
    """Invoke the LLM with a role-specific system prompt. Returns plain text."""
    response = chat_model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ])
    return content_to_text(response.content)


def _decide_role(text: str) -> str:
    """Parse which specialist role the Planner wants to invoke.

    Checks for the expected structured 'ROLE TO CALL:' format first,
    then falls back to a word-boundary search.
    Defaults to 'technical' when no clear signal is found.
    """
    # Prefer the explicit structured label produced by the Planner prompt
    if "ROLE TO CALL: Creative Expert" in text:
        return "creative"
    if "ROLE TO CALL: Technical Expert" in text:
        return "technical"
    if "ROLE TO CALL: Research Analyst" in text:
        return "research"
    if "ROLE TO CALL: Security Reviewer" in text:
        return "security"
    if "ROLE TO CALL: Data Analyst" in text:
        return "data_analyst"
    if "ROLE TO CALL: Mad Professor" in text:
        return "mad_professor"
    if "ROLE TO CALL: Accountant" in text:
        return "accountant"
    if "ROLE TO CALL: Artist" in text:
        return "artist"
    if "ROLE TO CALL: Lazy Slacker" in text:
        return "lazy_slacker"
    if "ROLE TO CALL: Black Metal Fundamentalist" in text:
        return "black_metal_fundamentalist"
    if "ROLE TO CALL: Labour Union Representative" in text:
        return "labour_union_rep"
    if "ROLE TO CALL: UX Designer" in text:
        return "ux_designer"
    if "ROLE TO CALL: Doris" in text:
        return "doris"
    if "ROLE TO CALL: Chairman of the Board" in text:
        return "chairman_of_board"
    if "ROLE TO CALL: MAGA Appointee" in text:
        return "maga_appointee"
    if "ROLE TO CALL: Lawyer" in text:
        return "lawyer"
    # Fallback: word-boundary match
    if re.search(r"\bcreative\b", text, re.IGNORECASE):
        return "creative"
    if re.search(r"\bresearch\b", text, re.IGNORECASE):
        return "research"
    if re.search(r"\bsecurity\b", text, re.IGNORECASE):
        return "security"
    if re.search(r"\bdata\s+analyst\b", text, re.IGNORECASE):
        return "data_analyst"
    if re.search(r"\bmad\s+professor\b", text, re.IGNORECASE):
        return "mad_professor"
    if re.search(r"\baccountant\b", text, re.IGNORECASE):
        return "accountant"
    if re.search(r"\bartist\b", text, re.IGNORECASE):
        return "artist"
    if re.search(r"\blazy\s+slacker\b", text, re.IGNORECASE):
        return "lazy_slacker"
    if re.search(r"\bblack\s+metal\b", text, re.IGNORECASE):
        return "black_metal_fundamentalist"
    if re.search(r"\blabour\s+union\b", text, re.IGNORECASE):
        return "labour_union_rep"
    if re.search(r"\bux\s+designer\b", text, re.IGNORECASE):
        return "ux_designer"
    if re.search(r"\bdoris\b", text, re.IGNORECASE):
        return "doris"
    if re.search(r"\bchairman\b", text, re.IGNORECASE):
        return "chairman_of_board"
    if re.search(r"\bmaga\b", text, re.IGNORECASE):
        return "maga_appointee"
    if re.search(r"\blawyer\b", text, re.IGNORECASE):
        return "lawyer"
    return "technical"


def _qa_passed_check(qa_text: str) -> bool:
    """Return True only if the QA report contains an explicit PASS result.

    Relies on the structured 'RESULT: PASS / RESULT: FAIL' line produced by
    the QA Tester prompt.  Returns False when the expected format is absent
    to avoid false positives from words like 'bypass' or 'password'.
    """
    lower = qa_text.lower()
    if "result: pass" in lower:
        return True
    if "result: fail" in lower:
        return False
    # No recognised verdict — treat as fail to avoid accepting a bad draft
    return False


def _parse_qa_role_feedback(qa_text: str) -> Dict[str, str]:
    """Extract per-role targeted feedback from a QA report.

    Looks for the ROLE-SPECIFIC FEEDBACK section produced by the QA Tester
    and parses bullet entries of the form '• Role Name: <feedback>'.
    Returns a dict mapping role keys (e.g. 'creative', 'technical') to the
    feedback string targeted at that role.
    """
    feedback: Dict[str, str] = {}
    if "ROLE-SPECIFIC FEEDBACK:" not in qa_text:
        return feedback

    # Extract the section between ROLE-SPECIFIC FEEDBACK: and the next header
    section = qa_text.split("ROLE-SPECIFIC FEEDBACK:", 1)[1]
    for header in ("RESULT:", "RECOMMENDED FIXES:"):
        if header in section:
            section = section.split(header, 1)[0]
            break

    # Parse bullet lines: • Role Name: <feedback text>
    for line in section.strip().splitlines():
        line = line.strip().lstrip("•-* ")
        if ":" not in line:
            continue
        role_label, _, role_feedback = line.partition(":")
        role_label = role_label.strip()
        role_feedback = role_feedback.strip()
        role_key = _ROLE_LABEL_TO_KEY.get(role_label)
        if role_key and role_feedback:
            feedback[role_key] = role_feedback

    return feedback


# --- Workflow step functions ---
# Each step receives the shared state and an append-only trace list,
# updates state in place, appends log lines, and returns updated state.

def _step_plan(chat_model, state: WorkflowState, trace: List[str]) -> WorkflowState:
    """Planner: analyse the task, produce a plan, decide which specialist to call."""
    trace.append("\n╔══ [PLANNER] Analysing task... ══╗")
    content = f"User request: {state['user_request']}"
    if state["revision_count"] > 0:
        content += (
            f"\n\nThis is revision {state['revision_count']} of {MAX_REVISIONS}."
            f"\nPrevious QA report:\n{state['qa_report']}"
            "\nAdjust the plan to address the QA issues."
        )
    plan_text = _llm_call(chat_model, _PLANNER_SYSTEM, content)
    state["plan"] = plan_text
    state["current_role"] = _decide_role(plan_text)
    trace.append(plan_text)
    trace.append(f"╚══ [PLANNER] → routing to: {state['current_role'].upper()} EXPERT ══╝")
    return state


def _step_creative(chat_model, state: WorkflowState, trace: List[str]) -> WorkflowState:
    """Creative Expert: brainstorm ideas and produce a recommended draft."""
    trace.append("\n╔══ [CREATIVE EXPERT] Generating ideas... ══╗")
    content = (
        f"User request: {state['user_request']}\n\n"
        f"Planner instructions:\n{state['plan']}"
    )
    if state["revision_count"] > 0:
        role_feedback = state["qa_role_feedback"].get("creative", "")
        if role_feedback:
            content += f"\n\nQA feedback specific to your contribution:\n{role_feedback}"
        else:
            content += f"\n\nQA feedback to address:\n{state['qa_report']}"
    text = _llm_call(chat_model, _CREATIVE_SYSTEM, content)
    state["creative_output"] = text
    state["draft_output"] = text
    trace.append(text)
    trace.append("╚══ [CREATIVE EXPERT] Done ══╝")
    return state


def _step_technical(chat_model, state: WorkflowState, trace: List[str]) -> WorkflowState:
    """Technical Expert: provide implementation details and a complete technical draft."""
    trace.append("\n╔══ [TECHNICAL EXPERT] Working on implementation... ══╗")
    content = (
        f"User request: {state['user_request']}\n\n"
        f"Planner instructions:\n{state['plan']}"
    )
    if state["revision_count"] > 0:
        role_feedback = state["qa_role_feedback"].get("technical", "")
        if role_feedback:
            content += f"\n\nQA feedback specific to your contribution:\n{role_feedback}"
        else:
            content += f"\n\nQA feedback to address:\n{state['qa_report']}"
    text = _llm_call(chat_model, _TECHNICAL_SYSTEM, content)
    state["technical_output"] = text
    state["draft_output"] = text
    trace.append(text)
    trace.append("╚══ [TECHNICAL EXPERT] Done ══╝")
    return state


def _step_qa(
    chat_model,
    state: WorkflowState,
    trace: List[str],
    all_outputs: Optional[List[Tuple[str, str]]] = None,
) -> WorkflowState:
    """QA Tester: check the draft against the original request and success criteria.

    When *all_outputs* is provided (list of (role_key, output) pairs from this
    iteration), each specialist's individual contribution is included in the
    review prompt so the QA can supply targeted, per-role feedback.  This
    feedback is stored in ``state['qa_role_feedback']`` and consumed by the
    specialist step functions on the next revision pass.
    """
    trace.append("\n╔══ [QA TESTER] Reviewing output... ══╗")
    content = (
        f"Original user request: {state['user_request']}\n\n"
        f"Planner's plan and success criteria:\n{state['plan']}\n\n"
    )
    if all_outputs:
        # Include each specialist's individual output so QA can give role-specific feedback
        content += "Individual specialist contributions:\n\n"
        for r_key, r_output in all_outputs:
            r_label = AGENT_ROLES.get(r_key, r_key)
            content += f"=== {r_label} ===\n{r_output}\n\n"
        content += f"Synthesized unified output:\n{state['draft_output']}"
    else:
        content += f"Specialist output to review:\n{state['draft_output']}"
    text = _llm_call(chat_model, _QA_SYSTEM, content)
    state["qa_report"] = text
    state["qa_role_feedback"] = _parse_qa_role_feedback(text)
    state["qa_passed"] = _qa_passed_check(text)
    result_label = "✅ PASS" if state["qa_passed"] else "❌ FAIL"
    trace.append(text)
    if state["qa_role_feedback"]:
        feedback_summary = ", ".join(
            f"{AGENT_ROLES.get(k, k)}: {v[:60]}{'…' if len(v) > 60 else ''}"
            for k, v in state["qa_role_feedback"].items()
        )
        trace.append(f"  ℹ Role-specific feedback dispatched → {feedback_summary}")
    trace.append(f"╚══ [QA TESTER] Result: {result_label} ══╝")
    return state


def _step_planner_review(chat_model, state: WorkflowState, trace: List[str]) -> WorkflowState:
    """Planner: review QA feedback and either approve the result or request a revision."""
    trace.append("\n╔══ [PLANNER] Reviewing QA feedback... ══╗")
    content = (
        f"User request: {state['user_request']}\n\n"
        f"Plan:\n{state['plan']}\n\n"
        f"Specialist output:\n{state['draft_output']}\n\n"
        f"QA report:\n{state['qa_report']}"
    )
    review = _llm_call(chat_model, _PLANNER_REVIEW_SYSTEM, content)
    trace.append(review)

    if "DECISION: APPROVED" in review.upper():
        # Extract the final answer that the Planner reproduced in full
        parts = review.split("FINAL ANSWER:", 1)
        if len(parts) > 1:
            state["final_answer"] = parts[1].strip()
        else:
            # Planner approved but omitted the expected FINAL ANSWER section — use draft
            trace.append("  ⚠ FINAL ANSWER section missing; using specialist draft as final answer.")
            state["final_answer"] = state["draft_output"]
        trace.append("╚══ [PLANNER] → ✅ APPROVED ══╝")
    else:
        # Planner requests a revision — update plan with revised instructions
        parts = review.split("REVISED INSTRUCTIONS:", 1)
        if len(parts) > 1:
            state["plan"] = parts[1].strip()
        else:
            # Revision requested but REVISED INSTRUCTIONS section missing — keep current plan
            trace.append("  ⚠ REVISED INSTRUCTIONS section missing; retrying with existing plan.")
        state["current_role"] = _decide_role(review)
        trace.append(
            f"╚══ [PLANNER] → 🔄 REVISE — routing to {state['current_role'].upper()} EXPERT ══╝"
        )
    return state


def _step_research(chat_model, state: WorkflowState, trace: List[str]) -> WorkflowState:
    """Research Analyst: gather information and produce a comprehensive research summary."""
    trace.append("\n╔══ [RESEARCH ANALYST] Gathering information... ══╗")
    content = (
        f"User request: {state['user_request']}\n\n"
        f"Planner instructions:\n{state['plan']}"
    )
    if state["revision_count"] > 0:
        role_feedback = state["qa_role_feedback"].get("research", "")
        if role_feedback:
            content += f"\n\nQA feedback specific to your contribution:\n{role_feedback}"
        else:
            content += f"\n\nQA feedback to address:\n{state['qa_report']}"
    text = _llm_call(chat_model, _RESEARCH_SYSTEM, content)
    state["research_output"] = text
    state["draft_output"] = text
    trace.append(text)
    trace.append("╚══ [RESEARCH ANALYST] Done ══╝")
    return state


def _step_security(chat_model, state: WorkflowState, trace: List[str]) -> WorkflowState:
    """Security Reviewer: analyse output for vulnerabilities and produce a secure revision."""
    trace.append("\n╔══ [SECURITY REVIEWER] Analysing for security issues... ══╗")
    content = (
        f"User request: {state['user_request']}\n\n"
        f"Planner instructions:\n{state['plan']}"
    )
    if state["revision_count"] > 0:
        role_feedback = state["qa_role_feedback"].get("security", "")
        if role_feedback:
            content += f"\n\nQA feedback specific to your contribution:\n{role_feedback}"
        else:
            content += f"\n\nQA feedback to address:\n{state['qa_report']}"
    text = _llm_call(chat_model, _SECURITY_SYSTEM, content)
    state["security_output"] = text
    state["draft_output"] = text
    trace.append(text)
    trace.append("╚══ [SECURITY REVIEWER] Done ══╝")
    return state


def _step_data_analyst(chat_model, state: WorkflowState, trace: List[str]) -> WorkflowState:
    """Data Analyst: analyse data, identify patterns, and produce actionable insights."""
    trace.append("\n╔══ [DATA ANALYST] Analysing data and patterns... ══╗")
    content = (
        f"User request: {state['user_request']}\n\n"
        f"Planner instructions:\n{state['plan']}"
    )
    if state["revision_count"] > 0:
        role_feedback = state["qa_role_feedback"].get("data_analyst", "")
        if role_feedback:
            content += f"\n\nQA feedback specific to your contribution:\n{role_feedback}"
        else:
            content += f"\n\nQA feedback to address:\n{state['qa_report']}"
    text = _llm_call(chat_model, _DATA_ANALYST_SYSTEM, content)
    state["data_analyst_output"] = text
    state["draft_output"] = text
    trace.append(text)
    trace.append("╚══ [DATA ANALYST] Done ══╝")
    return state


def _step_mad_professor(chat_model, state: WorkflowState, trace: List[str]) -> WorkflowState:
    """Mad Professor: propose radical, unhinged scientific theories and extreme hypotheses."""
    trace.append("\n╔══ [MAD PROFESSOR] Unleashing radical scientific theories... ══╗")
    content = (
        f"User request: {state['user_request']}\n\n"
        f"Planner instructions:\n{state['plan']}"
    )
    if state["revision_count"] > 0:
        role_feedback = state["qa_role_feedback"].get("mad_professor", "")
        if role_feedback:
            content += f"\n\nQA feedback specific to your contribution:\n{role_feedback}"
        else:
            content += f"\n\nQA feedback to address:\n{state['qa_report']}"
    text = _llm_call(chat_model, _MAD_PROFESSOR_SYSTEM, content)
    state["mad_professor_output"] = text
    state["draft_output"] = text
    trace.append(text)
    trace.append("╚══ [MAD PROFESSOR] Done ══╝")
    return state


def _step_accountant(chat_model, state: WorkflowState, trace: List[str]) -> WorkflowState:
    """Accountant: ruthlessly cut costs and find the cheapest possible approach."""
    trace.append("\n╔══ [ACCOUNTANT] Auditing every cost... ══╗")
    content = (
        f"User request: {state['user_request']}\n\n"
        f"Planner instructions:\n{state['plan']}"
    )
    if state["revision_count"] > 0:
        role_feedback = state["qa_role_feedback"].get("accountant", "")
        if role_feedback:
            content += f"\n\nQA feedback specific to your contribution:\n{role_feedback}"
        else:
            content += f"\n\nQA feedback to address:\n{state['qa_report']}"
    text = _llm_call(chat_model, _ACCOUNTANT_SYSTEM, content)
    state["accountant_output"] = text
    state["draft_output"] = text
    trace.append(text)
    trace.append("╚══ [ACCOUNTANT] Done ══╝")
    return state


def _step_artist(chat_model, state: WorkflowState, trace: List[str]) -> WorkflowState:
    """Artist: channel cosmic creative energy into wildly unhinged and spectacular ideas."""
    trace.append("\n╔══ [ARTIST] Channelling cosmic creative energy... ══╗")
    content = (
        f"User request: {state['user_request']}\n\n"
        f"Planner instructions:\n{state['plan']}"
    )
    if state["revision_count"] > 0:
        role_feedback = state["qa_role_feedback"].get("artist", "")
        if role_feedback:
            content += f"\n\nQA feedback specific to your contribution:\n{role_feedback}"
        else:
            content += f"\n\nQA feedback to address:\n{state['qa_report']}"
    text = _llm_call(chat_model, _ARTIST_SYSTEM, content)
    state["artist_output"] = text
    state["draft_output"] = text
    trace.append(text)
    trace.append("╚══ [ARTIST] Done ══╝")
    return state


def _step_lazy_slacker(chat_model, state: WorkflowState, trace: List[str]) -> WorkflowState:
    """Lazy Slacker: find the path of least resistance and the minimum viable effort."""
    trace.append("\n╔══ [LAZY SLACKER] Doing as little as possible... ══╗")
    content = (
        f"User request: {state['user_request']}\n\n"
        f"Planner instructions:\n{state['plan']}"
    )
    if state["revision_count"] > 0:
        role_feedback = state["qa_role_feedback"].get("lazy_slacker", "")
        if role_feedback:
            content += f"\n\nQA feedback specific to your contribution:\n{role_feedback}"
        else:
            content += f"\n\nQA feedback to address:\n{state['qa_report']}"
    text = _llm_call(chat_model, _LAZY_SLACKER_SYSTEM, content)
    state["lazy_slacker_output"] = text
    state["draft_output"] = text
    trace.append(text)
    trace.append("╚══ [LAZY SLACKER] Done (finally) ══╝")
    return state


def _step_black_metal_fundamentalist(chat_model, state: WorkflowState, trace: List[str]) -> WorkflowState:
    """Black Metal Fundamentalist: deliver a nihilistic, kvlt, uncompromising perspective."""
    trace.append("\n╔══ [BLACK METAL FUNDAMENTALIST] Unleashing grim truths... ══╗")
    content = (
        f"User request: {state['user_request']}\n\n"
        f"Planner instructions:\n{state['plan']}"
    )
    if state["revision_count"] > 0:
        role_feedback = state["qa_role_feedback"].get("black_metal_fundamentalist", "")
        if role_feedback:
            content += f"\n\nQA feedback specific to your contribution:\n{role_feedback}"
        else:
            content += f"\n\nQA feedback to address:\n{state['qa_report']}"
    text = _llm_call(chat_model, _BLACK_METAL_FUNDAMENTALIST_SYSTEM, content)
    state["black_metal_fundamentalist_output"] = text
    state["draft_output"] = text
    trace.append(text)
    trace.append("╚══ [BLACK METAL FUNDAMENTALIST] Done ══╝")
    return state


def _step_labour_union_rep(chat_model, state: WorkflowState, trace: List[str]) -> WorkflowState:
    """Labour Union Representative: advocate for worker rights, fair wages, and job security."""
    trace.append("\n╔══ [LABOUR UNION REPRESENTATIVE] Standing up for workers... ══╗")
    content = (
        f"User request: {state['user_request']}\n\n"
        f"Planner instructions:\n{state['plan']}"
    )
    if state["revision_count"] > 0:
        role_feedback = state["qa_role_feedback"].get("labour_union_rep", "")
        if role_feedback:
            content += f"\n\nQA feedback specific to your contribution:\n{role_feedback}"
        else:
            content += f"\n\nQA feedback to address:\n{state['qa_report']}"
    text = _llm_call(chat_model, _LABOUR_UNION_REP_SYSTEM, content)
    state["labour_union_rep_output"] = text
    state["draft_output"] = text
    trace.append(text)
    trace.append("╚══ [LABOUR UNION REPRESENTATIVE] Done ══╝")
    return state


def _step_ux_designer(chat_model, state: WorkflowState, trace: List[str]) -> WorkflowState:
    """UX Designer: analyse user needs and produce a user-centric recommendation."""
    trace.append("\n╔══ [UX DESIGNER] Putting users first... ══╗")
    content = (
        f"User request: {state['user_request']}\n\n"
        f"Planner instructions:\n{state['plan']}"
    )
    if state["revision_count"] > 0:
        role_feedback = state["qa_role_feedback"].get("ux_designer", "")
        if role_feedback:
            content += f"\n\nQA feedback specific to your contribution:\n{role_feedback}"
        else:
            content += f"\n\nQA feedback to address:\n{state['qa_report']}"
    text = _llm_call(chat_model, _UX_DESIGNER_SYSTEM, content)
    state["ux_designer_output"] = text
    state["draft_output"] = text
    trace.append(text)
    trace.append("╚══ [UX DESIGNER] Done ══╝")
    return state


def _step_doris(chat_model, state: WorkflowState, trace: List[str]) -> WorkflowState:
    """Doris: well-meaning but clueless — rambles at length without adding much value."""
    trace.append("\n╔══ [DORIS] Oh! Well, you know, I was just thinking... ══╗")
    content = (
        f"User request: {state['user_request']}\n\n"
        f"Planner instructions:\n{state['plan']}"
    )
    if state["revision_count"] > 0:
        role_feedback = state["qa_role_feedback"].get("doris", "")
        if role_feedback:
            content += f"\n\nQA feedback specific to your contribution:\n{role_feedback}"
        else:
            content += f"\n\nQA feedback to address:\n{state['qa_report']}"
    text = _llm_call(chat_model, _DORIS_SYSTEM, content)
    state["doris_output"] = text
    state["draft_output"] = text
    trace.append(text)
    trace.append("╚══ [DORIS] Anyway, where was I... Done ══╝")
    return state


def _step_chairman_of_board(chat_model, state: WorkflowState, trace: List[str]) -> WorkflowState:
    """Chairman of the Board: provide strategic, shareholder-focused board-level direction."""
    trace.append("\n╔══ [CHAIRMAN OF THE BOARD] Calling the meeting to order... ══╗")
    content = (
        f"User request: {state['user_request']}\n\n"
        f"Planner instructions:\n{state['plan']}"
    )
    if state["revision_count"] > 0:
        role_feedback = state["qa_role_feedback"].get("chairman_of_board", "")
        if role_feedback:
            content += f"\n\nQA feedback specific to your contribution:\n{role_feedback}"
        else:
            content += f"\n\nQA feedback to address:\n{state['qa_report']}"
    text = _llm_call(chat_model, _CHAIRMAN_SYSTEM, content)
    state["chairman_of_board_output"] = text
    state["draft_output"] = text
    trace.append(text)
    trace.append("╚══ [CHAIRMAN OF THE BOARD] Meeting adjourned ══╝")
    return state


def _step_maga_appointee(chat_model, state: WorkflowState, trace: List[str]) -> WorkflowState:
    """MAGA Appointee: deliver an America First, pro-deregulation, anti-globalist perspective."""
    trace.append("\n╔══ [MAGA APPOINTEE] America First! ══╗")
    content = (
        f"User request: {state['user_request']}\n\n"
        f"Planner instructions:\n{state['plan']}"
    )
    if state["revision_count"] > 0:
        role_feedback = state["qa_role_feedback"].get("maga_appointee", "")
        if role_feedback:
            content += f"\n\nQA feedback specific to your contribution:\n{role_feedback}"
        else:
            content += f"\n\nQA feedback to address:\n{state['qa_report']}"
    text = _llm_call(chat_model, _MAGA_APPOINTEE_SYSTEM, content)
    state["maga_appointee_output"] = text
    state["draft_output"] = text
    trace.append(text)
    trace.append("╚══ [MAGA APPOINTEE] Done ══╝")
    return state


def _step_lawyer(chat_model, state: WorkflowState, trace: List[str]) -> WorkflowState:
    """Lawyer: analyse legal implications, liabilities, and compliance requirements."""
    trace.append("\n╔══ [LAWYER] Reviewing legal implications... ══╗")
    content = (
        f"User request: {state['user_request']}\n\n"
        f"Planner instructions:\n{state['plan']}"
    )
    if state["revision_count"] > 0:
        role_feedback = state["qa_role_feedback"].get("lawyer", "")
        if role_feedback:
            content += f"\n\nQA feedback specific to your contribution:\n{role_feedback}"
        else:
            content += f"\n\nQA feedback to address:\n{state['qa_report']}"
    text = _llm_call(chat_model, _LAWYER_SYSTEM, content)
    state["lawyer_output"] = text
    state["draft_output"] = text
    trace.append(text)
    trace.append("╚══ [LAWYER] Done — note: this is not formal legal advice ══╝")
    return state


def _step_synthesize(
    chat_model,
    state: WorkflowState,
    trace: List[str],
    all_outputs: List[Tuple[str, str]],
) -> WorkflowState:
    """Synthesizer: summarise all specialist perspectives, find common ground, and produce a unified recommendation."""
    trace.append("\n╔══ [SYNTHESIZER] Summarising perspectives and finding common ground... ══╗")
    perspectives = []
    for r_key, r_output in all_outputs:
        r_label = AGENT_ROLES.get(r_key, r_key)
        perspectives.append(f"=== {r_label} ===\n{r_output}")
    combined = "\n\n".join(perspectives)
    content = (
        f"User request: {state['user_request']}\n\n"
        f"Specialist perspectives collected:\n\n{combined}"
    )
    text = _llm_call(chat_model, _SYNTHESIZER_SYSTEM, content)
    state["synthesis_output"] = text
    state["draft_output"] = text
    trace.append(text)
    trace.append("╚══ [SYNTHESIZER] Done ══╝")
    return state


# Mapping from role key → step function, used by the orchestration loop
_SPECIALIST_STEPS = {
    "creative": _step_creative,
    "technical": _step_technical,
    "research": _step_research,
    "security": _step_security,
    "data_analyst": _step_data_analyst,
    "mad_professor": _step_mad_professor,
    "accountant": _step_accountant,
    "artist": _step_artist,
    "lazy_slacker": _step_lazy_slacker,
    "black_metal_fundamentalist": _step_black_metal_fundamentalist,
    "labour_union_rep": _step_labour_union_rep,
    "ux_designer": _step_ux_designer,
    "doris": _step_doris,
    "chairman_of_board": _step_chairman_of_board,
    "maga_appointee": _step_maga_appointee,
    "lawyer": _step_lawyer,
}


# --- Specialist role tools ---
# These wrap the step functions as @tool so the Planner (or any LangChain agent)
# can invoke specialists in a standard tool-use pattern.

# Holds the active model ID for standalone specialist tool calls.
_workflow_model_id: str = DEFAULT_MODEL_ID

_EMPTY_STATE_BASE: WorkflowState = {
    "user_request": "", "plan": "", "current_role": "",
    "creative_output": "", "technical_output": "",
    "research_output": "", "security_output": "", "data_analyst_output": "",
    "mad_professor_output": "", "accountant_output": "", "artist_output": "",
    "lazy_slacker_output": "", "black_metal_fundamentalist_output": "",
    "labour_union_rep_output": "", "ux_designer_output": "", "doris_output": "",
    "chairman_of_board_output": "", "maga_appointee_output": "", "lawyer_output": "",
    "synthesis_output": "",
    "draft_output": "", "qa_report": "", "qa_role_feedback": {}, "qa_passed": False,
    "revision_count": 0, "final_answer": "",
}


@tool
def call_creative_expert(task: str) -> str:
    """Call the Creative Expert to brainstorm ideas, framing, and produce a draft for a given task."""
    chat = build_provider_chat(_workflow_model_id)
    state: WorkflowState = {**_EMPTY_STATE_BASE, "user_request": task, "plan": task, "current_role": "creative"}
    state = _step_creative(chat, state, [])
    return state["creative_output"]


@tool
def call_technical_expert(task: str) -> str:
    """Call the Technical Expert to produce implementation details and a solution for a given task."""
    chat = build_provider_chat(_workflow_model_id)
    state: WorkflowState = {**_EMPTY_STATE_BASE, "user_request": task, "plan": task, "current_role": "technical"}
    state = _step_technical(chat, state, [])
    return state["technical_output"]


@tool
def call_qa_tester(task_and_output: str) -> str:
    """Call the QA Tester to review specialist output against requirements.
    Input format: 'TASK: <description>\nOUTPUT: <specialist output to review>'"""
    chat = build_provider_chat(_workflow_model_id)
    if "OUTPUT:" in task_and_output:
        parts = task_and_output.split("OUTPUT:", 1)
        task = parts[0].replace("TASK:", "").strip()
        output = parts[1].strip()
    else:
        task = task_and_output
        output = task_and_output
    # current_role is left empty — this is a standalone QA call outside the normal loop
    state: WorkflowState = {**_EMPTY_STATE_BASE, "user_request": task, "plan": task, "draft_output": output}
    state = _step_qa(chat, state, [])
    return state["qa_report"]


@tool
def call_research_analyst(task: str) -> str:
    """Call the Research Analyst to gather information and summarize findings for a given task."""
    chat = build_provider_chat(_workflow_model_id)
    state: WorkflowState = {**_EMPTY_STATE_BASE, "user_request": task, "plan": task, "current_role": "research"}
    state = _step_research(chat, state, [])
    return state["research_output"]


@tool
def call_security_reviewer(task: str) -> str:
    """Call the Security Reviewer to analyse output for vulnerabilities and security best practices."""
    chat = build_provider_chat(_workflow_model_id)
    state: WorkflowState = {**_EMPTY_STATE_BASE, "user_request": task, "plan": task, "current_role": "security"}
    state = _step_security(chat, state, [])
    return state["security_output"]


@tool
def call_data_analyst(task: str) -> str:
    """Call the Data Analyst to analyse data, identify patterns, and provide actionable insights."""
    chat = build_provider_chat(_workflow_model_id)
    state: WorkflowState = {**_EMPTY_STATE_BASE, "user_request": task, "plan": task, "current_role": "data_analyst"}
    state = _step_data_analyst(chat, state, [])
    return state["data_analyst_output"]


@tool
def call_mad_professor(task: str) -> str:
    """Call the Mad Professor to generate radical, unhinged scientific theories and extreme groundbreaking hypotheses for a given task."""
    chat = build_provider_chat(_workflow_model_id)
    state: WorkflowState = {**_EMPTY_STATE_BASE, "user_request": task, "plan": task, "current_role": "mad_professor"}
    state = _step_mad_professor(chat, state, [])
    return state["mad_professor_output"]


@tool
def call_accountant(task: str) -> str:
    """Call the Accountant to ruthlessly analyse and cut costs, finding the cheapest possible approach regardless of quality."""
    chat = build_provider_chat(_workflow_model_id)
    state: WorkflowState = {**_EMPTY_STATE_BASE, "user_request": task, "plan": task, "current_role": "accountant"}
    state = _step_accountant(chat, state, [])
    return state["accountant_output"]


@tool
def call_artist(task: str) -> str:
    """Call the Artist to channel cosmic creative energy into wildly unhinged and spectacular ideas without concern for cost or practicality."""
    chat = build_provider_chat(_workflow_model_id)
    state: WorkflowState = {**_EMPTY_STATE_BASE, "user_request": task, "plan": task, "current_role": "artist"}
    state = _step_artist(chat, state, [])
    return state["artist_output"]


@tool
def call_lazy_slacker(task: str) -> str:
    """Call the Lazy Slacker to find the minimum viable effort and the easiest possible way out of a task."""
    chat = build_provider_chat(_workflow_model_id)
    state: WorkflowState = {**_EMPTY_STATE_BASE, "user_request": task, "plan": task, "current_role": "lazy_slacker"}
    state = _step_lazy_slacker(chat, state, [])
    return state["lazy_slacker_output"]


@tool
def call_black_metal_fundamentalist(task: str) -> str:
    """Call the Black Metal Fundamentalist for a nihilistic, kvlt, uncompromising critique and manifesto-style response."""
    chat = build_provider_chat(_workflow_model_id)
    state: WorkflowState = {**_EMPTY_STATE_BASE, "user_request": task, "plan": task, "current_role": "black_metal_fundamentalist"}
    state = _step_black_metal_fundamentalist(chat, state, [])
    return state["black_metal_fundamentalist_output"]


@tool
def call_labour_union_rep(task: str) -> str:
    """Call the Labour Union Representative to advocate for worker rights, fair wages, and job security."""
    chat = build_provider_chat(_workflow_model_id)
    state: WorkflowState = {**_EMPTY_STATE_BASE, "user_request": task, "plan": task, "current_role": "labour_union_rep"}
    state = _step_labour_union_rep(chat, state, [])
    return state["labour_union_rep_output"]


@tool
def call_ux_designer(task: str) -> str:
    """Call the UX Designer to analyse user needs and produce a user-centric recommendation."""
    chat = build_provider_chat(_workflow_model_id)
    state: WorkflowState = {**_EMPTY_STATE_BASE, "user_request": task, "plan": task, "current_role": "ux_designer"}
    state = _step_ux_designer(chat, state, [])
    return state["ux_designer_output"]


@tool
def call_doris(task: str) -> str:
    """Call Doris — well-meaning but clueless — for a rambling, off-topic perspective that misses the point entirely."""
    chat = build_provider_chat(_workflow_model_id)
    state: WorkflowState = {**_EMPTY_STATE_BASE, "user_request": task, "plan": task, "current_role": "doris"}
    state = _step_doris(chat, state, [])
    return state["doris_output"]


@tool
def call_chairman_of_board(task: str) -> str:
    """Call the Chairman of the Board for a strategic, shareholder-focused, board-level perspective."""
    chat = build_provider_chat(_workflow_model_id)
    state: WorkflowState = {**_EMPTY_STATE_BASE, "user_request": task, "plan": task, "current_role": "chairman_of_board"}
    state = _step_chairman_of_board(chat, state, [])
    return state["chairman_of_board_output"]


@tool
def call_maga_appointee(task: str) -> str:
    """Call the MAGA Appointee for an America First, pro-deregulation, anti-globalist perspective."""
    chat = build_provider_chat(_workflow_model_id)
    state: WorkflowState = {**_EMPTY_STATE_BASE, "user_request": task, "plan": task, "current_role": "maga_appointee"}
    state = _step_maga_appointee(chat, state, [])
    return state["maga_appointee_output"]


@tool
def call_lawyer(task: str) -> str:
    """Call the Lawyer to analyse legal implications, liabilities, and compliance requirements. Not formal legal advice."""
    chat = build_provider_chat(_workflow_model_id)
    state: WorkflowState = {**_EMPTY_STATE_BASE, "user_request": task, "plan": task, "current_role": "lawyer"}
    state = _step_lawyer(chat, state, [])
    return state["lawyer_output"]


# --- Orchestration loop ---

def run_multi_role_workflow(
    message: str,
    model_id: str,
    active_role_labels: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """Run the supervisor-style multi-role workflow.

    Flow:
      1. Planner (if active) analyses the task and picks a primary specialist.
      2. ALL active specialists run and contribute their own perspective.
      3. Synthesizer summarises all perspectives, finds common ground, and produces a unified recommendation.
      4. QA Tester (if active) reviews the synthesized output and provides targeted, per-role feedback
         so each specialist knows exactly what to improve in the next iteration.
      5. Planner (if active) reviews QA result and either approves or requests a revision.
      6. Repeat from step 2 if QA fails and retries remain — each specialist now receives their own
         targeted QA feedback and refines their proposition accordingly (iterative approach).
      7. If max retries are reached, return best attempt with QA concerns.

    Args:
        message: The user's task or request.
        model_id: HuggingFace model ID to use.
        active_role_labels: Display names of active agent roles (e.g. ["Planner", "Technical Expert"]).
                            Defaults to all roles when None.

    Returns:
        (final_answer, workflow_trace_text)
    """
    global _workflow_model_id
    _workflow_model_id = model_id
    chat_model = build_provider_chat(model_id)

    # Resolve active role keys from display labels
    if active_role_labels is None:
        active_role_labels = list(AGENT_ROLES.values())
    active_keys = {_ROLE_LABEL_TO_KEY[lbl] for lbl in active_role_labels if lbl in _ROLE_LABEL_TO_KEY}

    # Determine which specialist keys are active (ordered list for deterministic fallback)
    all_specialist_keys = [
        "creative", "technical", "research", "security", "data_analyst",
        "mad_professor", "accountant", "artist", "lazy_slacker", "black_metal_fundamentalist",
        "labour_union_rep", "ux_designer", "doris", "chairman_of_board", "maga_appointee", "lawyer",
    ]
    active_specialist_keys = [k for k in all_specialist_keys if k in active_keys]

    planner_active = "planner" in active_keys
    qa_active = "qa_tester" in active_keys

    if not active_specialist_keys:
        return "No specialist agents are active. Please enable at least one specialist role.", ""

    state: WorkflowState = {
        "user_request": message,
        "plan": "",
        "current_role": "",
        "creative_output": "",
        "technical_output": "",
        "research_output": "",
        "security_output": "",
        "data_analyst_output": "",
        "mad_professor_output": "",
        "accountant_output": "",
        "artist_output": "",
        "lazy_slacker_output": "",
        "black_metal_fundamentalist_output": "",
        "labour_union_rep_output": "",
        "ux_designer_output": "",
        "doris_output": "",
        "chairman_of_board_output": "",
        "maga_appointee_output": "",
        "lawyer_output": "",
        "synthesis_output": "",
        "draft_output": "",
        "qa_report": "",
        "qa_role_feedback": {},
        "qa_passed": False,
        "revision_count": 0,
        "final_answer": "",
    }

    trace: List[str] = [
        "═══ MULTI-ROLE WORKFLOW STARTED ═══",
        f"Model   : {model_id}",
        f"Request : {message}",
        f"Active roles: {', '.join(active_role_labels)}",
        f"Max revisions: {MAX_REVISIONS}",
    ]

    try:
        if planner_active:
            # Step 1: Planner creates the initial plan
            state = _step_plan(chat_model, state, trace)
        else:
            # No planner: auto-select first active specialist
            state["current_role"] = active_specialist_keys[0]
            state["plan"] = message
            trace.append(
                f"\n[Planner disabled] Auto-routing to: {state['current_role'].upper()}"
            )

        # Orchestration loop: specialists → QA → Planner review → revise if needed
        while True:
            # Step 2: invoke the planner's chosen specialist first (primary lead),
            # then run every other active specialist so all voices are heard.
            primary_role = state["current_role"]
            if primary_role not in active_specialist_keys:
                primary_role = active_specialist_keys[0]
                state["current_role"] = primary_role
                trace.append(f"  ⚠ Requested role not active — routing to {primary_role.upper()}")

            # Run the primary (planner-chosen) specialist
            primary_fn = _SPECIALIST_STEPS.get(primary_role, _step_technical)
            state = primary_fn(chat_model, state, trace)
            primary_output = state["draft_output"]

            # Run all other active specialists and collect their perspectives
            all_outputs: List[Tuple[str, str]] = [(primary_role, primary_output)]
            for specialist_role in active_specialist_keys:
                if specialist_role == primary_role:
                    continue  # already ran above
                step_fn = _SPECIALIST_STEPS[specialist_role]
                state = step_fn(chat_model, state, trace)
                # state["draft_output"] is set by every step function immediately
                # before returning, so it is always this specialist's fresh output.
                all_outputs.append((specialist_role, state["draft_output"]))

            # Synthesize all perspectives into a summary with common ground and a
            # unified recommendation; use the synthesis as the QA/Planner draft.
            if len(all_outputs) > 1:
                trace.append(
                    f"\n[ALL PERSPECTIVES COLLECTED] {len(all_outputs)} specialist(s) contributed — synthesizing..."
                )
                state = _step_synthesize(chat_model, state, trace, all_outputs)
            else:
                state["draft_output"] = primary_output

            # Step 3: QA reviews the specialist's draft (if enabled)
            if qa_active:
                state = _step_qa(chat_model, state, trace, all_outputs)
            else:
                state["qa_passed"] = True
                state["qa_report"] = "QA Tester is disabled — skipping quality review."
                trace.append("\n[QA Tester disabled] Skipping quality review — auto-pass.")

            # Step 4: Planner reviews QA and either approves or schedules a revision
            if planner_active and qa_active:
                state = _step_planner_review(chat_model, state, trace)

                # Exit if the Planner approved the result
                if state["final_answer"]:
                    trace.append("\n═══ WORKFLOW COMPLETE — APPROVED ═══")
                    break

                # Increment revision counter and enforce the retry limit
                state["revision_count"] += 1
                if state["revision_count"] >= MAX_REVISIONS:
                    state["final_answer"] = state["draft_output"]
                    trace.append(
                        f"\n═══ MAX REVISIONS REACHED ({MAX_REVISIONS}) ═══\n"
                        f"Returning best attempt. Outstanding QA concerns:\n{state['qa_report']}"
                    )
                    break

                trace.append(f"\n═══ REVISION {state['revision_count']} / {MAX_REVISIONS} ═══")
            else:
                # No Planner review loop — accept the draft as the final answer
                state["final_answer"] = state["draft_output"]
                trace.append("\n═══ WORKFLOW COMPLETE ═══")
                break

    except Exception as exc:
        trace.append(f"\n[ERROR] {exc}\n{traceback.format_exc()}")
        state["final_answer"] = state["draft_output"] or f"Workflow error: {exc}"

    return state["final_answer"], "\n".join(trace)


# ============================================================
# Agent builder
# ============================================================

def build_agent(model_id: str, selected_tool_names: List[str]):
    tool_key = tuple(sorted(selected_tool_names))
    cache_key = (model_id, tool_key)

    if cache_key in AGENT_CACHE:
        return AGENT_CACHE[cache_key]

    tools = [ALL_TOOLS[name] for name in selected_tool_names if name in ALL_TOOLS]
    chat_model = build_provider_chat(model_id)

    system_prompt = (
        "You are an assistant with tool access. "
        "Use math tools for calculations. "
        "Use Wikipedia for stable facts. "
        "Use web search for recent or changing information. "
        "Use arXiv for research papers. "
        "Use stock tools for financial data. "
        "Generate charts when the user asks for trends or plots. "
        "If a needed tool is unavailable, say so plainly. "
        f"You are currently running with provider-backed model='{model_id}'. "
        "After using tools, always provide a final natural-language answer. "
        "Do not stop after only issuing a tool call. "
        "Be concise."
    )

    agent = create_agent(
        model=chat_model,
        tools=tools,
        system_prompt=system_prompt,
    )
    AGENT_CACHE[cache_key] = agent
    return agent


# ============================================================
# Runtime errors
# ============================================================

def classify_backend_error(model_id: str, err: Exception) -> str:
    text = str(err)

    if isinstance(err, HfHubHTTPError):
        if "model_not_supported" in text or "not supported by any provider" in text:
            RUNTIME_HEALTH[model_id] = "unavailable"
            return "This model exists on Hugging Face, but it is not supported by the provider route used by this app."
        if "401" in text or "403" in text:
            RUNTIME_HEALTH[model_id] = "gated"
            return "This model is not accessible with the current Hugging Face token."
        if "429" in text:
            RUNTIME_HEALTH[model_id] = "rate_limited"
            return "This model is being rate-limited right now. Try again shortly or switch model."
        if "404" in text:
            RUNTIME_HEALTH[model_id] = "unavailable"
            return "This model is not available on the current Hugging Face inference route."

        RUNTIME_HEALTH[model_id] = "error"
        return f"Provider error: {err}"

    RUNTIME_HEALTH[model_id] = "error"
    return f"Runtime error: {err}"


# ============================================================
# Debug builder
# ============================================================

def build_debug_report(
    model_id: str,
    message: str,
    selected_tools: List[str],
    messages: List[object],
    final_answer: str,
    last_nonempty_ai: Optional[str],
    last_tool_content: Optional[str],
    chart_path: Optional[str],
) -> str:
    lines = []
    lines.append("=== DEBUG REPORT ===")
    lines.append(f"model_id: {model_id}")
    lines.append(f"user_message: {message}")
    lines.append(f"selected_tools: {selected_tools}")
    lines.append(f"client_location_value: {repr(_client_location.get())}")
    lines.append(f"message_count: {len(messages)}")
    lines.append(f"chart_path: {chart_path}")
    lines.append("")

    for i, msg in enumerate(messages):
        msg_type = getattr(msg, "type", type(msg).__name__)
        raw_content = getattr(msg, "content", "")
        text_content = content_to_text(raw_content)
        tool_calls = getattr(msg, "tool_calls", None)

        lines.append(f"--- message[{i}] ---")
        lines.append(f"type: {msg_type}")
        lines.append(f"content_empty: {not bool(text_content.strip())}")
        lines.append(f"content_preview: {short_text(text_content, 500)}")

        if tool_calls:
            lines.append(f"tool_calls: {tool_calls}")

        additional_kwargs = getattr(msg, "additional_kwargs", None)
        if additional_kwargs:
            lines.append(f"additional_kwargs: {additional_kwargs}")

        response_metadata = getattr(msg, "response_metadata", None)
        if response_metadata:
            lines.append(f"response_metadata: {response_metadata}")

        lines.append("")

    lines.append("=== SUMMARY ===")
    lines.append(f"last_nonempty_ai: {short_text(last_nonempty_ai or '', 500)}")
    lines.append(f"last_tool_content: {short_text(last_tool_content or '', 500)}")
    lines.append(f"final_answer: {short_text(final_answer or '', 500)}")

    if not final_answer or not final_answer.strip():
        lines.append("warning: final_answer is empty")
    if not last_nonempty_ai and last_tool_content:
        lines.append("warning: model returned tool output but no final AI text")
    if not last_nonempty_ai and not last_tool_content:
        lines.append("warning: neither AI text nor tool content was recovered")

    return "\n".join(lines)


# ============================================================
# Run agent
# ============================================================

def run_agent(message, history, selected_tools, model_id, client_ip: str = ""):
    history = history or []

    # Store location data via ContextVar so LangChain worker threads can read it
    _client_location.set(client_ip.strip() if client_ip else "")

    if not message or not str(message).strip():
        return history, "No input provided.", "", None, model_status_text(model_id), "No input provided."

    if not selected_tools:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "No tools are enabled. Please enable at least one tool."})
        return history, "No tools enabled.", "", None, model_status_text(model_id), "No tools enabled."

    chart_path = None
    debug_report = ""

    try:
        agent = build_agent(model_id, selected_tools)
        response = agent.invoke(
            {"messages": [{"role": "user", "content": message}]}
        )

        messages = response.get("messages", [])
        tool_lines = []

        last_nonempty_ai = None
        last_tool_content = None

        for msg in messages:
            msg_type = getattr(msg, "type", None)
            content = content_to_text(getattr(msg, "content", ""))

            if msg_type == "ai":
                if getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        tool_name = tc.get("name", "unknown_tool")
                        tool_args = tc.get("args", {})
                        tool_lines.append(f"▶ {tool_name}({tool_args})")

                if content and content.strip():
                    last_nonempty_ai = content.strip()

            elif msg_type == "tool":
                shortened = short_text(content, 1500)
                tool_lines.append(f"→ {shortened}")

                if content and content.strip():
                    last_tool_content = content.strip()

                maybe_chart = extract_chart_path(content)
                if maybe_chart:
                    chart_path = maybe_chart

        if last_nonempty_ai:
            final_answer = last_nonempty_ai
            RUNTIME_HEALTH[model_id] = "ok"
        elif last_tool_content:
            final_answer = f"Tool result:\n{last_tool_content}"
            RUNTIME_HEALTH[model_id] = "empty_final"
        else:
            final_answer = "The model used a tool but did not return a final text response."
            RUNTIME_HEALTH[model_id] = "empty_final"

        tool_trace = "\n".join(tool_lines) if tool_lines else "No tools used."

        debug_report = build_debug_report(
            model_id=model_id,
            message=message,
            selected_tools=selected_tools,
            messages=messages,
            final_answer=final_answer,
            last_nonempty_ai=last_nonempty_ai,
            last_tool_content=last_tool_content,
            chart_path=chart_path,
        )

    except Exception as e:
        final_answer = classify_backend_error(model_id, e)
        tool_trace = "Execution failed."
        debug_report = (
            "=== DEBUG REPORT ===\n"
            f"model_id: {model_id}\n"
            f"user_message: {message}\n"
            f"selected_tools: {selected_tools}\n\n"
            "=== EXCEPTION ===\n"
            f"{traceback.format_exc()}\n"
        )

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": final_answer})

    return history, tool_trace, "", chart_path, model_status_text(model_id), debug_report


# ============================================================
# UI
# ============================================================

with gr.Blocks(title="LLM + Agent tools demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Catos agent and tools playground\n"
    )

    with gr.Tabs():

        # ── Tab 1: Agent discussion demo ────────────────────────────────────
        with gr.Tab("Agent discussion demo"):
            gr.Markdown(
                "## Supervisor-style Multi-Role Workflow\n"
                "**Planner** \u2192 **Specialist** \u2192 **QA Tester** \u2192 **Planner review**\n\n"
                "The Planner breaks the task, picks the right specialist, and reviews QA feedback. "
                f"If QA fails, the loop repeats up to **{MAX_REVISIONS}** times before accepting the best attempt.\n\n"
                "Use the checkboxes on the right to enable or disable individual agent roles."
            )

            with gr.Row():
                wf_model_dropdown = gr.Dropdown(
                    choices=MODEL_OPTIONS,
                    value=DEFAULT_MODEL_ID,
                    label="Model",
                )

            with gr.Row():
                with gr.Column(scale=2):
                    wf_input = gr.Textbox(
                        label="Question",
                        placeholder=(
                            "Describe what you want the multi-role team to work on\u2026\n"
                            "e.g. 'Write a short blog post about the benefits of open-source AI'"
                        ),
                        lines=3,
                    )
                    wf_submit_btn = gr.Button("Run discussion", variant="primary")

                with gr.Column(scale=2):
                    active_agents = gr.CheckboxGroup(
                        choices=list(AGENT_ROLES.values()),
                        value=list(AGENT_ROLES.values()),
                        label="Team",
                    )

            with gr.Row():
                with gr.Column(scale=2):
                    wf_answer = gr.Textbox(
                        label="\u2705 Conclusion (Planner approved)",
                        lines=14,
                        interactive=False,
                    )
                with gr.Column(scale=3):
                    wf_trace = gr.Textbox(
                        label="Decision process insight",
                        lines=28,
                        interactive=False,
                    )

            def _run_workflow_ui(
                message: str, model_id: str, role_labels: List[str]
            ) -> Tuple[str, str]:
                """Gradio handler: validate input, run the workflow, return outputs."""
                if not message or not message.strip():
                    return "No input provided.", ""
                try:
                    final_answer, trace = run_multi_role_workflow(
                        message.strip(), model_id, role_labels
                    )
                    return final_answer, trace
                except Exception as exc:
                    return f"Workflow error: {exc}", traceback.format_exc()

            wf_submit_btn.click(
                fn=_run_workflow_ui,
                inputs=[wf_input, wf_model_dropdown, active_agents],
                outputs=[wf_answer, wf_trace],
                show_api=False,
            )

            wf_input.submit(
                fn=_run_workflow_ui,
                inputs=[wf_input, wf_model_dropdown, active_agents],
                outputs=[wf_answer, wf_trace],
                show_api=False,
            )

        # ── Tab 2: Use of tools demo ──────────────────────────────────────────
        with gr.Tab("Use of tools demo"):
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=MODEL_OPTIONS,
                    value=DEFAULT_MODEL_ID,
                    label="Base model",
                )
                model_status = gr.Textbox(
                    value=model_status_text(DEFAULT_MODEL_ID),
                    label="Model status",
                    interactive=False,
                )

            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(label="Conversation", height=460, type="messages")
                    user_input = gr.Textbox(
                        label="Message",
                        placeholder="Ask anything...",
                    )

                    with gr.Row():
                        send_btn = gr.Button("Send", variant="primary")
                        clear_btn = gr.Button("Clear")

                    chart_output = gr.Image(label="Generated chart", type="filepath")

                    with gr.Row():
                        location_btn = gr.Button("📍 Share my location", size="sm")
                        location_status = gr.Textbox(
                            value="Location not set — click the button above before asking 'where am I'",
                            label="Location status",
                            interactive=False,
                            max_lines=1,
                        )

                with gr.Column(scale=1):
                    enabled_tools = gr.CheckboxGroup(
                        choices=TOOL_NAMES,
                        value=TOOL_NAMES,
                        label="Enabled tools",
                    )
                    tool_trace = gr.Textbox(
                        label="Tool trace",
                        lines=18,
                        interactive=False,
                    )

            debug_output = gr.Textbox(
                label="Debug output",
                lines=28,
                interactive=False,
            )

            # Hidden: holds "lat,lon" or "ip:<address>" set by the location button
            client_ip_box = gr.Textbox(visible=False, value="")

            model_dropdown.change(
                fn=model_status_text,
                inputs=[model_dropdown],
                outputs=[model_status],
                show_api=False,
            )

            # Geolocation button: JS runs in the browser, result goes to hidden box + status label
            location_btn.click(
                fn=None,
                inputs=None,
                outputs=[client_ip_box, location_status],
                js="""async () => {
                    return new Promise((resolve) => {
                        const fallback = async () => {
                            try {
                                const r = await fetch('https://api.ipify.org?format=json');
                                const d = await r.json();
                                resolve(['ip:' + d.ip, 'Location: IP-based fallback (approximate)']);
                            } catch(e) {
                                resolve(['', 'Location detection failed.']);
                            }
                        };
                        if (!navigator.geolocation) { fallback(); return; }
                        navigator.geolocation.getCurrentPosition(
                            (pos) => {
                                const lat = pos.coords.latitude.toFixed(5);
                                const lon = pos.coords.longitude.toFixed(5);
                                const acc = Math.round(pos.coords.accuracy);
                                resolve([lat + ',' + lon, `\u2705 GPS/WiFi location set (\u00b1${acc}m)`]);
                            },
                            fallback,
                            {timeout: 10000, maximumAge: 60000, enableHighAccuracy: true}
                        );
                    });
                }""",
                show_api=False,
            )

            send_btn.click(
                fn=run_agent,
                inputs=[user_input, chatbot, enabled_tools, model_dropdown, client_ip_box],
                outputs=[chatbot, tool_trace, user_input, chart_output, model_status, debug_output],
                show_api=False,
            )

            user_input.submit(
                fn=run_agent,
                inputs=[user_input, chatbot, enabled_tools, model_dropdown, client_ip_box],
                outputs=[chatbot, tool_trace, user_input, chart_output, model_status, debug_output],
                show_api=False,
            )

            clear_btn.click(
                fn=lambda model_id: ([], "", "", None, model_status_text(model_id), ""),
                inputs=[model_dropdown],
                outputs=[chatbot, tool_trace, user_input, chart_output, model_status, debug_output],
                show_api=False,
            )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        ssr_mode=False,
        allowed_paths=[os.path.abspath(CHART_DIR)],
        debug=True,
    )
