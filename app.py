import os
import re
import uuid
import random
import warnings
import traceback
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

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

# Thread-local storage so each Gradio request can carry the real client IP
_request_context = threading.local()


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
    """Determine the user's approximate physical location based on their public IP address."""
    client_ip = getattr(_request_context, "client_ip", "") or ""
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
            f"ISP: {data.get('isp', 'N/A')}"
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

    # Store client IP in thread-local so get_user_location can read it
    _request_context.client_ip = client_ip.strip() if client_ip else ""

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

with gr.Blocks(title="Provider Multi-Model Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Provider Multi-Model Agent\n"
        "Provider-backed models only, with selectable tools and extended debugging."
    )

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

    # Populated by JavaScript on page load with the browser's real public IP
    client_ip_box = gr.Textbox(visible=False, value="")

    demo.load(
        fn=None,
        inputs=None,
        outputs=[client_ip_box],
        js="""async () => {
            try {
                const r = await fetch('https://api.ipify.org?format=json');
                const d = await r.json();
                return d.ip;
            } catch(e) {
                return '';
            }
        }""",
    )

    model_dropdown.change(
        fn=model_status_text,
        inputs=[model_dropdown],
        outputs=[model_status],
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
