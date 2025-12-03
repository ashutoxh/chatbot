# app.py
# George Soros RAG Chatbot - Modern Gradio Interface

import base64
import html
import random
from pathlib import Path
from typing import List

import gradio as gr
from gradio import ChatMessage

try:
    import markdown
except Exception:
    markdown = None

# George Soros Quotes
SOROS_QUOTES = [
    "The financial markets generally are unpredictable. So that one has to have different scenarios... The idea that you can actually predict what's going to happen contradicts my way of looking at the market.",
    "Markets are constantly in a state of uncertainty and flux, and money is made by discounting the obvious and betting on the unexpected.",
    "I'm only rich because I know when I'm wrong. I basically have survived by recognizing my mistakes.",
    "The worst investment you can make is a good investment at the wrong price.",
    "It's not whether you're right or wrong that's important, but how much money you make when you're right and how much you lose when you're wrong.",
]

QUICK_PROMPTS = [
    "Break down Soros's reflexivity principle.",
    "How does Soros size a position when conviction is low?",
    "Summarize Soros's playbook for crisis investing.",
    "What risk controls does Soros rely on most?",
    "Explain how Soros knows when he is wrong.",
]

TEAM_MEMBERS = [
    "Ashutosh Singh",
    "Durga Sreshta Kamani",
    "Junyi Zhang",
    "Mohit Jain",
    "Neeha Girja",
    "Sujay S N",
]

_rag_engine_module = None
_hero_image_data_url = None
HERO_IMAGE_PATH = Path(__file__).parent / "img" / "Soros.png"
README_PATH = Path(__file__).parent / "README.md"
_readme_html_cache = None


def _get_rag_engine():
    """Lazy import rag_engine so the UI stays responsive."""
    global _rag_engine_module
    if _rag_engine_module is None:
        import rag_engine  # local import prevents heavy startup cost
        _rag_engine_module = rag_engine
    return _rag_engine_module


def _get_hero_image_data_url() -> str:
    """Return a base64 data URL for the Soros portrait once."""
    global _hero_image_data_url
    if _hero_image_data_url is not None:
        return _hero_image_data_url
    
    if HERO_IMAGE_PATH.exists():
        encoded = base64.b64encode(HERO_IMAGE_PATH.read_bytes()).decode("utf-8")
        _hero_image_data_url = f"data:image/png;base64,{encoded}"
    else:
        _hero_image_data_url = ""
    return _hero_image_data_url


def _load_readme_html() -> str:
    """Render README.md as HTML so the UI mirrors the docs."""
    global _readme_html_cache
    if _readme_html_cache is not None:
        return _readme_html_cache
    
    try:
        text = README_PATH.read_text(encoding="utf-8")
    except Exception as exc:
        _readme_html_cache = f"<p>Unable to load README.md: {html.escape(str(exc))}</p>"
        return _readme_html_cache
    
    if markdown is not None:
        _readme_html_cache = markdown.markdown(text)
    else:
        escaped = html.escape(text).replace("\n", "<br/>")
        _readme_html_cache = f"<pre>{escaped}</pre>"
    return _readme_html_cache


def _load_readme_html() -> str:
    """Render README.md as HTML so the UI mirrors the docs."""
    global _readme_html_cache
    if _readme_html_cache is not None:
        return _readme_html_cache
    
    try:
        text = README_PATH.read_text(encoding="utf-8")
    except Exception as exc:
        _readme_html_cache = f"<p>Unable to load README.md: {html.escape(str(exc))}</p>"
        return _readme_html_cache
    
    if markdown is not None:
        _readme_html_cache = markdown.markdown(text)
    else:
        escaped = html.escape(text).replace("\n", "<br/>")
        _readme_html_cache = f"<pre>{escaped}</pre>"
    return _readme_html_cache


def _trim_text(value: str, limit: int = 220) -> str:
    text = (value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _format_context_cards(retrieved: List[dict]) -> str:
    """Render retrieved Q&A pairs as Claude-style glass cards."""
    if not retrieved:
        return """
        <div class="context-empty">
            <p>No matching research notes yet.</p>
            <p>Ask about Soros's reflexivity loop, risk posture, or macro plays.</p>
        </div>
        """
    
    cards = []
    for item in retrieved:
        label = html.escape(item.get("label", "Core Insight") or "Core Insight")
        question = html.escape(item.get("question", ""))
        answer = html.escape(_trim_text(item.get("answer", "")))
        score = float(item.get("score", 0)) * 100
        cards.append(
            f"""
            <div class="context-card">
                <div class="context-pill">{label}</div>
                <div class="context-question">{question}</div>
                <div class="context-answer">{answer}</div>
                <div class="context-meta">Relevance: {score:.1f}%</div>
            </div>
            """
        )
    
    return f"<div class='context-grid'>{''.join(cards)}</div>"


def _build_insight_summary(retrieved: List[dict]) -> str:
    """Return a concise Markdown insight panel."""
    if not retrieved:
        return (
            "### Insight Deck\n"
            "- Start with Soros's reflexivity, risk cutting, or crisis pivots\n"
            "- Reference how he sizes trades when conviction shifts\n"
            "- Blend macro stance with micro catalysts for best answers"
        )
    
    top = retrieved[0]
    labels = [item.get("label", "").strip() for item in retrieved if item.get("label")]
    unique_labels = ", ".join(dict.fromkeys([label.title() for label in labels])) or "Core Strategy"
    question = top.get("question", "This topic")
    
    bullets = [
        f"- Anchor point: **{question}**",
        f"- Thematic threads: **{unique_labels}**",
    ]
    bullets.append("- Guidance pulled directly from curated Soros field notes.")
    
    return "### Insight Deck\n" + "\n".join(bullets)


def _build_dataset_overview() -> str:
    """Show dataset meta-stats for credibility."""
    try:
        engine = _get_rag_engine()
        df, _ = engine._load_data()
        total = len(df)
        label_count = len(df["Label"].unique()) if "Label" in df else "-"
        sample_questions = ", ".join(df["Questions"].sample(min(3, total)).tolist()) if total else ""
        return f"""
        <div class="stats-card">
            <div class="stats-title">Dataset Pulse</div>
            <div class="stats-metric">{total}</div>
            <div class="stats-subtitle">Soros research notes indexed</div>
            <div class="stats-row">
                <span>Strategy tags</span>
                <strong>{label_count}</strong>
            </div>
            <div class="stats-foot">
                Sample topics: {html.escape(_trim_text(sample_questions, 140))}
            </div>
        </div>
        """
    except Exception as exc:
        return f"""
        <div class="stats-card error">
            Unable to load dataset overview.<br/>{html.escape(str(exc))}
        </div>
        """


def _build_quote_block() -> str:
    quote = random.choice(SOROS_QUOTES)
    return f"""
    <div class="quote-card">
        <div class="quote-icon">*</div>
        <p>{html.escape(quote)}</p>
        <span>- George Soros</span>
    </div>
    """


def _coerce_history(history: List) -> List[ChatMessage]:
    """Ensure history is a flat list of ChatMessage objects."""
    if not history:
        return []
    
    normalized: List[ChatMessage] = []
    for item in history:
        if isinstance(item, ChatMessage):
            normalized.append(item)
        elif isinstance(item, dict) and "role" in item and "content" in item:
            normalized.append(ChatMessage(role=item["role"], content=item["content"]))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            normalized.append(ChatMessage(role="user", content=str(item[0])))
            normalized.append(ChatMessage(role="assistant", content=str(item[1])))
    return normalized


def _handle_chat(message: str, history: List[List[str]]):
    """Main chat handler returning updated history + side-panels."""
    history_messages = _coerce_history(history)
    message = (message or "").strip()
    if not message:
        return history_messages, "", _format_context_cards([]), _build_insight_summary([])
    
    history_messages.append(ChatMessage(role="user", content=message))
    
    try:
        engine = _get_rag_engine()
        if not engine.MODEL_READY:
            engine._load_models()
        
        result = engine.get_answer(message, top_k=4)
        answer = result.get("answer", "I couldn't generate an answer. Please try again.")
        retrieved = result.get("retrieved", [])
        
        history_messages.append(ChatMessage(role="assistant", content=answer))
        return history_messages, "", _format_context_cards(retrieved), _build_insight_summary(retrieved)
    except Exception as exc:
        error_msg = f"[WARN]  Unable to process request: {exc}"
        history_messages.append(ChatMessage(role="assistant", content=error_msg))
        return history_messages, "", _format_context_cards([]), error_msg

def create_interface():
    """Create and configure the Gradio interface with a Claude-inspired layout."""
    
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    :root {
        --surface: #07090d;
        --surface-panel: rgba(12, 14, 20, 0.95);
        --surface-glow: rgba(170, 133, 73, 0.12);
        --stroke-subtle: rgba(255, 255, 255, 0.05);
        --accent-amber: #d9b36a;
        --accent-amber-soft: rgba(217, 179, 106, 0.16);
        --accent-fern: #5e9c7e;
    }
    
    html, body {
        background: var(--surface);
    }
    
    .gradio-container {
        min-height: 100vh;
        padding: 28px !important;
        background:
            radial-gradient(circle at 10% -20%, rgba(217, 179, 106, 0.15), transparent 40%),
            radial-gradient(circle at 80% -10%, rgba(94, 156, 126, 0.12), transparent 38%),
            linear-gradient(135deg, #050608, #0b0d13);
        font-family: 'Space Grotesk', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        --background-fill-primary: rgba(9, 10, 14, 0.94);
        --background-fill-secondary: rgba(13, 15, 21, 0.92);
        --block-background-fill: rgba(11, 12, 18, 0.94);
        --border-color-primary: rgba(255, 255, 255, 0.05);
        --border-color-accent: rgba(217, 179, 106, 0.48);
        --border-color-accent-subdued: rgba(217, 179, 106, 0.26);
        --color-accent-soft: var(--accent-amber-soft);
        --body-text-color: #f5f6fa;
        --body-text-color-subdued: rgba(226, 232, 240, 0.8);
        --chatbot-text-size: 1rem;
    }
    
    .gradio-container header,
    .gradio-container footer,
    .gradio-container [data-testid="settings-button"],
    .gradio-container [data-testid="share-btn"],
    .gradio-container a[href*="gradio"],
    .gradio-container a[href*="api"],
    .gradio-container .main-menu {
        display: none !important;
    }
    
    .shell {
        max-width: 1240px;
        margin: 0 auto;
        display: flex;
        flex-direction: column;
        gap: 18px;
    }
    
    .hero {
        padding: 26px 32px;
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        background: linear-gradient(135deg, rgba(9, 10, 14, 0.98), rgba(7, 8, 11, 0.94));
        color: #f5f6fa;
        position: relative;
        overflow: hidden;
        box-shadow: 0 30px 70px rgba(0, 0, 0, 0.55);
    }
    
    .hero::after {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at 18% 5%, rgba(217, 179, 106, 0.28), transparent 55%);
        pointer-events: none;
    }
    
    .hero-top-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
    }
    
    .hero-cta {
        display: inline-flex;
        gap: 10px;
        align-items: center;
        padding: 9px 18px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.06);
        font-weight: 600;
        color: var(--accent-amber);
        font-size: 0.95rem;
    }
    
    .hero-body {
        margin-top: 18px;
    }
    
    .hero-text {
        flex: 1 1 320px;
    }
    
    .hero-text h1 {
        font-size: 2.1rem;
        margin-bottom: 0.65rem;
        letter-spacing: -0.015em;
    }
    
    .hero-text p {
        color: rgba(243, 245, 248, 0.82);
        max-width: 720px;
        line-height: 1.55;
        margin-bottom: 1.15rem;
    }
    
    .hero-tagline {
        font-size: 0.96rem;
        color: rgba(243, 245, 248, 0.85);
        letter-spacing: 0.02em;
    }
    
    .hero-team-block {
        margin-top: 18px;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    
    .hero-team-title {
        font-size: 0.85rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: rgba(248, 249, 252, 0.55);
    }
    
    .hero-team {
        list-style: none;
        padding: 0;
        margin: 0;
        display: flex;
        gap: 10px;
        font-size: 0.95rem;
        color: rgba(245, 246, 250, 0.86);
        overflow-x: auto;
        white-space: nowrap;
        scrollbar-width: none;
    }
    
    .hero-team::-webkit-scrollbar {
        display: none;
    }
    
    .hero-team li {
        padding: 10px 18px;
        border-radius: 999px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        background: rgba(255, 255, 255, 0.02);
        flex: 0 0 auto;
    }
    
    .readme-card {
        margin-top: 16px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.02);
        padding: 0;
        overflow: hidden;
    }
    
    .readme-card summary {
        cursor: pointer;
        padding: 10px 16px;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: rgba(248, 249, 252, 0.8);
        list-style: none;
    }
    
    .readme-card summary::-webkit-details-marker {
        display: none;
    }
    
    .readme-card[open] summary {
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .readme-body {
        padding: 16px;
        color: rgba(229, 233, 240, 0.85);
        font-size: 0.92rem;
        line-height: 1.5;
    }
    
    .readme-body h4 {
        margin-top: 0;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.85rem;
        color: rgba(248, 249, 252, 0.7);
    }
    
    .readme-body ul {
        padding-left: 20px;
    }
    
    .readme-body li {
        margin-bottom: 6px;
    }
    
    .hero-portrait-wrap {
        flex: 0 0 180px;
        display: flex;
        justify-content: flex-end;
        align-items: center;
    }
    
    .hero-portrait {
        width: 180px;
        max-width: 180px;
        border-radius: 22px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 25px 55px rgba(0, 0, 0, 0.5);
        background: rgba(0, 0, 0, 0.2);
    }
    
    .chat-stack {
        display: flex;
        flex-direction: column;
        gap: 18px;
    }
    
    .chat-window {
        background: rgba(11, 12, 18, 0.96);
        border-radius: 28px;
        border: 1px solid rgba(255, 255, 255, 0.04);
        padding: 30px;
        box-shadow: 0 30px 90px rgba(0, 0, 0, 0.6);
    }
    
    .chatgpt-feed {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    
    .chatgpt-feed .wrap {
        display: flex;
        flex-direction: column;
        gap: 12px;
        padding: 4px 0 12px;
    }
    
    .chatgpt-feed :where(.message-row.svelte-1nr59td) {
        padding: 0 6px;
        margin: 0;
    }
    
    .chatgpt-feed :where(.bubble.svelte-1nr59td) {
        margin: 0;
        width: 100%;
    }
    
    .chatgpt-feed :where(.message.svelte-1nr59td) {
        width: 100%;
        border: none !important;
        background: transparent !important;
        padding: 0 !important;
    }
    
    .chatgpt-feed :where(.avatar-container.svelte-1nr59td) {
        display: none !important;
    }
    
    .chatgpt-feed :where(.user.svelte-1nr59td),
    .chatgpt-feed :where(.bot.svelte-1nr59td) {
        border-radius: 22px !important;
        border-width: 1px !important;
        max-width: 78% !important;
        font-size: 1rem !important;
        line-height: 1.6;
        padding: 14px 18px !important;
        box-shadow: 0 18px 38px rgba(0, 0, 0, 0.45);
    }
    
    .chatgpt-feed :where(.user.svelte-1nr59td) {
        margin-left: auto;
        background: rgba(19, 20, 29, 0.95) !important;
        border-color: rgba(217, 179, 106, 0.45) !important;
        color: #f5f6fa !important;
    }
    
    .chatgpt-feed :where(.bot.svelte-1nr59td) {
        margin-right: auto;
        background: rgba(10, 11, 17, 0.92) !important;
        border-color: rgba(148, 163, 184, 0.22) !important;
        color: rgba(230, 233, 240, 0.92) !important;
    }
    
    .chatgpt-feed :where(.message.svelte-1nr59td p) {
        margin: 0.2rem 0;
    }
    
    .composer-card {
        background: rgba(9, 10, 14, 0.96);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.04);
        padding: 22px;
        box-shadow: 0 28px 80px rgba(0, 0, 0, 0.55);
        display: flex;
        flex-direction: column;
        gap: 16px;
    }
    
    .composer-input textarea {
        min-height: 110px !important;
        border-radius: 20px !important;
        background: rgba(13, 14, 19, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        color: #f8fafc !important;
        font-size: 1rem !important;
        padding: 16px !important;
        resize: none !important;
    }
    
    .composer-actions {
        display: flex !important;
        justify-content: flex-end;
        gap: 10px;
        flex-wrap: wrap;
    }
    
    .primary-btn, .ghost-btn {
        min-height: 46px !important;
        border-radius: 999px !important;
        padding: 0 20px !important;
        font-size: 0.95rem !important;
        border: none !important;
    }
    
    .primary-btn {
        background: linear-gradient(120deg, #d4af67, #b88a43);
        color: #070808;
        font-weight: 600;
        box-shadow: 0 12px 24px rgba(184, 138, 67, 0.4);
    }
    
    .ghost-btn {
        background: rgba(255, 255, 255, 0.05);
        color: rgba(241, 245, 255, 0.86);
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
    }
    
    .prompt-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    
    .prompt-chip {
        border-radius: 999px !important;
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        color: rgba(248, 250, 252, 0.82) !important;
        font-size: 0.92rem !important;
        padding: 8px 16px !important;
    }
    
    .context-section {
        margin-top: 6px;
        background: rgba(8, 9, 13, 0.96);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.04);
        padding: 22px;
        box-shadow: 0 26px 80px rgba(0, 0, 0, 0.45);
    }
    
    .context-heading h3 {
        margin-bottom: 14px;
        color: var(--accent-amber);
        font-size: 1.05rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    
    .context-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 16px;
    }
    
    .context-card {
        padding: 18px;
        border-radius: 18px;
        background: rgba(9, 10, 15, 0.94);
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.015);
    }
    
    .context-pill {
        display: inline-flex;
        padding: 4px 12px;
        border-radius: 999px;
        background: rgba(94, 156, 126, 0.18);
        color: #c6f2dd;
        font-size: 0.75rem;
        margin-bottom: 10px;
    }
    
    .context-question {
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 6px;
    }
    
    .context-answer {
        color: rgba(226, 232, 240, 0.85);
        line-height: 1.55;
        font-size: 0.92rem;
    }
    
    .context-meta {
        margin-top: 12px;
        font-size: 0.78rem;
        color: rgba(148, 163, 184, 0.9);
        letter-spacing: 0.06em;
    }
    
    .context-empty {
        padding: 20px;
        border-radius: 18px;
        border: 1px dashed rgba(255, 255, 255, 0.18);
        color: rgba(226, 232, 240, 0.8);
        background: rgba(9, 12, 25, 0.7);
        text-align: center;
    }
    
    .intel-grid {
        margin-top: 12px;
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 16px;
    }
    
    .intel-card {
        background: rgba(8, 9, 13, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 18px;
        padding: 18px;
        color: #e2e8f0;
        box-shadow: 0 28px 70px rgba(0, 0, 0, 0.4);
    }
    
    .stats-card {
        background: rgba(7, 8, 12, 0.95);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 18px;
        color: #f8fafc;
    }
    
    .stats-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        color: rgba(248, 250, 252, 0.7);
    }
    
    .stats-metric {
        font-size: 2.8rem;
        font-weight: 600;
        margin: 12px 0 4px;
    }
    
    .stats-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.9rem;
        color: rgba(248, 250, 252, 0.8);
        margin-top: 8px;
    }
    
    .stats-foot {
        margin-top: 14px;
        font-size: 0.85rem;
        color: rgba(226, 232, 240, 0.72);
    }
    
    .quote-card {
        background: rgba(8, 9, 13, 0.94);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 18px;
        padding: 18px;
        color: rgba(226, 232, 240, 0.9);
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    
    .quote-icon {
        font-size: 1.6rem;
        color: var(--accent-amber);
    }
    
    @media (max-width: 900px) {
        .gradio-container {
            padding: 16px !important;
        }
        
        .chat-window, .composer-card {
            padding: 18px;
        }
        
        .chatgpt-feed :where(.user.svelte-1nr59td),
        .chatgpt-feed :where(.bot.svelte-1nr59td) {
            max-width: 100% !important;
        }
    }
    """
    
    hero_image_url = _get_hero_image_data_url()
    hero_image_tag = (
        f"""
        <div class="hero-portrait-wrap">
            <img src="{hero_image_url}" alt="George Soros portrait" class="hero-portrait"/>
        </div>
        """
        if hero_image_url
        else ""
    )
    team_list = "".join(f"<li>{html.escape(member)}</li>" for member in TEAM_MEMBERS)
    readme_html = _load_readme_html()
    
    hero_html = f"""
    <section class="hero">
        <div class="hero-top-row">
            <div class="hero-cta">George Soros - Reflexivity Intelligence Suite</div>
            {hero_image_tag}
        </div>
        <div class="hero-body">
            <div class="hero-text">
                <h1>Ask Soros-level questions. Get dossier-backed answers.</h1>
                <p>
                    A modern RAG copilot tuned on Soros's research notebooks, trade journals,
                    and macro reflections. Blend qualitative insight with tactical detail -
                    the interface responds like a disciplined macro analyst.
                </p>
                <div class="hero-tagline">
                    Made by Team Soros for CSYE 7380 Theory and Practical Applications of AI Generative Modeling,
                    Fall 2025, under the guidance of Prof. Yizhen Zhao.
                </div>
                <div class="hero-team-block">
                    <div class="hero-team-title">Team Soros</div>
                    <ul class="hero-team">
                        {team_list}
                    </ul>
                </div>
            </div>
        </div>
    </section>
    """
    readme_card_html = f"""
    <details class="readme-card">
        <summary>README</summary>
        <div class="readme-body">
            {readme_html}
        </div>
    </details>
    """
    
    dataset_html = _build_dataset_overview()
    quote_html = _build_quote_block()
    
    with gr.Blocks(title="George Soros Reflexivity Copilot", fill_height=True, analytics_enabled=False) as demo:
        gr.HTML(f"<style>{custom_css}</style>")
        with gr.Column(elem_classes="shell"):
            gr.HTML(hero_html)
            gr.HTML(readme_card_html)
            
            with gr.Column(elem_classes="chat-stack"):
                with gr.Column(elem_classes="chat-window"):
                    chatbot = gr.Chatbot(
                        label="Dialogue",
                        height=600,
                        show_label=False,
                        elem_id="chatbot",
                        elem_classes="chatgpt-feed",
                        autoscroll=True,
                    )
                
                with gr.Column(elem_classes="composer-card"):
                    user_input = gr.Textbox(
                        placeholder="Ask about Soros's frameworks, risk takedowns, crisis plays...",
                        lines=3,
                        elem_id="soros-input",
                        show_label=False,
                        elem_classes="composer-input",
                    )
                    with gr.Row(elem_classes="composer-actions"):
                        clear_btn = gr.Button("Reset", elem_classes="ghost-btn")
                        send_btn = gr.Button("Send", elem_classes="primary-btn")
                    with gr.Row(elem_classes="prompt-row"):
                        for prompt in QUICK_PROMPTS:
                            gr.Button(prompt, elem_classes="prompt-chip").click(
                                fn=lambda p=prompt: p,
                                inputs=None,
                                outputs=user_input,
                            )
            
            with gr.Column(elem_classes="context-section"):
                gr.Markdown("### Context intelligence", elem_classes="context-heading")
                context_panel = gr.HTML(
                    value=_format_context_cards([]),
                    elem_classes="context-body",
                )
            
            with gr.Row(elem_classes="intel-grid"):
                insight_panel = gr.Markdown(
                    value=_build_insight_summary([]),
                    elem_classes="intel-card",
                )
                gr.HTML(dataset_html, elem_classes="intel-card")
                gr.HTML(quote_html, elem_classes="intel-card")
        
        send_btn.click(
            _handle_chat,
            inputs=[user_input, chatbot],
            outputs=[chatbot, user_input, context_panel, insight_panel],
        )
        user_input.submit(
            _handle_chat,
            inputs=[user_input, chatbot],
            outputs=[chatbot, user_input, context_panel, insight_panel],
        )
        clear_btn.click(
            lambda: ([], "", _format_context_cards([]), _build_insight_summary([])),
            inputs=None,
            outputs=[chatbot, user_input, context_panel, insight_panel],
        )
    
    return demo

if __name__ == "__main__":
    print("[START]  Starting George Soros RAG Chatbot...")
    print("[INFO]  Preloading Soros dataset + semantic index (CPU-only, cross-platform)...\n")
    
    # Load models BEFORE starting the server
    # Try to load models directly first (faster if it works)
    models_loaded = False
    try:
        print("Step 1: Importing RAG engine module...")
        import rag_engine
        print("OK Module imported")
        
        print("Step 2: Loading dataset...")
        rag_engine._load_data()
        print("OK Dataset loaded")
        
        print("Step 3: Building lightweight semantic index...")
        print("   - Initializing TF-IDF vectorizer...")
        print("   - Fitting Soros corpus (1-2 gram features)...")
        print("   - Ready for instant CPU inference...")
        
        success = rag_engine._load_models()
        
        if success and rag_engine.MODEL_READY:
            print("\n[OK]  All models loaded successfully!")
            models_loaded = True
        else:
            print(f"\n[WARN]   Models not ready: {rag_engine.MODEL_ERROR}")
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as e:
        # If direct loading fails, try subprocess (isolates segfaults)
        error_type = type(e).__name__
        print(f"\n[WARN]   Direct loading failed ({error_type}): {e}")
        print("   Trying subprocess isolation...")
        
        import subprocess
        import sys
        try:
            result = subprocess.run(
                [sys.executable, "load_models.py"],
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode == 0 and "SUCCESS" in result.stdout:
                # Models loaded in subprocess, now import them
                import rag_engine
                if rag_engine.MODEL_READY:
                    print("[OK]  Models loaded successfully via subprocess!")
                    models_loaded = True
                else:
                    print(f"[WARN]   Models loaded but not ready: {rag_engine.MODEL_ERROR}")
            else:
                print(f"[WARN]   Subprocess failed: {result.stderr or result.stdout}")
        except subprocess.TimeoutExpired:
            print("[WARN]   Model loading timed out (>2 minutes)")
        except Exception as sub_e:
            print(f"[WARN]   Subprocess error: {sub_e}")
    
    if models_loaded:
        print("\n[WEB]  Starting web interface (models ready - chat is ready!)...\n")
    else:
        print("\n[WEB]  Starting web interface (semantic index failed to load)...\n")
        print("   Chat will attempt to rebuild the index on first question.\n")
    
    demo = create_interface()
    # Try port 7860, but let Gradio find another if it's taken
    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True,
        )
    except OSError:
        # Port 7860 is taken, let Gradio auto-select
        print("[WARN]   Port 7860 is in use, using auto-selected port...")
        demo.launch(
            server_name="127.0.0.1",
            server_port=None,
            share=False,
            show_error=True,
        )
