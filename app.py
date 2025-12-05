# app.py
# George Soros RAG Chatbot - Modern Gradio Interface

import base64
import html
import io
import random
import tempfile
import os
from pathlib import Path
from typing import List

import gradio as gr
from gradio import ChatMessage
from datetime import datetime, timedelta

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import markdown
except Exception:
    markdown = None

try:
    import pairs_trading
    from pairs_trading import (
        run_pairs_analysis,
        PairsTradingError,
        InvalidDateRangeError,
        InsufficientDataError,
        CointegrationRequirementError
    )
except ImportError:
    pairs_trading = None

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
    # Don't cache - always reload to show latest README updates
    try:
        text = README_PATH.read_text(encoding="utf-8")
    except Exception as exc:
        return f"<p>Unable to load README.md: {html.escape(str(exc))}</p>"
    
    if markdown is not None:
        return markdown.markdown(text)
    else:
        escaped = html.escape(text).replace("\n", "<br/>")
        return f"<pre>{escaped}</pre>"


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


def _handle_pairs_trading_toggle():
    """Toggle visibility of pairs trading form."""
    return gr.update(visible=True)


def _format_pairs_trading_result(result, show_optional_cointegration=False) -> str:
    """Format pairs trading result as HTML summary with proper UI cards."""
    stats = result.summary_stats
    p_value = stats['trading_p_value']
    is_good_pair = stats['trading_cointegration_passed']
    
    # Format PnL with color
    pnl_value = stats['total_pnl']
    pnl_class = "positive" if pnl_value >= 0 else "negative"
    pnl_sign = "+" if pnl_value >= 0 else ""
    
    summary_html = f"""
    <div class="pairs-results-container">
        <div class="results-header">
            <h2 class="results-title">Pairs Trading Analysis</h2>
            <div class="results-subtitle">{result.stock1} vs {result.stock2}</div>
            <div class="results-date-range">{result.start_date} to {result.end_date}</div>
        </div>
    """
    
    # Only show pair quality assessment and cointegration details when checkbox is checked
    if show_optional_cointegration:
        # Pair quality indicator with styling
        if is_good_pair:
            pair_status_html = f"""
        <div class="pair-status-card good-pair">
            <div class="pair-status-icon">✅</div>
            <div class="pair-status-content">
                <div class="pair-status-title">GOOD PAIR</div>
                <div class="pair-status-desc">Shows strong cointegration (p-value < 0.05)</div>
            </div>
        </div>
        """
        else:
            pair_status_html = f"""
        <div class="pair-status-card weak-pair">
            <div class="pair-status-icon">⚠️</div>
            <div class="pair-status-content">
                <div class="pair-status-title">WEAK PAIR</div>
                <div class="pair-status-desc">Limited cointegration (p-value >= 0.05). Strategy may be less effective.</div>
            </div>
        </div>
        """
        
        summary_html += pair_status_html
        
        summary_html += f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Profit/Loss</div>
                <div class="metric-value {pnl_class}">{pnl_sign}{pnl_value:.6f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">{stats['num_trades']}</div>
                <div class="metric-subtext">Long: {stats['num_long']} | Short: {stats['num_short']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">P-Value</div>
                <div class="metric-value">{p_value:.4f}</div>
                <div class="metric-subtext">{'Cointegrated' if is_good_pair else 'Not Cointegrated'}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Test Statistic</div>
                <div class="metric-value">{stats['trading_score']:.4f}</div>
            </div>
        </div>
        """
    else:
        # When checkbox is unchecked, show only performance metrics (no cointegration details)
        summary_html += f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Profit/Loss</div>
                <div class="metric-value {pnl_class}">{pnl_sign}{pnl_value:.6f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">{stats['num_trades']}</div>
                <div class="metric-subtext">Long: {stats['num_long']} | Short: {stats['num_short']}</div>
            </div>
        </div>
        """
    
    summary_html += f"""
        <div class="spread-metrics-grid">
            <div class="metric-card-small">
                <div class="metric-label-small">Mean Spread</div>
                <div class="metric-value-small">{stats['mean_spread']:.4f}</div>
            </div>
            <div class="metric-card-small">
                <div class="metric-label-small">Spread Std Dev</div>
                <div class="metric-value-small">{stats['std_spread']:.4f}</div>
            </div>
            <div class="metric-card-small">
                <div class="metric-label-small">Upper Threshold</div>
                <div class="metric-value-small">{stats['upper_threshold']:.4f}</div>
            </div>
            <div class="metric-card-small">
                <div class="metric-label-small">Lower Threshold</div>
                <div class="metric-value-small">{stats['lower_threshold']:.4f}</div>
            </div>
        </div>
    """
    
    # Only show optional cointegration test details if checkbox was checked
    if show_optional_cointegration:
        summary_html += f"""
        <div class="optional-test-card">
            <div class="optional-test-header">Optional Cointegration Test (User Requested)</div>
            <div class="optional-test-content">
                <div class="optional-test-row">
                    <span class="optional-test-label">Test Statistic:</span>
                    <span class="optional-test-value">{result.cointegration_score:.4f}</span>
                </div>
                <div class="optional-test-row">
                    <span class="optional-test-label">P-Value:</span>
                    <span class="optional-test-value">{result.cointegration_p_value:.4f}</span>
                </div>
                <div class="optional-test-row">
                    <span class="optional-test-label">Result:</span>
                    <span class="optional-test-value {'passed' if result.cointegration_passed else 'failed'}">
                        {'PASSED - Pair is cointegrated' if result.cointegration_passed else 'NOT COINTEGRATED - Pair may not be suitable'}
                    </span>
                </div>
            </div>
        </div>
        """
    
    summary_html += """
        <div class="interpretation-card">
            <div class="interpretation-header">Strategy Interpretation</div>
            <div class="interpretation-text">
                The strategy monitors the spread between the two stocks. When the spread deviates significantly 
                from its mean (beyond the thresholds), positions are taken to profit from mean reversion.
            </div>
        </div>
    </div>
    """
    
    return summary_html


def _handle_pairs_trading_analysis(
    stock1: str,
    stock2: str,
    start_date: str,
    end_date: str,
    run_cointegration: bool
):
    """Handle pairs trading analysis request."""
    print(f"[DEBUG] Pairs trading analysis called: {stock1} vs {stock2}, {start_date} to {end_date}, cointegration={run_cointegration}")
    
    if not pairs_trading:
        error_msg = "**Pairs Trading Module Not Available**\n\nPlease ensure all dependencies are installed:\n```bash\npip install -r requirements.txt\n```"
        print("[ERROR] pairs_trading module not available")
        return (
            "",
            error_msg,
            None,
            None,
            None
        )
    
    try:
        print("[DEBUG] Calling run_pairs_analysis...")
        result = run_pairs_analysis(
            stock1=stock1,
            stock2=stock2,
            start_date=start_date,
            end_date=end_date,
            run_cointegration_test=run_cointegration
        )
        print("[DEBUG] Analysis completed successfully")
        
        summary_md = _format_pairs_trading_result(result)
        
        # Convert base64 plots to PIL Images for Gradio
        # Gradio's Image component with type="pil" handles PIL Images directly
        try:
            if Image is not None:
                # Decode base64 to PIL Image
                spread_bytes = base64.b64decode(result.spread_plot_base64)
                spread_pil = Image.open(io.BytesIO(spread_bytes))
                # Convert RGBA to RGB if needed
                if spread_pil.mode == 'RGBA':
                    rgb_img = Image.new('RGB', spread_pil.size, (255, 255, 255))
                    rgb_img.paste(spread_pil, mask=spread_pil.split()[3])
                    spread_img = rgb_img
                else:
                    spread_img = spread_pil.convert('RGB')
                
                pnl_bytes = base64.b64decode(result.pnl_plot_base64)
                pnl_pil = Image.open(io.BytesIO(pnl_bytes))
                # Convert RGBA to RGB if needed
                if pnl_pil.mode == 'RGBA':
                    rgb_img = Image.new('RGB', pnl_pil.size, (255, 255, 255))
                    rgb_img.paste(pnl_pil, mask=pnl_pil.split()[3])
                    pnl_img = rgb_img
                else:
                    pnl_img = pnl_pil.convert('RGB')
                
                print(f"[DEBUG] Images converted to PIL - spread size: {spread_img.size}, pnl size: {pnl_img.size}")
            else:
                # Fallback: return base64 strings if PIL not available
                spread_img = f"data:image/png;base64,{result.spread_plot_base64}"
                pnl_img = f"data:image/png;base64,{result.pnl_plot_base64}"
                print("[DEBUG] Using base64 strings (PIL not available)")
        except Exception as img_error:
            # If image conversion fails, return None and let Gradio handle it
            import traceback
            print(f"[WARN] Image conversion failed: {img_error}")
            print(traceback.format_exc())
            spread_img = None
            pnl_img = None
        
        # Format trading signals dataframe
        signals_df = result.trading_signals.copy()
        signals_df.index.name = 'Date'
        signals_df = signals_df.reset_index()
        signals_df['Date'] = signals_df['Date'].astype(str)
        
        print("[DEBUG] Returning results")
        return (
            summary_md,
            "",
            spread_img,
            pnl_img,
            signals_df
        )
        
    except CointegrationRequirementError as e:
        # This should no longer be raised, but keep for backwards compatibility
        error_msg = f"**Strategy Blocked**\n\n{str(e)}\n\nPlease select a different pair or date range."
        print(f"[ERROR] Cointegration requirement failed: {e}")
        return "", error_msg, None, None, None
    except InvalidDateRangeError as e:
        error_msg = f"**Invalid Date Range**\n\n{str(e)}"
        print(f"[ERROR] Invalid date range: {e}")
        return "", error_msg, None, None, None
    except InsufficientDataError as e:
        error_msg = f"**Data Error**\n\n{str(e)}"
        print(f"[ERROR] Insufficient data: {e}")
        return "", error_msg, None, None, None
    except PairsTradingError as e:
        error_msg = f"**Error**\n\n{str(e)}"
        print(f"[ERROR] Pairs trading error: {e}")
        return "", error_msg, None, None, None
    except Exception as e:
        import traceback
        error_msg = f"**Unexpected Error**\n\n{str(e)}\n\n```\n{traceback.format_exc()}\n```\n\nPlease check your inputs and try again."
        print(f"[ERROR] Unexpected error: {e}")
        print(traceback.format_exc())
        return "", error_msg, None, None, None

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
    
    .main-content-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 24px;
        margin-top: 24px;
    }
    
    .pairs-trading-section {
        background: rgba(8, 9, 13, 0.96);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.04);
        padding: 28px;
        box-shadow: 0 28px 80px rgba(0, 0, 0, 0.55);
        display: flex;
        flex-direction: column;
        height: fit-content;
    }
    
    .pairs-trading-section h2 {
        margin: 0 0 20px 0;
        font-size: 1.5rem;
        color: var(--accent-amber);
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    
    .chat-section {
        background: rgba(8, 9, 13, 0.96);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.04);
        padding: 28px;
        box-shadow: 0 28px 80px rgba(0, 0, 0, 0.55);
        display: flex;
        flex-direction: column;
    }
    
    .chat-section h2 {
        margin: 0 0 20px 0;
        font-size: 1.5rem;
        color: var(--accent-fern);
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    
    .pairs-trading-cta {
        width: 100%;
        min-height: 56px !important;
        border-radius: 16px !important;
        background: linear-gradient(120deg, #d4af67, #b88a43) !important;
        color: #070808 !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        border: none !important;
        box-shadow: 0 12px 24px rgba(184, 138, 67, 0.4) !important;
        margin-bottom: 20px !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
        cursor: pointer !important;
        pointer-events: auto !important;
        z-index: 10 !important;
    }
    
    .pairs-trading-cta:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 32px rgba(184, 138, 67, 0.5) !important;
    }
    
    .pairs-trading-cta:disabled {
        opacity: 0.6;
        cursor: not-allowed !important;
    }
    
    .pairs-trading-form {
        display: flex;
        flex-direction: column;
        gap: 18px;
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(-10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .pairs-trading-form .input-group {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    
    .pairs-trading-form label {
        font-size: 0.9rem;
        color: rgba(248, 250, 252, 0.85);
        font-weight: 500;
    }
    
    .pairs-trading-form input,
    .pairs-trading-form select {
        background: rgba(13, 14, 19, 0.95) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
        padding: 12px 16px !important;
        font-size: 0.95rem !important;
    }
    
    .pairs-trading-form .checkbox-group {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .pairs-cointegration-checkbox {
        cursor: pointer !important;
        pointer-events: auto !important;
        padding: 12px !important;
        border-radius: 8px !important;
        background: rgba(13, 14, 19, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        position: relative !important;
        z-index: 100 !important;
        transition: all 0.2s ease !important;
    }
    
    .pairs-cointegration-checkbox * {
        pointer-events: auto !important;
        cursor: pointer !important;
    }
    
    .pairs-cointegration-checkbox label {
        cursor: pointer !important;
        pointer-events: auto !important;
        user-select: none;
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
    }
    
    .pairs-cointegration-checkbox input[type="checkbox"] {
        cursor: pointer !important;
        pointer-events: auto !important;
        width: 22px !important;
        height: 22px !important;
        margin: 0 !important;
        accent-color: var(--accent-fern) !important;
        flex-shrink: 0 !important;
        position: relative !important;
        z-index: 101 !important;
        appearance: auto !important;
        -webkit-appearance: checkbox !important;
    }
    
    .pairs-cointegration-checkbox input[type="checkbox"]:checked {
        accent-color: var(--accent-fern) !important;
        background-color: var(--accent-fern) !important;
    }
    
    .pairs-cointegration-checkbox:hover {
        background: rgba(13, 14, 19, 0.7) !important;
        border-color: rgba(94, 156, 126, 0.3) !important;
    }
    
    .pairs-cointegration-checkbox:has(input[type="checkbox"]:checked) {
        background: rgba(94, 156, 126, 0.15) !important;
        border-color: rgba(94, 156, 126, 0.4) !important;
    }
    
    .pairs-trading-run-btn {
        min-height: 48px !important;
        border-radius: 12px !important;
        background: linear-gradient(120deg, #5e9c7e, #4a7c65) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: none !important;
        box-shadow: 0 8px 16px rgba(94, 156, 126, 0.3) !important;
        cursor: pointer !important;
        pointer-events: auto !important;
        z-index: 10 !important;
        width: 100% !important;
    }
    
    .pairs-trading-run-btn:hover {
        background: linear-gradient(120deg, #6bac8e, #5a8c75) !important;
        box-shadow: 0 12px 24px rgba(94, 156, 126, 0.4) !important;
    }
    
    .pairs-trading-run-btn:disabled {
        opacity: 0.6;
        cursor: not-allowed !important;
    }
    
    .pairs-trading-results {
        margin-top: 24px;
        display: flex;
        flex-direction: column;
        gap: 24px;
        animation: fadeIn 0.4s ease-in;
    }
    
    .pairs-trading-plot {
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(9, 10, 15, 0.94);
        padding: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    
    .pairs-trading-plot img {
        width: 100%;
        height: auto;
        border-radius: 8px;
    }
    
    .pairs-trading-dataframe {
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(9, 10, 15, 0.94);
        padding: 20px;
        overflow-x: auto;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    
    .pairs-results-container {
        display: flex;
        flex-direction: column;
        gap: 20px;
    }
    
    .results-header {
        text-align: center;
        padding: 20px 0;
        border-bottom: 2px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 10px;
    }
    
    .results-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--accent-amber);
        margin: 0 0 8px 0;
        letter-spacing: -0.02em;
    }
    
    .results-subtitle {
        font-size: 1.3rem;
        font-weight: 600;
        color: #f8fafc;
        margin: 0 0 4px 0;
    }
    
    .results-date-range {
        font-size: 0.95rem;
        color: rgba(248, 250, 252, 0.7);
    }
    
    .pair-status-card {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 20px 24px;
        border-radius: 16px;
        border: 2px solid;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    
    .pair-status-card.good-pair {
        background: rgba(94, 156, 126, 0.15);
        border-color: rgba(94, 156, 126, 0.5);
    }
    
    .pair-status-card.weak-pair {
        background: rgba(217, 179, 106, 0.15);
        border-color: rgba(217, 179, 106, 0.5);
    }
    
    .pair-status-icon {
        font-size: 2.5rem;
        flex-shrink: 0;
    }
    
    .pair-status-content {
        flex: 1;
    }
    
    .pair-status-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    
    .pair-status-desc {
        font-size: 0.95rem;
        color: rgba(248, 250, 252, 0.85);
        line-height: 1.5;
    }
    
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin: 20px 0;
    }
    
    .metric-card {
        background: rgba(9, 10, 15, 0.94);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: rgba(248, 250, 252, 0.7);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    
    .metric-value.positive {
        color: #5e9c7e;
    }
    
    .metric-value.negative {
        color: #d4af67;
    }
    
    .metric-subtext {
        font-size: 0.8rem;
        color: rgba(248, 250, 252, 0.6);
        margin-top: 4px;
    }
    
    .spread-metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 12px;
        margin: 16px 0;
    }
    
    .metric-card-small {
        background: rgba(9, 10, 15, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
    
    .metric-label-small {
        font-size: 0.75rem;
        color: rgba(248, 250, 252, 0.65);
        margin-bottom: 6px;
    }
    
    .metric-value-small {
        font-size: 1.2rem;
        font-weight: 600;
        color: #f8fafc;
    }
    
    .optional-test-card {
        background: rgba(94, 156, 126, 0.1);
        border: 1px solid rgba(94, 156, 126, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
    }
    
    .optional-test-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--accent-fern);
        margin-bottom: 16px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .optional-test-content {
        display: flex;
        flex-direction: column;
        gap: 12px;
    }
    
    .optional-test-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .optional-test-row:last-child {
        border-bottom: none;
    }
    
    .optional-test-label {
        font-size: 0.9rem;
        color: rgba(248, 250, 252, 0.8);
        font-weight: 500;
    }
    
    .optional-test-value {
        font-size: 0.95rem;
        color: #f8fafc;
        font-weight: 600;
    }
    
    .optional-test-value.passed {
        color: var(--accent-fern);
    }
    
    .optional-test-value.failed {
        color: var(--accent-amber);
    }
    
    .interpretation-card {
        background: rgba(8, 9, 13, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
    }
    
    .interpretation-header {
        font-size: 1rem;
        font-weight: 600;
        color: var(--accent-amber);
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .interpretation-text {
        font-size: 0.95rem;
        color: rgba(248, 250, 252, 0.85);
        line-height: 1.6;
    }
    
    @media (max-width: 1200px) {
        .main-content-grid {
            grid-template-columns: 1fr;
        }
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
            
            # Main Content Grid - Pairs Trading and Chat side by side
            with gr.Row(elem_classes="main-content-grid"):
                # Pairs Trading Section
                with gr.Column(elem_classes="pairs-trading-section"):
                    gr.Markdown("## Pairs Trading Strategy", elem_classes="pairs-trading-title")
                    pairs_trading_visible = gr.State(value=False)
                    pairs_trading_cta_btn = gr.Button(
                        "Test George Soros' Pair Trading Strategy",
                        elem_classes="pairs-trading-cta"
                    )
                    
                    with gr.Column(visible=False, elem_classes="pairs-trading-form") as pairs_trading_form:
                        with gr.Row():
                            with gr.Column(elem_classes="input-group"):
                                pairs_stock1 = gr.Textbox(
                                    label="Stock 1 Ticker",
                                    placeholder="e.g., XOM",
                                    value="XOM"
                                )
                            with gr.Column(elem_classes="input-group"):
                                pairs_stock2 = gr.Textbox(
                                    label="Stock 2 Ticker",
                                    placeholder="e.g., CVX",
                                    value="CVX"
                                )
                        
                        with gr.Row():
                            with gr.Column(elem_classes="input-group"):
                                # Default to 1 year ago to today
                                default_start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
                                default_end = datetime.now().strftime("%Y-%m-%d")
                                pairs_start_date = gr.Textbox(
                                    label="Start Date (YYYY-MM-DD)",
                                    placeholder="e.g., 2023-01-01",
                                    value=default_start
                                )
                            with gr.Column(elem_classes="input-group"):
                                pairs_end_date = gr.Textbox(
                                    label="End Date (YYYY-MM-DD)",
                                    placeholder="e.g., 2023-12-31",
                                    value=default_end
                                )
                        
                        with gr.Row():
                            pairs_run_cointegration = gr.Checkbox(
                                label="Run optional cointegration test",
                                value=False,
                                info="Check this to see detailed cointegration test results in the output. Note: Pair quality assessment always runs regardless of this setting.",
                                interactive=True,
                                elem_classes="pairs-cointegration-checkbox",
                                container=True
                            )
                        
                        pairs_run_btn = gr.Button(
                            "Run Pairs Trading Analysis",
                            elem_classes="pairs-trading-run-btn"
                        )
                        
                        pairs_error_msg = gr.Markdown(visible=False)
                        
                        with gr.Column(visible=False, elem_classes="pairs-trading-results") as pairs_results:
                            pairs_summary = gr.HTML()
                            
                            with gr.Row():
                                pairs_spread_plot = gr.Image(
                                    label="Spread Analysis", 
                                    elem_classes="pairs-trading-plot",
                                    type="pil"
                                )
                                pairs_pnl_plot = gr.Image(
                                    label="Cumulative PnL", 
                                    elem_classes="pairs-trading-plot",
                                    type="pil"
                                )
                            
                            pairs_signals_df = gr.Dataframe(
                                label="Recent Trading Signals (Last 30 Days)",
                                elem_classes="pairs-trading-dataframe",
                                wrap=True
                            )
                
                # Chat Section
                with gr.Column(elem_classes="chat-section"):
                    gr.Markdown("## Soros RAG Chatbot", elem_classes="chat-title")
                    with gr.Column(elem_classes="chat-stack"):
                        with gr.Column(elem_classes="chat-window"):
                            chatbot = gr.Chatbot(
                                label="Dialogue",
                                height=500,
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
            
            # Context and Intelligence Section (below main grid)
            with gr.Column(elem_classes="context-section"):
                gr.Markdown("### Context Intelligence", elem_classes="context-heading")
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
        
        # Pairs Trading Event Handlers
        def toggle_form():
            """Toggle form visibility - always show on click."""
            print("[DEBUG] Toggle form button clicked")
            return gr.update(visible=True)
        
        pairs_trading_cta_btn.click(
            fn=toggle_form,
            inputs=None,
            outputs=[pairs_trading_form],
            show_progress=False
        )
        
        def handle_analysis_with_visibility(stock1, stock2, start_date, end_date, run_cointegration):
            """Handle analysis and return results with visibility updates."""
            print(f"[DEBUG] Run button clicked - stock1={stock1}, stock2={stock2}, start={start_date}, end={end_date}, cointegration_checkbox={run_cointegration} (type: {type(run_cointegration)})")
            
            if not pairs_trading:
                error_msg = "**Pairs Trading Module Not Available**\n\nPlease ensure all dependencies are installed."
                return "", error_msg, None, None, None, gr.update(visible=False)
            
            # Ensure boolean conversion
            run_cointegration = bool(run_cointegration) if run_cointegration is not None else False
            print(f"[DEBUG] Cointegration checkbox value (after conversion): {run_cointegration}")
            
            try:
                # Call the analysis handler
                result_obj = run_pairs_analysis(
                    stock1=stock1,
                    stock2=stock2,
                    start_date=start_date,
                    end_date=end_date,
                    run_cointegration_test=run_cointegration
                )
                
                # Format summary with optional cointegration flag
                summary_html = _format_pairs_trading_result(result_obj, show_optional_cointegration=run_cointegration)
                
                # Convert images
                try:
                    if Image is not None:
                        spread_bytes = base64.b64decode(result_obj.spread_plot_base64)
                        spread_pil = Image.open(io.BytesIO(spread_bytes))
                        if spread_pil.mode == 'RGBA':
                            rgb_img = Image.new('RGB', spread_pil.size, (255, 255, 255))
                            rgb_img.paste(spread_pil, mask=spread_pil.split()[3])
                            spread_img = rgb_img
                        else:
                            spread_img = spread_pil.convert('RGB')
                        
                        pnl_bytes = base64.b64decode(result_obj.pnl_plot_base64)
                        pnl_pil = Image.open(io.BytesIO(pnl_bytes))
                        if pnl_pil.mode == 'RGBA':
                            rgb_img = Image.new('RGB', pnl_pil.size, (255, 255, 255))
                            rgb_img.paste(pnl_pil, mask=pnl_pil.split()[3])
                            pnl_img = rgb_img
                        else:
                            pnl_img = pnl_pil.convert('RGB')
                    else:
                        spread_img = f"data:image/png;base64,{result_obj.spread_plot_base64}"
                        pnl_img = f"data:image/png;base64,{result_obj.pnl_plot_base64}"
                except Exception as img_error:
                    print(f"[WARN] Image conversion failed: {img_error}")
                    spread_img = None
                    pnl_img = None
                
                # Format trading signals dataframe
                signals_df = result_obj.trading_signals.copy()
                signals_df.index.name = 'Date'
                signals_df = signals_df.reset_index()
                signals_df['Date'] = signals_df['Date'].astype(str)
                
                summary = summary_html
                error_msg = ""
                
                # Determine visibility
                show_results = bool(summary)
                show_error = False
                
                print(f"[DEBUG] Final visibility - show_results={show_results}, summary_len={len(summary) if summary else 0}")
                print(f"[DEBUG] Optional cointegration shown: {run_cointegration}")
                
                # Return all outputs including visibility updates
                return (
                    summary,  # pairs_summary
                    gr.update(value=error_msg, visible=show_error),  # pairs_error_msg (content + visibility)
                    spread_img,  # pairs_spread_plot
                    pnl_img,  # pairs_pnl_plot
                    signals_df,  # pairs_signals_df
                    gr.update(visible=show_results),  # pairs_results visibility
                )
                
            except CointegrationRequirementError as e:
                error_msg = f"**Strategy Blocked**\n\n{str(e)}\n\nPlease select a different pair or date range."
                print(f"[ERROR] Cointegration requirement failed: {e}")
                return "", gr.update(value=error_msg, visible=True), None, None, None, gr.update(visible=False)
            except InvalidDateRangeError as e:
                error_msg = f"**Invalid Date Range**\n\n{str(e)}"
                print(f"[ERROR] Invalid date range: {e}")
                return "", gr.update(value=error_msg, visible=True), None, None, None, gr.update(visible=False)
            except InsufficientDataError as e:
                error_msg = f"**Data Error**\n\n{str(e)}"
                print(f"[ERROR] Insufficient data: {e}")
                return "", gr.update(value=error_msg, visible=True), None, None, None, gr.update(visible=False)
            except PairsTradingError as e:
                error_msg = f"**Error**\n\n{str(e)}"
                print(f"[ERROR] Pairs trading error: {e}")
                return "", gr.update(value=error_msg, visible=True), None, None, None, gr.update(visible=False)
            except Exception as e:
                import traceback
                error_msg = f"**Unexpected Error**\n\n{str(e)}\n\nPlease check your inputs and try again."
                print(f"[ERROR] Unexpected error: {e}")
                print(traceback.format_exc())
                return "", gr.update(value=error_msg, visible=True), None, None, None, gr.update(visible=False)
        
        # First show results container, then populate it
        def show_results_first():
            """Show results container first to prevent rendering issues."""
            return gr.update(visible=True)
        
        pairs_run_btn.click(
            fn=show_results_first,
            inputs=None,
            outputs=[pairs_results]
        ).then(
            fn=handle_analysis_with_visibility,
            inputs=[pairs_stock1, pairs_stock2, pairs_start_date, pairs_end_date, pairs_run_cointegration],
            outputs=[pairs_summary, pairs_error_msg, pairs_spread_plot, pairs_pnl_plot, pairs_signals_df, pairs_results],
            show_progress=True,
            api_name="run_pairs_analysis"
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
