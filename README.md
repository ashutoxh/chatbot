# George Soros RAG Chatbot

## Overview

This repository hosts a retrieval augmented chatbot focused on George Soros's research notes. The system loads a curated Q&A workbook, builds an in-memory TF-IDF index, and returns grounded answers with traceable source cards inside a modern Gradio interface. Everything runs on CPU only and avoids heavyweight model downloads so it works on any platform.

## Component Map

- `data/Soros_sample.xlsx` - Primary knowledge base containing question, answer, and label columns.
- `rag_engine.py` - Retrieval layer that lazily loads the dataset, builds a TF-IDF (1-2 gram, 5k feature) matrix, and composes grounded answers without calling a generative LLM.
- `pairs_trading.py` - Pairs trading strategy engine implementing George Soros's pair trading methodology with cointegration testing, spread analysis, and backtesting capabilities.
- `app.py` - Gradio UI with a chat-first layout, context cards, quick prompts, hero messaging, and pairs trading analysis interface.
- `load_models.py` - Helper script that preloads the retrieval index in an isolated process to avoid interpreter crashes.
- `requirements.txt` - Dependency list including Gradio, pandas, scikit-learn, yfinance, statsmodels, and matplotlib.

## Data and Retrieval Flow

1. **Dataset ingestion**: `rag_engine._load_data()` reads `Soros_sample.xlsx`, validates required columns, and caches both the dataframe and a normalized question lookup map.
2. **Index construction**: `_load_models()` initializes a TF-IDF vectorizer (CPU only) and stores the sparse matrix in memory. This acts as the semantic search model.
3. **Query handling**: `_retrieve_relevant_qa()` transforms an incoming query into the same TF-IDF space, finds the top matches, and optionally injects an exact match based on the normalized question lookup.
4. **Answer composition**: `_compose_answer()` inspects the retrieved snippets. For fact-style prompts (name, birth, death, etc.) it responds with the highest confidence snippet. For broader prompts it stitches the top answers into a narrative summary with explicit attribution.
5. **UI update**: `app.py` displays the assistant reply in the chat feed, refreshes the context cards, and updates the insight and dataset panels.

## Why This Is a Custom RAG Stack

- **Custom corpus**: The knowledge base is a hand-curated Soros Q&A workbook under `data/Soros_sample.xlsx`, not a hosted API or generic dataset. `_load_data()` normalizes and caches every question for exact-match overrides.
- **Custom retriever**: `_load_models()` trains our own TF-IDF (1–2 gram, 5k feature) index on that corpus at runtime, yielding a lightweight, CPU-only semantic searcher tailored to the dataset.
- **Custom answer composer**: `_compose_answer()` decides when to return verbatim factual snippets versus stitched summaries, always citing the retrieved rows. There is no LLM call—responses are pure retrieval + deterministic formatting.
- **Tight UI wiring**: `_handle_chat()` in `app.py` calls `rag_engine.get_answer()`, surfaces the retrieved evidence as context cards, and keeps the conversation grounded in the Soros dossier.

## Models and Indexing Details

- **Vectorizer**: `sklearn.feature_extraction.text.TfidfVectorizer` with `(1, 2)` n-grams, 5,000 features, English stop words, and lowercase normalization.
- **Similarity metric**: cosine similarity over the TF-IDF matrix.
- **Device**: CPU only, ensuring parity across macOS (Intel or Apple Silicon), Linux, and Windows.
- **Answering strategy**: deterministic snippet selection and formatting. There is no remote API call or generative decoder, so responses always originate from the curated dataset.

## User Interface

- **Chat window**: Custom CSS recreates a ChatGPT-style bubble flow with alternating assistant and user messages, autoscroll, and glassmorphism panels.
- **Palette and chrome**: The UI uses softer graphite (#07090d) surfaces with muted amber (#d9b36a) and fern (#5e9c7e) accents, and it hides the default Gradio header/footer (API logo, settings button, etc.) for an immersive feel.
- **Hero banner**: A base64-embedded Soros portrait anchors the top header alongside a Team Soros attribution line for CSYE 7380 (Fall 2025, Prof. Yizhen Zhao) and the full roster: Ashutosh Singh, Durga Sreshta Kamani, Junyi Zhang, Mohit Jain, Neeha Girja, and Sujay S N.
- **Composer**: Tall multiline textbox with dedicated send and reset buttons plus quick prompt chips for common Soros topics.
- **Context intelligence**: Retrieved Q&A cards summarize label, question, trimmed answer, and relevance score.
- **Insight deck**: Markdown summary describing the anchor question, thematic labels, and guidance for deeper exploration.
- **Dataset pulse and quote cards**: Provide credibility (indexed row counts, label coverage) alongside rotating Soros quotes.
- **Pairs Trading Tester**: A collapsible analysis tool that allows users to test George Soros's pair trading strategy on any two stock tickers. Users can select a date range (up to 5 years back), optionally run cointegration tests, and view backtest results including spread dynamics, cumulative profit/loss, and trading signals. The strategy enforces a p-value requirement (< 0.05) during the trading period to ensure statistical validity.

## Running Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the app:
   ```bash
   python app.py
   ```
3. Visit `http://localhost:7860` (or the auto-selected port if 7860 is busy). The first question may take a few seconds while the TF-IDF matrix warms up.

## Testing

Run unit tests for the pairs trading module:

```bash
pytest tests/test_pairs_trading.py -v
```

Tests cover date validation, spread calculation, cointegration requirements, and error handling. The tests use mocked data to avoid requiring live API calls during testing.

## Pairs Trading Strategy

The pairs trading tester implements a market-neutral strategy based on George Soros's pair trading methodology:

1. **Stock Selection**: Users input any two stock ticker symbols (e.g., XOM and CVX).
2. **Date Range**: Select start and end dates for backtesting, with automatic validation ensuring the range doesn't exceed 5 years from the current date.
3. **Cointegration Testing**: The system automatically performs an Engle-Granger cointegration test on the selected period. The strategy will only proceed if the p-value is below 0.05, ensuring the pair shows sufficient statistical cointegration for mean reversion trading.
4. **Optional Cointegration Display**: Users can optionally request to see detailed cointegration test results, but this is informational only - the p-value requirement is always enforced.
5. **Spread Calculation**: The system uses OLS regression to calculate the spread between the two stocks and identifies mean reversion thresholds (mean ± 1.18 standard deviations).
6. **Trading Signals**: When the spread deviates beyond thresholds, the strategy generates long/short signals to profit from expected mean reversion.
7. **Performance Metrics**: Results include total profit/loss, number of trades, spread visualization, and cumulative PnL charts.

The strategy is implemented in `pairs_trading.py` and integrated into the main UI via a collapsible section accessible via the "Click here to test George Soros' pair trading strategy" button.

## Operational Notes

- Model loading happens at startup; if anything fails, `load_models.py` can rebuild the index in a subprocess without crashing the Gradio server.
- All retrieval and formatting logic lives in `rag_engine.py`, making it straightforward to unit test without the UI.
- The dataset can be updated by editing `data/Soros_sample.xlsx`. New rows are automatically indexed the next time `_load_models()` runs.
- Errors surface inside the chat as textual warnings, so users see diagnostics instead of broken pages.
- Pairs trading analysis requires valid stock tickers and date ranges. Invalid inputs or pairs that don't meet the p-value requirement will display helpful error messages.

## Project Rules

1. Never use emojis anywhere in code, documentation, UI copy, or console logs. Stick to ASCII symbols or words.
2. Update this README whenever a feature, component, or workflow changes so it remains the authoritative reference.
3. Keep explanations descriptive instead of versioned release notes; this document should describe how the system works today.

See `PROJECT_RULES.md` for the canonical wording of these rules.
