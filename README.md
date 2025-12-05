# George Soros RAG Chatbot

## Overview

This repository hosts a retrieval augmented chatbot focused on George Soros's research notes. The system loads a curated Q&A workbook and supports two retrieval modes: (1) TF-IDF-based semantic search (lightweight, CPU-only, no external dependencies) and (2) Transformer embeddings via Hugging Face Router API (requires API token, provides better semantic understanding). Both modes return grounded answers with traceable source cards inside a modern Gradio interface.

## Component Map

- `data/Soros_sample.xlsx` - Primary knowledge base containing question, answer, and label columns.
- `rag_engine.py` - Retrieval layer that lazily loads the dataset, supports both TF-IDF (1-2 gram, 5k feature) and transformer embeddings (via Hugging Face Router API), and composes grounded answers without calling a generative LLM.
- `pairs_trading.py` - Pairs trading strategy engine implementing George Soros's pair trading methodology with cointegration testing, spread analysis, and backtesting capabilities.
- `app.py` - Gradio UI with a chat-first layout, context cards, quick prompts, hero messaging, and pairs trading analysis interface.
- `load_models.py` - Helper script that preloads the retrieval index in an isolated process to avoid interpreter crashes.
- `requirements.txt` - Dependency list including Gradio, pandas, scikit-learn, yfinance, statsmodels, matplotlib, and requests (for Hugging Face API).

## Data and Retrieval Flow

1. **Dataset ingestion**: `rag_engine._load_data()` reads `Soros_sample.xlsx`, validates required columns, and caches both the dataframe and a normalized question lookup map.
2. **Index construction**: `_load_models()` initializes either:
   - **TF-IDF mode (default)**: Builds a TF-IDF vectorizer (CPU only) and stores the sparse matrix in memory
   - **Transformer mode (optional)**: Generates embeddings for the corpus using Hugging Face Router API (requires API token)
3. **Query handling**: `_retrieve_relevant_qa()` processes queries based on selected mode:
   - **TF-IDF**: Transforms query into TF-IDF space (statistical word counting, no transformer), finds top matches via cosine similarity
   - **Transformer**: Gets query embedding from Hugging Face API (uses neural network transformer model), computes cosine similarity against corpus embeddings
   - Both modes optionally inject exact matches based on normalized question lookup
4. **Answer composition**: `_compose_answer()` inspects the retrieved snippets. For fact-style prompts (name, birth, death, etc.) it responds with the highest confidence snippet. For broader prompts it stitches the top answers into a narrative summary with explicit attribution.
5. **UI update**: `app.py` displays the assistant reply in the chat feed, refreshes the context cards, and updates the insight and dataset panels.

## Why This Is a Custom RAG Stack

- **Custom corpus**: The knowledge base is a hand-curated Soros Q&A workbook under `data/Soros_sample.xlsx`, not a hosted API or generic dataset. `_load_data()` normalizes and caches every question for exact-match overrides.
- **Dual retrieval modes**: `_load_models()` supports both TF-IDF (lightweight, CPU-only, no external dependencies) and transformer embeddings (via Hugging Face Router API for better semantic understanding). Users can switch between modes via UI checkbox.
- **Custom answer composer**: `_compose_answer()` decides when to return verbatim factual snippets versus stitched summaries, always citing the retrieved rows. There is no generative LLM call—responses are pure retrieval + deterministic formatting.
- **Tight UI wiring**: `_handle_chat()` in `app.py` calls `rag_engine.get_answer()` with the selected retrieval mode, surfaces the retrieved evidence as context cards, and keeps the conversation grounded in the Soros dossier.

## Models and Indexing Details

### TF-IDF Mode (Default)

**Important: TF-IDF does NOT use a transformer model.** It is a traditional statistical text analysis method that predates neural networks and transformers by decades.

- **Method**: Term Frequency-Inverse Document Frequency (TF-IDF) - a bag-of-words approach
- **Implementation**: `sklearn.feature_extraction.text.TfidfVectorizer` from scikit-learn
- **How it works**:
  1. Counts how often each word/ngram appears in documents (Term Frequency)
  2. Weights words by how rare they are across the corpus (Inverse Document Frequency)
  3. Creates sparse vectors representing documents as weighted word counts
  4. Uses cosine similarity to find similar documents
- **No neural networks**: Pure statistical computation, no machine learning model training
- **No transformers**: Does not use transformer architecture, attention mechanisms, or embeddings
- **Vectorizer**: `TfidfVectorizer` with `(1, 2)` n-grams, 5,000 features, English stop words, and lowercase normalization
- **Similarity metric**: cosine similarity over the TF-IDF matrix
- **Device**: CPU only, ensuring parity across macOS (Intel or Apple Silicon), Linux, and Windows
- **Dependencies**: None beyond standard Python libraries (scikit-learn, pandas)
- **Code location**: `rag_engine.py` lines 288-295 - uses `sklearn.feature_extraction.text.TfidfVectorizer`, not any transformer model

### Transformer Mode (Optional)
- **Model**: `sentence-transformers/paraphrase-MiniLM-L6-v2` via Hugging Face Router API
- **Endpoint**: `https://router.huggingface.co/hf-inference/models/sentence-transformers/paraphrase-MiniLM-L6-v2/pipeline/feature-extraction`
- **Similarity metric**: cosine similarity over transformer embeddings (384-dimensional vectors)
- **Authentication**: Requires Hugging Face API token (set via `HUGGINGFACE_API_TOKEN` or `HF_TOKEN` environment variable)
- **Advantages**: Better semantic understanding, handles paraphrasing and conceptual similarity better than TF-IDF
- **Limitations**: Requires internet connection and API token, API rate limits apply

### Answering Strategy
Both modes use deterministic snippet selection and formatting. There is no generative LLM call, so responses always originate from the curated dataset.

### Why TF-IDF and Transformer Often Give the Same Answers

For this specific use case (curated Q&A dataset about George Soros), TF-IDF and transformer embeddings frequently produce identical or very similar results. Here's why:

1. **Small, Curated Dataset**: The knowledge base is a hand-curated Excel workbook with specific questions and answers. The vocabulary is limited and domain-specific, making keyword-based matching highly effective.

2. **Fact-Based Queries**: Most questions are factual (birth date, birthplace, investment philosophy, etc.). TF-IDF with 1-2 gram n-grams effectively captures these exact and near-exact matches.

3. **Deterministic Answer Composition**: The `_compose_answer()` function uses the same logic regardless of retrieval method. It selects the top-scoring result and formats it deterministically. Since both methods often retrieve the same top result, the final answer is identical.

4. **Exact Match Override**: The system includes an exact-match lookup that overrides similarity scores when a normalized question matches exactly. This ensures consistent answers for known questions regardless of retrieval method.

5. **Limited Paraphrasing**: The dataset doesn't contain extensive paraphrasing or conceptual variations. Questions are relatively straightforward, so semantic understanding (transformer's strength) provides less advantage here.

6. **TF-IDF Effectiveness**: With 1-2 gram n-grams and 5,000 features, TF-IDF captures phrase-level patterns that work well for structured Q&A. For example, "investment philosophy" as a bigram is highly discriminative.

**When Transformer Might Help**: Transformer embeddings would show more benefit with:
- Large, diverse datasets with extensive paraphrasing
- Conceptual queries requiring semantic understanding beyond keywords
- User queries that don't match dataset vocabulary closely
- Multilingual or cross-lingual scenarios

**Recommendation**: For this Soros Q&A chatbot, TF-IDF (default mode) is sufficient and recommended because:
- No external dependencies or API tokens required
- Faster (no network calls)
- Lower latency
- Works offline
- Produces equivalent results for this dataset

Transformer mode is available for experimentation and comparison, but TF-IDF is the practical choice for production use with this specific dataset.

## User Interface

- **Chat window**: Custom CSS recreates a ChatGPT-style bubble flow with alternating assistant and user messages, autoscroll, and glassmorphism panels.
- **Palette and chrome**: The UI uses softer graphite (#07090d) surfaces with muted amber (#d9b36a) and fern (#5e9c7e) accents, and it hides the default Gradio header/footer (API logo, settings button, etc.) for an immersive feel.
- **Hero banner**: A base64-embedded Soros portrait anchors the top header alongside a Team Soros attribution line for CSYE 7380 (Fall 2025, Prof. Yizhen Zhao) and the full roster: Ashutosh Singh, Durga Sreshta Kamani, Junyi Zhang, Mohit Jain, Neeha Girja, and Sujay S N.
- **Composer**: Tall multiline textbox with dedicated send and reset buttons plus quick prompt chips for common Soros topics.
- **Retrieval mode selector**: Checkbox labeled "Use Transformer Embeddings (Hugging Face API)" allows users to switch between TF-IDF (unchecked, default) and transformer embeddings (checked). When checked, requires Hugging Face API token to be set. Note: For this curated Q&A dataset, both modes typically produce the same results (see "Why TF-IDF and Transformer Often Give the Same Answers" section below).
- **Context intelligence**: Retrieved Q&A cards summarize label, question, trimmed answer, and relevance score.
- **Insight deck**: Markdown summary describing the anchor question, thematic labels, and guidance for deeper exploration.
- **Dataset pulse and quote cards**: Provide credibility (indexed row counts, label coverage) alongside rotating Soros quotes.
- **Pairs Trading Tester**: A collapsible analysis tool that allows users to test George Soros's pair trading strategy on any two stock tickers. Users can select a date range (up to 5 years back), optionally show cointegration test results and pair quality assessment via checkbox, and view backtest results including spread dynamics, cumulative profit/loss, and trading signals. The analysis always runs regardless of p-value, but pair quality indicators (good/weak pair) are only displayed when the optional cointegration checkbox is checked.

## Running Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. (Optional) For transformer embeddings: Set Hugging Face API token:
   ```bash
   export HUGGINGFACE_API_TOKEN="your_token_here"
   # OR
   export HF_TOKEN="your_token_here"
   ```
   Get your token from: https://huggingface.co/settings/tokens
   - Token is **required** only if you want to use transformer embeddings (checkbox in UI)
   - TF-IDF mode (default) works without a token and is recommended for this dataset
   - **Note**: For this curated Q&A dataset, transformer embeddings typically produce the same results as TF-IDF (see explanation below). TF-IDF is faster, requires no external dependencies, and works offline.

3. Start the app:
   ```bash
   python app.py
   ```
4. Visit `http://localhost:7860` (or the auto-selected port if 7860 is busy). The first question may take a few seconds while the TF-IDF matrix warms up.

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
3. **Analysis Execution**: The strategy always runs the backtest regardless of cointegration statistics, allowing users to evaluate any pair.
4. **Optional Cointegration Display**: Users can check the "Run optional cointegration test" checkbox to view:
   - Pair quality assessment (GOOD PAIR / WEAK PAIR indicator based on p-value < 0.05)
   - Detailed cointegration test statistics (test statistic, p-value, interpretation)
   - Optional cointegration test results card
   - When unchecked, only performance metrics (PnL, trades, spread metrics) and charts are shown
5. **Spread Calculation**: The system uses OLS regression to calculate the spread between the two stocks and identifies mean reversion thresholds (mean ± 1.18 standard deviations).
6. **Trading Signals**: When the spread deviates beyond thresholds, the strategy generates long/short signals to profit from expected mean reversion.
7. **Performance Metrics**: Results include total profit/loss, number of trades, spread visualization, and cumulative PnL charts displayed in a modern card-based UI.

The strategy is implemented in `pairs_trading.py` and integrated into the main UI via a collapsible section accessible via the "Test George Soros' Pair Trading Strategy" button. The UI features a two-column layout with pairs trading on the left and the chatbot on the right for simultaneous use.

## Operational Notes

- Model loading happens at startup; if anything fails, `load_models.py` can rebuild the index in a subprocess without crashing the Gradio server.
- All retrieval and formatting logic lives in `rag_engine.py`, making it straightforward to unit test without the UI.
- The dataset can be updated by editing `data/Soros_sample.xlsx`. New rows are automatically indexed the next time `_load_models()` runs.
- Errors surface inside the chat as textual warnings, so users see diagnostics instead of broken pages.
- Pairs trading analysis requires valid stock tickers and date ranges. Invalid inputs will display helpful error messages. The analysis runs for any pair regardless of cointegration statistics, but pair quality indicators are only shown when the optional cointegration checkbox is checked.

## Project Rules

1. Never use emojis anywhere in code, documentation, UI copy, or console logs. Stick to ASCII symbols or words.
2. Update this README whenever a feature, component, or workflow changes so it remains the authoritative reference.
3. Keep explanations descriptive instead of versioned release notes; this document should describe how the system works today.

See `PROJECT_RULES.md` for the canonical wording of these rules.
