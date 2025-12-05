# rag_engine.py
# RAG Engine for George Soros Q&A Dataset
# Supports both TF-IDF and Transformer-based retrieval (via Hugging Face API) with deterministic answer composition

import os
import textwrap
from datetime import datetime, timedelta
from typing import List, Optional

# Lightweight, cross-platform dependencies only
_pandas = None
_TfidfVectorizer = None
_cosine_similarity = None
_requests = None
_transformer_embeddings = None

# Use CPU for all platforms - most compatible
def _get_device():
    """Always use CPU for maximum compatibility."""
    return "cpu"

def _import_pandas():
    """Import pandas only (safe and fast)."""
    global _pandas
    if _pandas is None:
        import pandas as pd
        _pandas = pd


def _import_vectorizer():
    """Import scikit-learn pieces lazily."""
    global _TfidfVectorizer, _cosine_similarity
    if _TfidfVectorizer is None or _cosine_similarity is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        _TfidfVectorizer = TfidfVectorizer
        _cosine_similarity = cosine_similarity


def _import_transformer_api():
    """Import requests for API calls."""
    global _requests
    if '_requests' not in globals():
        try:
            import requests
            globals()['_requests'] = requests
            return True
        except ImportError:
            return False
    return True

# ==============================
# PATHS & DATA LOADING (LAZY)
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Soros_sample.xlsx")
# Using Hugging Face Router API instead of local model
# Correct router endpoint format: https://router.huggingface.co/hf-inference/models/{model_id}/pipeline/{task}
# For feature extraction (embeddings), we use the feature-extraction pipeline
TRANSFORMER_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/paraphrase-MiniLM-L6-v2/pipeline/feature-extraction"
# Optional: Set HUGGINGFACE_API_TOKEN or HF_TOKEN environment variable for authentication
# Get token from: https://huggingface.co/settings/tokens
HF_API_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN") or os.environ.get("HF_TOKEN")

_df = None
_corpus_texts = None
_question_lookup = None
_question_tokens = None
_EXCEL_EPOCH = datetime(1899, 12, 30)
_STOP_TERMS = {"george", "soros", "george soros"}

_EXCEL_EPOCH = datetime(1899, 12, 30)


def _normalize_question(text: str) -> str:
    text = (text or "").lower()
    clean = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    return " ".join(clean.split())


def _excel_serial_to_date(value: str) -> str:
    """Convert an Excel serial (days since 1899-12-30) into a readable date."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return ""
    if not 1 <= num <= 80000:
        return ""
    date = _EXCEL_EPOCH + timedelta(days=int(num))
    return date.strftime("%B %d, %Y")


def _format_answer_value(answer: str, question: str) -> str:
    """Normalize stored answers (e.g., convert Excel serial dates when question asks 'when')."""
    text = (answer or "").strip()
    if not text:
        return text
    q_lower = (question or "").lower()
    if any(keyword in q_lower for keyword in ("when", "date", "marry", "married")):
        converted = _excel_serial_to_date(text)
        if converted:
            return converted
    return text

def _load_data():
    """Lazy load data to avoid issues at import time."""
    global _df, _corpus_texts, _question_lookup, _question_tokens
    
    if _df is not None:
        return _df, _corpus_texts
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Soros dataset not found at: {DATA_PATH}")
    
    _import_pandas()
    pd = _pandas
    
    print(f"[DATA]  Loading Soros dataset from: {DATA_PATH}")
    _df = pd.read_excel(DATA_PATH)
    
    required_cols = {"Questions", "Answers"}
    missing = required_cols - set(_df.columns)
    if missing:
        raise ValueError(f"Soros_sample.xlsx is missing columns: {missing}")
    
    # Embedding corpus = Q + A for semantic search
    _corpus_texts = [
        f"Question: {q}\nAnswer: {a}"
        for q, a in zip(_df["Questions"].astype(str), _df["Answers"].astype(str))
    ]
    _question_lookup = {}
    for idx, q in enumerate(_df["Questions"].astype(str)):
        normalized = _normalize_question(q)
        _question_lookup[normalized] = idx
        stripped = " ".join(tok for tok in normalized.split() if tok and tok not in _STOP_TERMS)
        if stripped and stripped not in _question_lookup:
            _question_lookup[stripped] = idx
    _question_tokens = [
        {tok for tok in _normalize_question(q).split() if tok and tok not in _STOP_TERMS}
        for q in _df["Questions"].astype(str)
    ]
    
    print(f"[OK]  Loaded {len(_corpus_texts)} Q&A entries from Soros dataset.")
    return _df, _corpus_texts


# ==============================
# MODELS - EAGER LOADING AT STARTUP
# ==============================

MODEL_READY = False
MODEL_ERROR = None
_vectorizer = None
_corpus_matrix = None
_models_loading = False

TRANSFORMER_MODEL_READY = False
TRANSFORMER_MODEL_ERROR = None
_transformer_loading = False

def _load_models(use_transformer: bool = False):
    """Prepare TF-IDF vectorizer + matrix (default) or Transformer embeddings via Hugging Face API."""
    global MODEL_READY, MODEL_ERROR, _vectorizer, _corpus_matrix, _models_loading
    global TRANSFORMER_MODEL_READY, TRANSFORMER_MODEL_ERROR, _transformer_embeddings, _transformer_loading
    
    if use_transformer:
        # Load transformer model
        if TRANSFORMER_MODEL_READY:
            return True
        
        if _transformer_loading:
            return False
        
        _transformer_loading = True
        
        try:
            if not _import_transformer_api():
                raise ImportError("requests not installed. Install with: pip install requests")
            
            _import_pandas()
            df, corpus_texts = _load_data()
            
            print("[MODEL]  Using Hugging Face Router API for transformer embeddings...")
            if not HF_API_TOKEN:
                raise RuntimeError(
                    "Hugging Face API token required for router API.\n"
                    "Set HUGGINGFACE_API_TOKEN or HF_TOKEN environment variable.\n"
                    "Get token from: https://huggingface.co/settings/tokens"
                )
            print("[MODEL]  Using authenticated API access")
            print("[MODEL]  Generating embeddings for corpus via API...")
            
            # Generate embeddings using Hugging Face API
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            requests = globals().get('_requests')
            if not requests:
                import requests
                globals()['_requests'] = requests
            
            # Prepare headers with authentication (token already validated above)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {HF_API_TOKEN}"
            }
            
            # Batch API calls (API has rate limits, so we'll do in batches)
            batch_size = 10
            all_embeddings = []
            
            for i in range(0, len(corpus_texts), batch_size):
                batch = corpus_texts[i:i+batch_size]
                batch_num = i // batch_size + 1
                try:
                    print(f"[DEBUG] API CALL - Corpus Embeddings Batch {batch_num}/{len(corpus_texts)//batch_size + 1}")
                    print(f"[DEBUG]   URL: {TRANSFORMER_API_URL}")
                    print(f"[DEBUG]   Batch size: {len(batch)} documents")
                    print(f"[DEBUG]   Has token: {HF_API_TOKEN is not None}")
                    
                    response = requests.post(
                        TRANSFORMER_API_URL,
                        json={"inputs": batch},
                        headers=headers,
                        timeout=30
                    )
                    
                    print(f"[DEBUG]   Response status: {response.status_code}")
                    print(f"[DEBUG]   Response headers: {dict(response.headers)}")
                    
                    response.raise_for_status()
                    batch_embeddings = response.json()
                    
                    print(f"[DEBUG]   Response type: {type(batch_embeddings)}")
                    if isinstance(batch_embeddings, list):
                        print(f"[DEBUG]   Response length: {len(batch_embeddings)}")
                        if len(batch_embeddings) > 0:
                            print(f"[DEBUG]   First embedding shape: {len(batch_embeddings[0]) if isinstance(batch_embeddings[0], list) else 'not a list'}")
                    
                    all_embeddings.extend(batch_embeddings)
                    print(f"[MODEL]  Processed {min(i+batch_size, len(corpus_texts))}/{len(corpus_texts)} documents...")
                except Exception as e:
                    print(f"[ERROR] API CALL FAILED - Batch {batch_num}")
                    print(f"[ERROR]   Exception type: {type(e).__name__}")
                    print(f"[ERROR]   Exception message: {str(e)}")
                    if hasattr(e, 'response') and e.response is not None:
                        print(f"[ERROR]   Response status: {e.response.status_code}")
                        print(f"[ERROR]   Response text: {e.response.text[:200]}")
                    # Fallback: use zeros for failed batches
                    embedding_dim = 384  # paraphrase-MiniLM-L6-v2 dimension
                    all_embeddings.extend([[0.0] * embedding_dim] * len(batch))
                    print(f"[WARN]  Using zero embeddings fallback for batch {batch_num}")
            
            embeddings = np.array(all_embeddings)
            _transformer_embeddings = embeddings
            
            print("[OK]  Transformer embeddings ready (via API).")
            
            TRANSFORMER_MODEL_READY = True
            TRANSFORMER_MODEL_ERROR = None
            return True
        except Exception as e:
            TRANSFORMER_MODEL_READY = False
            TRANSFORMER_MODEL_ERROR = str(e)
            print(f"[ERROR]  Error loading transformer model: {TRANSFORMER_MODEL_ERROR}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            _transformer_loading = False
    else:
        # Load TF-IDF (default)
        if MODEL_READY:
            return True
        
        if _models_loading:
            return False
        
        _models_loading = True
        
        try:
            _import_pandas()
            _import_vectorizer()
            
            df, corpus_texts = _load_data()
            
            print("[MODEL]  Building TF-IDF vectorizer (1-2 grams, 5k features)...")
            vectorizer = _TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                stop_words="english",
                lowercase=True,
            )
            corpus_matrix = vectorizer.fit_transform(corpus_texts)
            
            _vectorizer = vectorizer
            _corpus_matrix = corpus_matrix
            
            print("[OK]  Vectorizer ready (cross-platform, CPU-only).")
            
            MODEL_READY = True
            MODEL_ERROR = None
            return True
        except Exception as e:
            MODEL_READY = False
            MODEL_ERROR = str(e)
            print(f"[ERROR]  Error building vectorizer: {MODEL_ERROR}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            _models_loading = False


# ==============================
# RETRIEVAL
# ==============================

def _retrieve_relevant_qa(query: str, top_k: int = 5, use_transformer: bool = False):
    """Retrieve the most relevant Q&A items using cosine similarity.
    
    Args:
        query: User query string
        top_k: Number of results to return
        use_transformer: If True, use transformer embeddings; if False, use TF-IDF
    """
    if not _load_models(use_transformer=use_transformer):
        error_msg = TRANSFORMER_MODEL_ERROR if use_transformer else MODEL_ERROR
        raise RuntimeError(f"Models not ready: {error_msg}")
    
    df, corpus_texts = _load_data()
    
    query = (query or "").strip()
    if not query:
        return []

    normalized_query_tokens = {
        token
        for token in _normalize_question(query).split()
        if token and token not in _STOP_TERMS and not token.startswith("soros") and len(token) > 2
    }

    # Use transformer embeddings or TF-IDF
    if use_transformer:
        print(f"[DEBUG] RETRIEVAL MODE: Using TRANSFORMER embeddings")
        print(f"[DEBUG]   Transformer embeddings loaded: {_transformer_embeddings is not None}")
        if _transformer_embeddings is not None:
            print(f"[DEBUG]   Embeddings shape: {_transformer_embeddings.shape}")
        
        if _transformer_embeddings is None:
            raise RuntimeError(f"Transformer embeddings not initialized: {TRANSFORMER_MODEL_ERROR}")
        
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        # Get query embedding from API
        requests = globals().get('_requests')
        if not requests:
            import requests
            globals()['_requests'] = requests
        
        # Prepare headers with authentication (required for router API)
        if not HF_API_TOKEN:
            raise RuntimeError(
                "Hugging Face API token required. Set HUGGINGFACE_API_TOKEN or HF_TOKEN environment variable.\n"
                "Get token from: https://huggingface.co/settings/tokens"
            )
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {HF_API_TOKEN}"
        }
        
        try:
            print(f"[DEBUG] API CALL - Query Embedding")
            print(f"[DEBUG]   URL: {TRANSFORMER_API_URL}")
            print(f"[DEBUG]   Query: {query[:100]}...")
            print(f"[DEBUG]   Has token: {HF_API_TOKEN is not None}")
            
            response = requests.post(
                TRANSFORMER_API_URL,
                json={"inputs": [query]},
                headers=headers,
                timeout=10
            )
            
            print(f"[DEBUG]   Response status: {response.status_code}")
            print(f"[DEBUG]   Response headers: {dict(response.headers)}")
            
            response.raise_for_status()
            query_embedding_result = response.json()
            
            print(f"[DEBUG]   Response type: {type(query_embedding_result)}")
            if isinstance(query_embedding_result, list):
                print(f"[DEBUG]   Response length: {len(query_embedding_result)}")
            
            # API returns list with one embedding
            if isinstance(query_embedding_result, list) and len(query_embedding_result) > 0:
                query_embedding = np.array([query_embedding_result[0]])
                print(f"[DEBUG]   Query embedding shape: {query_embedding.shape}")
            else:
                raise ValueError(f"Unexpected API response format: {type(query_embedding_result)}")
        except Exception as e:
            print(f"[ERROR] API CALL FAILED - Query Embedding")
            print(f"[ERROR]   Exception type: {type(e).__name__}")
            print(f"[ERROR]   Exception message: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"[ERROR]   Response status: {e.response.status_code}")
                print(f"[ERROR]   Response text: {e.response.text[:500]}")
            raise RuntimeError(f"Failed to get query embedding from API: {e}")
        
        scores = cosine_similarity(query_embedding, _transformer_embeddings)[0]
        print(f"[DEBUG]   Computed similarity scores shape: {scores.shape}")
        print(f"[DEBUG]   Top score: {scores.max():.4f}, Min score: {scores.min():.4f}")
    else:
        print(f"[DEBUG] RETRIEVAL MODE: Using TF-IDF")
        if _vectorizer is None or _corpus_matrix is None:
            raise RuntimeError(f"Vectorizer not initialized: {MODEL_ERROR}")
        
        query_vec = _vectorizer.transform([query])
        scores = _cosine_similarity(query_vec, _corpus_matrix)[0]
        print(f"[DEBUG]   TF-IDF scores computed, shape: {scores.shape}")
        print(f"[DEBUG]   Top score: {scores.max():.4f}, Min score: {scores.min():.4f}")

    top_k = min(top_k, len(corpus_texts))
    ranked_idx = scores.argsort()[::-1][:top_k * 2]

    if scores[ranked_idx[0]] < 0.10:
        return []
    
    seen_questions = set()
    results = []
    for idx in ranked_idx:
        score = float(scores[idx])
        question_text = str(df.iloc[idx]["Questions"])
        if question_text in seen_questions:
            continue
        seen_questions.add(question_text)
        formatted_answer = _format_answer_value(str(df.iloc[idx]["Answers"]), question_text)
        results.append(
            {
                "score": score,
                "qa_text": corpus_texts[idx],
                "question": question_text,
                "answer": formatted_answer,
                "label": str(df.iloc[idx].get("Label", "")),
                "index": idx,
            }
        )
        if len(results) >= top_k:
            break

    normalized_query = _normalize_question(query)
    exact_idx = _question_lookup.get(normalized_query) if _question_lookup else None
    if exact_idx is not None:
        question_text = str(df.iloc[exact_idx]["Questions"])
        if question_text not in seen_questions:
            formatted_answer = _format_answer_value(str(df.iloc[exact_idx]["Answers"]), question_text)
            results.insert(
                0,
                {
                    "score": 1.0,
                    "qa_text": corpus_texts[exact_idx],
                    "question": question_text,
                    "answer": formatted_answer,
                    "label": str(df.iloc[exact_idx].get("Label", "")),
                    "index": exact_idx,
                },
            )
            seen_questions.add(question_text)
            if len(results) > top_k:
                results = results[:top_k]

    if normalized_query_tokens and _question_tokens:
        def overlap_value(item):
            idx = item.get("index")
            if idx is None or idx >= len(_question_tokens):
                return 0
            return len(_question_tokens[idx] & normalized_query_tokens)
        results.sort(key=lambda item: (overlap_value(item), item["score"]), reverse=True)

    for item in results:
        item.pop("index", None)

    return results


# ==============================
# ANSWER COMPOSITION (no LLM)
# ==============================

def _compose_answer(query: str, retrieved_qas: List[dict]) -> str:
    """
    Build a response by weaving retrieved Soros notes. For short factual prompts,
    respond directly with the top snippet to avoid hallucinated narratives.
    """
    if not retrieved_qas:
        return (
            "I could not find anything about that topic in the Soros research notes. "
            "Please try another question focused on his investing philosophy, risk views, "
            "macroeconomic insights, or philanthropy work documented in the dataset."
        )
    
    query_lower = query.lower()
    normalized_query_tokens = set()
    for token in _normalize_question(query).split():
        if token and token not in _STOP_TERMS and not token.startswith("soros") and len(token) > 2:
            normalized_query_tokens.add(token)
    top_entry = retrieved_qas[0]
    priority_terms = (
        "where",
        "name",
        "born",
        "birthday",
        "birth",
        "die",
        "alive",
        "age",
        "when",
        "marry",
        "married",
        "wife",
        "wives",
        "spouse",
        "spouses",
        "back",
        "pain",
    )
    for term in priority_terms:
        if term in query_lower:
            for candidate in retrieved_qas:
                question = (candidate.get("question") or "").lower()
                if term in question:
                    top_entry = candidate
                    break
            break
    top_answer = (top_entry.get("answer") or "").strip()
    label = top_entry.get("label", "").strip() or "Soros dossier"
    
    def annotate(text: str) -> str:
        suffix = "" if text.endswith((".", "!", "?")) else "."
        return f"{text}{suffix}\n\n_Source: {label}_"
    
    fact_keywords = (
        "name",
        "born",
        "birthday",
        "birth",
        "where",
        "die",
        "alive",
        "age",
        "when",
        "marry",
        "married",
        "date",
        "wife",
        "wives",
        "spouse",
        "spouses",
        "back",
        "pain",
    )
    fact_keyword_set = set(fact_keywords)
    if normalized_query_tokens.intersection(fact_keyword_set) and top_answer:
        filtered_terms = {
            term
            for term in normalized_query_tokens
            if term
            and term not in _STOP_TERMS
            and not term.startswith("soros")
            and term != "george"
            and len(term) > 2
        }
        target_terms = {term for term in fact_keywords if term in normalized_query_tokens}
        required_terms = {
            term
            for term in ("die", "wife", "wives", "spouse", "spouses", "back", "pain")
            if term in normalized_query_tokens
        }
        filtered_answers = []
        seen = set()
        for item in retrieved_qas:
            answer_text = (item.get("answer") or "").strip()
            question_text = (item.get("question") or "").lower()
            if not answer_text:
                continue
            tokenized_question = {
                tok
                for tok in _normalize_question(item.get("question", "")).split()
                if tok and tok not in _STOP_TERMS and len(tok) > 2
            }
            active_terms = target_terms or filtered_terms
            if required_terms and not (tokenized_question & required_terms):
                continue
            if active_terms and not (tokenized_question & active_terms):
                continue
            normalized_answer = answer_text.rstrip(".")
            if normalized_answer and normalized_answer not in seen:
                filtered_answers.append(normalized_answer)
                seen.add(normalized_answer)
        if not filtered_answers:
            filtered_answers = [top_answer.strip().rstrip(".")]
        normalized = filtered_answers[0]
        normalized_lower = normalized.lower()
        if "name" in query_lower:
            if "soros" in normalized_lower and ("name" in normalized_lower or "gy" in normalized_lower):
                response = normalized
            else:
                response = f"George Soros's birth name is {normalized}"
        elif "where" in query_lower and "born" in query_lower:
            if "soros" in normalized_lower and "born" in normalized_lower:
                response = normalized
            else:
                response = f"George Soros was born in {normalized}"
        elif "born" in query_lower or "birthday" in query_lower:
            if "soros" in normalized_lower and "born" in normalized_lower:
                response = normalized
            else:
                response = f"George Soros was born on {normalized}"
        elif "die" in query_lower or "alive" in query_lower:
            response = normalized if normalized_lower.startswith("george") else f"George Soros {normalized}"
        elif len(filtered_answers) > 1:
            response = "; ".join(filtered_answers)
        else:
            response = normalized
        return annotate(response)
    
    top_answers = [item.get("answer", "").strip() for item in retrieved_qas[:3] if item.get("answer")]
    combined = " ".join(top_answers)
    combined = " ".join(combined.split())  # collapse whitespace
    
    narrative = (
        f"{textwrap.fill(combined, width=100)}\n\n"
    )
    return narrative


# ==============================
# PUBLIC API
# ==============================

def get_answer(query: str, top_k: int = 5, use_transformer: bool = False):
    """
    Main RAG function: retrieve -> prompt -> generate.
    Returns structured response with answer and retrieved context.
    
    Args:
        query: User query string
        top_k: Number of results to return
        use_transformer: If True, use transformer embeddings; if False, use TF-IDF
    """
    print(f"[DEBUG] get_answer called - use_transformer={use_transformer}, top_k={top_k}")
    query = (query or "").strip()
    if not query:
        return {
            "answer": "Please enter a valid question.",
            "retrieved": [],
            "error": None,
        }

    try:
        print(f"[DEBUG] Calling _retrieve_relevant_qa with use_transformer={use_transformer}")
        retrieved = _retrieve_relevant_qa(query, top_k=top_k, use_transformer=use_transformer)
        print(f"[DEBUG] _retrieve_relevant_qa returned {len(retrieved)} results")

        if not retrieved:
            answer_text = (
                "This topic is not included in the George Soros dataset. "
                "Try asking about his investing philosophy, risk management, "
                "macro vs micro views, or personal history covered in the notes."
            )
            return {
                "answer": answer_text,
                "retrieved": [],
                "error": None,
            }

        answer_text = _compose_answer(query, retrieved)

        return {
            "answer": answer_text,
            "retrieved": retrieved,
            "error": None,
        }
    except Exception as e:
        return {
            "answer": f"Error processing your question: {str(e)}",
            "retrieved": [],
            "error": str(e),
        }

