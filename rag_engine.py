# rag_engine.py
# RAG Engine for George Soros Q&A Dataset
# Lightweight TF-IDF retrieval with deterministic answer composition

import os
import textwrap
from datetime import datetime, timedelta
from typing import List

# Lightweight, cross-platform dependencies only
_pandas = None
_TfidfVectorizer = None
_cosine_similarity = None

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

# ==============================
# PATHS & DATA LOADING (LAZY)
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Soros_sample.xlsx")

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

def _load_models():
    """Prepare TF-IDF vectorizer + matrix (our lightweight 'model')."""
    global MODEL_READY, MODEL_ERROR, _vectorizer, _corpus_matrix, _models_loading
    
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

def _retrieve_relevant_qa(query: str, top_k: int = 5):
    """Retrieve the most relevant Q&A items using cosine similarity."""
    if not _load_models():
        raise RuntimeError(f"Models not ready: {MODEL_ERROR}")
    
    if _vectorizer is None or _corpus_matrix is None:
        raise RuntimeError(f"Vectorizer not initialized: {MODEL_ERROR}")
    
    df, corpus_texts = _load_data()
    
    query = (query or "").strip()
    if not query:
        return []

    normalized_query_tokens = {
        token
        for token in _normalize_question(query).split()
        if token and token not in _STOP_TERMS and not token.startswith("soros") and len(token) > 2
    }

    query_vec = _vectorizer.transform([query])
    scores = _cosine_similarity(query_vec, _corpus_matrix)[0]

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

def get_answer(query: str, top_k: int = 5):
    """
    Main RAG function: retrieve -> prompt -> generate.
    Returns structured response with answer and retrieved context.
    """
    query = (query or "").strip()
    if not query:
        return {
            "answer": "Please enter a valid question.",
            "retrieved": [],
            "error": None,
        }

    try:
        retrieved = _retrieve_relevant_qa(query, top_k=top_k)

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

