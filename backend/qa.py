import re
import os
from typing import Dict, Any, List
from pypdf import PdfReader
import spacy
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------------------------------
# ||LLM based QA dependencies

import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ||----------------------------------------------------------------------------------------------------

nlp = spacy.load("en_core_web_sm")
KNOWLEDGE_FILE_PATH = "knowledge.pdf"

# -----------------------------------------------------------
# 1️⃣ Utility: Load PDF Text
# -----------------------------------------------------------
def load_pdf_content(file_path: str) -> str:
    if not os.path.exists(file_path):
        return ""
    text = ""
    reader = PdfReader(file_path)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return re.sub(r"\n+", "\n", text.strip())

# -----------------------------------------------------------
# 2️⃣ Extract structured facts with robust regex
# -----------------------------------------------------------
def extract_structured_facts(text: str) -> Dict[str, str]:
    facts = {}
    patterns = {
        "father": r"Father:\s*(.*?)\s*\(",
        "mother": r"Mother:\s*(.*?)\s*\(",
        "wife": r"Wife:\s*(.*?)\s*\(",
        "children": r"Children:\s*(.*?)\s*Diet",
        "school": r"School:\s*(.*?)\s*Coach",
        "coach": r"Coach:\s*(.*?)–",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            facts[key] = m.group(1).strip()
    return facts

# -----------------------------------------------------------
# 3️⃣ Simple fact lookup rules
# -----------------------------------------------------------
def extract_from_facts(question: str, facts: Dict[str, str]) -> str | None:
    q = question.lower()
    tokens = re.findall(r"\w+", q)

    if ("father" in tokens or "dad" in tokens) and "father" in facts:
        return f"His father is {facts['father']}."
    if ("mother" in tokens or "mom" in tokens) and "mother" in facts:
        return f"His mother is {facts['mother']}."
    if ("wife" in tokens or "spouse" in tokens) and "wife" in facts:
        return f"His wife is {facts['wife']}."
    if any(x in tokens for x in ["child", "children", "kids", "daughter", "son"]) and "children" in facts:
        return f"His children are {facts['children']}."
    if "coach" in tokens and "coach" in facts:
        return f"His coach is {facts['coach']}."
    if "school" in tokens and "school" in facts:
        return f"He studied at {facts['school']}."
    return None

# -----------------------------------------------------------
# 4️⃣ NLP Extractive QA (TF-IDF + cosine)
# -----------------------------------------------------------
def nlp_similarity_qa(question: str, text: str) -> str:
    sentences = [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 10]
    if not sentences:
        return "No data found."

    # Expand question with related terms
    expansion = {
        "achievement": ["award", "record", "title", "trophy", "milestone"],
        "career": ["matches", "runs", "centuries", "captain", "debut"],
        "record": ["highest", "best", "world record"],
    }
    q_lower = question.lower()
    extra = [w for k, v in expansion.items() if k in q_lower for w in v]
    expanded_q = question + " " + " ".join(extra)

    # Add named entities & noun chunks
    doc = nlp(question)
    enriched_q = expanded_q + " " + " ".join([ent.text for ent in doc.ents] + [nc.text for nc in doc.noun_chunks])

    # TF-IDF + cosine
    vec = TfidfVectorizer(stop_words="english")
    matrix = vec.fit_transform(sentences + [enriched_q])
    sim = cosine_similarity(matrix[-1], matrix[:-1])[0]
    best_idx = sim.argmax()
    if sim[best_idx] < 0.1:
        return "No relevant match found."
    return sentences[best_idx]

# -----------------------------------------------------------
# 5️⃣ Unified QA entry
# -----------------------------------------------------------
def nlp_extractive_qa(question: str) -> str:
    """Hybrid QA: structured facts + similarity."""
    text = load_pdf_content(KNOWLEDGE_FILE_PATH)
    if not text:
        return "Could not load knowledge base."

    # Try rule-based extraction first
    facts = extract_structured_facts(text)
    rule_ans = extract_from_facts(question, facts)
    if rule_ans:
        return rule_ans

    # Fallback to NLP similarity
    return nlp_similarity_qa(question, text)

# -----------------------------------------------------------
# Keep placeholders for future LLM integration
# -----------------------------------------------------------
def rule_based_qa(question: str) -> str:
    return nlp_extractive_qa(question)  # reuse same improved method

# -----------------------------------------------------------
# 7️⃣ Gemini LLM-Based Extractive QA (RAG style)
# -----------------------------------------------------------

# Configure Gemini with your API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """
    Split large document text into overlapping chunks for retrieval.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def retrieve_relevant_chunks(question: str, chunks: list[str], top_k: int = 3) -> list[str]:
    """
    Use TF-IDF + cosine similarity to retrieve top-k most relevant text chunks.
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(chunks + [question])
    question_vec = tfidf_matrix[-1]
    similarities = cosine_similarity(question_vec, tfidf_matrix[:-1])[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices if similarities[i] > 0.05]


def llm_extractive_qa(question: str) -> str:
    """
    LLM-based extractive QA using Gemini API.
    Retrieves top chunks and asks Gemini to extract a factual answer.
    """
    text = load_pdf_content(KNOWLEDGE_FILE_PATH)
    if not text:
        return "Could not load knowledge base."

    chunks = chunk_text(text)
    relevant_chunks = retrieve_relevant_chunks(question, chunks)

    if not relevant_chunks:
        return "No relevant context found in the document."

    context = "\n\n".join(relevant_chunks)

    prompt = f"""
    You are a factual question-answering assistant.
    Extract the most accurate and concise answer from the provided context only.
    Do not add external information.

    Question: {question}

    Context:
    {context}

    Answer (in 2–3 sentences, based only on the context):
    """

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error querying Gemini: {e}"


# || LLM based Genrative QA system section

def llm_generative_qa(question: str) -> str:
    return "LLM Generative QA not implemented yet."
