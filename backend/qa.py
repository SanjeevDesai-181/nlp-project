import re
import os
from typing import Dict, Any, List
from pypdf import PdfReader
import spacy
# nltk is needed for sent_tokenize, ensure it's installed and punkt is downloaded
try:
    from nltk.tokenize import sent_tokenize
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt')
except ImportError:
    print("🔴 ERROR: NLTK not found. Please install it (`pip install nltk`) for sentence tokenization.")
    # Define a fallback tokenizer if nltk isn't available
    def sent_tokenize(text):
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import google.generativeai as genai
import time # For potential rate limiting

# --- Configuration ---
KNOWLEDGE_FILE_PATH = "knowledge.pdf"
try:
    NLP_MODEL = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy en_core_web_sm model...")
    spacy.cli.download("en_core_web_sm")
    NLP_MODEL = spacy.load("en_core_web_sm")

# --- Global Variables for RAG ---
PDF_CHUNKS = []
CHUNK_EMBEDDINGS = None
EMBEDDING_MODEL_NAME = "models/text-embedding-004" # Google's embedding model

# --- Gemini API Configuration ---
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("🔴 ERROR: GEMINI_API_KEY environment variable not set.")
else:
    genai.configure(api_key=API_KEY)
    print("✅ Gemini API Key configured.")

# -----------------------------------------------------------
# Utility: Lemmatization Function
# -----------------------------------------------------------
def lemmatize_text(text: str) -> str:
    """Lemmatizes the text using SpaCy."""
    if not NLP_MODEL:
        return text # Return original if SpaCy isn't loaded
    doc = NLP_MODEL(text.lower()) # Process lowercase text
    # Join lemmas, excluding punctuation and spaces
    return " ".join([token.lemma_ for token in doc if not token.is_punct and not token.is_space])

# -----------------------------------------------------------
# 1️⃣ Utility: Load PDF Text
# -----------------------------------------------------------
def load_pdf_content(file_path: str) -> str:
    # (Function remains the same as before)
    if not os.path.exists(file_path):
        print(f"🔴 ERROR: Knowledge file not found at {file_path}")
        return ""
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            try:
                text += page.extract_text() + "\n"
            except Exception as page_e:
                print(f"⚠️ Warning: Could not extract text from a page. Error: {page_e}")
    except Exception as e:
        print(f"🔴 ERROR: Failed to read PDF: {e}")
        return ""
    return re.sub(r"\n+", "\n", text.strip())

# -----------------------------------------------------------
# 2️⃣ Extract structured facts with robust regex (EXPANDED)
# -----------------------------------------------------------
def extract_structured_facts(text: str) -> Dict[str, str]:
    # (Function remains the same as before)
    facts = {}
    patterns = {
        "full_name": r"Full Name:\s*(.*?)\s*Date of Birth",
        "profession": r"Profession:\s*(.*?)\s*Batting Style",
        "playing_role": r"Playing Role:\s*(.*?)\n",
        "father": r"Father:\s*(.*?)\s*\(",
        "mother": r"Mother:\s*(.*?)\s*(?:Rohit Sharma was raised|\()",
        "wife": r"Wife:\s*(.*?)\s*\(",
        "children": r"Children:\s*(.*?)\s*Diet",
        "school": r"School:\s*(.*?)\s*Coach",
        "coach": r"Coach:\s*(.*?)(?:–|- helped him)",
        "icc_titles": r"Major ICC Titles Won as Player:\s*(.*?)\s*Arjuna Award",
        "national_awards": r"(Arjuna Award\s*–\s*\d{4}.*?Rajiv Gandhi Khel Ratna\s*–\s*\d{4})",
        "other_honors": r"(ICC ODI Cricketer.*?Padma Shri Nominee.*?)\n",
        "highest_odi_record": r"Highest ODI Score:\s*(.*?)\s*— World Record",
        "odi_double_centuries": r"Most Double Centuries in ODIs:\s*(.*?)\n",
        "world_cup_record": r"Most Centuries in a Single World Cup:\s*(.*?)\n",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            facts[key] = re.sub(r'\s+', ' ', m.group(1).strip())
    return facts

# -----------------------------------------------------------
# 3️⃣ Simple fact lookup rules (EXPANDED + Lemmatized Question Tokens)
# -----------------------------------------------------------
def extract_from_facts(question: str, facts: Dict[str, str]) -> str | None:
    q_lower = question.lower()
    # --- CHANGE: Lemmatize question tokens for matching ---
    q_lemmatized = lemmatize_text(q_lower)
    tokens = set(re.findall(r"\w+", q_lemmatized)) # Match on lemmatized tokens
    # --- END CHANGE ---

    # --- Handle "Who is Rohit Sharma?" ---
    # Use original tokens for name matching, lemmatized for keywords
    original_tokens = set(re.findall(r"\w+", q_lower))
    if ("who" in tokens and ("rohit" in original_tokens or "sharma" in original_tokens or "he" in tokens)) or \
       ("tell" in tokens and "about" in tokens and ("rohit" in original_tokens or "sharma" in original_tokens)):
        name = facts.get("full_name", "Rohit Sharma")
        prof = facts.get("profession", "a Cricketer")
        role = facts.get("playing_role", "Opening Batter and Captain")
        return f"{name} is an Indian {prof}, known for his role as an {role}."

    # --- Handle Profession/Role ---
    if "profession" in tokens or ("what" in tokens and "do" in tokens and "he" in tokens): # Simplified rule
         prof = facts.get("profession", "Cricketer (Batsman, occasional right-arm offbreak bowler)")
         role = facts.get("playing_role", "Opening Batter and Captain")
         return f"His profession is {prof}, and his primary playing role is {role}."

    # --- Handle Achievements/Awards/Titles ---
    # Use lemmatized tokens for broader matching
    if any(t in tokens for t in ["achievement", "award", "title", "record", "honor"]):
        response_parts = []
        if facts.get("icc_titles"):
            response_parts.append(f"Major ICC Titles: {facts['icc_titles']}.")
        if facts.get("national_awards"):
            response_parts.append(f"National Awards: {facts['national_awards']}.")
        if facts.get("highest_odi_record"):
             response_parts.append(f"He holds the World Record for Highest ODI Score ({facts['highest_odi_record']}).")
        if facts.get("odi_double_centuries"):
             response_parts.append(f"He has the most double centuries in ODIs ({facts['odi_double_centuries']}).")
        if facts.get("world_cup_record"):
             response_parts.append(f"He scored the most centuries in a single World Cup ({facts['world_cup_record']}).")
        if facts.get("other_honors"):
             response_parts.append(f"Other honors include: {facts['other_honors']}.")

        if response_parts:
            # Check if only asking about a specific award type
            if "icc" in tokens or "world" in tokens or "cup" in tokens:
                return facts.get("icc_titles", "ICC title information not found.")
            if "national" in tokens or "arjuna" in tokens or "ratna" in tokens:
                return facts.get("national_awards", "National award information not found.")
            # Otherwise return summary
            return "Some of his major achievements include: " + " ".join(response_parts)
        else:
             return None # Fall through if no achievement facts extracted

    # Check existing facts dictionary before returning (using lemmatized tokens)
    if ("father" in tokens or "dad" in tokens) and facts.get("father"):
        return f"His father is {facts['father']}."
    if ("mother" in tokens or "mom" in tokens) and facts.get("mother"):
        return f"His mother is {facts['mother']}."
    if ("wife" in tokens or "spouse" in tokens) and facts.get("wife"):
        return f"His wife is {facts['wife']}."
    if any(x in tokens for x in ["child", "children", "kid", "daughter", "son"]) and facts.get("children"):
        children_info = facts['children'].replace(' (born ', ', born ').replace(') and a son (born ', '; son born ')
        return f"His children are: {children_info}."
    if "coach" in tokens and facts.get("coach"):
        return f"His coach was {facts['coach']}."
    if "school" in tokens and facts.get("school"):
        return f"He attended {facts['school']}."
    return None

# -----------------------------------------------------------
# 4️⃣ NLP Extractive QA (TF-IDF + cosine) - Fallback Method (WITH LEMMATIZATION)
# -----------------------------------------------------------
def nlp_similarity_qa(question: str, text: str) -> str:
    try:
        original_sentences = [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 15]
    except Exception as e:
        print(f"⚠️ Warning: NLTK sentence tokenization failed. Error: {e}")
        original_sentences = [p.strip() for p in text.split('\n') if len(p.strip()) > 15]

    if not original_sentences:
        return "No processable sentences found in the document."

    # --- CHANGE: Lemmatize sentences and question for TF-IDF ---
    lemmatized_sentences = [lemmatize_text(s) for s in original_sentences]
    lemmatized_question = lemmatize_text(question)
    # --- END CHANGE ---

    # TF-IDF + cosine on lemmatized text
    try:
        vec = TfidfVectorizer(stop_words="english", min_df=1)
        # --- CHANGE: Use lemmatized versions for fitting and transforming ---
        matrix = vec.fit_transform(lemmatized_sentences)
        question_vec = vec.transform([lemmatized_question])
        # --- END CHANGE ---

        if question_vec.nnz == 0:
             print("⚠️ Warning: Lemmatized question vector is empty.")
             # Fallback: simple keyword check on original sentences
             q_tokens = set(re.findall(r'\w+', question.lower()))
             for sentence in original_sentences: # Check original sentences
                 s_tokens = set(re.findall(r'\w+', sentence.lower()))
                 if q_tokens.intersection(s_tokens):
                     return f"Keyword Match: '{sentence}'" # Return original
             return "Could not find a relevant keyword match (question too generic?)."

        sim = cosine_similarity(question_vec, matrix)[0]
        best_idx = sim.argmax()

        if sim[best_idx] < 0.1:
            print(f"ℹ️ Top TF-IDF similarity score ({sim[best_idx]:.2f}) on lemmatized text below threshold (0.1).")
            return "Could not find a highly relevant sentence using keyword matching."
        print(f"ℹ️ Found TF-IDF match (lemmatized) with score {sim[best_idx]:.2f}.")
        # --- CHANGE: Return the ORIGINAL sentence corresponding to the best lemmatized match ---
        return f"Contextual Match: '{original_sentences[best_idx]}'"
        # --- END CHANGE ---

    except ValueError as ve:
        print(f"⚠️ Warning: TF-IDF Vectorizer failed on lemmatized text. Error: {ve}")
        q_tokens = set(re.findall(r'\w+', question.lower()))
        for sentence in original_sentences: # Check original sentences
            s_tokens = set(re.findall(r'\w+', sentence.lower()))
            if q_tokens.intersection(s_tokens):
                return f"Keyword Match: '{sentence}'" # Return original
        return "Could not find a relevant keyword match in the document."
    except Exception as e:
        print(f"🔴 ERROR: Unexpected error during TF-IDF similarity: {e}")
        return "An error occurred during relevance analysis."

# -----------------------------------------------------------
# 5️⃣ Unified Rule-Based/NLP QA entry point
# -----------------------------------------------------------
def rule_based_qa(question: str) -> str:
    # (Function remains the same as before - calls extract_from_facts then nlp_similarity_qa)
    text = load_pdf_content(KNOWLEDGE_FILE_PATH)
    if not text:
        return "Could not load knowledge base."
    facts = extract_structured_facts(text)
    rule_ans = extract_from_facts(question, facts)
    if rule_ans:
        print("✅ Answered using high-precision rule.")
        return rule_ans
    print("ℹ️ No specific rule matched, falling back to NLP similarity search...")
    return nlp_similarity_qa(question, text)

# -----------------------------------------------------------
# 6️⃣ Chunking and Embedding Generation (RAG Setup)
# --- NO LEMMATIZATION ADDED HERE ---
# -----------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    # (Function remains the same - uses original text for chunks)
    try:
        sentences = sent_tokenize(text)
    except Exception as e:
        print(f"⚠️ Warning: NLTK sentence tokenization failed during chunking. Error: {e}")
        sentences = [p.strip() for p in text.split('\n\n') if p.strip()]

    chunks = []
    current_chunk_words = []
    current_chunk = ""
    for sentence in sentences:
        sentence_words = sentence.split()
        if len(current_chunk_words) + len(sentence_words) <= chunk_size:
            current_chunk_words.extend(sentence_words)
            current_chunk += " " + sentence
        else:
            if current_chunk.strip():
                 chunks.append(current_chunk.strip())
            overlap_word_count = min(overlap, len(current_chunk_words))
            overlap_words = current_chunk_words[-overlap_word_count:]
            current_chunk_words = overlap_words + sentence_words
            current_chunk = " ".join(overlap_words) + " " + sentence
            if len(sentence_words) > chunk_size:
                 print(f"⚠️ Warning: Sentence longer than chunk size: '{sentence[:50]}...'")
                 if current_chunk.strip():
                     chunks.append(current_chunk.strip())
                 chunks.append(sentence)
                 current_chunk_words = []
                 current_chunk = ""

    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    final_chunks = [ch for ch in chunks if len(ch.split()) > 10]
    print(f"📄 Created {len(final_chunks)} chunks from the document.")
    return final_chunks


def generate_embeddings(texts: list[str]) -> np.ndarray | None:
    # (Function remains the same - embeds original text)
    if not API_KEY:
        print("🔴 ERROR: Cannot generate embeddings, API Key not configured.")
        return None
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=texts,
            task_type="retrieval_document"
        )
        print(f"✨ Generated embeddings for {len(texts)} texts.")
        return np.array(result['embedding'])
    except Exception as e:
        print(f"🔴 ERROR: Failed to generate embeddings via Gemini API: {e}")
        return None

def initialize_document_data():
    # (Function remains the same - chunks and embeds original text)
    global PDF_CHUNKS, CHUNK_EMBEDDINGS
    if PDF_CHUNKS and CHUNK_EMBEDDINGS is not None:
         print("✅ Document data already initialized.")
         return
    print("⏳ Initializing document data (loading, chunking, embedding)...")
    text = load_pdf_content(KNOWLEDGE_FILE_PATH)
    if text:
        PDF_CHUNKS = chunk_text(text)
        if PDF_CHUNKS:
            CHUNK_EMBEDDINGS = generate_embeddings(PDF_CHUNKS)
            if CHUNK_EMBEDDINGS is not None:
                 print("✅ Document data initialization complete.")
            else:
                 print("🔴 ERROR: Failed to generate embeddings during initialization.")
        else:
            print("🔴 ERROR: No chunks created during initialization.")
    else:
        print("🔴 ERROR: Failed to load document for initialization.")

try:
    nltk.data.find('tokenizers/punkt')
    initialize_document_data()
except Exception as e:
    print(f"🔴 ERROR during initial setup (NLTK or Document Data): {e}")

# -----------------------------------------------------------
# 7️⃣ Semantic Retrieval for RAG
# --- NO LEMMATIZATION ADDED HERE ---
# -----------------------------------------------------------
def retrieve_relevant_chunks_semantic(question: str, top_k: int = 3) -> list[str]:
    # (Function remains the same - generates embedding for original question)
    global PDF_CHUNKS, CHUNK_EMBEDDINGS
    if CHUNK_EMBEDDINGS is None or len(PDF_CHUNKS) == 0:
        print("🔴 ERROR: Document embeddings not available.")
        return []
    try:
        question_embedding_result = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=question,
            task_type="retrieval_query"
        )
        question_embedding = np.array(question_embedding_result['embedding'])
    except Exception as e:
        print(f"🔴 ERROR: Failed to generate question embedding: {e}")
        return []

    if question_embedding.ndim == 0:
        print("🔴 ERROR: Question embedding generation failed (empty result).")
        return []
    if question_embedding.ndim == 1:
        question_embedding = question_embedding.reshape(1, -1)
    if CHUNK_EMBEDDINGS.ndim == 1:
         print("🔴 ERROR: Chunk embeddings are not 2D.")
         return []

    try:
        similarities = cosine_similarity(question_embedding, CHUNK_EMBEDDINGS)[0]
    except ValueError as ve:
         print(f"🔴 ERROR: Cosine similarity failed. Dim mismatch? Error: {ve}")
         return []

    k = min(top_k, len(similarities))
    if k <= 0: return []
    sorted_indices = np.argsort(similarities)[::-1]
    top_indices = sorted_indices[:k]
    threshold = 0.3
    relevant_chunks = []
    scores = []
    for i in top_indices:
         if i < len(PDF_CHUNKS) and similarities[i] > threshold:
              relevant_chunks.append(PDF_CHUNKS[i]) # Return original chunk
              scores.append(similarities[i])
    if relevant_chunks:
        print(f"🔍 Retrieved {len(relevant_chunks)} relevant chunks (Semantic Score: {scores[0]:.3f}).")
    else:
        max_score = similarities[top_indices[0]] if len(top_indices) > 0 else -1
        print(f"⚠️ No chunks found above semantic threshold ({threshold}). Max score: {max_score:.3f}")
    return relevant_chunks

# -----------------------------------------------------------
# 8️⃣ Gemini LLM-Based Extractive QA (Using Semantic RAG)
# --- NO LEMMATIZATION ADDED HERE ---
# -----------------------------------------------------------
def llm_extractive_qa(question: str) -> str:
    # (Function remains the same - uses original question and retrieves original chunks)
    if not API_KEY:
        return "🔴 ERROR: Gemini API Key not configured."
    if CHUNK_EMBEDDINGS is None:
         print("⚠️ Embeddings not ready, attempting re-initialization...")
         initialize_document_data()
         if CHUNK_EMBEDDINGS is None:
              return "🔴 ERROR: Document not processed for semantic search."

    relevant_chunks = retrieve_relevant_chunks_semantic(question, top_k=4)

    if not relevant_chunks:
        print("⚠️ Semantic retrieval failed, falling back to TF-IDF keyword match...")
        text = load_pdf_content(KNOWLEDGE_FILE_PATH)
        if not text: return "Could not load knowledge base for fallback search."
        fallback_answer = nlp_similarity_qa(question, text) # This now uses lemmatization internally
        if "Could not find" in fallback_answer or "No relevant" in fallback_answer or "error" in fallback_answer.lower() or "No processable" in fallback_answer:
             return "Could not find relevant information in the document using multiple methods."
        else:
             clean_fallback = fallback_answer.replace('Contextual Match:', '').replace('Keyword Match:', '').strip().strip("'")
             return f"Found a possible match using keyword search: '{clean_fallback}' (Semantic search failed)."

    context = "\n\n---\n\n".join(relevant_chunks)
    prompt = f"""
    **Your Task:** Answer the user's question based *only* on the provided text Context.
    **Constraint:** Find the specific sentence or phrase within the Context that directly answers the Question. Extract this information verbatim or as a very concise summary (max 2 sentences) derived *solely* from the Context.
    **Important:** Do *not* add any external knowledge or information not present in the Context. If the answer is not found in the Context, you MUST respond exactly with: "The answer is not available in the provided context."

    **Context:**
    \"\"\"
    {context}
    \"\"\"

    **Question:** {question}

    **Extracted Answer:**
    """
    print(f"⚙️ Calling Gemini ({len(relevant_chunks)} context chunks) for Q: '{question}'")
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        safety_settings = {} # Simplified
        response = model.generate_content(prompt, safety_settings=safety_settings)
        try:
             answer_text = response.text.strip()
             if not answer_text:
                  print("⚠️ Warning: Gemini returned empty text.")
                  return "The model returned an empty response. Please try rephrasing."
             return answer_text
        except ValueError as ve:
            print(f"⚠️ Warning: Gemini response blocked/invalid. Error: {ve}")
            block_reason = "Unknown"
            try: block_reason = response.prompt_feedback.block_reason
            except Exception: pass
            return f"Response blocked ({block_reason}). Try rephrasing."
    except Exception as e:
        print(f"🔴 ERROR: Error querying Gemini API: {e}")
        error_str = str(e).lower()
        if "api key not valid" in error_str: return "🔴 ERROR: Invalid Gemini API Key."
        if "quota" in error_str or "rate limit" in error_str: return "🟡 INFO: API rate limit possibly exceeded. Wait & try again."
        return f"Error communicating with AI model: {e}"

# -----------------------------------------------------------
# 9️⃣ Placeholder for LLM Generative QA
# -----------------------------------------------------------
def llm_generative_qa(question: str) -> str:
    # (Function remains the same)
    return (f"[LLM Generative Mode - Not Implemented]\n"
            f"This mode would use semantic retrieval but ask Gemini to generate a *new*, fluent answer.")

# --- Example Usage (for testing directly) ---
if __name__ == "__main__":
    if CHUNK_EMBEDDINGS is None:
         print("Running direct test, ensuring initialization...")
         initialize_document_data()

    # Test cases... (remain the same)
    test_questions = [
        "Who is Rohit Sharma's wife?",
        "What are his major achievements in world cups?",
        "Tell me about his charity work.",
        "How many runs did he score in IPL?",
        "Who is Rohit Sharma?",
        "What awards did he receive?",
        "What is his profession?",
        "Does he meditate?",
        "Who is Sachin Tendulkar?" # OOD
    ]

    for q in test_questions:
        print(f"\n--- Testing Rule-Based ---")
        print(f"Q: {q}")
        print(f"A: {rule_based_qa(q)}")

        print(f"\n--- Testing LLM Extractive ---")
        print(f"Q: {q}")
        print(f"A: {llm_extractive_qa(q)}")

