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
        # Basic sentence split based on punctuation followed by space/newline
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import google.generativeai as genai
import time # For potential rate limiting

# --- Configuration ---
KNOWLEDGE_FILE_PATH = "knowledge.pdf"
try:
    # Use a larger SpaCy model if available for potentially better lemmatization/NER
    # NLP_MODEL = spacy.load("en_core_web_md")
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
    # Fallback or disable LLM modes
else:
    # Configure the SDK
    try:
        genai.configure(api_key=API_KEY)
        print("✅ Gemini API Key configured.")
    except Exception as config_e:
        print(f"🔴 ERROR: Failed to configure Gemini API: {config_e}")
        API_KEY = None # Mark API as unusable

# -----------------------------------------------------------
# Utility: Lemmatization Function
# -----------------------------------------------------------
def lemmatize_text(text: str) -> str:
    """Lemmatizes the text using SpaCy."""
    if not NLP_MODEL:
        return text # Return original if SpaCy isn't loaded
    doc = NLP_MODEL(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_punct and not token.is_space and token.lemma_ != '-PRON-'])

# -----------------------------------------------------------
# 1️⃣ Utility: Load PDF Text
# -----------------------------------------------------------
def load_pdf_content(file_path: str) -> str:
    # --- UNCHANGED ---
    if not os.path.exists(file_path):
        print(f"🔴 ERROR: Knowledge file not found at {file_path}")
        return ""
    text = ""
    try:
        reader = PdfReader(file_path)
        print(f"📖 Reading PDF: {file_path} ({len(reader.pages)} pages)")
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                     text += page_text + "\n"
                else:
                     print(f"⚠️ Warning: No text extracted from page {i+1}.")
            except Exception as page_e:
                print(f"⚠️ Warning: Could not extract text from page {i+1}. Error: {page_e}")
        print(f"📑 Extracted total text length: {len(text)} characters.")
    except Exception as e:
        print(f"🔴 ERROR: Failed to read PDF: {e}")
        return ""
    text = re.sub(r'\s*\n\s*', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

# -----------------------------------------------------------
# 2️⃣ Extract structured facts with robust regex (EXPANDED)
# -----------------------------------------------------------
def extract_structured_facts(text: str) -> Dict[str, str]:
    # --- UNCHANGED ---
    facts = {}
    patterns = {
        "full_name": r"Full Name:\s*(.*?)\s*Date of Birth",
        "profession": r"Profession:\s*(.*?)(?:\n|Batting Style)",
        "playing_role": r"Playing Role:\s*(.*?)(?:\n|Father:)",
        "father": r"Father:\s*(.*?)\s*\(",
        "mother": r"Mother:\s*(.*?)\s*(?:Rohit Sharma was raised|\n)",
        "wife": r"Wife:\s*(.*?)\s*\(",
        "children": r"Children:\s*(.*?)(?:\n|Diet:)",
        "school": r"School:\s*(.*?)(?:\n|Coach:)",
        "coach": r"Coach:\s*(.*?)(?:–|- helped him)",
        "icc_titles": r"Major ICC Titles Won as Player:\s*(.*?)\s*Arjuna Award",
        "national_awards": r"(Arjuna Award\s*–\s*\d{4}.*?Rajiv Gandhi Khel Ratna\s*–\s*\d{4})",
        "other_honors": r"(ICC ODI Cricketer.*?Padma Shri Nominee.*?)(?:\n|\Z)",
        "highest_odi_record": r"Highest ODI Score:\s*(.*?)\s*— World Record",
        "odi_double_centuries": r"Most Double Centuries in ODIs:\s*(.*?)(?:\n|Most Centuries)",
        "world_cup_record": r"Most Centuries in a Single World Cup:\s*(.*?)(?:\n|Most Sixes)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            clean_text = re.sub(r'\s+', ' ', m.group(1).strip())
            facts[key] = clean_text
    return facts

# -----------------------------------------------------------
# 3️⃣ Simple fact lookup rules (EXPANDED + Lemmatized Question Tokens)
# -----------------------------------------------------------
def extract_from_facts(question: str, facts: Dict[str, str]) -> str | None:
     # --- UNCHANGED ---
    q_lower = question.lower()
    q_lemmatized = lemmatize_text(q_lower)
    tokens = set(re.findall(r"\w+", q_lemmatized))
    original_tokens = set(re.findall(r"\w+", q_lower))

    # General Info
    if ("who" in tokens and ("rohit" in original_tokens or "sharma" in original_tokens or "he" in tokens)) or \
       ("tell" in tokens and "about" in tokens and ("rohit" in original_tokens or "sharma" in original_tokens)):
        name = facts.get("full_name", "Rohit Sharma")
        prof = facts.get("profession", "a Cricketer")
        role = facts.get("playing_role", "")
        role_text = f", playing as an {role}" if role else ""
        return f"{name} is an Indian {prof}{role_text}."

    if "profession" in tokens or ("what" in tokens and "do" in tokens and "he" in tokens):
         prof = facts.get("profession", "Cricketer (Batsman, occasional right-arm offbreak bowler)")
         role = facts.get("playing_role", "Opening Batter and Captain")
         return f"His profession is {prof}, primarily playing as an {role}."

    # Achievements
    if any(t in tokens for t in ["achievement", "award", "title", "record", "honor"]):
        ach_list = []
        if facts.get("icc_titles"): ach_list.append(f"Major ICC Titles: {facts['icc_titles']}.")
        if facts.get("national_awards"): ach_list.append(f"National Awards: {facts['national_awards']}.")
        if facts.get("highest_odi_record"): ach_list.append(f"World Record for Highest ODI Score ({facts['highest_odi_record']}).")
        if facts.get("odi_double_centuries"): ach_list.append(f"Most ODI double centuries ({facts['odi_double_centuries']}).")
        if facts.get("world_cup_record"): ach_list.append(f"Most centuries in a single World Cup ({facts['world_cup_record']}).")

        if ach_list:
            if "icc" in tokens or "world" in tokens or "cup" in tokens: return facts.get("icc_titles", "ICC title information not found.")
            if "national" in tokens or "arjuna" in tokens or "ratna" in tokens: return facts.get("national_awards", "National award information not found.")
            # Return a slightly more structured summary for rule-based
            summary = "Some key achievements include: " + " ".join(ach_list)
            # Limit length if needed, but for rule-based, full info is often better
            return summary
        else:
            return None # Fall through if no specific facts extracted

    # Family & Personal
    if ("father" in tokens or "dad" in tokens) and facts.get("father"): return f"His father is {facts['father']}."
    if ("mother" in tokens or "mom" in tokens) and facts.get("mother"): return f"His mother is {facts['mother']}."
    if ("wife" in tokens or "spouse" in tokens) and facts.get("wife"): return f"His wife is {facts['wife']}."
    if any(x in tokens for x in ["child", "children", "kid", "daughter", "son"]) and facts.get("children"):
        children_info = facts['children'].replace(' (born ', ', born ').replace(') and a son (born ', '; son born ')
        return f"His children are: {children_info}."
    if "coach" in tokens and facts.get("coach"): return f"His coach was {facts['coach']}."
    if "school" in tokens and facts.get("school"): return f"He attended {facts['school']}."

    return None

# -----------------------------------------------------------
# 4️⃣ NLP Extractive QA (TF-IDF + cosine) - Fallback Method (WITH LEMMATIZATION)
# -----------------------------------------------------------
def nlp_similarity_qa(question: str, text: str) -> str:
    # --- UNCHANGED ---
    try:
        original_sentences = [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 20]
    except Exception as e:
        print(f"⚠️ Warning: NLTK sentence tokenization failed. Error: {e}")
        original_sentences = [p.strip() for p in text.split('\n') if len(p.strip()) > 20]

    if not original_sentences:
        return "No processable sentences found in the document for similarity search."

    lemmatized_sentences = [lemmatize_text(s) for s in original_sentences]
    lemmatized_question = lemmatize_text(question)

    try:
        vec = TfidfVectorizer(stop_words="english", min_df=1, ngram_range=(1, 2))
        matrix = vec.fit_transform(lemmatized_sentences)
        question_vec = vec.transform([lemmatized_question])

        if question_vec.nnz == 0:
             print("⚠️ Warning: Lemmatized question vector empty (TF-IDF).")
             q_tokens = set(re.findall(r'\w+', question.lower()))
             for sentence in original_sentences:
                 s_tokens = set(re.findall(r'\w+', sentence.lower()))
                 if len(q_tokens.intersection(s_tokens)) > 1:
                     return f"Keyword Match: '{sentence}'"
             return "Could not find relevant keywords (question too generic?)."

        sim = cosine_similarity(question_vec, matrix)[0]
        best_idx = sim.argmax()
        threshold = 0.08
        if sim[best_idx] < threshold:
            print(f"ℹ️ Top TF-IDF score ({sim[best_idx]:.2f}) on lemmatized text below threshold ({threshold}).")
            return "Could not find a highly relevant sentence using keyword matching."

        print(f"ℹ️ Found TF-IDF match (lemmatized) score {sim[best_idx]:.2f}.")
        return f"Contextual Match: '{original_sentences[best_idx]}'"

    except Exception as e:
        print(f"🔴 ERROR: Unexpected error during TF-IDF similarity: {e}")
        return "An error occurred during relevance analysis."

# -----------------------------------------------------------
# 5️⃣ Unified Rule-Based/NLP QA entry point
# -----------------------------------------------------------
def rule_based_qa(question: str) -> str:
    # --- UNCHANGED ---
    """Hybrid QA: structured facts + NLP similarity as fallback."""
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
# -----------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 40) -> list[str]:
    # --- UNCHANGED ---
    """Split large document text into overlapping chunks using sentences."""
    try:
        if NLP_MODEL:
             doc = NLP_MODEL(text)
             sentences = [sent.text.strip() for sent in doc.sents]
        else:
             sentences = sent_tokenize(text)
    except Exception as e:
        print(f"⚠️ Warning: Sentence tokenization failed during chunking. Error: {e}")
        sentences = [p.strip() for p in text.split('\n\n') if p.strip()]

    chunks = []
    current_chunk_words = []
    for sentence in sentences:
        sentence_words = sentence.split()
        if not sentence_words: continue

        if len(current_chunk_words) + len(sentence_words) > chunk_size and current_chunk_words:
            chunks.append(" ".join(current_chunk_words))
            overlap_count = min(overlap, len(current_chunk_words))
            current_chunk_words = current_chunk_words[-overlap_count:] + sentence_words
        else:
            current_chunk_words.extend(sentence_words)

        if len(sentence_words) > chunk_size:
             print(f"⚠️ Warning: Sentence longer than chunk size: '{sentence[:50]}...'")
             if current_chunk_words[:-len(sentence_words)]:
                 chunks.append(" ".join(current_chunk_words[:-len(sentence_words)]))
             chunks.append(sentence)
             current_chunk_words = []

    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))

    final_chunks = [ch for ch in chunks if len(ch.split()) > 5]
    print(f"📄 Created {len(final_chunks)} chunks from the document.")
    return final_chunks

def generate_embeddings(texts: list[str], task: str = "retrieval_document") -> np.ndarray | None:
    # --- UNCHANGED ---
    """Generate embeddings using Google AI, specifying task type."""
    if not API_KEY:
        print("🔴 ERROR: Cannot generate embeddings, API Key not configured.")
        return None
    try:
        try:
            result = genai.embed_content(
                model=EMBEDDING_MODEL_NAME, content=texts, task_type=task
            )
            print(f"✨ Generated embeddings for {len(texts)} texts (task: {task}).")
            return np.array(result['embedding'])
        except Exception as e:
            # Basic rate limit handling
            if "resource_exhausted" in str(e).lower() or "quota" in str(e).lower() or "429" in str(e):
                print("🟡 INFO: Rate limit likely hit. Waiting 5 seconds...")
                time.sleep(5)
                result = genai.embed_content(
                    model=EMBEDDING_MODEL_NAME, content=texts, task_type=task
                )
                print(f"✨ Generated embeddings for {len(texts)} texts after retry.")
                return np.array(result['embedding'])
            else: raise e # Re-raise other errors
    except Exception as e:
        print(f"🔴 ERROR: Failed to generate embeddings via Gemini API: {e}")
        return None


def initialize_document_data():
    # --- UNCHANGED ---
    """Load, chunk, and embed the document ONCE."""
    global PDF_CHUNKS, CHUNK_EMBEDDINGS
    if PDF_CHUNKS and CHUNK_EMBEDDINGS is not None:
         print("✅ Document data already initialized.")
         return
    print("⏳ Initializing document data (loading, chunking, embedding)...")
    text = load_pdf_content(KNOWLEDGE_FILE_PATH)
    if text:
        PDF_CHUNKS = chunk_text(text)
        if PDF_CHUNKS:
            batch_size = 100 # Gemini API limit
            all_embeddings = []
            print(f"⏳ Generating embeddings for {len(PDF_CHUNKS)} chunks in batches of {batch_size}...")
            for i in range(0, len(PDF_CHUNKS), batch_size):
                 batch_texts = PDF_CHUNKS[i:i+batch_size]
                 print(f"  - Processing batch {i//batch_size + 1}...")
                 batch_embeddings = generate_embeddings(batch_texts, task="retrieval_document")
                 if batch_embeddings is not None:
                     all_embeddings.append(batch_embeddings)
                 else:
                      print(f"🔴 ERROR: Failed embeddings for batch index {i}. Aborting initialization.")
                      CHUNK_EMBEDDINGS = None; return # Stop if embedding fails
            if all_embeddings:
                CHUNK_EMBEDDINGS = np.vstack(all_embeddings)
                print(f"✅ Document data initialization complete. Embeddings shape: {CHUNK_EMBEDDINGS.shape}")
            else: print("🔴 ERROR: No embeddings generated."); CHUNK_EMBEDDINGS = None
        else: print("🔴 ERROR: No chunks created during initialization.")
    else: print("🔴 ERROR: Failed to load document for initialization.")

try:
    if 'nltk' in globals(): nltk.data.find('tokenizers/punkt')
    initialize_document_data()
except Exception as e:
    print(f"🔴 ERROR during initial setup: {e}")

# -----------------------------------------------------------
# 7️⃣ Semantic Retrieval for RAG (IMPROVED)
# -----------------------------------------------------------
def retrieve_relevant_chunks_semantic(question: str, top_k: int = 6) -> list[str]:
    # --- UNCHANGED (Using improved settings from previous step) ---
    """Retrieve top-k relevant text chunks using semantic embeddings."""
    global PDF_CHUNKS, CHUNK_EMBEDDINGS
    if CHUNK_EMBEDDINGS is None or len(PDF_CHUNKS) == 0:
        print("🔴 ERROR: Document embeddings not available.")
        return []
    question_embedding = generate_embeddings([question], task="retrieval_query")
    if question_embedding is None or question_embedding.ndim == 0:
        print("🔴 ERROR: Failed to generate question embedding.")
        return []
    if question_embedding.ndim == 1: question_embedding = question_embedding.reshape(1, -1)
    if CHUNK_EMBEDDINGS.ndim != 2:
         print(f"🔴 ERROR: Chunk embeddings are not 2D. Shape: {CHUNK_EMBEDDINGS.shape}")
         return []
    try:
        if question_embedding.shape[1] != CHUNK_EMBEDDINGS.shape[1]:
             print(f"🔴 ERROR: Embedding dimension mismatch! Q:{question_embedding.shape} Chunks:{CHUNK_EMBEDDINGS.shape}")
             return []
        similarities = cosine_similarity(question_embedding, CHUNK_EMBEDDINGS)[0]
    except ValueError as ve:
         print(f"🔴 ERROR: Cosine similarity failed. Dim mismatch? Error: {ve}")
         return []
    k = min(top_k, len(similarities))
    if k <= 0: return []
    sorted_indices = np.argsort(similarities)[::-1]
    top_indices = sorted_indices[:k]
    threshold = 0.35 # Keep slightly lower threshold from previous step
    relevant_chunks = []
    scores = []
    for i in top_indices:
         # Add bounds check
         if 0 <= i < len(PDF_CHUNKS) and similarities[i] > threshold:
              relevant_chunks.append(PDF_CHUNKS[i])
              scores.append(similarities[i])
         elif i >= len(PDF_CHUNKS):
              print(f"⚠️ Warning: Index {i} out of bounds for PDF_CHUNKS (length {len(PDF_CHUNKS)}).")


    if relevant_chunks:
        score_str = f"{scores[0]:.3f}" if scores else "N/A"
        print(f"🔍 Retrieved {len(relevant_chunks)} semantically relevant chunks (Top score: {score_str}).")
    else:
        # Check if top_indices is valid before accessing similarities
        max_score = similarities[top_indices[0]] if top_indices.size > 0 and top_indices[0] < len(similarities) else -1
        print(f"⚠️ No chunks found above semantic threshold ({threshold}). Max score: {max_score:.3f}")
    return relevant_chunks

# -----------------------------------------------------------
# 8️⃣ Gemini LLM-Based Extractive QA (Using Improved Semantic RAG + MODIFIED PROMPT)
# -----------------------------------------------------------
def llm_extractive_qa(question: str) -> str:
    """
    LLM-based extractive QA using Gemini API with SEMANTIC retrieval.
    Uses a slightly relaxed prompt to extract relevant sentences/points.
    """
    if not API_KEY: return "🔴 ERROR: Gemini API Key not configured."
    if CHUNK_EMBEDDINGS is None:
         print("⚠️ Embeddings not ready, attempting re-initialization...")
         initialize_document_data()
         if CHUNK_EMBEDDINGS is None: return "🔴 ERROR: Document not processed."

    # Use improved retrieval settings
    relevant_chunks = retrieve_relevant_chunks_semantic(question, top_k=6)

    # --- Fallback logic remains the same ---
    if not relevant_chunks:
        print("⚠️ Semantic retrieval failed, falling back to TF-IDF keyword match...")
        text = load_pdf_content(KNOWLEDGE_FILE_PATH)
        if not text: return "Could not load knowledge base for fallback search."
        fallback_answer = nlp_similarity_qa(question, text) # This uses lemmatization
        # Check fallback quality
        if "Could not find" in fallback_answer or "No relevant" in fallback_answer or "error" in fallback_answer.lower() or "No processable" in fallback_answer:
             # If both methods fail, report failure
             return "Could not find relevant information in the document using multiple methods."
        else:
             # Return TF-IDF match with caveat
             clean_fallback = fallback_answer.replace('Contextual Match:', '').replace('Keyword Match:', '').strip().strip("'")
             return f"Based on keyword matching, a possible relevant sentence is: '{clean_fallback}' (Semantic search did not find a strong match)."

    context = "\n\n---\n\n".join(relevant_chunks)

    # --- **** MODIFIED EXTRACTIVE PROMPT **** ---
    # Relaxed from strict verbatim extraction to extracting key sentences/points
    prompt = f"""
    **Your Task:** Answer the user's Question based *only* on the provided text Context.

    **Instructions:**
    1.  Carefully read the Context and the Question.
    2.  Identify and extract the most relevant sentence(s) or key phrases from the Context that directly address the Question.
    3.  Combine these extractions into a concise answer. You can list key points if appropriate.
    4.  Your answer MUST be composed *only* of text found within the Context. Do not synthesize new sentences or add external information.
    5.  If the Context does not contain information to answer the Question, respond *only* with the exact phrase: "The answer is not available in the provided context."

    **Context:**
    \"\"\"
    {context}
    \"\"\"

    **Question:** {question}

    **Extracted Answer (Relevant sentences/phrases from Context):**
    """
    # --- **** END OF MODIFIED PROMPT **** ---

    print(f"⚙️ Calling Gemini (Relaxed Extractive) ({len(relevant_chunks)} chunks) for Q: '{question}'")
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        safety_settings = {}
        response = model.generate_content(prompt, safety_settings=safety_settings)

        # Robust response handling (same as before)
        try:
             answer_text = response.text.strip()
             if not answer_text:
                  print("⚠️ Warning: Gemini returned empty text (Relaxed Extractive).")
                  finish_reason = "Unknown"
                  try: finish_reason = response.candidates[0].finish_reason
                  except Exception: pass
                  return f"The model returned an empty response (Finish reason: {finish_reason}). Please try rephrasing."

             # Check if the model explicitly stated unavailability
             if "answer is not available" in answer_text.lower():
                  print("ℹ️ LLM indicated answer not extractable even with relaxed prompt.")
                  return "The answer is not available in the provided context."

             # Post-processing: remove potential introductory phrases if model adds them despite instructions
             answer_text = re.sub(r"^\s*Based on the context,?\s*", "", answer_text, flags=re.IGNORECASE).strip()
             answer_text = re.sub(r"^\s*The relevant sentences? are:?\s*", "", answer_text, flags=re.IGNORECASE).strip()
             return answer_text # Return the potentially longer, extracted answer

        except ValueError as ve: # Safety block
            print(f"⚠️ Warning: Gemini response blocked (Relaxed Extractive). Error: {ve}")
            block_reason = "Unknown"
            try: block_reason = response.prompt_feedback.block_reason
            except Exception: pass
            return f"Response blocked due to safety filters ({block_reason}). Try rephrasing."
        except AttributeError as ae:
             print(f"⚠️ Warning: Could not access response text (Relaxed Extractive). Error: {ae}")
             return "Could not parse the model's response structure."

    except Exception as e:
        # General API error handling (same as before)
        print(f"🔴 ERROR: Error querying Gemini API (Relaxed Extractive): {e}")
        error_str = str(e).lower()
        if "api key not valid" in error_str: return "🔴 ERROR: Invalid Gemini API Key."
        if "quota" in error_str or "rate limit" in error_str: return "🟡 INFO: API rate limit possibly exceeded. Wait & try again."
        return f"An error occurred while communicating with the AI model."


# -----------------------------------------------------------
# 9️⃣ LLM Generative QA (IMPLEMENTED)
# -----------------------------------------------------------
def llm_generative_qa(question: str) -> str:
    # --- UNCHANGED ---
    """
    LLM-based generative QA using Gemini API with SEMANTIC retrieval.
    """
    if not API_KEY: return "🔴 ERROR: Gemini API Key not configured."
    if CHUNK_EMBEDDINGS is None:
         print("⚠️ Embeddings not ready for Generative QA, attempting re-initialization...")
         initialize_document_data()
         if CHUNK_EMBEDDINGS is None: return "🔴 ERROR: Document not processed."

    relevant_chunks = retrieve_relevant_chunks_semantic(question, top_k=5)

    if not relevant_chunks:
        print("⚠️ Semantic retrieval failed for Generative QA. Trying TF-IDF fallback...")
        text = load_pdf_content(KNOWLEDGE_FILE_PATH)
        if not text: return "Could not load knowledge base for fallback search."
        fallback_match = nlp_similarity_qa(question, text)
        if "Could not find" in fallback_match or "No relevant" in fallback_match or "error" in fallback_match.lower() or "No processable" in fallback_match:
             print("⚠️ No relevant context found by any method. Asking Gemini without specific context...")
             context = "No specific context from the document was found to be relevant to this question."
        else:
             context = fallback_match.replace('Contextual Match:', '').replace('Keyword Match:', '').strip().strip("'")
             print(f"⚠️ Using TF-IDF fallback context for Generative QA: '{context[:100]}...'")
    else:
        context = "\n\n---\n\n".join(relevant_chunks)

    prompt = f"""
    **Your Role:** You are a helpful assistant answering questions based *only* on the provided Context.
    **Task:** Read the following Context and Question carefully. Generate a clear, concise, and natural-sounding answer (typically 2-4 sentences) that directly addresses the Question using *only* the information present in the Context.
    **Instructions:**
    1.  Synthesize information from different parts of the Context if necessary.
    2.  Do NOT add any external knowledge, opinions, or details not found in the Context.
    3.  If the Context does not contain enough information to answer the Question accurately, state that clearly (e.g., "Based on the provided text, I cannot answer..." or "The document does not contain information about...").
    4.  Ensure the answer flows well and reads naturally.

    **Context:**
    \"\"\"
    {context}
    \"\"\"

    **Question:** {question}

    **Answer:**
    """
    print(f"⚙️ Calling Gemini (Generative) for Q: '{question}'")
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        safety_settings = {}
        response = model.generate_content(prompt, safety_settings=safety_settings)
        try:
             answer_text = response.text.strip()
             if not answer_text:
                  print("⚠️ Warning: Gemini (Generative) returned empty text.")
                  finish_reason = "Unknown"
                  try: finish_reason = response.candidates[0].finish_reason
                  except Exception: pass
                  return f"The model returned an empty response (Finish reason: {finish_reason}). Try rephrasing."
             answer_text = re.sub(r"^\s*Based on the provided context,?\s*", "", answer_text, flags=re.IGNORECASE).strip()
             answer_text = re.sub(r"^\s*According to the text,?\s*", "", answer_text, flags=re.IGNORECASE).strip()
             return answer_text
        except ValueError as ve:
            print(f"⚠️ Warning: Gemini (Generative) response blocked/invalid. Error: {ve}")
            block_reason = "Unknown"
            try: block_reason = response.prompt_feedback.block_reason
            except Exception: pass
            return f"Response blocked due to safety filters ({block_reason}). Try rephrasing."
        except AttributeError as ae:
             print(f"⚠️ Warning: Could not access generative response text. Error: {ae}")
             return "Could not parse the model's generative response."
    except Exception as e:
        print(f"🔴 ERROR: Error querying Gemini API (Generative): {e}")
        error_str = str(e).lower()
        if "api key not valid" in error_str: return "🔴 ERROR: Invalid Gemini API Key."
        if "quota" in error_str or "rate limit" in error_str: return "🟡 INFO: API rate limit possibly exceeded. Wait & try again."
        return f"An error occurred while communicating with the AI model for generation."


# --- Example Usage (for testing directly) ---
if __name__ == "__main__":
    # --- UNCHANGED ---
    if CHUNK_EMBEDDINGS is None:
         print("Running direct test, ensuring initialization...")
         initialize_document_data()

    test_questions = [
        "Who is Rohit Sharma's wife?",
        "Summarize Rohit Sharma's major achievements and records.", # Good for generative
        "Tell me about his charity work with animals.", # Good for generative
        "How many runs did he score in IPL in total?",
        "Who is Rohit Sharma?",
        "What awards did he receive?",
        "What is his profession?",
        "Does he practice meditation?", # Requires broader context
        "Who is Virat Kohli?" # Out of Document
    ]

    for q in test_questions:
        print(f"\n--- Testing Rule-Based ---")
        print(f"Q: {q}")
        print(f"A: {rule_based_qa(q)}")

        print(f"\n--- Testing LLM Extractive ---")
        print(f"Q: {q}")
        print(f"A: {llm_extractive_qa(q)}")

        print(f"\n--- Testing LLM Generative ---") # Now implemented
        print(f"Q: {q}")
        print(f"A: {llm_generative_qa(q)}")

