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
    # Consider adding logic to disable LLM modes if key is missing
else:
    genai.configure(api_key=API_KEY)
    print("✅ Gemini API Key configured.")

# -----------------------------------------------------------
# 1️⃣ Utility: Load PDF Text
# -----------------------------------------------------------
def load_pdf_content(file_path: str) -> str:
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
    facts = {}
    # Use non-greedy matching .*? and flags for robustness
    patterns = {
        "full_name": r"Full Name:\s*(.*?)\s*Date of Birth",
        "profession": r"Profession:\s*(.*?)\s*Batting Style",
        "playing_role": r"Playing Role:\s*(.*?)\n",
        "father": r"Father:\s*(.*?)\s*\(",
        "mother": r"Mother:\s*(.*?)\s*(?:Rohit Sharma was raised|\()", # Handle variations
        "wife": r"Wife:\s*(.*?)\s*\(",
        "children": r"Children:\s*(.*?)\s*Diet",
        "school": r"School:\s*(.*?)\s*Coach",
        "coach": r"Coach:\s*(.*?)(?:–|- helped him)", # Handle variations
        # --- NEW: Extract blocks for achievements/titles/awards ---
        "icc_titles": r"Major ICC Titles Won as Player:\s*(.*?)\s*Arjuna Award",
        "national_awards": r"(Arjuna Award\s*–\s*\d{4}.*?Rajiv Gandhi Khel Ratna\s*–\s*\d{4})",
        "other_honors": r"(ICC ODI Cricketer.*?Padma Shri Nominee.*?)\n",
        # --- NEW: Extract Highest ODI Score for achievement questions ---
        "highest_odi_record": r"Highest ODI Score:\s*(.*?)\s*— World Record",
        "odi_double_centuries": r"Most Double Centuries in ODIs:\s*(.*?)\n",
        "world_cup_record": r"Most Centuries in a Single World Cup:\s*(.*?)\n",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            # Clean extracted text: remove extra spaces/newlines
            facts[key] = re.sub(r'\s+', ' ', m.group(1).strip())
    return facts

# -----------------------------------------------------------
# 3️⃣ Simple fact lookup rules (EXPANDED)
# -----------------------------------------------------------
def extract_from_facts(question: str, facts: Dict[str, str]) -> str | None:
    q_lower = question.lower()
    tokens = set(re.findall(r"\w+", q_lower)) # Use set for faster lookups

    # --- NEW: Handle "Who is Rohit Sharma?" ---
    if ("who" in tokens and ("rohit" in tokens or "sharma" in tokens or "he" in tokens)) or \
       ("tell" in tokens and "about" in tokens and ("rohit" in tokens or "sharma" in tokens)):
        name = facts.get("full_name", "Rohit Sharma")
        prof = facts.get("profession", "a Cricketer")
        role = facts.get("playing_role", "Opening Batter and Captain")
        return f"{name} is an Indian {prof}, known for his role as an {role}."

    # --- NEW: Handle Profession/Role ---
    if "profession" in tokens or ("what" in tokens and "does" in tokens and "he" in tokens and "do" in tokens):
         prof = facts.get("profession", "Cricketer (Batsman, occasional right-arm offbreak bowler)")
         role = facts.get("playing_role", "Opening Batter and Captain")
         return f"His profession is {prof}, and his primary playing role is {role}."

    # --- NEW: Handle Achievements/Awards/Titles ---
    if "achievement" in tokens or "award" in tokens or "title" in tokens or "record" in tokens or "honors" in tokens:
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
            return "Some of his major achievements include: " + " ".join(response_parts)
        else:
            # If specific extractions failed, let it fall through to similarity search
             return None


    # Check existing facts dictionary before returning
    if ("father" in tokens or "dad" in tokens) and facts.get("father"):
        return f"His father is {facts['father']}."
    if ("mother" in tokens or "mom" in tokens) and facts.get("mother"):
        return f"His mother is {facts['mother']}."
    if ("wife" in tokens or "spouse" in tokens) and facts.get("wife"):
        return f"His wife is {facts['wife']}."
    if any(x in tokens for x in ["child", "children", "kids", "daughter", "son"]) and facts.get("children"):
        # Clean up children string if needed
        children_info = facts['children'].replace(' (born ', ', born ').replace(') and a son (born ', '; son born ')
        return f"His children are: {children_info}."
    if "coach" in tokens and facts.get("coach"):
        return f"His coach was {facts['coach']}."
    if "school" in tokens and facts.get("school"):
        return f"He attended {facts['school']}."
    return None

# -----------------------------------------------------------
# 4️⃣ NLP Extractive QA (TF-IDF + cosine) - Fallback Method
# -----------------------------------------------------------
def nlp_similarity_qa(question: str, text: str) -> str:
    try:
        # Use NLTK sent_tokenize if available and working
        sentences = [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 15] # Increase min length
    except Exception as e:
        print(f"⚠️ Warning: NLTK sentence tokenization failed. Error: {e}")
        # Fallback split by newline, assuming paragraphs are separated by lines
        sentences = [p.strip() for p in text.split('\n') if len(p.strip()) > 15]

    if not sentences:
        return "No processable sentences found in the document."

    # TF-IDF + cosine
    try:
        vec = TfidfVectorizer(stop_words="english", min_df=1) # Lowered min_df for potentially smaller doc
        matrix = vec.fit_transform(sentences) # Fit only on sentences first
        question_vec = vec.transform([question]) # Transform question separately

        # Ensure question_vec is not empty (can happen if question is only stop words)
        if question_vec.nnz == 0:
             print("⚠️ Warning: Question vector is empty after stop word removal.")
             # Fallback: simple keyword check without TF-IDF
             q_tokens = set(re.findall(r'\w+', question.lower()))
             for sentence in sentences:
                 s_tokens = set(re.findall(r'\w+', sentence.lower()))
                 if q_tokens.intersection(s_tokens):
                     return f"Keyword Match: '{sentence}'"
             return "Could not find a relevant keyword match (question too generic?)."


        sim = cosine_similarity(question_vec, matrix)[0]
        best_idx = sim.argmax()

        # Set a reasonable similarity threshold
        if sim[best_idx] < 0.1: # Adjusted threshold slightly lower for broader fallback
            print(f"ℹ️ Top TF-IDF similarity score ({sim[best_idx]:.2f}) below threshold (0.1).")
            return "Could not find a highly relevant sentence using keyword matching."
        print(f"ℹ️ Found TF-IDF match with score {sim[best_idx]:.2f}.")
        return f"Contextual Match: '{sentences[best_idx]}'"

    except ValueError as ve:
         # Handle case where vocabulary might be empty after stop words/min_df
        print(f"⚠️ Warning: TF-IDF Vectorizer failed. Possibly due to short text or filtering. Error: {ve}")
        # Simple fallback: return first sentence containing any keyword
        q_tokens = set(re.findall(r'\w+', question.lower()))
        for sentence in sentences:
            s_tokens = set(re.findall(r'\w+', sentence.lower()))
            if q_tokens.intersection(s_tokens):
                return f"Keyword Match: '{sentence}'"
        return "Could not find a relevant keyword match in the document."
    except Exception as e:
        print(f"🔴 ERROR: Unexpected error during TF-IDF similarity: {e}")
        return "An error occurred during relevance analysis."

# -----------------------------------------------------------
# 5️⃣ Unified Rule-Based/NLP QA entry point
# -----------------------------------------------------------
def rule_based_qa(question: str) -> str:
    """Hybrid QA: structured facts + NLP similarity as fallback."""
    text = load_pdf_content(KNOWLEDGE_FILE_PATH)
    if not text:
        return "Could not load knowledge base."

    # Try rule-based extraction first (high precision)
    facts = extract_structured_facts(text)
    rule_ans = extract_from_facts(question, facts)
    if rule_ans:
        print("✅ Answered using high-precision rule.")
        return rule_ans

    # Fallback to NLP similarity (broader context matching)
    print("ℹ️ No specific rule matched, falling back to NLP similarity search...")
    return nlp_similarity_qa(question, text)

# -----------------------------------------------------------
# 6️⃣ Chunking and Embedding Generation (RAG Setup)
# -----------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split large document text into overlapping chunks using sentences."""
    try:
        sentences = sent_tokenize(text)
    except Exception as e:
        print(f"⚠️ Warning: NLTK sentence tokenization failed during chunking. Error: {e}")
        # Fallback to paragraph splitting if NLTK fails
        sentences = [p.strip() for p in text.split('\n\n') if p.strip()]

    chunks = []
    current_chunk_words = []
    current_chunk = ""

    for sentence in sentences:
        sentence_words = sentence.split()
        # Check if adding the next sentence exceeds chunk size
        if len(current_chunk_words) + len(sentence_words) <= chunk_size:
            current_chunk_words.extend(sentence_words)
            current_chunk += " " + sentence
        else:
            # Add the completed chunk if it's not empty
            if current_chunk.strip():
                 chunks.append(current_chunk.strip())
            # Start new chunk with overlap
            # Ensure overlap doesn't exceed current chunk length
            overlap_word_count = min(overlap, len(current_chunk_words))
            overlap_words = current_chunk_words[-overlap_word_count:]
            # Start the new chunk with the overlap words and the current sentence
            current_chunk_words = overlap_words + sentence_words
            current_chunk = " ".join(overlap_words) + " " + sentence
            # Handle cases where a single sentence is larger than chunk_size
            if len(sentence_words) > chunk_size:
                 print(f"⚠️ Warning: Sentence starting with '{sentence[:50]}...' is longer than chunk size {chunk_size}. Adding as its own chunk.")
                 if current_chunk.strip(): # Add previous chunk first if any
                     chunks.append(current_chunk.strip())
                 chunks.append(sentence) # Add long sentence as chunk
                 current_chunk_words = [] # Reset for next iteration
                 current_chunk = ""


    if current_chunk.strip(): # Add the last chunk
        chunks.append(current_chunk.strip())

    # Filter out potentially very short chunks created by overlap logic
    final_chunks = [ch for ch in chunks if len(ch.split()) > 10]
    print(f"📄 Created {len(final_chunks)} chunks from the document.")
    return final_chunks


def generate_embeddings(texts: list[str]) -> np.ndarray | None:
    """Generate embeddings for a list of texts using Google AI."""
    if not API_KEY:
        print("🔴 ERROR: Cannot generate embeddings, API Key not configured.")
        return None
    try:
        # Batching is handled automatically by embed_content for lists
        result = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=texts,
            task_type="retrieval_document" # Or "RETRIEVAL_QUERY" for the question
        )
        print(f"✨ Generated embeddings for {len(texts)} texts.")
        return np.array(result['embedding'])
    except Exception as e:
        print(f"🔴 ERROR: Failed to generate embeddings via Gemini API: {e}")
        # Consider adding more robust error handling (e.g., retries with backoff)
        # time.sleep(2**retry_count)
        return None

def initialize_document_data():
    """Load, chunk, and embed the document ONCE."""
    global PDF_CHUNKS, CHUNK_EMBEDDINGS
    # Check if already initialized to prevent re-running
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
                 print("🔴 ERROR: Failed to generate embeddings for document chunks during initialization.")
        else:
            print("🔴 ERROR: No chunks created from the document during initialization.")
    else:
        print("🔴 ERROR: Failed to load document content for initialization.")

# --- Run Initialization when module is loaded ---
# Ensure NLTK punkt is available before initializing
try:
    nltk.data.find('tokenizers/punkt')
    initialize_document_data()
except Exception as e:
    print(f"🔴 ERROR during initial setup (NLTK or Document Data): {e}")


# -----------------------------------------------------------
# 7️⃣ Semantic Retrieval for RAG
# -----------------------------------------------------------
def retrieve_relevant_chunks_semantic(question: str, top_k: int = 3) -> list[str]:
    """Retrieve top-k relevant text chunks using semantic embeddings."""
    global PDF_CHUNKS, CHUNK_EMBEDDINGS

    if CHUNK_EMBEDDINGS is None or len(PDF_CHUNKS) == 0:
        print("🔴 ERROR: Document embeddings not available. Cannot retrieve.")
        return []

    # Generate embedding specifically for the query
    question_embedding_result = genai.embed_content(
        model=EMBEDDING_MODEL_NAME,
        content=question,
        task_type="retrieval_query" # Use query type for question embedding
    )
    question_embedding = np.array(question_embedding_result['embedding'])

    if question_embedding is None or question_embedding.ndim == 0: # Check if embedding failed
        print("🔴 ERROR: Failed to generate embedding for the question.")
        return []

    # Ensure embeddings are 2D for cosine_similarity
    if question_embedding.ndim == 1:
        question_embedding = question_embedding.reshape(1, -1)
    if CHUNK_EMBEDDINGS.ndim == 1: # Should not happen if generate_embeddings is correct
         print("🔴 ERROR: Chunk embeddings are not in the expected format.")
         return []


    # Calculate cosine similarities
    try:
        similarities = cosine_similarity(question_embedding, CHUNK_EMBEDDINGS)[0]
    except ValueError as ve:
         print(f"🔴 ERROR: Cosine similarity calculation failed. Check embedding dimensions. Error: {ve}")
         return []


    # Get top_k indices, sorted by similarity DESC
    k = min(top_k, len(similarities))
    # Ensure indices are within bounds
    if k <= 0: return []
    # Use partition for potentially faster top-k selection than full sort
    # top_indices = np.argpartition(similarities, -k)[-k:]
    # sorted_top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]

    # Simpler argsort for clarity
    sorted_indices = np.argsort(similarities)[::-1] # Sort all indices descending
    top_indices = sorted_indices[:k] # Take the top k

    # Filter by a minimum similarity threshold
    threshold = 0.3 # Increased threshold for semantic search
    relevant_chunks = []
    scores = []
    for i in top_indices:
         # Check index bounds explicitly
         if i < len(PDF_CHUNKS) and similarities[i] > threshold:
              relevant_chunks.append(PDF_CHUNKS[i])
              scores.append(similarities[i])

    if relevant_chunks:
        print(f"🔍 Retrieved {len(relevant_chunks)} relevant chunks using semantic search (Top score: {scores[0]:.3f}).")
    else:
        print(f"⚠️ No chunks found above semantic similarity threshold ({threshold}). Max score: {similarities[top_indices[0]]:.3f}" if len(top_indices) > 0 else "No chunks retrieved.")

    return relevant_chunks

# -----------------------------------------------------------
# 8️⃣ Gemini LLM-Based Extractive QA (Using Semantic RAG)
# -----------------------------------------------------------
def llm_extractive_qa(question: str) -> str:
    """
    LLM-based extractive QA using Gemini API with SEMANTIC retrieval.
    Retrieves top chunks via embeddings and asks Gemini to extract a factual answer.
    """
    if not API_KEY:
        return "🔴 ERROR: Gemini API Key not configured. Cannot perform LLM QA."
    if CHUNK_EMBEDDINGS is None:
         # Attempt re-initialization if failed on load
         print("⚠️ Embeddings not ready, attempting re-initialization...")
         initialize_document_data()
         if CHUNK_EMBEDDINGS is None:
              return "🔴 ERROR: Document not processed for semantic search. Cannot perform LLM QA."

    relevant_chunks = retrieve_relevant_chunks_semantic(question, top_k=4) # Retrieve more chunks

    if not relevant_chunks:
        # Fallback to basic NLP similarity if semantic search fails
        print("⚠️ Semantic retrieval failed, falling back to TF-IDF keyword match...")
        text = load_pdf_content(KNOWLEDGE_FILE_PATH) # Reload text for fallback
        if not text: return "Could not load knowledge base for fallback search."

        fallback_answer = nlp_similarity_qa(question, text)
        # Check if fallback provided a real answer or an error/no-match message
        if "Could not find" in fallback_answer or "No relevant" in fallback_answer or "error" in fallback_answer.lower() or "No processable" in fallback_answer:
             # If TF-IDF also fails, give a generic failure message
             return "Could not find relevant information in the document using multiple methods."
        else:
             # Return the TF-IDF match but indicate it's lower confidence
             clean_fallback = fallback_answer.replace('Contextual Match:', '').replace('Keyword Match:', '').strip().strip("'")
             return f"Found a possible match using keyword search: '{clean_fallback}' (Semantic search failed to find a high-confidence match)."

    context = "\n\n---\n\n".join(relevant_chunks) # Add separators for clarity

    # Strict Extractive Prompt - Refined
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
        # Use the correct, stable model ID
        model = genai.GenerativeModel("gemini-2.5-flash")
        # Safety settings (optional, adjust if overly restrictive)
        safety_settings = {
            # genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            # genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            # genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            # genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        }
        response = model.generate_content(prompt, safety_settings=safety_settings)

        # Enhanced check for empty or blocked response
        try:
             # Accessing response.text directly can raise an exception if blocked
             answer_text = response.text.strip()
             if not answer_text:
                  print("⚠️ Warning: Gemini returned an empty text response.")
                  # Check candidate parts manually if text is empty
                  if response.candidates and response.candidates[0].content.parts:
                      # If parts exist but text is empty, maybe unusual content?
                       return "Model response received but contained no standard text."
                  else: # Truly empty
                       return "The model returned an empty response. Please try rephrasing."
             return answer_text # Return the valid text

        except ValueError as ve: # Often indicates blocking by safety filters
            print(f"⚠️ Warning: Gemini response blocked or invalid. Error: {ve}")
            # Try to get more specific block reason
            block_reason = "Unknown"
            try:
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    block_reason = response.prompt_feedback.block_reason
                elif response.candidates and response.candidates[0].finish_reason != 'STOP':
                     block_reason = f"Generation stopped: {response.candidates[0].finish_reason}"

            except Exception: pass # Ignore errors trying to get reason

            return f"The response was blocked or could not be generated ({block_reason}). Try rephrasing the question or check safety settings."

    except Exception as e:
        # General API error handling
        print(f"🔴 ERROR: Error querying Gemini API: {e}")
        # Add checks for specific errors like API key validity, rate limits, etc.
        error_str = str(e).lower()
        if "api key not valid" in error_str:
             return "🔴 ERROR: Invalid Gemini API Key. Please check your environment variable."
        elif "quota" in error_str or "rate limit" in error_str:
             return "🟡 INFO: API rate limit possibly exceeded. Please wait and try again."
        return f"An error occurred while communicating with the AI model: {e}"


# -----------------------------------------------------------
# 9️⃣ Placeholder for LLM Generative QA
# -----------------------------------------------------------
def llm_generative_qa(question: str) -> str:
    """Placeholder for the LLM-Based Generative QA system."""
    # This would be similar to llm_extractive_qa but with a different prompt
    # asking the model to synthesize or summarize rather than extract.
    return (f"[LLM Generative Mode - Not Implemented]\n"
            f"This mode is a placeholder. It would use semantic retrieval like the extractive mode, "
            f"but the prompt would ask Gemini to generate a *new*, fluent answer based on the context, "
            f"rather than just extracting text.")

# --- Example Usage (for testing directly) ---
if __name__ == "__main__":
    # Ensure initialization runs if testing directly
    if CHUNK_EMBEDDINGS is None:
         print("Running direct test, ensuring initialization...")
         initialize_document_data()

    test_question = "Who is Rohit Sharma's wife?"
    print(f"\n--- Testing Rule-Based ---")
    print(f"Q: {test_question}")
    print(f"A: {rule_based_qa(test_question)}")

    test_question_2 = "What are his major achievements in world cups?"
    print(f"\n--- Testing LLM Extractive ---")
    print(f"Q: {test_question_2}")
    print(f"A: {llm_extractive_qa(test_question_2)}")

    test_question_3 = "Tell me about his charity work."
    print(f"\n--- Testing LLM Extractive (Charity) ---")
    print(f"Q: {test_question_3}")
    print(f"A: {llm_extractive_qa(test_question_3)}")

    test_question_4 = "How many runs did he score in IPL?" # Requires table lookup
    print(f"\n--- Testing Rule-Based (Table Fallback - might fail) ---")
    print(f"Q: {test_question_4}")
    print(f"A: {rule_based_qa(test_question_4)}") # Rule-based might fail here

    test_question_5 = "Who is Rohit Sharma?" # Test general question
    print(f"\n--- Testing Rule-Based (General) ---")
    print(f"Q: {test_question_5}")
    print(f"A: {rule_based_qa(test_question_5)}") # Should hit the new rule

    test_question_6 = "What awards did he receive?" # Test achievements rule
    print(f"\n--- Testing Rule-Based (Awards) ---")
    print(f"Q: {test_question_6}")
    print(f"A: {rule_based_qa(test_question_6)}")

    test_question_7 = "What is his profession?" # Test profession rule
    print(f"\n--- Testing Rule-Based (Profession) ---")
    print(f"Q: {test_question_7}")
    print(f"A: {rule_based_qa(test_question_7)}")

    test_question_8 = "Does he meditate?" # Test TF-IDF fallback
    print(f"\n--- Testing Rule-Based (Fallback) ---")
    print(f"Q: {test_question_8}")
    print(f"A: {rule_based_qa(test_question_8)}")

    test_question_9 = "Who is Sachin Tendulkar?" # Test LLM OOD
    print(f"\n--- Testing LLM Extractive (Out of Document) ---")
    print(f"Q: {test_question_9}")
    print(f"A: {llm_extractive_qa(test_question_9)}")

