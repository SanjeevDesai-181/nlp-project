import re
import os
from typing import Dict, Any, List
from pypdf import PdfReader # Used for simple text extraction

# --- Configuration ---
KNOWLEDGE_FILE_PATH = "knowledge.pdf"

# --- Utility Functions for PDF Loading and Fact Extraction ---

def load_pdf_content(file_path: str) -> str:
    """Reads the entire text content from the PDF file."""
    if not os.path.exists(file_path):
        print(f"ERROR: Knowledge file not found at {file_path}")
        return ""
    
    full_text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""
    
    return full_text

def get_structured_facts(text_content: str) -> Dict[str, str]:
    """
    Extracts high-confidence, single-line facts using simple pattern matching.
    In a simple rule-based system, this provides quick lookups.
    """
    facts = {}
    
    # Example regex patterns for structured data in your file
    facts["dob"] = re.search(r"Date of Birth: (.*?)\n", text_content, re.IGNORECASE)
    facts["birth_place"] = re.search(r"Birth Place: (.*?)\n", text_content, re.IGNORECASE)
    facts["father"] = re.search(r"Father: (.*?) \(", text_content, re.IGNORECASE)
    facts["wife"] = re.search(r"Wife: (.*?)\n", text_content, re.IGNORECASE)
    facts["ipl_titles"] = re.search(r"IPL Titles as Captain: (.*?)\.", text_content, re.IGNORECASE)
    facts["odi_score"] = re.search(r"Highest ODI Score: (.*?) World Record", text_content, re.IGNORECASE)
    facts["career_centuries"] = re.search(r"Career Centuries: (.*?)\.", text_content, re.IGNORECASE)
    facts["t20i_retirement"] = re.search(r"T20I Retirement: (.*?)\n", text_content, re.IGNORECASE)

    # Clean and store the found facts
    extracted_facts = {}
    for key, match in facts.items():
        if match:
            extracted_facts[key] = match.group(1).strip()
            
    return extracted_facts


# --- Utility Function for Rule Matching ---

def extract_from_facts(question_tokens: List[str], facts: Dict[str, str]) -> str:
    """Checks for specific, pre-defined factual queries against the extracted facts."""
    
    q = " ".join(question_tokens)
    
    # 1. Family Facts Lookup
    if ("father" in question_tokens or "dad" in question_tokens) and 'father' in facts:
        return f"His father's name is {facts['father']}."
    if ("wife" in question_tokens or "spouse" in question_tokens) and 'wife' in facts:
        return f"His wife is {facts['wife']}."
    if ("birth" in question_tokens and ("date" in question_tokens or "when" in question_tokens)) and 'dob' in facts:
        return f"His date of birth is {facts['dob']}."
        
    # 2. Cricket Records/Stats Lookup
    if "highest" in question_tokens and "odi" in question_tokens and "score" in question_tokens and 'odi_score' in facts:
        return f"His highest ODI score is {facts['odi_score']}, a world record."
    if "ipl" in question_tokens and ("titles" in question_tokens or "captain" in question_tokens) and 'ipl_titles' in facts:
        return f"He has won {facts['ipl_titles']} IPL titles as captain."
    if "total" in question_tokens and "centuries" in question_tokens and 'career_centuries' in facts:
        return f"His career centuries are: {facts['career_centuries']}."
        
    # 3. Retirements Lookup
    if "retirement" in question_tokens and "t20" in q.lower() and 't20i_retirement' in facts:
        return f"He retired from T20I cricket in {facts['t20i_retirement']}."
        
    return None # Return None if no specific fact match is found

# --- Core Simple Rule-Based QA System ---

def rule_based_qa(question: str) -> str:
    """
    Simple rule-based QA using token matching and pre-defined fact lookups, 
    reading from the external PDF file.
    """
    # Load and structure content from the PDF
    pdf_content = load_pdf_content(KNOWLEDGE_FILE_PATH)
    if not pdf_content:
        return "[Rule-Based - Error] Could not load or extract text from knowledge.pdf."
        
    structured_facts = get_structured_facts(pdf_content)
    
    question_lower = question.lower()
    question_tokens = re.findall(r'\w+', question_lower)
    
    # 1. Try to extract from structured facts (High Confidence)
    fact_answer = extract_from_facts(question_tokens, structured_facts)
    if fact_answer:
        return f"[Rule-Based - High Confidence Lookup] {fact_answer}"
        
    # 2. Fallback to basic keyword search (Lower Confidence)
    sentences = re.split(r'[.!?\n]', pdf_content)
    best_match = None
    best_score = 0
    
    for sentence in sentences:
        score = sum(1 for token in question_tokens if token in sentence.lower())
        
        # We need a decent keyword overlap and a non-empty sentence
        if score > best_score and len(sentence.strip()) > 10:
            best_score = score
            best_match = sentence.strip()
            
    if best_score > 1 and best_match:
        return f"[Rule-Based - Keyword Match] Contextual Match: '{best_match}'"
    
    return "[Rule-Based - No Match] The system could not find a relevant fact or keyword match in the document."

# --- Placeholder Functions for LLM Modes ---

def llm_extractive_qa(question: str) -> str:
    """
    Placeholder for the LLM-Based Extractive QA system (RAG with strict prompt).
    """
    return f"[LLM Extractive - Placeholder] Ready to process question: '{question}'. Will retrieve context from PDF and extract the answer."

def llm_generative_qa(question: str) -> str:
    """
    Placeholder for the LLM-Based Generative QA system (RAG with synthesis).
    """
    return f"[LLM Generative - Placeholder] Ready to process question: '{question}'. Will retrieve context from PDF and generate a fluent summary."
