import re
import os
from typing import Dict, Any, List
# We use pypdf, which is the successor to PyPDF2
from pypdf import PdfReader 

# --- Configuration ---
# Ensure your PDF file is named 'knowledge.pdf' and is in the same directory
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
            # Safely extract text from each page
            try:
                # Add a space between pages to avoid words merging
                full_text += page.extract_text() + "\n"
            except:
                pass
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""
    
    # Clean up common PDF extraction issues (like extra newlines)
    full_text = re.sub(r'\n+', '\n', full_text)
    return full_text

def get_structured_facts(text_content: str) -> Dict[str, Any]:
    """
    Extracts key facts and the career table data using precise regex patterns.
    """
    facts = {}
    
    # 1. Direct Fact Extraction (using non-greedy matches and flags)
    facts["dob"] = re.search(r"Date of Birth: (.*?)Birth Place:", text_content, re.IGNORECASE | re.DOTALL)
    facts["father"] = re.search(r"Father: (.*?) \(worked", text_content, re.IGNORECASE | re.DOTALL)
    facts["mother"] = re.search(r"Mother: (.*?)Rohit Sharma was raised", text_content, re.IGNORECASE | re.DOTALL)
    facts["wife"] = re.search(r"Wife: (.*?) \(", text_content, re.IGNORECASE | re.DOTALL)
    facts["marriage_date"] = re.search(r"Marriage Date: (.*?)Children:", text_content, re.IGNORECASE | re.DOTALL)
    # --- NEW --- Added regex for children
    facts["children"] = re.search(r"Children: (.*?)Diet:", text_content, re.IGNORECASE | re.DOTALL)
    facts["ipl_titles"] = re.search(r"IPL Titles as Captain: (.*?)Matches Played:", text_content, re.IGNORECASE | re.DOTALL)
    facts["highest_odi_score"] = re.search(r"Highest ODI Score: (.*?)— World Record", text_content, re.IGNORECASE | re.DOTALL)
    facts["t20i_retirement"] = re.search(r"T20I Retirement: (.*?)after winning the T20 World Cup", text_content, re.IGNORECASE | re.DOTALL)
    facts["test_retirement"] = re.search(r"Test Retirement: (.*?)\n", text_content, re.IGNORECASE | re.DOTALL)
    facts["icc_titles"] = re.search(r"Major ICC Titles Won as Player:(.*?)Arjuna Award", text_content, re.IGNORECASE | re.DOTALL)
    facts["charity_work"] = re.search(r"Collaborates with PETA India and WWF-India (.*?)Invested in Prozo", text_content, re.IGNORECASE | re.DOTALL)
    facts["school"] = re.search(r"School: (.*?)Coach:", text_content, re.IGNORECASE | re.DOTALL)
    facts["coach"] = re.search(r"Coach: (.*?) - helped him", text_content, re.IGNORECASE | re.DOTALL)
    facts["domestic_debut"] = re.search(r"Domestic Debut: (.*?)First-Class Debut:", text_content, re.IGNORECASE | re.DOTALL)
    facts["career_runs_total"] = re.search(r"Career Runs \(as of 2025\): (.*?)\n", text_content, re.IGNORECASE | re.DOTALL)

    
    # 2. Table Extraction Logic (targeting the multi-line, comma-separated block)
    # This regex is specifically built for the table structure in your PDF
    table_match = re.search(
        r"Format,Matches,Runs,Average,100s,50s,Highest Score,Strike Rate,Fours(.*?)\n"
        r"(Test,.*?)\n"
        r"(ODI,.*?)\n"
        r"(T20I,.*?)\n"
        r"(IPL,.*?)", 
        text_content, re.DOTALL
    )
    
    if table_match:
        # Define headers based on the full structure in the input
        header = ["Format", "Matches", "Runs", "Average", "100s", "50s", "Highest Score", "Strike Rate", "Fours", "Sixes"]
        stats = {}
        # The groups[1:] contains the 4 data rows (Test, ODI, T20I, IPL)
        for row_str in table_match.groups()[1:]: 
            # Split by comma, remove quotes and newlines
            row_data = [item.strip('"\n ') for item in row_str.split(',')]
            if row_data:
                format_key = row_data[0].lower() # 'test', 'odi', etc.
                stats[format_key] = dict(zip(header[1:], row_data[1:]))
        facts['career_stats'] = stats

    # Clean and store the found facts
    extracted_facts = {}
    for key, match in facts.items():
        if isinstance(match, re.Match):
            # Clean up captured group (removing extra whitespace/newlines)
            extracted_facts[key] = match.group(1).strip().replace('\n', ' ')
        elif key == 'career_stats':
             extracted_facts[key] = match 
            
    return extracted_facts

# --- Utility Function for Rule Matching (PASSING question_tokens and question_lower) ---

def extract_from_facts(question_tokens: List[str], question_lower: str, facts: Dict[str, Any]) -> str | None:
    """Checks for specific, pre-defined factual queries against the extracted facts."""
    
    # 1. Family & Personal Facts Lookup
    if ("father" in question_tokens or "dad" in question_tokens) and 'father' in facts:
        return f"His father's name is {facts['father']}."
    # --- NEW --- Added rule for mother
    if ("mother" in question_tokens or "mom" in question_tokens) and 'mother' in facts:
        return f"His mother's name is {facts['mother']}."
    # --- NEW --- Added rule for children/kids
    if ("children" in question_tokens or "kids" in question_tokens or "daughter" in question_tokens or "son" in question_tokens) and 'children' in facts:
        return f"His children are: {facts['children']}."
    if ("wife" in question_tokens or "spouse" in question_tokens) and 'wife' in facts:
        return f"His wife is {facts['wife']}."
    if ("birth" in question_tokens and ("date" in question_tokens or "when" in question_tokens)) and 'dob' in facts:
        return f"His date of birth is {facts['dob']}."
    if ("school" in question_tokens) and 'school' in facts:
        return f"He attended {facts['school']} in Mumbai."
    if ("coach" in question_tokens or "trained" in question_tokens) and 'coach' in facts:
        return f"His coach was {facts['coach']}."
        
    # 2. Career & Records Lookup
    if "highest" in question_tokens and "odi" in question_tokens and "score" in question_tokens and 'highest_odi_score' in facts:
        return f"His highest ODI score is {facts['highest_odi_score']}, a world record."
    if "ipl" in question_tokens and ("titles" in question_tokens or "captain" in question_tokens) and 'ipl_titles' in facts:
        return f"He has won {facts['ipl_titles']} IPL Titles as captain, and is the only player to win six IPL titles."
    if "retirement" in question_tokens and "t20" in question_lower and 't20i_retirement' in facts:
        return f"He retired from T20I cricket in {facts['t20i_retirement']}."
    if "retirement" in question_tokens and "test" in question_lower and 'test_retirement' in facts:
        return f"His planned Test Retirement date is {facts['test_retirement']}."
    if ("icc" in question_tokens or "trophy" in question_tokens or "world cup" in question_lower) and 'icc_titles' in facts:
        # Clean up the ICC titles block for display
        titles = facts['icc_titles'].replace('\n', ', ').strip()
        return f"His major ICC titles include: {titles}"
    if ("debut" in question_tokens or "first match" in question_lower) and 'domestic_debut' in facts:
        return f"His Domestic Debut was in the {facts['domestic_debut']}."
    if ("total" in question_tokens and "career" in question_tokens and ("score" in question_tokens or "runs" in question_tokens)) and 'career_runs_total' in facts:
        return f"His total career runs (as of 2025) are: {facts['career_runs_total']}."
    
    # 3. Non-Cricket Facts (Charity/Business)
    if "charity" in question_tokens or "ambassador" in question_tokens or "wwf" in question_lower and 'charity_work' in facts:
        work = facts['charity_work'].replace('\n', ' ').strip()
        return f"He collaborates with {work}."
    if "prozo" in question_tokens or "invested" in question_tokens:
        return "He invested in Prozo, an Indian supply chain technology company, in 2025."


    # 4. Statistical Table Lookup
    if 'career_stats' in facts and any(stat in question_lower for stat in ['runs', 'matches', 'average', 'centuries', '100s', '50s', 'fours', 'sixes', 'strike rate']):
        
        # Determine the requested format
        format_key = 'odi' if 'odi' in question_lower else \
                     'test' if 'test' in question_lower else \
                     't20i' if 't20i' in question_lower else \
                     'ipl' if 'ipl' in question_lower else \
                     None
                     
        if format_key and format_key in facts['career_stats']:
            stats = facts['career_stats'][format_key]
            
            # Match the requested statistic
            if 'runs' in question_lower:
                return f"In {format_key.upper()}, he scored {stats.get('Runs', 'N/A')} runs."
            if 'matches' in question_lower:
                return f"He played {stats.get('Matches', 'N/A')} {format_key.upper()} matches."
            if 'average' in question_lower:
                return f"His {format_key.upper()} batting average is {stats.get('Average', 'N/A')}."
            if 'centuries' in question_lower or '100s' in question_lower:
                return f"He has scored {stats.get('100s', 'N/A')} centuries and {stats.get('50s', 'N/A')} half-centuries in {format_key.upper()}."
            if 'sixes' in question_lower:
                 return f"He has hit {stats.get('Sixes', 'N/A')} sixes in {format_key.upper()}."
        
    return None # Return None if no specific fact match is found

# --- Core Simple Rule-Based QA System ---

def rule_based_qa(question: str) -> str:
    """
    Simple rule-based QA using token matching and pre-defined fact lookups, 
    reading from the external PDF file.
    """
    pdf_content = load_pdf_content(KNOWLEDGE_FILE_PATH)
    if not pdf_content:
        return "Could not load or extract text from knowledge.pdf."
        
    structured_facts = get_structured_facts(pdf_content)
    
    # --- Define variables here to avoid NameError ---
    question_lower = question.lower()
    question_tokens = re.findall(r'\w+', question_lower)
    
    # 1. Try to extract from structured facts (High Confidence)
    # --- Pass both question_tokens and question_lower ---
    fact_answer = extract_from_facts(question_tokens, question_lower, structured_facts)
    if fact_answer:
        # Return clean answer
        return fact_answer
        
    # 2. Fallback to basic keyword search (Lower Confidence)
    # Split by sentences
    sentences = re.split(r'[.!?\n]', pdf_content)
    best_match = None
    best_score = 0
    
    for sentence in sentences:
        # Score based on how many question tokens are in the sentence
        score = sum(1 for token in question_tokens if token in sentence.lower())
        
        # We need a decent keyword overlap and a non-empty sentence
        if score > 1 and len(sentence.strip()) > 15: # Require score > 1 and min length
            if score > best_score:
                best_score = score
                best_match = sentence.strip()
            
    if best_match:
        # Return clean contextual match
        return f"Contextual Match: '{best_match}'"
    
    return "The system could not find a relevant fact or keyword match in the document."

# --- Placeholder Functions for LLM Modes ---

def llm_extractive_qa(question: str) -> str:
    """
    Placeholder for the LLM-Based Extractive QA system (RAG with strict prompt).
    """
    # NOTE: You must implement the RAG pipeline here (Vector DB lookup + LLM Extraction)
    # This placeholder clearly states it's not implemented.
    return (f"[LLM Extractive Mode - Not Implemented]\n"
            f"This mode is a placeholder. To make it functional, you must:\n"
            f"1. Implement PDF chunking and vector embedding (e.g., with LangChain and ChromaDB).\n"
            f"2. Pass the retrieved context and question to an LLM with a strict 'extract-only' prompt.")

def llm_generative_qa(question: str) -> str:
    """
    Placeholder for the LLM-Based Generative QA system (RAG with synthesis).
    """
    # NOTE: You must implement the RAG pipeline here (Vector DB lookup + LLM Generation)
    return (f"[LLM Generative Mode - Not Implemented]\n"
            f"This mode is a placeholder. To make it functional, you must:\n"
            f"1. Implement PDF chunking and vector embedding (e.g., with LangChain and ChromaDB).\n"
            f"2. Pass the retrieved context and question to an LLM (like Gemini) and ask it to generate a new answer.")

