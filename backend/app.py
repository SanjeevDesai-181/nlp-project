from flask import Flask, request, jsonify
from flask_cors import CORS 
import os 
# Assuming qa_engine.py is in the same directory
from qa_engine import (
    rule_based_qa, 
    llm_extractive_qa, 
    llm_generative_qa
)

app = Flask(__name__)
# Enable CORS so your JavaScript frontend can make requests to this API
CORS(app) 

# Note on LLM Initialization: 
# If you integrate the LLM components in qa_engine.py later, you would add an 
# initialization call here (e.g., initialize_qa_engine()) to run once at startup.


@app.route('/api/answer', methods=['POST'])
def get_answer():
    """
    Main API endpoint to handle QA requests.
    Expects JSON: {"question": "...", "mode": "..."} from the frontend.
    """
    try:
        # Get JSON data from the request body
        data = request.get_json()
        question = data.get('question')
        mode = data.get('mode')
        
        # Basic validation
        if not question or not mode:
            return jsonify({"error": "Missing 'question' or 'mode' in request."}), 400

        # --- Dispatch to the correct QA mode logic ---
        if mode == 'rule-based':
            answer = rule_based_qa(question)
        elif mode == 'llm-extractive':
            # This calls the placeholder function in qa_engine.py
            answer = llm_extractive_qa(question)
        elif mode == 'llm-generative':
            # This calls the placeholder function in qa_engine.py
            answer = llm_generative_qa(question)
        else:
            answer = "Invalid QA mode selected."

        # --- Return the result to the frontend ---
        return jsonify({
            "mode": mode,
            "question": question,
            "answer": answer
        })

    except Exception as e:
        print(f"An error occurred: {e}")
        # Return a generic server error response
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    # Run the Flask app on default port 5000
    print("--- Running Flask Server ---")
    print("API endpoint: http://127.0.0.1:5000/api/answer")
    print("Ensure 'knowledge.pdf' is in the same directory.")
    # Set debug=True for development to allow automatic reloading on code changes
    app.run(debug=True)
