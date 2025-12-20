import os
import re
import json
import time
import fitz  # PyMuPDF
import google.generativeai as genai
from dotenv import load_dotenv
from solidity_parser import parser

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY is not set. Please add it to your .env file.")
    exit()

genai.configure(api_key=GOOGLE_API_KEY)

# Settings
MODEL_NAME = "gemini-2.5-flash-preview-05-20"
TEMPERATURE = 0.3
RETRIES = 3
RETRY_DELAY = 5
MIN_CLAUSE_LENGTH = 30

# File parsers
def read_pdf(path):
    try:
        return "\n".join([p.get_text() for p in fitz.open(path)])
    except Exception as e:
        print(f"PDF error: {e}")
        return ""

def read_txt(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"TXT error: {e}")
        return ""

def read_sol(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            code = f.read()
        ast = parser.parse(code, loc=True)
        lines = code.splitlines(True)

        def get_offset(line, col):
            return sum(len(lines[i]) for i in range(line - 1)) + col

        functions = []

        def visit(node):
            if isinstance(node, dict):
                if node.get("type") == "FunctionDefinition" and node.get("loc"):
                    s, e = node["loc"]["start"], node["loc"]["end"]
                    start = get_offset(s["line"], s["column"])
                    end = get_offset(e["line"], e["column"]) + 1
                    snippet = code[start:end].strip()
                    if len(snippet) >= MIN_CLAUSE_LENGTH:
                        functions.append(snippet)
                for v in node.values():
                    if isinstance(v, (list, dict)):
                        visit(v)

            elif isinstance(node, list):
                for item in node:
                    visit(item)

        visit(ast)
        return functions

    except Exception as e:
        print(f"Solidity error: {e}")
        return []

def extract_clauses(text):
    return [c.strip() for c in re.split(r'\n\s*\n', text) if len(c.strip()) >= MIN_CLAUSE_LENGTH]

def get_clauses(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_clauses(read_pdf(file_path))
    elif ext == ".txt":
        return extract_clauses(read_txt(file_path))
    elif ext == ".sol":
        return read_sol(file_path)
    else:
        print(f"Unsupported format: {ext}")
        return []

# LLM interaction
def annotate_clause(clause):
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = f"""
Analyze the following contract clause or code block and provide the analysis in JSON format.

Clause/Code Block:
---
{clause}
---

Tasks:
1. Risk Level: High/Medium/Low.
2. Summary: Concise purpose and implications.
3. Key Terms/Operations: List important terms or actions.

JSON:
{{
  "risk_level": "...",
  "summary": "...",
  "key_terms_operations": ["...", "..."]
}}
"""
    for i in range(RETRIES):
        try:
            resp = model.generate_content(prompt, generation_config={"temperature": TEMPERATURE})
            text = resp.text.strip().strip("```json").strip("```").strip()
            return json.loads(text)
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
            if i < RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** i))
    return None

# Pipeline
def process_file(file_path, output_dir="preprocessed_jsonl"):
    clauses = get_clauses(file_path)
    if not clauses:
        print("No clauses found.")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    filename = f"prepared_{os.path.splitext(os.path.basename(file_path))[0]}.jsonl"
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, clause in enumerate(clauses, 1):
            print(f"Annotating clause {i}/{len(clauses)}...")
            result = annotate_clause(clause)
            if result:
                record = {
                    "original_clause_or_definition": clause,
                    "risk_level": result.get("risk_level", "N/A"),
                    "summary": result.get("summary", "N/A"),
                    "key_terms_operations": result.get("key_terms_operations", [])
                }
                f.write(json.dumps(record) + "\n")

    print(f"Saved results to {output_path}")

# Batch processing
def process_all(base_dir="raw_data/sample_contracts"):
    output_dir = "preprocessed_jsonl"
    for subdir in ["sol", "text"]:
        dir_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(dir_path):
            print(f"Missing dir: {dir_path}")
            continue
        for file in os.listdir(dir_path):
            if file.endswith((".sol", ".txt")):
                path = os.path.join(dir_path, file)
                process_file(path, output_dir=output_dir)


# Entry point
if __name__ == "__main__":
    process_all()
