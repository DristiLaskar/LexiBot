# unified_main.py
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import torch
import logging
from typing import Optional, Dict, Any
import sys
from pathlib import Path  
import json
import re 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(
    title="Unified Legal Services API",
    description="Unified API for Legal Assistant and Smart Contract Analyzer services",
    version="2.0.0",
)

# --- CORS Setup ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add service directories to Python path
current_dir = Path(__file__).parent
legal_assistant_path = current_dir / "legal_assistant"
contract_analyzer_path = current_dir / "contract_analyzer"

sys.path.append(str(legal_assistant_path))
sys.path.append(str(contract_analyzer_path))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Application Setup ---
load_dotenv()

# --- Pydantic Models for Request/Response ---

# Legal Assistant Models
class LegalQueryRequest(BaseModel):
    country: str
    query: str

class LegalQueryResponse(BaseModel):
    response: str

# Contract Analyzer Models
class ContractAnalysisRequest(BaseModel):
    code: str

class ContractAnalysisResponse(BaseModel):
    risk_level: str
    summary: str
    key_terms_operations: list

# Health Check Response
class HealthResponse(BaseModel):
    status: str
    services: dict
    torch_device: str

# --- Global State Management ---
state = {
    "legal_assistant": {},
    "contract_analyzer": {}
}

# --- Service Import and Initialization Functions ---

def initialize_legal_assistant():
    """Initialize the legal assistant service"""
    try:
        logger.info("Initializing Legal Assistant service...")
        
        # Import legal assistant modules
        from legal_assistant.models.gemma_llm import GemmaLLM
        from legal_assistant.agents.country_agents import CountryLegalAgent
        from legal_assistant.vector_store.faiss_store import FAISSVectorStore
        
        # Load the Language Model (LLM)
        logger.info("Loading Gemma model for Legal Assistant...")
        llm = GemmaLLM()
        state["legal_assistant"]["llm"] = llm
        logger.info("✅ Legal Assistant Gemma model loaded successfully.")
        
        # Initialize Vector Stores and Agents for each country
        countries = ["india", "usa", "germany"]
        agents = {}
        
        for country in countries:
            logger.info(f"Initializing legal resources for {country.upper()}...")
            try:
                vector_store = FAISSVectorStore(country=country)
                agent = CountryLegalAgent(
                    country=country,
                    vector_store=vector_store,
                    llm=llm
                )
                agents[country] = agent
                logger.info(f"✅ Successfully initialized legal agent for {country.upper()}.")
            except Exception as e:
                logger.error(f"❌ Failed to initialize legal agent for {country.upper()}: {e}")
        
        state["legal_assistant"]["agents"] = agents
        state["legal_assistant"]["status"] = "ready"
        logger.info("✅ Legal Assistant service initialized successfully.")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Legal Assistant service: {e}")
        state["legal_assistant"]["status"] = "failed"
        state["legal_assistant"]["error"] = str(e)

def initialize_contract_analyzer():
    """Initialize the contract analyzer service"""
    try:
        logger.info("Initializing Contract Analyzer service...")
        
        # Set environment variables to disable Torch optimizations
        os.environ["TORCHINDUCTOR_DISABLE"] = "1"
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
        
        # Import contract analyzer modules
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        # Configuration - Fix the model path
        MODEL_NAME = "gemma3-1b-contract-analyzer"
        BASE_MODEL_ID = "google/gemma-3-1b-it"
        
        # Construct the full path to the model directory
        model_path = contract_analyzer_path / MODEL_NAME
        logger.info(f"Looking for model at: {model_path}")
        
        # Check if model exists in the contract_analyzer directory
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory '{MODEL_NAME}' not found at {model_path}")
        
        # Load model and tokenizer
        logger.info(f"Loading base model: {BASE_MODEL_ID}...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
        
        # Create offload directory for memory management
        offload_dir = contract_analyzer_path / "model_offload"
        offload_dir.mkdir(exist_ok=True)
        logger.info(f"Using offload directory: {offload_dir}")
        
        # Determine device strategy based on available memory
        device_strategy = "auto"
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "offload_folder": str(offload_dir),
        }
        
        # Check if CUDA is available and adjust strategy
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            logger.info(f"Available GPU memory: {gpu_memory:.2f} GB")
            
            if gpu_memory < 8:  # Less than 8GB VRAM
                logger.info("Limited GPU memory detected, using CPU fallback")
                load_kwargs["device_map"] = {"": "cpu"}
                load_kwargs["torch_dtype"] = torch.float32  # Use float32 for CPU
            else:
                load_kwargs["device_map"] = "auto"
        else:
            logger.info("CUDA not available, using CPU")
            load_kwargs["device_map"] = {"": "cpu"}
            load_kwargs["torch_dtype"] = torch.float32
        
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            **load_kwargs
        )
        
        logger.info(f"Loading PEFT adapter from: {model_path}...")
        # Use same device strategy for PEFT model
        peft_kwargs = {
            "torch_dtype": load_kwargs["torch_dtype"],
        }
        if "device_map" in load_kwargs:
            peft_kwargs["device_map"] = load_kwargs["device_map"]
            
        model = PeftModel.from_pretrained(
            base_model,
            str(model_path),
            **peft_kwargs
        )
        model.eval()
        
        state["contract_analyzer"]["model"] = model
        state["contract_analyzer"]["tokenizer"] = tokenizer
        state["contract_analyzer"]["status"] = "ready"
        logger.info("✅ Contract Analyzer service initialized successfully.")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Contract Analyzer service: {e}")
        state["contract_analyzer"]["status"] = "failed"
        state["contract_analyzer"]["error"] = str(e)

# --- Application Startup Event ---
@app.on_event("startup")
def load_resources():
    """Load all necessary resources on application startup."""
    logger.info("🚀 Unified application starting up...")
    
    # Initialize both services
    initialize_legal_assistant()
    initialize_contract_analyzer()
    
    # Log final status
    device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    logger.info(f"Running on device: {device}")
    
    legal_status = state["legal_assistant"].get("status", "unknown")
    contract_status = state["contract_analyzer"].get("status", "unknown")
    
    logger.info(f"Legal Assistant status: {legal_status}")
    logger.info(f"Contract Analyzer status: {contract_status}")
    logger.info("🎉 Unified application startup complete.")

# --- Legal Assistant Endpoints ---

@app.post("/query", response_model=LegalQueryResponse, tags=["Legal Assistant"])
async def get_legal_response(request: LegalQueryRequest):
    """
    Receives a legal query for a specific country and returns a response.
    """
    logger.info(f"Received legal query for country: '{request.country}'")
    
    # Check if legal assistant is ready
    if state["legal_assistant"].get("status") != "ready":
        error_msg = state["legal_assistant"].get("error", "Legal Assistant service not available")
        raise HTTPException(status_code=503, detail=f"Legal Assistant service unavailable: {error_msg}")
    
    # Input validation
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    agent = state["legal_assistant"]["agents"].get(request.country.lower())

    if not agent:
        logger.warning(f"No legal agent found for country: {request.country}")
        raise HTTPException(
            status_code=404, 
            detail=f"Legal assistant for country '{request.country}' not found or failed to initialize."
        )

    try:
        logger.info("Generating legal response...")
        response_text = agent.get_response(query=request.query)
        logger.info("Legal response generated successfully.")
        return LegalQueryResponse(response=response_text)
    except Exception as e:
        logger.error(f"Error processing legal query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"An internal error occurred while processing legal query: {e}"
        )

# --- Contract Analyzer Endpoints ---

@app.post("/analyze", response_model=ContractAnalysisResponse, tags=["Contract Analyzer"])
async def analyze_contract(request: ContractAnalysisRequest):
    """
    Analyzes Solidity smart contract code and returns security analysis.
    """
    logger.info("Received contract analysis request")
    
    # Check if contract analyzer is ready
    if state["contract_analyzer"].get("status") != "ready":
        error_msg = state["contract_analyzer"].get("error", "Contract Analyzer service not available")
        raise HTTPException(status_code=503, detail=f"Contract Analyzer service unavailable: {error_msg}")
    
    # Input validation
    if not request.code:
        raise HTTPException(status_code=400, detail="Solidity code cannot be empty.")

    try:
        model = state["contract_analyzer"]["model"]
        tokenizer = state["contract_analyzer"]["tokenizer"]
        
        # Create prompt
        prompt = create_contract_analysis_prompt(request.code)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        logger.info("Analyzing smart contract...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[-1]
        gen_ids = outputs[0][input_len:]
        analysis = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        logger.info("Contract analysis completed successfully.")
        
        # Parse the analysis text into structured JSON
        parsed_analysis = parse_contract_analysis(analysis)
        
        return ContractAnalysisResponse(**parsed_analysis)
        
    except Exception as e:
        logger.error(f"Error analyzing contract: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"An internal error occurred while analyzing contract: {e}"
        )

# --- Utility Functions ---

def create_contract_analysis_prompt(solidity_code: str) -> str:
    """Create a prompt for contract analysis"""
    return (
        "<start_of_turn>user\n"
        "You are a smart contract security analyzer. Analyze the following\n"
        "Solidity code and provide analysis in JSON format with risk_level,\n"
        "summary, and key_terms_operations.\n\n"
        "Analyze this Solidity code:\n"
        "```solidity\n"
        f"{solidity_code}\n"
        "```<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

def parse_contract_analysis(analysis_text: str) -> Dict[str, Any]:
    """
    Parse the contract analysis text and extract structured data.
    """
    logger.info(f"Raw analysis text: {analysis_text}")
    
    # Default values
    default_response = {
        "risk_level": "UNKNOWN",
        "summary": "Analysis could not be parsed properly",
        "key_terms_operations": []
    }
    
    try:
        # First, try to find JSON within the text
        json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                parsed_json = json.loads(json_str)
                
                # Extract and validate required fields
                result = {
                    "risk_level": str(parsed_json.get("risk_level", "UNKNOWN")).upper(),
                    "summary": str(parsed_json.get("summary", "No summary provided")),
                    "key_terms_operations": parsed_json.get("key_terms_operations", [])
                }
                
                # Ensure key_terms_operations is a list
                if not isinstance(result["key_terms_operations"], list):
                    result["key_terms_operations"] = []
                
                logger.info("Successfully parsed JSON from analysis")
                return result
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON: {e}")
        
        # If JSON parsing fails, try to extract information using regex patterns
        logger.info("Attempting to extract information using regex patterns")
        
        # Extract risk level
        risk_match = re.search(r'risk[_\s]*level["\s]*:[\s]*["\']?([^"\'}\n,]+)', analysis_text, re.IGNORECASE)
        risk_level = risk_match.group(1).strip().upper() if risk_match else "UNKNOWN"
        
        # Extract summary
        summary_match = re.search(r'summary["\s]*:[\s]*["\']([^"\'}\n]+)', analysis_text, re.IGNORECASE | re.DOTALL)
        summary = summary_match.group(1).strip() if summary_match else "No summary available"
        
        # Extract key terms/operations (look for arrays or lists)
        key_terms = []
        key_terms_match = re.search(r'key[_\s]*terms[_\s]*operations["\s]*:[\s]*\[([^\]]+)\]', analysis_text, re.IGNORECASE)
        if key_terms_match:
            terms_str = key_terms_match.group(1)
            # Split by comma and clean up
            key_terms = [term.strip().strip('"\'') for term in terms_str.split(',') if term.strip()]
        
        result = {
            "risk_level": risk_level,
            "summary": summary,
            "key_terms_operations": key_terms
        }
        
        logger.info("Successfully extracted information using regex")
        return result
        
    except Exception as e:
        logger.error(f"Error parsing contract analysis: {e}")
        return default_response

# --- Health Check Endpoint ---

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Comprehensive health check for both services."""
    return HealthResponse(
        status="ok",
        services={
            "legal_assistant": state["legal_assistant"].get("status", "unknown"),
            "contract_analyzer": state["contract_analyzer"].get("status", "unknown")
        },
        torch_device="cuda" if torch.cuda.is_available() else "cpu"
    )

# --- Service-specific Health Checks ---

@app.get("/health/legal", tags=["Health"])
def legal_health_check():
    """Health check specifically for Legal Assistant service."""
    status = state["legal_assistant"].get("status", "unknown")
    response = {"service": "legal_assistant", "status": status}
    
    if status == "failed":
        response["error"] = state["legal_assistant"].get("error", "Unknown error")
        
    return response

@app.get("/health/contract", tags=["Health"])
def contract_health_check():
    """Health check specifically for Contract Analyzer service."""
    status = state["contract_analyzer"].get("status", "unknown")
    response = {"service": "contract_analyzer", "status": status}
    
    if status == "failed":
        response["error"] = state["contract_analyzer"].get("error", "Unknown error")
        
    return response

# --- Documentation Endpoints ---

@app.get("/", tags=["Documentation"])
def root():
    """Root endpoint with service information."""
    return {
        "message": "Unified Legal Services API",
        "version": "2.0.0",
        "services": {
            "legal_assistant": {
                "endpoint": "/legal/query",
                "description": "Multi-country legal assistant powered by Gemma and FAISS",
                "supported_countries": ["india", "usa", "germany"]
            },
            "contract_analyzer": {
                "endpoint": "/contract/analyze",
                "description": "Smart contract security analyzer for Solidity code"
            }
        },
        "health_check": "/health",
        "documentation": "/docs"
    }

# --- Main Execution ---
if __name__ == "__main__":
    # To run the server, use the command:
    # uvicorn unified_main:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)