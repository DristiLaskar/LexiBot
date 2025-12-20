from langchain_core.language_models import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Optional, List, Any
import logging

class GemmaLLM(LLM):
    """Custom LLM wrapper for Gemma model, configured for concise answers."""

    model_name: str = "google/gemma-3-1b-it"
    tokenizer: Any = None
    model: Any = None
    max_length: int = 2048
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_model()

    def _load_model(self):
        """Load Gemma model and tokenizer"""
        try:
            print("Loading Gemma 3 1B model...")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with appropriate settings
            device = "cuda" if torch.cuda.is_available() else "cpu"

            if device == "cuda":
                # Use 8-bit quantization for GPU to save memory
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    load_in_8bit=True,
                    trust_remote_code=True
                )
            else:
                # CPU inference (fallback to CPU in case GPU fails)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                self.model.to(device)

            print(f"Gemma model loaded successfully on {device}")

        except Exception as e:
            logging.error(f"Error loading Gemma model: {e}")
            raise e

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Generate response from Gemma model"""
        try:
            instruction = "Provide a concise and direct answer to the following prompt."
            formatted_prompt = f"<bos><start_of_turn>user\n{instruction}\n\nPROMPT: {prompt}<end_of_turn>\n<start_of_turn>model\n"

            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length - 256  
            )

            
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=self.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            # Clean up response
            response = response.strip()

            # Apply stop sequences if provided
            if stop:
                for stop_seq in stop:
                    if stop_seq in response:
                        response = response.split(stop_seq)[0]

            return response

        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"

    @property
    def _llm_type(self) -> str:
        return "gemma"

    @property
    def _identifying_params(self) -> dict:
        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p
        }
