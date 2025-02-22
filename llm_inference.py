import torch
import os
import re  # Add missing import
import logging
import shutil
from time import time
from pathlib import Path
from typing import Optional, Dict, Tuple
from dotenv import load_dotenv
from huggingface_hub import login, scan_cache_dir
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Add a model cache at module level
_MODEL_CACHE: Dict[str, tuple] = {}  # {model_name: (tokenizer, model, pipe)}

# Load environment variables at module level
def get_project_root() -> Path:
    return Path(__file__).parent.resolve()

# Load .env file if 'HF_TOKEN' not in environment
if 'HF_TOKEN' not in os.environ:
    env_path = get_project_root() / '.env'
    if (env_path.exists()):
        load_dotenv(str(env_path))

HTML_CONVERSION_PROMPT = """Convert the following text into clean, semantic Markdown format. Follow these formatting rules strictly:

FORMATTING RULES:
1. Main titles (chapter titles, section starts) -> # Title
2. Subtitles or secondary headings -> ## Subtitle
3. Quotations -> Use > for quotes
4. Unordered lists -> Use * or -
5. Ordered lists (multiple lines with numbers forming a sequence) -> 1. First item (keep original numbers)
6. Indented text or text between --- -> Indent with 4 spaces or use ``` for code blocks
7. Regular paragraphs -> Plain text, separated by blank lines
8. Preserve meaningful line breaks using double spaces at line ends
9. Keep original text structure (blank lines, indentation)
10. Special formatting:
    - *italic* for emphasis
    - **bold** for strong emphasis
    - `code` for inline code
    - --- for horizontal rules

EXAMPLE:
INPUT:
CHAPTER I
Introduction
  This is a quote
  spanning lines
1. First item
2. Second item
  Indented text line 1
  Indented text line 2
Regular text

OUTPUT:
# CHAPTER I

## Introduction

> This is a quote
> spanning lines

1. First item
2. Second item

    Indented text line 1
    Indented text line 2

Regular text

TEXT TO CONVERT:
{text}

INSTRUCTIONS:
- Analyze text structure to apply appropriate Markdown syntax
- Preserve original text formatting where Markdown allows
- Return clean Markdown without explanations or extra formatting
- Use standard Markdown symbols (# for headings, > for quotes, etc.)"""

def get_hf_cache_dir() -> Path:
    """Get the Hugging Face cache directory."""
    return Path.home() / '.cache' / 'huggingface'

class ModelManager:
    """Singleton class to manage model loading and caching."""
    _instances: Dict[str, Tuple[AutoTokenizer, AutoModelForCausalLM, pipeline]] = {}
    _logger = logging.getLogger('llm_inference')

    @classmethod
    def get_model(cls, model_name: str, token: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM, pipeline]:
        if model_name not in cls._instances:
            cls._logger.info(f"Loading model {model_name} for the first time")
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=token,
                torch_dtype=torch.float16
            )

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                truncation=True,
                padding=True,
                pad_token_id=tokenizer.pad_token_id
            )
            cls._instances[model_name] = (tokenizer, model, pipe)
            cls._logger.info(f"Model {model_name} loaded and cached in memory")

        return cls._instances[model_name]

class LLMInference:
    def __init__(self, model_name: str, token: Optional[str] = None):
        self.logger = logging.getLogger('llm_inference')
        self.model_name = model_name
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.logger.info(f"Initializing LLM with model: {model_name} on device: {self.device}")
        
        # Simplified token handling - environment variable takes precedence
        self.token = os.getenv('HF_TOKEN') or token
        if not self.token:
            raise ValueError("HuggingFace token not found")
        
        # Removed login() call since we'll use token directly in from_pretrained
        
        # Get model from singleton manager
        self.tokenizer, self.model, self.pipe = ModelManager.get_model(model_name, self.token)

    def cleanup(self):
        """Cleanup GPU memory but keep model loaded."""
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None

    def run_inference(self, text: str) -> str:
        if not text or not text.strip():
            raise ValueError("Empty text provided for inference")
            
        self.logger.info(f"Running inference on text of length: {len(text)}")
        start_time = time()
        
        prompt = HTML_CONVERSION_PROMPT.format(text=text)
        result = self.pipe(
            prompt,
            max_length=len(text) + 1500,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.2
        )[0]['generated_text']
        
        markdown_content = result.split("TEXT TO CONVERT:")[-1].strip()
        processing_time = time() - start_time
        self.logger.info(f"Inference completed in {processing_time:.2f}s")
        
        if not markdown_content:
            raise ValueError("Model returned empty response")
        
        # Simplified pattern check for basic Markdown
        is_markdown = bool(re.search(r'[#>*\-\d\.]|    ', markdown_content))
        if not is_markdown:
            self.logger.warning("Response might not be proper Markdown, but proceeding")
        
        self.cleanup()
        return markdown_content
