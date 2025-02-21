import torch
import os
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

HTML_CONVERSION_PROMPT = """Convert the following text from a Gutenberg.org text file into valid, well-formed HTML. Follow these formatting rules strictly to produce clean, semantic HTML output. Use the text’s structure (e.g., blank lines, indentation, prefixes) to determine the correct tags.

FORMATTING RULES:
1. Main titles (e.g., chapter titles, section starts, all-caps or centered text) -> <h1>Title</h1>
2. Subtitles or secondary headings (e.g., title case text followed by a blank line) -> <h2>Subtitle</h2>
3. Quotations (lines starting with '>') -> <blockquote>Quote</blockquote>. Use <br> for internal line breaks.
4. Unordered lists (lines starting with '*' or '-') -> Group into <ul><li>Item</li></ul>. Each line is a separate <li>.
5. Ordered lists (lines starting with '1.', '2.', etc.) -> Group into <ol><li>Item</li></ol>. Each line is a separate <li>. Let HTML handle numbering.
6. Indented text or text between '---' lines -> <div class="verse">Text</div>. Use <br> for internal line breaks.
7. All other paragraphs (blocks of text separated by blank lines) -> <p>Text</p>. Keep paragraphs intact unless separated by a blank line.
8. Preserve meaningful line breaks and spacing:
   - Use <br> for intentional line breaks within blockquotes, verses, or paragraphs.
   - Avoid extra whitespace or tags between elements.
9. Every line of text must be wrapped in an HTML tag based on its context. No untagged text.
10. Ensure valid HTML:
    - Nest and close all tags correctly.
    - Wrap the full output in <html><body>...</body></html>.

EXAMPLE:
INPUT:
CHAPTER I
Introduction
> This is a quote
> spanning two lines
1. First item
2. Second item
  Indented text line 1
  Indented text line 2
Regular paragraph text

OUTPUT:
<html><body>
<h1>CHAPTER I</h1>
<h2>Introduction</h2>
<blockquote>This is a quote<br>spanning two lines</blockquote>
<ol><li>First item</li><li>Second item</li></ol>
<div class="verse">Indented text line 1<br>Indented text line 2</div>
<p>Regular paragraph text</p>
</body></html>

TEXT TO CONVERT:
{text}

INSTRUCTIONS:
- Analyze the input text’s structure to apply the correct tags (e.g., blank lines separate paragraphs, indentation signals verses).
- Group related lines into single HTML elements (e.g., consecutive list items in one <ul> or <ol>).
- Return only valid HTML. Do not include explanations, comments, or text outside the HTML structure.
- If a line’s purpose is unclear, wrap it in <p> tags as a default."""

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
        
        self.token = token or os.getenv('HF_TOKEN')
        if not self.token:
            raise ValueError("HuggingFace token not found")
        
        login(self.token)
        
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
        
        html_content = result.split("TEXT TO CONVERT:")[-1].strip()
        processing_time = time() - start_time
        self.logger.info(f"Inference completed in {processing_time:.2f}s")
        
        # Extract HTML content
        html_content = result.split("TEXT TO CONVERT:")[-1].strip()
        
        if not html_content:
            raise ValueError("Model returned empty response")
            
        if not any(tag in html_content.lower() for tag in ['<p>', '<h1>', '<h2>', '<blockquote>', '<div', '<ul>', '<ol>']):
            raise ValueError("Model response contains no valid HTML tags")
        
        self.cleanup()
        return html_content
