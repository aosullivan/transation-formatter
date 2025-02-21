import math
import html
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, Optional
from statistics import mean
from time import time
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
import sys
import psutil
from llm_inference import LLMInference
import re

def calculate_optimal_workers() -> int:
    """Calculate optimal batch size based on available memory."""
    memory = psutil.virtual_memory()
    memory_gb = memory.available / (1024 * 1024 * 1024)
    return max(1, min(4, int(memory_gb / 4)))

@dataclass
class ProcessingConfig:
    """Configuration settings for text processing."""
    chunk_size: int = 1000
    batch_size: int = calculate_optimal_workers()
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    timeout: int = 300

class ChunkTimeTracker:
    def __init__(self, total_chunks: int, num_workers: int):
        self.total_chunks = total_chunks
        self.num_workers = max(1, min(num_workers, total_chunks))
        self.completed_chunks = 0
        self.processing_times = []
        self.start_time = time()
        
    def add_chunk_time(self, processing_time: float) -> Tuple[float, float, float]:
        self.processing_times.append(processing_time)
        self.completed_chunks += 1
        
        avg_time = mean(self.processing_times[-self.num_workers:] 
                       if len(self.processing_times) >= self.num_workers 
                       else self.processing_times)
        effective_rate = avg_time / min(self.num_workers, self.total_chunks)
        remaining_chunks = self.total_chunks - self.completed_chunks
        estimated_remaining = effective_rate * remaining_chunks
        completion_percentage = (self.completed_chunks / self.total_chunks) * 100
        
        return avg_time, estimated_remaining, completion_percentage
    
    def get_elapsed_time(self) -> float:
        return time() - self.start_time

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup logging with a single log file."""
    Path(log_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/formatter_{timestamp}.log"
    
    logger = logging.getLogger('formatter')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    return logger

def run_inference(model_name: str, chunk: str, chunk_num: int) -> str:
    """Run inference on a text chunk."""
    llm = LLMInference(model_name)
    return llm.run_inference(chunk)

class TextProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger('proofreader')

    def process_chunk(self, chunk: str, chunk_num: int, total_chunks: int) -> Tuple[int, str, float]:
        """Process a single chunk of text."""
        start_time = time()
        
        html_content = run_inference(
            self.config.model_name,
            chunk,
            chunk_num
        )
        
        if not html_content:
            raise ValueError(f"Empty result from chunk {chunk_num}")
        
        self._log_result(
            chunk_num, 
            total_chunks, 
            chunk, 
            html_content, 
            time() - start_time
        )
        
        return chunk_num - 1, html_content, time() - start_time

    def _build_prompt(self, chunk: str) -> str:
        return f"""Convert this text to HTML following these rules:
1. Lines appearing as main titles become <h1> tags
2. Lines appearing as subtitles become <h2> tags
3. Lines starting with > or clear quotations become <blockquote> tags
4. Lines starting with * or - become <li> tags in a <ul>
5. Sequential numbered lines (e.g., 1., 2.) become <li> tags in an <ol>; standalone numbers become headings
6. Indented text or text between --- becomes <div class="verse">
7. Other text becomes <p> tags
8. Preserve original text and blank lines

Text:
{chunk}

Return only the HTML output."""

    def _extract_html(self, generated_text: str, chunk: str, chunk_num: int) -> str:
        """Extract and validate HTML from model output."""
        try:
            # Extract content after the prompt
            html_content = generated_text.split("TEXT TO CONVERT:")[-1].strip()
            
            # Basic validation
            if not any(tag in html_content.lower() for tag in ['<p>', '<h1>', '<h2>', '<blockquote>', '<div', '<ul>', '<ol>']):
                self.logger.warning(f"No HTML tags in chunk {chunk_num}, using enhanced fallback")
                return self._enhanced_formatting(chunk)

            # Fix common issues
            html_content = self._fix_common_issues(html_content)
            
            return html_content
        except Exception as e:
            self.logger.warning(f"Failed to extract HTML from chunk {chunk_num}: {str(e)}")
            return self._enhanced_formatting(chunk)

    def _fix_common_issues(self, html: str) -> str:
        """Fix common formatting issues in generated HTML."""
        fixes = [
            (r'\n{2,}', '\n'),  # Multiple newlines to single
            (r'<p>\s*</p>', ''),  # Empty paragraphs
            (r'(?<!>)\n(?!<)', ' '),  # Newlines within tags to spaces
            (r'<(p|h1|h2|blockquote)>([\s]*)</', r'<\1>No content</'),  # Empty tags
            (r'\s+</li>', '</li>'),  # Extra spaces before list closings
            (r'<li>\s+', '<li>'),  # Extra spaces after list openings
        ]
        
        result = html
        for pattern, replacement in fixes:
            result = re.sub(pattern, replacement, result)
        return result

    def _enhanced_formatting(self, text: str) -> str:
        """Enhanced fallback formatting with better structure detection."""
        lines = text.split('\n')
        html_lines = []
        current_list = None
        in_verse = False
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_list:
                    html_lines.append(f"</{current_list}>")
                    current_list = None
                if in_verse:
                    html_lines.append("</div>")
                    in_verse = False
                continue
            
            # Title detection
            if len(line) < 80 and (line.isupper() or line.istitle()):
                if current_list:
                    html_lines.append(f"</{current_list}>")
                    current_list = None
                html_lines.append(f"<h1>{line}</h1>")
                continue
            
            # List detection
            if line.startswith(('* ', '- ')):
                if current_list != 'ul':
                    if current_list:
                        html_lines.append(f"</{current_list}>")
                    html_lines.append("<ul>")
                    current_list = 'ul'
                html_lines.append(f"<li>{line[2:].strip()}</li>")
                continue
            
            # Numbered list detection
            if re.match(r'^\d+\.\s', line):
                if current_list != 'ol':
                    if current_list:
                        html_lines.append(f"</{current_list}>")
                    html_lines.append("<ol>")
                    current_list = 'ol'
                html_lines.append(f"<li>{re.sub(r'^\d+\.\s', '', line)}</li>")
                continue
            
            # Quote detection
            if line.startswith('>'):
                if current_list:
                    html_lines.append(f"</{current_list}>")
                    current_list = None
                html_lines.append(f"<blockquote>{line[1:].strip()}</blockquote>")
                continue
            
            # Verse detection
            if line.startswith(('    ', '\t')) or '---' in line:
                if not in_verse:
                    html_lines.append('<div class="verse">')
                    in_verse = True
                html_lines.append(line.strip())
                continue
            
            # Regular paragraph
            if current_list:
                html_lines.append(f"</{current_list}>")
                current_list = None
            if in_verse:
                html_lines.append("</div>")
                in_verse = False
            html_lines.append(f"<p>{line}</p>")
        
        # Clean up any open tags
        if current_list:
            html_lines.append(f"</{current_list}>")
        if in_verse:
            html_lines.append("</div>")
            
        return '\n'.join(html_lines)

    def _log_result(self, chunk_num: int, total_chunks: int, chunk: str, html_content: str, processing_time: float):
        """Log processing results."""
        self.logger.debug(
            f"Chunk {chunk_num}/{total_chunks}\n"
            f"Input length: {len(chunk)}, Output length: {len(html_content)}\n"
            f"Time: {processing_time:.2f}s\n"
            f"Output:\n{html_content}\n"
            f"{'-'*40}"
        )

    def _handle_error(self, chunk: str, chunk_num: int, total_chunks: int, error: Exception, start_time: float) -> Tuple[int, str, float]:
        processing_time = time() - start_time
        fallback_content = f'<p>{html.escape(chunk)}</p>'
        self.logger.error(f"Chunk {chunk_num}/{total_chunks} failed: {str(error)}")
        return chunk_num - 1, fallback_content, processing_time

def format_time(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))

def text_to_html(input_file: str, output_file: str, config: Optional[ProcessingConfig] = None) -> None:
    """Convert text file to HTML."""
    config = config or ProcessingConfig()
    logger = setup_logging()
    processor = TextProcessor(config)
    
    logger.info(f"Converting {input_file} to HTML")
    
    css_styles = """
    <style>
        body { font-family: 'Georgia', serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; color: #333; }
        h1 { font-size: 2.5em; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-bottom: 20px; }
        h2 { font-size: 1.8em; color: #34495e; margin: 30px 0 15px; }
        p { margin: 15px 0; text-align: justify; }
        ul, ol { margin: 15px 0 15px 40px; padding-left: 0; }
        li { margin: 10px 0; }
        blockquote { margin: 20px 0; padding: 15px 20px; background: #f8f9fa; border-left: 5px solid #3498db; font-style: italic; color: #555; }
        .verse { margin: 20px 0; padding: 15px; background: #f0f4f8; border-radius: 5px; font-family: 'Courier New', monospace; white-space: pre-wrap; }
    </style>
    """
    
    text = Path(input_file).read_text(encoding='utf-8')
    chunks = split_into_chunks(text, config.chunk_size)
    
    if not chunks:
        raise ValueError(f"No valid text chunks found in {input_file}")
    
    html_chunks = [''] * len(chunks)
    time_tracker = ChunkTimeTracker(len(chunks), 1)  # Single worker
    
    # Process chunks sequentially
    for i, chunk in enumerate(chunks):
        chunk_idx, html_content, processing_time = processor.process_chunk(chunk, i+1, len(chunks))
        html_chunks[chunk_idx] = html_content
        
        avg_time, remaining_time, completion_percentage = time_tracker.add_chunk_time(processing_time)
        logger.info(f"Chunk {chunk_idx + 1}/{len(chunks)} ({completion_percentage:.1f}%) - "
                   f"Avg: {format_time(avg_time)}, Remaining: {format_time(remaining_time)}")
    
    if any(not chunk for chunk in html_chunks):
        raise ValueError("Some chunks failed to process")
    
    html_content = fix_chunk_boundaries(html_chunks)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Formatted Document</title>
    {css_styles}
</head>
<body>
    {html_content}
</body>
</html>"""
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    logger.info(f"Generated {output_file} in {format_time(time_tracker.get_elapsed_time())}")

def split_into_chunks(text: str, chunk_size: int) -> list[str]:
    """Split text into chunks while preserving sentences."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        if end < len(text):
            sentence_end = end
            while sentence_end > start:
                if sentence_end < len(text) and text[sentence_end-1] in '.!?' and (
                    text[sentence_end] in ' \n\t' or text[sentence_end-2:sentence_end] == '..."'
                ):
                    break
                sentence_end -= 1
            
            if sentence_end <= start:
                sentence_end = end
                while sentence_end > start and text[sentence_end] not in ' \n\t':
                    sentence_end -= 1
            
            end = sentence_end
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end

    return chunks

def fix_chunk_boundaries(chunks: list[str]) -> str:
    """Fix HTML tags across chunk boundaries."""
    combined = '\n'.join(chunks)
    combined = combined.replace('</p>\n<p>', '\n')
    combined = combined.replace('</ul>\n<ul>', '\n')
    combined = combined.replace('</ol>\n<ol>', '\n')
    combined = combined.replace('</blockquote>\n<blockquote>', '\n')
    combined = combined.replace('</div>\n<div class="verse">', '\n')
    return combined

if __name__ == "__main__":
    logger = setup_logging()
    print("Processing log: " + str(Path(logger.handlers[0].baseFilename)))
    
    config = ProcessingConfig()
    text_to_html("texts/text.txt", "output.html", config)