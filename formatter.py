import re
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from statistics import mean
from time import time
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
import sys
import psutil
from llm_inference import LLMInference
import re
from collections import deque
import argparse
import json

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
    output_format: str = "md"  # Changed from html to md

@dataclass
class ProcessingState:
    """State information for processing job."""
    input_file: str
    output_file: str
    total_chunks: int
    completed_chunks: Dict[int, str]  # chunk_idx -> content
    start_time: float
    config: Dict[str, Any]
    
    @classmethod
    def load(cls, state_file: Path) -> 'ProcessingState':
        data = json.loads(state_file.read_text())
        return cls(**data)
    
    def save(self, state_file: Path):
        data = {
            'input_file': self.input_file,
            'output_file': self.output_file,
            'total_chunks': self.total_chunks,
            'completed_chunks': self.completed_chunks,
            'start_time': self.start_time,
            'config': self.config
        }
        state_file.write_text(json.dumps(data))

class ChunkTimeTracker:
    def __init__(self, total_chunks: int, num_workers: int, history_size: int = 20):
        self.total_chunks = total_chunks
        self.num_workers = max(1, min(num_workers, total_chunks))
        self.completed_chunks = 0
        self.processing_times = deque(maxlen=history_size)
        self.start_time = time()
        self.last_estimate = None
        
    def add_chunk_time(self, processing_time: float) -> Tuple[float, float, float]:
        """Add a new processing time and calculate estimates."""
        self.processing_times.append(processing_time)
        self.completed_chunks += 1
        
        # Calculate weighted moving average (recent times weighted more heavily)
        weights = [1 + (i/10) for i in range(len(self.processing_times))]
        weighted_times = [t * w for t, w in zip(self.processing_times, weights)]
        avg_time = sum(weighted_times) / sum(weights)
        
        # Calculate trend (are times increasing or decreasing?)
        if len(self.processing_times) >= 2:
            recent_avg = mean(list(self.processing_times)[-3:])
            overall_avg = mean(self.processing_times)
            trend_factor = recent_avg / overall_avg
            avg_time *= trend_factor
        
        # Calculate remaining time
        remaining_chunks = self.total_chunks - self.completed_chunks
        estimated_remaining = avg_time * remaining_chunks
        
        # Smooth the estimate if we have a previous one
        if self.last_estimate is not None:
            estimated_remaining = (estimated_remaining + self.last_estimate) / 2
        self.last_estimate = estimated_remaining
        
        completion_percentage = (self.completed_chunks / self.total_chunks) * 100
        
        return avg_time, estimated_remaining, completion_percentage
    
    def get_elapsed_time(self) -> float:
        return time() - self.start_time
    
    def get_completion_eta(self) -> Optional[datetime]:
        """Get estimated completion time."""
        if self.last_estimate:
            return datetime.now() + timedelta(seconds=round(self.last_estimate))
        return None

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
        self.logger = logging.getLogger('formatter')
        self.state_file = Path(f"state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        self.state = None

    def _save_state(self, markdown_chunks: list[str], total_chunks: int, input_file: str, output_file: str):
        """Save current processing state."""
        completed = {i: chunk for i, chunk in enumerate(markdown_chunks) if chunk}
        state = ProcessingState(
            input_file=str(input_file),
            output_file=str(output_file),
            total_chunks=total_chunks,
            completed_chunks=completed,
            start_time=time(),
            config=self.config.__dict__
        )
        state.save(self.state_file)
        self.state = state

    def process_chunk(self, chunk: str, chunk_num: int, total_chunks: int) -> Tuple[int, str, float]:
        """Process a single chunk of text."""
        start_time = time()
        
        markdown_content = run_inference(
            self.config.model_name,
            chunk,
            chunk_num
        )
        
        if not markdown_content:
            raise ValueError(f"Empty result from chunk {chunk_num}")
        
        if not self._validate_markdown(markdown_content):
            markdown_content = self._fallback_formatting(chunk)
        
        self._log_result(
            chunk_num, 
            total_chunks, 
            chunk, 
            markdown_content, 
            time() - start_time
        )
        
        return chunk_num - 1, markdown_content, time() - start_time

    def _fallback_formatting(self, text: str) -> str:
        """Convert text to Markdown as fallback."""
        lines = text.split('\n')
        md_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                md_lines.append('')
                continue
            
            # Title detection
            if len(line) < 80 and (line.isupper() or line.istitle()):
                md_lines.append(f"# {line}")
                continue
            
            # Quote detection
            if line.startswith('>'):
                md_lines.append(line)  # Already markdown format
                continue
            
            # List detection
            if line.startswith(('* ', '- ')):
                md_lines.append(line)  # Already markdown format
                continue
            
            # Numbered list detection
            if re.match(r'^\d+\.\s', line):
                md_lines.append(line)  # Already markdown format
                continue
            
            # Indented text
            if line.startswith(('    ', '\t')):
                md_lines.append(f"    {line.lstrip()}")
                continue
            
            # Regular paragraph
            md_lines.append(line)
            
        return '\n\n'.join(md_lines)

    def _validate_markdown(self, text: str) -> bool:
        """Validate that the text contains Markdown formatting."""
        markdown_patterns = [
            r'^#+ ',          # Headers
            r'^\* ',          # Unordered lists
            r'^\d+\. ',       # Ordered lists
            r'^> ',           # Blockquotes
            r'    ',          # Indented code/verse
            r'\*\*.*\*\*',    # Bold
            r'\*[^\*]+\*'     # Italic
        ]
        return any(re.search(pattern, text, re.MULTILINE) for pattern in markdown_patterns)

    def _log_result(self, chunk_num: int, total_chunks: int, chunk: str, markdown_content: str, processing_time: float):
        """Log processing results."""
        self.logger.debug(
            f"Chunk {chunk_num}/{total_chunks}\n"
            f"Input length: {len(chunk)}, Output length: {len(markdown_content)}\n"
            f"Time: {processing_time:.2f}s\n"
            f"Output:\n{markdown_content}\n"
            f"{'-'*40}"
        )

def format_time(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))  # Fixed syntax: seconds=round(seconds)

def text_to_markdown(input_file: str, output_file: str, config: Optional[ProcessingConfig] = None, 
                    resume_file: Optional[str] = None, force_restart: bool = False) -> None:
    """Convert text file to Markdown with resume capability."""
    config = config or ProcessingConfig()
    logger = setup_logging()
    processor = TextProcessor(config)
    
    # Handle resume/restart
    if resume_file and not force_restart:
        try:
            state_path = Path(resume_file)
            if state_path.exists():
                state = ProcessingState.load(state_path)
                logger.info(f"Resuming from state file: {resume_file}")
                markdown_chunks = [''] * state.total_chunks
                for idx, content in state.completed_chunks.items():
                    markdown_chunks[idx] = content
                chunks = split_into_chunks(Path(state.input_file).read_text(encoding='utf-8'), 
                                        config.chunk_size)
                start_idx = len(state.completed_chunks)
                processor.state = state
                processor.state_file = state_path
            else:
                logger.warning(f"State file not found: {resume_file}")
                chunks = split_into_chunks(Path(input_file).read_text(encoding='utf-8'), 
                                        config.chunk_size)
                markdown_chunks = [''] * len(chunks)
                start_idx = 0
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            chunks = split_into_chunks(Path(input_file).read_text(encoding='utf-8'), 
                                    config.chunk_size)
            markdown_chunks = [''] * len(chunks)
            start_idx = 0
    else:
        chunks = split_into_chunks(Path(input_file).read_text(encoding='utf-8'), 
                                config.chunk_size)
        markdown_chunks = [''] * len(chunks)
        start_idx = 0
    
    if not chunks:
        raise ValueError(f"No valid text chunks found in {input_file}")
    
    time_tracker = ChunkTimeTracker(len(chunks), 1)
    
    # Save initial state
    processor._save_state(markdown_chunks, len(chunks), input_file, output_file)
    
    # Process chunks sequentially
    for i in range(start_idx, len(chunks)):
        try:
            chunk_idx, content, processing_time = processor.process_chunk(chunks[i], i+1, len(chunks))
            markdown_chunks[chunk_idx] = content
            
            # Save state after each chunk
            processor._save_state(markdown_chunks, len(chunks), input_file, output_file)
            
            avg_time, remaining_time, completion_percentage = time_tracker.add_chunk_time(processing_time)
            eta = time_tracker.get_completion_eta()
            eta_str = f", ETA: {eta.strftime('%H:%M:%S')}" if eta else ""
            logger.info(
                f"Chunk {chunk_idx + 1}/{len(chunks)} ({completion_percentage:.1f}%) - "
                f"Avg: {format_time(avg_time)}, Remaining: {format_time(remaining_time)}{eta_str}"
            )
            
        except KeyboardInterrupt:
            logger.info("\nProcessing interrupted. State saved - use --resume to continue later.")
            return
        except Exception as e:
            logger.error(f"Error processing chunk {i+1}: {e}")
            logger.info("State saved - use --resume to continue later.")
            raise
    
    if any(not chunk for chunk in markdown_chunks):
        raise ValueError("Some chunks failed to process")
    
    # Join chunks with double newlines for clear section separation
    markdown_content = '\n\n'.join(markdown_chunks)
    
    # Ensure output file has .md extension
    if not output_file.endswith('.md'):
        output_file = Path(output_file).with_suffix('.md')
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
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
    """Fix Markdown chunk boundaries."""
    # Join chunks with double newlines
    return '\n\n'.join(chunk.strip() for chunk in chunks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert text to Markdown with resume capability')
    parser.add_argument('--input', '-i', 
                       default="texts/text.txt",
                       help='Input text file (default: texts/text.txt)')
    parser.add_argument('--output', '-o', 
                       default="output.md",
                       help='Output markdown file (default: output.md)')
    parser.add_argument('--resume', '-r',
                       help='Resume from state file')
    parser.add_argument('--restart',
                       action='store_true',
                       help='Force restart, ignore existing state')
    
    args = parser.parse_args()
    logger = setup_logging()
    config = ProcessingConfig()
    
    # Create texts directory if it doesn't exist
    Path("texts").mkdir(exist_ok=True)
    
    try:
        text_to_markdown(args.input, args.output, config, args.resume, args.restart)
    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input}")
        logger.info(f"Please place your input file in: {Path(args.input).absolute()}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)