import os
from llm_inference import LLMInference
from dotenv import load_dotenv

def test_llm_inference():
    load_dotenv()
    token = os.getenv('HF_TOKEN')
    if not token:
        raise ValueError("No HF_TOKEN found in environment or .env file")
        
    llm = LLMInference("meta-llama/Llama-3.2-3B-Instruct", token=token)
    test_text = """Title
This is a paragraph.

> This is a quote.

1. First item
2. Second item

* Bullet point
"""
    
    result = llm.run_inference(test_text)
    print(f"LLM Output:\n{result}")
    
    # Add your assertions here
    assert "<h1>" in result
    assert "<p>" in result
    assert "<blockquote>" in result
    assert "<ol>" in result
    assert "<ul>" in result

if __name__ == "__main__":
    test_llm_inference()
