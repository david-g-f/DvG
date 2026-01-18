import logging
import sys
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer
from huggingface_hub import login 

# Script to check HF access for using the model(s)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def checkAccess():
    load_dotenv()
    token = os.getenv("HFTOKEN")

    if not token:
        logging.error("No HuggingFace token found in environment variables. Please append")
        return
    
    try:
        login(token=token)
        logging.info("Login to HuggingFace successful. Attempting to load Llama 3.1 now...")
        tokenizer = AutoTokenizer.from_pretrained(os.getenv("LLMID"))
        logging.info("Access to LLM granted and ready to use.")

    except Exception as err:
        logging.error("Access denied as an exception has been raised: {err}")

if __name__ == "__main__":
    checkAccess()