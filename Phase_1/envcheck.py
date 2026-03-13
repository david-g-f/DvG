import torch
import sys
import logging
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def verifyStation():
    logging.info("--- Verifying Environment + Workstation")

    if not torch.cuda.is_available():
        logging.error("CUDA not detected, can't proceed with project.")
        return
    
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    logging.info(f"Current GPU: {gpu}.")
    logging.info(f"VRAM: {vram:.2f} GB")

    logging.info("Memory Swapping will be used to manage the VRAM constraint.")
    
    libs = ['transformers', 'peft', 'bitsandbytes', 'accelerate', 'unsloth'] # Necessary libraries for using and fine tuning SLM
    for lib in libs:
        try:
            __import__(lib)
            logging.info(f"{lib} is installed properly.")
        except ImportError:
            logging.error(f"{lib} is missing. Run pip install {lib}.")
        
    
    model = "microsoft/Phi-3-mini-4k-instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
        logging.info("Tokenizer access for Phi-3 granted and verified.")
    except Exception as err:
        logging.error("Failed to access model or Hugging Face: {err}")

    logging.info("Workstation verification complete.")

if __name__ == "__main__":
    verifyStation()