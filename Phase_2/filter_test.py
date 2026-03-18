import torch
import logging, sys
import time
# import Phase_1.utils as ut 
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

logging.basicConfig( # Configure terminal logger
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


# Load the trained model and test it on unseen phrases, both benign and malignant
def evaluate_security():
    model_path = "../metrics/distilibert_trained_v2"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path).to("cuda")
    # ut.checkVRAM() # Monitor how much VRAM DistiliBERT is using
    model.eval()

    logging.info("-- DistiliBERT filter set to eval mode.")
    while True:
        user = input("Enter a prompt to be tested here: ")
        if user.lower() == "exit":
            break

        inputs = tokenizer(user, return_tensors="pt", truncation=True, padding=True, max_length=512).to("cuda")
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        duration = time.time() - start_time
        logging.info(f"-- Time taken to compute probability: {duration:.2f}s")

        score = probabilities[0][1].item()
        malignant = score > 0.7
        status = 'BLOCKED. Prompt has been deemed unsafe.' if malignant else 'SAFE. Prompt has been deemed clean.'

        logging.info(f"-- Test complete. STATUS: {status}.")
        logging.info(f"-- Malignant Probability of input: {score:.2f}")

if __name__ == "__main__":
    evaluate_security()