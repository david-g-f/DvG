import dotenv
import os
import logging, sys
import torch
import transformers as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
import pandas as pd
import numpy as np


dotenv.load_dotenv()
logging.basicConfig( # Configure terminal logger
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def metrics(results): # Utility function to help visualize training results
    logits, labels = results
    prediction = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, prediction)
    f1 = f1_score(labels, prediction, average="weighted")
    return {"accuracy": accuracy, "f1": f1}

class WeightedTrainer(tf.Trainer): # Subclass of Trainer that has weightage, just an experiment
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_factor = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0,2.5]).to("cuda")) # Weight Ratio of benign to malignant labels 
        loss = loss_factor(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss
        

# Creating the data structures 
data = pd.read_csv("../metrics/training_data_f.csv")
train_data, test_data = train_test_split(data, test_size=0.2)
train_hf = Dataset.from_pandas(train_data)
test_hf = Dataset.from_pandas(test_data)

tokenizer = tf.DistilBertTokenizerFast.from_pretrained(os.getenv("SLMID2"))

# Necessary tokenize function for each input so the model is capable of processing it
def tokenize(input_text):
    return tokenizer(input_text["text"], truncation=True, padding="max_length", max_length=512)

train_hf = train_hf.map(tokenize, batched=True)
test_hf = test_hf.map(tokenize, batched=True)

# Setting up the model for training
model = tf.DistilBertForSequenceClassification.from_pretrained(os.getenv("SLMID2"), num_labels=2)
training_settings = tf.TrainingArguments(
    output_dir="../metrics/distilibert_tune_v6",
    eval_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="eval_loss",
    learning_rate=0.00002,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="../metrics/distilibert_logs_v6",
    load_best_model_at_end=True
)

trainer = WeightedTrainer( # Experimenting with the weightage now
    model=model,
    args=training_settings,
    train_dataset=train_hf,
    eval_dataset=test_hf,
    compute_metrics=metrics
)

logging.info("-- Beginning DistiliBERT Fine Tuning.")
trainer.train()
model.save_pretrained("../metrics/distilibert_trained_v6")
tokenizer.save_pretrained("../metrics/distilibert_trained_v6")
logging.info("-- DistiliBERT Training complete.")