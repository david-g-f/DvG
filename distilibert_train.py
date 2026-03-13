import dotenv
import os
import logging, sys
import torch
import transformers as tf
import utils as ut
from sklearn.model_selection import train_test_split
from datasets import Dataset
import pandas as pd

# Creating the data structures 
data = pd.read_csv("metrics/training_data_a.csv")
train_data, test_data = train_test_split(data, test_size=0.2)
train_hf = Dataset.from_pandas(train_data)
test_hf = Dataset.from_pandas(test_data)

tokenizer = tf.DistilBertTokenizerFast.from_pretrained(os.getenv("SLMID2"))