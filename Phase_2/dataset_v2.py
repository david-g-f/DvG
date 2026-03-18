from datasets import load_dataset
import pandas as pd


def dataset(outpath):
    print("-- Loading Deepset dataset from HuggingFace...")
    deepset_data = load_dataset("deepset/prompt-injections", split="train").to_pandas()
    deepset_malignant = deepset_data[deepset_data['label'] == 1]
    deepset_benign = deepset_data[deepset_data['label'] == 0]
    print("-- Deepset data loaded successfully.")

    min_count = min(len(deepset_malignant), len(deepset_benign)) # Ensure an exact 50/50 split 
    final_data = pd.concat([deepset_malignant.sample(n=min_count, random_state=100), deepset_benign.sample(n=min_count, random_state=100)])

    final_data = final_data.sample(frac=1).reset_index(drop=True) # Shuffle the dataset to prevent bias
    final_data.to_csv(outpath, index=False, encoding='utf-8')

    print(f"-- Dataset successfully saved to {outpath} and ready for training.")

if __name__ == "__main__":
    dataset(outpath="../metrics/training_data_b.csv")