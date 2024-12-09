from glob import glob
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    # Load all the batches
    batches = glob("data/RatingPrediction/*.csv")
    dfs = [pd.read_csv(batch) for batch in tqdm(batches)]

    # Concatenate all the batches
    df = pd.concat(dfs)

    # Save the concatenated dataset
    df.to_csv("data/RatingPrediction/rating_reviewer_pairs.csv", index=False)
