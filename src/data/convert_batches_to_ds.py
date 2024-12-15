from glob import glob
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    # Load all the batches
    print("Loading all the batches...")
    batches = sorted(glob("data/RatingPrediction/batch_no_*.csv"))
    dfs = [pd.read_csv(batch) for batch in tqdm(batches)]

    # Load all the foreign batches
    print("Loading all the foreign batches...")
    batches_foreign = sorted(glob("data/RatingPrediction/foreign_batch_*.csv"))
    dfs_foreign = [pd.read_csv(batch) for batch in tqdm(batches_foreign)]

    # Concatenate all the batches
    df = pd.concat(dfs)
    df.reset_index(drop=True, inplace=True)
    df_foreign = pd.concat(dfs_foreign)
    df_foreign.reset_index(drop=True, inplace=True)

    # Concatenate the foreign dataset
    df = pd.concat([df, df_foreign], axis=1)

    # Save the concatenated dataset
    df.to_csv("data/RatingPrediction/rating_reviewer_pairs_foreign.csv", index=False)
