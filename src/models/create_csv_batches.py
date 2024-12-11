import csv
from src.models.rating_prediction import *
from datetime import datetime
from src.data.some_dataloader import *


def filter_beer_ratings(df, user_threshold, beer_threshold):
    """
    Filters the DataFrame to include only:
    - Ratings from users who have more than `user_threshold` ratings.
    - Ratings for beers that have more than `beer_threshold` ratings.
    - Reviews where the 'text' column is not NaN.

    Args:
        df (pd.DataFrame): The original DataFrame containing beer ratings.
        user_threshold (int): Minimum number of ratings a user must have.
        beer_threshold (int): Minimum number of ratings a beer must have.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    # Drop rows where the 'text' column is NaN
    df = df.dropna(subset=["text"])

    # Count the number of ratings per user
    user_counts = df["user_id"].value_counts()

    # Keep only users with more than `user_threshold` ratings
    valid_users = user_counts[user_counts > user_threshold].index
    df = df[df["user_id"].isin(valid_users)]

    # Count the number of ratings per beer
    beer_counts = df["beer_id"].value_counts()

    # Keep only beers with more than `beer_threshold` ratings
    valid_beers = beer_counts[beer_counts > beer_threshold].index
    df = df[df["beer_id"].isin(valid_beers)]

    # keeping only entries that contain locational data
    len_before = len(df)
    df = df[df.index.isin(joined_ba_df["ratings_idx"])]
    len_after = len(df)
    print(
        "We filtered out",
        len_before - len_after,
        "ratings because they had no locational data.",
    )

    return df


output_csv_file = "features_and_label.csv"

column_names = [
    "label_actual_rating",
    "user_mean_until_now",
    "user_style_mean_until_now",
    "user_num_ratings_until_now",
    "user_is_exp",
    "user_brewery_distance",
    "beer_mean_rating",
    "good_interaction_term",
    "bad_interaction_term",
]

for prefix in [
    "good_distr_user_",
    "bad_distr_user_",
    "good_distr_beer_",
    "bad_distr_beer_",
]:
    column_names += [prefix + word for word in words]

column_names += ["beer_in_category_" + category for category in possible_cat_vals]
column_names += [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
print("The CSV will have", len(column_names), "columns")


def filter_and_create_stats():
    ratings_to_predict = filter_beer_ratings(
        df_ba_ratings, user_threshold=20, beer_threshold=50
    )
    print("There are ", len(ratings_to_predict), "ratings worth predicting")

    # shuffling the data just to be sure there is no bias through the order
    ratings_to_predict = ratings_to_predict.sample(frac=1, random_state=42)

    print("Creating features for ratings")
    user_ids = ratings_to_predict["user_id"].unique().tolist()
    beer_ids = ratings_to_predict["beer_id"].unique().tolist()

    user_stats, beer_stats = init_features(
        df_ba_ratings,  # here we need the original one to take all the ratings in to account when creating the features
        # not just the ones we're trying to predict
        user_ids,
        beer_ids,
        words,
        lower_threshold=2.7,
        upper_threshold=3.8,
    )
    return user_stats, beer_stats, ratings_to_predict


def get_last_completed_batch(log_file):
    if not os.path.exists(log_file):
        return 0

    with open(log_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
        if lines:
            last_line = lines[-1].strip()
            try:
                return int(last_line.split(",")[0])
            except (IndexError, ValueError):
                return 0
    return 0


def log_batch_completion(log_file, batch_number):
    with open(log_file, "a", encoding="utf-8") as file:
        completion_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"{batch_number},{completion_time}\n")


def save_csv_in_batches(
    user_stats,
    beer_stats,
    ratings_to_predict,
    output_csv_file,
    log_file="batch_log.txt",
    batch_size=5000,
):
    num_batches = (
        len(ratings_to_predict) + batch_size - 1
    ) // batch_size  # Calculate the number of batches

    last_completed_batch = get_last_completed_batch(log_file)

    for batch_number in range(last_completed_batch + 1, num_batches + 1):
        batch_start = (batch_number - 1) * batch_size
        batch_end = min(batch_start + batch_size, len(ratings_to_predict))

        batch_ratings = ratings_to_predict.iloc[batch_start:batch_end]
        batch_file_name = f"batch_no_{batch_number}.csv"

        with open(batch_file_name, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            writer.writerow(column_names)

            for idx in tqdm(batch_ratings.index, desc=f"Writing batch {batch_number}"):
                features = get_features(idx, user_stats, beer_stats)
                writer.writerow(features)

        log_batch_completion(log_file, batch_number)
        print(f"Batch {batch_number} saved successfully in '{batch_file_name}'.")


"""
------------ Creating the label-feature pairs -------------
user_stats1, beer_stats1, ratings_to_predict1 = filter_and_create_stats()
save_csv_in_batches(
    user_stats1, beer_stats1, ratings_to_predict1, output_csv_file, batch_size=1000
)
"""


def create_foreign_batches():
    ratings_to_predict = filter_beer_ratings(
        df_ba_ratings, user_threshold=20, beer_threshold=50
    )
    # shuffling the data just to be sure there is no bias through the order
    ratings_to_predict = ratings_to_predict.sample(frac=1, random_state=42)

    users_ba_df = pd.read_csv("../../data/BeerAdvocate/users.csv")
    breweries_ba_df = pd.read_csv("../../data/BeerAdvocate/breweries.csv")

    users_ba_df = users_ba_df.drop_duplicates(subset="user_id", keep="first")
    ratings_to_predict = ratings_to_predict.merge(
        users_ba_df[["user_id", "location"]], on="user_id", how="left"
    )
    ratings_to_predict.rename(columns={"location": "user_location"}, inplace=True)

    breweries_ba_df.drop_duplicates(subset="brewery_id", keep="first")
    ratings_to_predict = ratings_to_predict.merge(
        breweries_ba_df[["brewery_id", "location"]], on="brewery_id", how="left"
    )
    ratings_to_predict.rename(columns={"location": "brewery_location"}, inplace=True)

    ratings_to_predict["foreign_us_aggr"] = ratings_to_predict.apply(
        lambda row: (
            1
            if (
                row["user_location"] == row["brewery_location"]
                or (
                    "United States" in row["user_location"]
                    and "United States" in row["brewery_location"]
                )
            )
            else 0
        ),
        axis=1,
    )

    ratings_to_predict["foreign_us_split"] = ratings_to_predict.apply(
        lambda row: (1 if (row["user_location"] == row["brewery_location"]) else 0),
        axis=1,
    )

    batch_size = 1000
    for i in range(230):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_df = ratings_to_predict.iloc[start_idx:end_idx]

        file_name = f"foreign_batch_{i + 1}.csv"
        batch_df.to_csv(
            file_name, index=False, columns=["foreign_us_aggr", "foreign_us_split"]
        )

        print(f"Batch {i + 1} saved as: {file_name}")


# originally I forgot to put the foreign feature in there, so here is the data generating for that:
create_foreign_batches()
