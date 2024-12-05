import csv
from src.models.rating_prediction import *
from src.models.experience_words import *


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

    return df


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

ratings_to_predict = filter_beer_ratings(
    df_ba_ratings, user_threshold=650, beer_threshold=650
)

ratings_to_predict = ratings_to_predict.sample(n=1000, random_state=42)
print("Creating features for", len(ratings_to_predict), "ratings")
user_ids = ratings_to_predict["user_id"].tolist()
beer_ids = ratings_to_predict["beer_id"].tolist()

user_stats, beer_stats = init_features(
    df_ba_ratings,  # here we need the original one to take all the ratings in to account when creating the features
    # not just the ones we're trying to predict
    user_ids,
    beer_ids,
    words,
    lower_threshold=2.7,
    upper_threshold=3.8,
)

output_csv_file = "features_and_label.csv"
with open(output_csv_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow(column_names)

    for idx in tqdm(ratings_to_predict.index, desc="Writing to CSV"):
        features = get_features(idx, user_stats, beer_stats)
        writer.writerow(features)

print(f"The features where saved successfully in '{output_csv_file}'.")
