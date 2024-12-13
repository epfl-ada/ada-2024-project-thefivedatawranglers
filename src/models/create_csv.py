import csv
from src.models.rating_prediction import *


def filter_beer_ratings(df, user_threshold, beer_threshold):
    """
    Filters the DataFrame to include only:
    - Ratings from users who have more than `user_threshold` ratings.
    - Ratings for beers that have more than `beer_threshold` ratings.
    - Reviews where the 'text' column is not NaN.
    :param df: the original DataFrame containing beer ratings.
    :param user_threshold: minimum number of ratings a user must have.
    :param beer_threshold: minimum number of ratings a beer must have.
    :return: the filtered dataframe
    """

    # drop rows where the 'text' column is NaN
    df = df.dropna(subset=["text"])

    # count the number of ratings per user and keep only users with more than `user_threshold` ratings
    user_counts = df["user_id"].value_counts()
    valid_users = user_counts[user_counts > user_threshold].index
    df = df[df["user_id"].isin(valid_users)]

    # count the number of ratings per beer and keep only beers with more than `beer_threshold` ratings
    beer_counts = df["beer_id"].value_counts()
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

# single value columns
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

# the distribution columns
for prefix in [
    "good_distr_user_",
    "bad_distr_user_",
    "good_distr_beer_",
    "bad_distr_beer_",
]:
    column_names += [prefix + word for word in words]

# category one hot columns
column_names += ["beer_in_category_" + category for category in possible_cat_vals]

# month one-hot columns
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


# nb stands for non-batch version
def filter_and_create_stats_nb():
    """
    Creates the user stats, beer stats and filters the ratings to only those surpassing our criteria
    :return: user stats, beer stats, ratings to predict
    """
    ratings_to_predict = filter_beer_ratings(
        df_ba_ratings, user_threshold=20, beer_threshold=50
    )
    print("There are ", len(ratings_to_predict), "ratings worth predicting")

    # shuffling the data just to be sure there is no bias through the order
    ratings_to_predict = ratings_to_predict.sample(frac=1, random_state=42)
    ratings_to_predict = ratings_to_predict.iloc[0:1000]

    print("Creating features for", len(ratings_to_predict), "ratings")
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


def save_csv(user_stats, beer_stats, ratings_to_predict):
    """
    Saves the features for the NN in a CSV file
    :param user_stats:
    :param beer_stats:
    :param ratings_to_predict:
    :return:
    """
    with open(output_csv_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(column_names)

        for idx in tqdm(ratings_to_predict.index, desc="Writing to CSV"):
            features = get_features(idx, user_stats, beer_stats)
            writer.writerow(features)

    print(f"The features where saved successfully in '{output_csv_file}'.")


# ------------ Original feature creation non-batch-wise ------------
"""user_stats1, beer_stats1, ratings_to_predict1 = filter_and_create_stats_nb()
save_csv(user_stats1, beer_stats1, ratings_to_predict1)"""
