import pandas as pd
from collections import Counter
import numpy as np
import os
import ast

from src.data.some_dataloader import *
from src.models.distance_analysis import (
    retrieve_location_data,
    calculate_distances,
    join_users_breweries_ratings,
)
from src.models.experience_words import *
from tqdm import tqdm


df_ba_ratings, df_rb_ratings = load_rating_data(
    ba_path="../../data/BeerAdvocate/BA_ratings.csv",
    rb_path="../../data" "/RateBeer" "/RB_ratings.csv",
)

df_ba_users, df_rb_users = load_user_data(
    ba_path="../../data/BeerAdvocate/users.csv", rb_path="../../data/RateBeer/users.csv"
)

df_brewery_ba = load_brewery_data(brewery_path="../../data/BeerAdvocate/breweries.csv")
df_brewery_rb = load_brewery_data(brewery_path="../../data/RateBeer/breweries.csv")

joined_ba_df = join_users_breweries_ratings(
    df_ba_users, df_brewery_ba, df_ba_ratings, ratebeer=False
)
joined_rb_df = join_users_breweries_ratings(
    df_rb_users, df_brewery_rb, df_rb_ratings, ratebeer=True
)

df_location = retrieve_location_data(
    joined_ba_df, joined_rb_df, path="../../data/locations.csv"
)

joined_ba_df = calculate_distances(joined_ba_df, df_location)
joined_rb_df = calculate_distances(joined_rb_df, df_location)

df_ba_ratings["month"] = pd.to_datetime(df_ba_ratings["date"], unit="s").dt.month
df_rb_ratings["month"] = pd.to_datetime(df_rb_ratings["date"], unit="s").dt.month

months = df_ba_ratings.month.unique()

exp_user_ids_ba = get_experienced_users2(df_ba_ratings, exp_words1)
df_ba_ratings["experienced_user"] = (
    df_ba_ratings["user_id"].isin(exp_user_ids_ba).astype(int)
)


categories = {
    "Lager": [
        "Euro Pale Lager",
        "German Pilsener",
        "Munich Helles Lager",
        "Czech Pilsener",
        "Vienna Lager",
        "Light Lager",
        "Munich Dunkel Lager",
        "Schwarzbier",
        "Euro Dark Lager",
        "Märzen / Oktoberfest",
        "Doppelbock",
        "Eisbock",
        "Maibock / Helles Bock",
        "Baltic Porter",
        "Euro Strong Lager",
    ],
    "Ale": [
        "English Pale Ale",
        "American Pale Ale (APA)",
        "English Bitter",
        "Extra Special / Strong Bitter (ESB)",
        "Belgian Pale Ale",
        "Irish Red Ale",
        "American Amber / Red Ale",
        "Scottish Ale",
        "English Brown Ale",
        "American Brown Ale",
        "Old Ale",
        "English Strong Ale",
        "American Strong Ale",
        "Scotch Ale / Wee Heavy",
        "English Barleywine",
        "American Barleywine",
        "Belgian Dark Ale",
        "Belgian Strong Dark Ale",
        "Quadrupel (Quad)",
        "Dubbel",
        "Tripel",
    ],
    "IPA": [
        "English India Pale Ale (IPA)",
        "American IPA",
        "American Double / Imperial IPA",
        "Belgian IPA",
    ],
    "Stout": [
        "Irish Dry Stout",
        "Milk / Sweet Stout",
        "Oatmeal Stout",
        "Foreign / Export Stout",
        "Russian Imperial Stout",
        "American Stout",
        "English Stout",
    ],
    "Porter": ["English Porter", "American Porter", "Baltic Porter"],
    "Wheat Beer": [
        "Hefeweizen",
        "Kristalweizen",
        "Dunkelweizen",
        "Weizenbock",
        "Witbier",
        "Berliner Weissbier",
        "Gose",
        "Roggenbier",
    ],
    "Belgian Styles": [
        "Saison / Farmhouse Ale",
        "Bière de Garde",
        "Lambic - Fruit",
        "Lambic - Unblended",
        "Gueuze",
        "Faro",
    ],
    "Specialty": [
        "Smoked Beer",
        "Herbed / Spiced Beer",
        "Pumpkin Ale",
        "Chile Beer",
        "Scottish Gruit / Ancient Herbed Ale",
        "American Wild Ale",
        "Bière de Champagne / Bière Brut",
        "Wheatwine",
        "Sahti",
        "Kvass",
        "Braggot",
    ],
    "Hybrid": ["Kölsch", "Altbier", "California Common / Steam Beer", "Cream Ale"],
    "Light/Low Alcohol": [
        "Low Alcohol Beer",
        "American Adjunct Lager",
        "American Pale Lager",
        "Japanese Rice Lager",
        "Happoshu",
    ],
}

positive_words = [
    "malty",
    "hoppy",
    "fruity",
    "sweet",
    "smooth",
    "crisp",
    "refreshing",
    "balanced",
    "caramel",
    "chocolatey",
    "nutty",
    "citrusy",
    "spicy",
    "creamy",
    "full-bodied",
    "light",
    "dry",
    "velvety",
    "excellent",
    "amazing",
    "delicious",
    "perfect",
    "great",
    "fantastic",
    "lovely",
    "enjoyable",
    "favorite",
    "wonderful",
    "classic",
    "authentic",
    "well-crafted",
    "artisanal",
    "clean",
]

negative_words = [
    "bitter",
    "sour",
    "bland",
    "stale",
    "metallic",
    "burnt",
    "overpowering",
    "flat",
    "watery",
    "cloying",
    "harsh",
    "astringent",
    "thin",
    "weak",
    "unbalanced",
    "bad",
    "disappointing",
    "boring",
    "unpleasant",
    "off-putting",
    "weird",
    "mediocre",
    "not great",
    "subpar",
    "average",
    "artificial",
    "generic",
    "industrial",
    "chemical",
]

exp_words1 = [
    "Lacing",
    "Ester",
    "Diacetyl",
    "Phenol",
    "Dry Hop",
    "DMS",
    "Oxidation",
    "catty",
    "resinous",
    "astringent",
    "Effervescent",
    "Tannic",
    "Brettanomyces",
    "lactic",
    "autolysis",
    "Krausen",
]

words = list(set(positive_words + negative_words + exp_words1))
print(f"We calculate the distribution for {len(words)} words")

# Find the 3 most frequent beer styles
top_styles = df_ba_ratings["style"].value_counts().nlargest(3).index.tolist()
possible_cat_vals = list(categories.keys()) + top_styles


def split_ratings_by_threshold(df, user_id, upper_threshold, lower_threshold):
    """
    Splits the ratings of a user into two DataFrames based on thresholds.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the ratings.
        user_id (int or str): The ID of the user whose ratings are to be filtered.
        upper_threshold (float): The threshold above which ratings are considered "good".
        lower_threshold (float): The threshold below which ratings are considered "bad".

    Returns:
        tuple: Two DataFrames - one for ratings >= upper_threshold, and one for ratings <= lower_threshold.
    """
    # Filter DataFrame for the specific user
    user_ratings = df[df["user_id"] == user_id]

    # Create a DataFrame for ratings >= upper_threshold
    good_ratings = user_ratings[user_ratings["rating"] >= upper_threshold]

    # Create a DataFrame for ratings <= lower_threshold
    bad_ratings = user_ratings[user_ratings["rating"] <= lower_threshold]

    return good_ratings, bad_ratings


def count_word_frequencies(df_ratings, word_list):
    """
    Counts the frequencies of words from a given word list in a text column of a DataFrame.

    Parameters:
        df_ratings (pd.DataFrame): The DataFrame containing the text column.
        word_list (list): A list of words whose frequencies should be counted.

    Returns:
        dict: A dictionary with words as keys and their frequencies as values.
    """
    # Combine all texts in the column into one large string
    all_text = " ".join(
        df_ratings["text"].dropna()
    ).lower()  # Convert to lowercase for consistent counting

    # Tokenize the text (split into words)
    words = all_text.split()

    # Filter only the words from the provided word list
    filtered_words = [word for word in words if word in word_list]

    # Count the frequencies of the filtered words
    word_frequencies = Counter(filtered_words)

    # Return the result as a dictionary
    return dict(word_frequencies)


def pre_stats(df_ratings, style, date):
    df_ratings = df_ratings[df_ratings["date"] < date]
    mean = df_ratings["rating"].mean()
    style_mean = df_ratings[df_ratings["style"] == style]["rating"].mean()
    num_ratings = len(df_ratings)
    return mean, style_mean, num_ratings


def is_experienced(user_id, exp_user_ids):
    return user_id in exp_user_ids


def get_top_styles(df_ratings, threshold):
    top_30_styles = df_ratings["style"].value_counts().head(threshold)


def map_to_category(style):
    """Replace beer styles with their categories, except for the top 3 styles"""
    if style in top_styles:
        return style
    for category, styles in categories.items():
        if style in styles:
            return category
    return "Other"  # Catch-all for uncategorized styles


def categorize_beer_styles(df):
    # Apply the mapping
    df["style"] = df["style"].apply(map_to_category)
    return df


def safe_normalize_to_list(distribution, word_list):
    """
    Normalizes a dictionary of word frequencies and converts it into a list based on a predefined word list.

    Parameters:
        distribution (dict): A dictionary with words as keys and their frequencies as values.
        word_list (list): A predefined list of words to structure the output list.

    Returns:
        list: A list of normalized frequencies corresponding to the order in word_list.
    """
    # Calculate the total for normalization
    total = sum(distribution.values())

    # Normalize the distribution and handle division by zero
    normalized_distribution = {
        key: value / total if total > 0 else 0 for key, value in distribution.items()
    }

    # Convert the normalized dictionary into a list based on the order in word_list
    normalized_list = [normalized_distribution.get(word, 0) for word in word_list]

    return normalized_list


def init_features(
    df_ratings,
    user_ids,
    beer_ids,
    word_list,
    lower_threshold=2.5,
    upper_threshold=3.8,
):
    # Load or initialize user statistics
    user_stats_file = "user_stats.csv"
    print("Trying to load user_stats.csv...")
    if os.path.exists(user_stats_file):
        existing_user_stats_df = pd.read_csv(user_stats_file)

        # we need to interpret the lists that are actually saved as strings as lists again
        existing_user_stats_df["good_distr_user"] = existing_user_stats_df[
            "good_distr_user"
        ].apply(ast.literal_eval)
        existing_user_stats_df["bad_distr_user"] = existing_user_stats_df[
            "bad_distr_user"
        ].apply(ast.literal_eval)

        existing_user_ids = set(existing_user_stats_df["user_id"])
    else:
        existing_user_stats_df = pd.DataFrame()
        existing_user_ids = set()

    new_user_ids = set(user_ids) - existing_user_ids
    user_stats = {}

    for user_id in tqdm(new_user_ids, desc="Computing missing user stats"):
        user_ratings = df_ratings[df_ratings["user_id"] == user_id]

        # Filter "Good" and "Bad" ratings
        good_ratings_user_df = user_ratings[user_ratings["rating"] >= upper_threshold]
        bad_ratings_user_df = user_ratings[user_ratings["rating"] <= lower_threshold]

        # Calculate distributions
        good_distr_user = count_word_frequencies(
            good_ratings_user_df, word_list=word_list
        )
        bad_distr_user = count_word_frequencies(
            bad_ratings_user_df, word_list=word_list
        )
        # Normalize distributions
        good_distr_user = safe_normalize_to_list(good_distr_user, word_list)
        bad_distr_user = safe_normalize_to_list(bad_distr_user, word_list)

        user_stats[user_id] = {
            "user_id": user_id,
            "good_distr_user": good_distr_user,
            "bad_distr_user": bad_distr_user,
        }

    # Append new user stats to the CSV file
    if user_stats:
        new_user_stats_df = pd.DataFrame(user_stats.values())
        user_stats_df = pd.concat(
            [existing_user_stats_df, new_user_stats_df], ignore_index=True
        )
        user_stats_df.to_csv(user_stats_file, index=False)
    else:
        user_stats_df = existing_user_stats_df

    # Load or initialize beer statistics
    beer_stats_file = "beer_stats.csv"
    if os.path.exists(beer_stats_file):
        existing_beer_stats_df = pd.read_csv(beer_stats_file)

        # we need to interpret the lists that are actually saved as strings as lists again
        existing_beer_stats_df["good_distr_beer"] = existing_beer_stats_df[
            "good_distr_beer"
        ].apply(ast.literal_eval)
        existing_beer_stats_df["bad_distr_beer"] = existing_beer_stats_df[
            "bad_distr_beer"
        ].apply(ast.literal_eval)
        existing_beer_stats_df["one_hot_cat"] = existing_beer_stats_df[
            "one_hot_cat"
        ].apply(ast.literal_eval)

        existing_beer_ids = set(existing_beer_stats_df["beer_id"])
    else:
        existing_beer_stats_df = pd.DataFrame()
        existing_beer_ids = set()

    new_beer_ids = set(beer_ids) - existing_beer_ids
    beer_stats = {}

    for beer_id in tqdm(new_beer_ids, desc="Computing beer stats"):
        beer_ratings = df_ratings[df_ratings["beer_id"] == beer_id]
        good_ratings_beer_df = beer_ratings[beer_ratings["rating"] >= upper_threshold]
        bad_ratings_beer_df = beer_ratings[beer_ratings["rating"] <= lower_threshold]
        style_cat = map_to_category(beer_ratings.iloc[0]["style"])

        # Calculate word distributions
        good_distr_beer = count_word_frequencies(
            good_ratings_beer_df, word_list=word_list
        )
        good_distr_beer = safe_normalize_to_list(good_distr_beer, word_list)
        bad_distr_beer = count_word_frequencies(
            bad_ratings_beer_df, word_list=word_list
        )
        bad_distr_beer = safe_normalize_to_list(bad_distr_beer, word_list)

        # Calculate average rating
        mean_rating = beer_ratings["rating"].mean()

        one_hot_cat = [
            1 if category == style_cat else 0 for category in possible_cat_vals
        ]

        beer_stats[beer_id] = {
            "beer_id": beer_id,
            "good_distr_beer": good_distr_beer,
            "bad_distr_beer": bad_distr_beer,
            "mean_rating": mean_rating,
            "one_hot_cat": one_hot_cat,
        }

    # Append new beer stats to the CSV file
    if beer_stats:
        new_beer_stats_df = pd.DataFrame(beer_stats.values())
        beer_stats_df = pd.concat(
            [existing_beer_stats_df, new_beer_stats_df], ignore_index=True
        )
        beer_stats_df.to_csv(beer_stats_file, index=False)
    else:
        beer_stats_df = existing_beer_stats_df

    return user_stats_df, beer_stats_df


def get_features(rating_idx, user_stats, beer_stats, rate_beer=False):
    # setting variables depending on with which dataset we are working
    if rate_beer:
        df_ratings = df_rb_ratings
        joined_df = joined_rb_df
    else:
        df_ratings = df_ba_ratings
        joined_df = joined_ba_df

    # things about this specific rating
    rating_row = df_ratings.loc[rating_idx]
    label = rating_row["rating"]
    user_id = rating_row["user_id"]
    beer_id = rating_row["beer_id"]
    timestamp_date = rating_row["date"]
    month = rating_row["month"]
    style = rating_row["style"]

    is_exp = rating_row["experienced_user"]

    user_info = user_stats[user_stats["user_id"] == user_id]
    beer_info = beer_stats[beer_stats["beer_id"] == beer_id]

    # using default values if an error occurs and reporting the error
    if user_info.empty:
        print("For some reason the user data is missing.")
        user_good_distr = [0] * 77
        user_bad_distr = [0] * 77
    else:
        user_good_distr = user_info["good_distr_user"].values[0]
        user_bad_distr = user_info["bad_distr_user"].values[0]
    if beer_info.empty:
        print("For some reason the beer data is missing.")
        beer_good_distr = [0] * 77
        beer_bad_distr = [0] * 77
        beer_mean_rating = 0
        beer_one_hot_cat = [0] * 13
    else:
        beer_good_distr = beer_info["good_distr_beer"].values[0]
        beer_bad_distr = beer_info["bad_distr_beer"].values[0]
        beer_mean_rating = beer_info["mean_rating"].values[0]
        beer_one_hot_cat = beer_info["one_hot_cat"].values[0]

    # this calculates some statistics for this user but only for the ratings he did before doing this rating now
    user_mean, user_style_mean, user_num_ratings = pre_stats(
        df_ratings[df_ratings["user_id"] == user_id], style, timestamp_date
    )

    distance = joined_df[joined_df["ratings_idx"] == rating_idx][
        "distance_user_brewery"
    ].iloc[0]
    # print("Distance: ", distance)

    # we also want to add an interaction feature between the beer-sided distribution of the words and th user-sided one
    # here we use the multiplication as an interaction term
    interaction_good = np.array(user_good_distr) @ np.array(beer_good_distr)
    interaction_bad = np.array(user_bad_distr) @ np.array(beer_bad_distr)

    # now the month as a one-hot encoding
    one_hot_months = {
        m: [1 if m == month else 0 for month in range(1, 13)] for m in range(1, 13)
    }
    one_hot_month = one_hot_months.get(month, [0] * 12)

    return (
        [
            label,
            user_mean,
            user_style_mean,
            user_num_ratings,
            is_exp,
            distance,
            beer_mean_rating,
            interaction_good,
            interaction_bad,
        ]  # 8 vals
        + user_good_distr  # 77 vals
        + user_bad_distr  # 77 vals
        + beer_good_distr  # 77 vals
        + beer_bad_distr  # 77 vals
        + beer_one_hot_cat  # 13 vals
        + one_hot_month  # 12 vals
    )  # results in a total of 341 features + 1 label => 342 values


# we will transform the distributions in a latent space of dimension 5, so the big NN will only see 55 vals
