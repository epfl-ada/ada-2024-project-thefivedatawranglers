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


# we aggregate beer styles in meta-beer-styles
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

# this is a list of word that are popular to describe a positive or neutral sentiment about beer
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

# this is a list of word that are popular to describe a negative or neutral sentiment about beer
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

# the words for which we calculate the distributions.e Made up of experience-words, positive-words and negative-words
words = list(set(positive_words + negative_words + exp_words1))
print(f"We calculate the distribution for {len(words)} words")

# Find the 3 most frequent beer styles
top_styles = df_ba_ratings["style"].value_counts().nlargest(3).index.tolist()
# We aggregate the beer styles so that the NN doesn't get too many features as one-hots.
# originally there were ~100 beer styles. We aggregate them to 10.
# But we take the 3 most common beer styles as their owm aggregated categories
# (as they're way more popular than the others)
possible_cat_vals = list(categories.keys()) + top_styles


def split_ratings_by_threshold(df_ratings, user_id, upper_threshold, lower_threshold):
    """
    Splits the ratings of a user into two DataFrames based on thresholds of the rating
    :param df_ratings: the dataframe containing the ratings
    :param user_id: the id of the user whose ratings are to be filtered
    :param upper_threshold: the threshold above which ratings are considered "good"
    :param lower_threshold: the threshold below which ratings are considered "bad"
    :return: good_ratings, bad_ratings: two dataframes - one for ratings >= upper_threshold,
         and one for ratings <= lower_threshold
    """

    # just use the ratings of this user
    user_ratings = df_ratings[df_ratings["user_id"] == user_id]

    good_ratings = user_ratings[user_ratings["rating"] >= upper_threshold]
    bad_ratings = user_ratings[user_ratings["rating"] <= lower_threshold]

    return good_ratings, bad_ratings


def count_word_frequencies(df_ratings, word_list):
    """
    Counts the frequencies of words from a given word list in a text column of a dataframe
    :param df_ratings: the dataframe containing the text column
    :param word_list: a list of words whose frequencies should be counted
    :return: a dictionary with words as keys and their frequencies as values.
    """

    # combine all texts in the column into one large string
    all_text = " ".join(df_ratings["text"].dropna()).lower()

    # split into words
    text_words = all_text.split()
    # filter only the words from the provided word list
    # (don't know if that's the most efficient way, but it does the job)
    filtered_words = [word for word in text_words if word in word_list]

    word_frequencies = Counter(filtered_words)

    return dict(word_frequencies)


def pre_stats(df_ratings, style, date):
    """
    Calculates stats for the uer based on only those ratings he made before the rating we want to predict now.
    We don't want to look in the future in our rating, both because that's not possible in reality but also because we
    want to have stats of the user how he is right now (as we see in the experienced user analysis the behaviour of
    users tends to change with more rations)
    :param df_ratings: the ratings dataframe
    :param style: the style of the beer for which we are predicting
    :param date: the date the rating was made (as a unix time stamp)
    :return:
    """
    # we can use "<" because it's a unix time stamp
    df_ratings = df_ratings[df_ratings["date"] < date]

    mean = df_ratings["rating"].mean()
    style_mean = df_ratings[df_ratings["style"] == style]["rating"].mean()
    num_ratings = len(df_ratings)

    return mean, style_mean, num_ratings


def is_experienced(user_id, exp_user_ids):
    return user_id in exp_user_ids


def get_top_styles(df_ratings, threshold):
    return df_ratings["style"].value_counts().head(threshold)


def map_to_category(style):
    """
    Replace beer styles with their categories, except for the top 3 styles
    :param style: the beer style of the beer that we want to predict
    :return: the aggregated style of that beer
    """
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
    normalizes the word frequencies and converts them into a list based on the predefined word list
    :param distribution: a dictionary with words as keys and their frequencies as values
    :param word_list: the word list
    :return: the normalized list
    """

    # calculate the total for normalization
    total = sum(distribution.values())

    # normalize the distribution
    normalized_distribution = {
        key: value / total if total > 0 else 0 for key, value in distribution.items()
    }

    # convert the normalized dictionary into a list based on the order in word_list
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
    """
    Computes all the features we can calculate about the users and the beers without knowing
    which rating we want to predict. Stuff like the beer category and the word distributions
    (that are quite costly to compute) never change for a given beer/user
    (other than for example the pre-stats) so we can use precompute those features for all the beers
    and users for which we want to make predictions in the future.
    :param df_ratings: the ratings dataframe
    :param user_ids: the users for which we want to compute the stats
    :param beer_ids: the beers for which we want to compute the stats
    :param word_list: the word_list defined above for which we create the distributions
    :param lower_threshold: the threshold defining what we consider a bad rating
    :param upper_threshold: the threshold defining what we consider a good rating
    :return: two dataframes containing the information about the users/beers respectively
    """

    # Load or initialize user statistics
    user_stats_file = "user_stats.csv"
    print(
        "Trying to load user_stats.csv..."
    )  # this is a very verbose function because it takes so long

    # if we (ever) computed stats already we want to load them from a file (because it takes so long)
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

    new_user_ids = (
        set(user_ids) - existing_user_ids
    )  # only computing stuff we didn't do already
    user_stats = {}

    # computation for the user stats
    for user_id in tqdm(new_user_ids, desc="Computing missing user stats"):
        user_ratings = df_ratings[df_ratings["user_id"] == user_id]

        # filter "Good" and "Bad" ratings
        good_ratings_user_df = user_ratings[user_ratings["rating"] >= upper_threshold]
        bad_ratings_user_df = user_ratings[user_ratings["rating"] <= lower_threshold]

        # calculate distributions
        good_distr_user = count_word_frequencies(
            good_ratings_user_df, word_list=word_list
        )
        bad_distr_user = count_word_frequencies(
            bad_ratings_user_df, word_list=word_list
        )
        # normalize distributions
        good_distr_user = safe_normalize_to_list(good_distr_user, word_list)
        bad_distr_user = safe_normalize_to_list(bad_distr_user, word_list)

        user_stats[user_id] = {
            "user_id": user_id,
            "good_distr_user": good_distr_user,
            "bad_distr_user": bad_distr_user,
        }

    # append new user stats to the CSV file from which we can load them in the future
    if user_stats:
        new_user_stats_df = pd.DataFrame(user_stats.values())
        user_stats_df = pd.concat(
            [existing_user_stats_df, new_user_stats_df], ignore_index=True
        )
        user_stats_df.to_csv(user_stats_file, index=False)
    else:
        user_stats_df = (
            existing_user_stats_df  # we can't append if there is nothing to append to
        )

    # load or initialize beer statistics (there is also a save file)
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

    # computation for the beer stats
    for beer_id in tqdm(new_beer_ids, desc="Computing beer stats"):
        beer_ratings = df_ratings[df_ratings["beer_id"] == beer_id]
        good_ratings_beer_df = beer_ratings[beer_ratings["rating"] >= upper_threshold]
        bad_ratings_beer_df = beer_ratings[beer_ratings["rating"] <= lower_threshold]
        style_cat = map_to_category(beer_ratings.iloc[0]["style"])

        # calculate word distributions
        good_distr_beer = count_word_frequencies(
            good_ratings_beer_df, word_list=word_list
        )
        good_distr_beer = safe_normalize_to_list(good_distr_beer, word_list)
        bad_distr_beer = count_word_frequencies(
            bad_ratings_beer_df, word_list=word_list
        )
        bad_distr_beer = safe_normalize_to_list(bad_distr_beer, word_list)

        # calculate average rating
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

    # append new beer stats to the CSV file (the save file)
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
    """
    For a given rating it returns all the features that we give to the neural network (except the foreign feature...
    I forgot that originally and added it in as a separate small function in the create_csv_batches.py)
    :param rating_idx: the index of the rating which we try to predict
    :param user_stats: the user_stats dataframe
    :param beer_stats: the beer_stats dataframe
    :param rate_beer: a flag saying whether we do the computation for the RateBeer or the BeerAdvocate dataframe
    :return:
    """
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
        beer_good_distribution = [0] * 77
        beer_bad_distribution = [0] * 77
        beer_mean_rating = 0
        beer_one_hot_cat = [0] * 13
    else:
        beer_good_distribution = beer_info["good_distr_beer"].values[0]
        beer_bad_distribution = beer_info["bad_distr_beer"].values[0]
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
    interaction_good = np.array(user_good_distr) @ np.array(beer_good_distribution)
    interaction_bad = np.array(user_bad_distr) @ np.array(beer_bad_distribution)

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
        + beer_good_distribution  # 77 vals
        + beer_bad_distribution  # 77 vals
        + beer_one_hot_cat  # 13 vals
        + one_hot_month  # 12 vals
    )  # results in a total of 341 features + 1 label => 342 values


# we will transform the distributions in a latent space of dimension 5, so the big NN will only see 55 vals
