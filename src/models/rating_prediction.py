import pandas as pd
from collections import Counter

from src.data.some_dataloader import *
from src.models.distance_analysis import (
    haversine_distance,
    retrieve_location_data,
    calculate_distances,
    join_users_breweries_ratings,
)


df_ba_ratings, df_rb_ratings = load_rating_data(
    ba_path="../../data/BeerAdvocate/BA_ratings.csv",
    rb_path="../../data" "/RateBeer" "/RB_ratings.csv",
)

df_ba_users, df_rb_users = load_user_data()

df_brewery_ba = load_brewery_data(brewery_path="data/BeerAdvocate/breweries.csv")
df_brewery_rb = load_brewery_data(brewery_path="data/RateBeer/breweries.csv")

joined_ba_df = join_users_breweries_ratings(
    df_ba_users, df_brewery_ba, df_ba_ratings, ratebeer=False
)
joined_rb_df = join_users_breweries_ratings(
    df_rb_users, df_brewery_rb, df_rb_ratings, ratebeer=True
)

df_location = retrieve_location_data(joined_ba_df, joined_rb_df)

joined_ba_df = calculate_distances(joined_ba_df, df_location)
joined_rb_df = calculate_distances(joined_rb_df, df_location)


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
    "overly carbonated",
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

# Find the 3 most frequent beer styles
top_styles = df_ba_ratings["style"].value_counts().nlargest(3).index.tolist()


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


words = list(set(positive_words + negative_words + exp_words1))


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


def get_features(rating_idx, exp_user_ids, rate_beer=False):
    # setting variables depending on with which dataset we are working
    if rate_beer:
        df_ratings = df_rb_ratings
        joined_df = joined_rb_df
    else:
        df_ratings = df_ba_ratings
        joined_df = joined_ba_df

    # things about this specific rating
    user_id = df_ratings[rating_idx]["user_id"]
    style = df_ratings[rating_idx]["style"]
    style_cat = map_to_category(style)
    timestamp_date = df_ratings[rating_idx]["date"]
    beer_id = df_ratings[rating_idx]["beer_id"]

    # calculate the word freq distributions for good and bad ratings of this user
    good_ratings_df, bad_ratings_df = split_ratings_by_threshold(
        df_ratings, user_id, upper_threshold=3.8, lower_threshold=2.7
    )
    good_distr = count_word_frequencies(good_ratings_df, word_list=words)
    bad_distr = count_word_frequencies(bad_ratings_df, word_list=words)
    # normalizing the distributions
    good_total = sum(good_distr.values())
    bad_total = sum(bad_distr.values())
    good_distr = {word: freq / good_distr for word, freq in good_distr.items()}
    bad_distr = {word: freq / bad_distr for word, freq in bad_distr.items()}

    df_ratings_cat = categorize_beer_styles(df_ratings)

    # this calculates some statistics for this user but only for the ratings he did before doing this rating now
    user_mean, user_style_mean, user_num_ratings = pre_stats(
        df_ratings, style, timestamp_date
    )

    is_exp = is_experienced(user_id, exp_user_ids)

    df_ratings_this_beer = df_ratings[df_ratings["beer_id"] == beer_id]
    beer_mean = df_ratings_this_beer["rating"].mean()

    distance = joined_df[joined_df["ratings_idx"] == rating_idx][
        "distance_user_brewery"
    ]
