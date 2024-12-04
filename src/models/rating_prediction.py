import pandas as pd
from src.data.some_dataloader import *
from collections import Counter

df_ba_ratings, df_rb_ratings = load_rating_data(
    ba_path="../../data/BeerAdvocate/BA_ratings.csv",
    rb_path="../../data" "/RateBeer" "/RB_ratings.csv",
)


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


def count_word_frequencies(df, text_column, word_list):
    """
    Counts the frequencies of words from a given word list in a text column of a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the text column.
        text_column (str): The name of the column that contains the text data.
        word_list (list): A list of words whose frequencies should be counted.

    Returns:
        dict: A dictionary with words as keys and their frequencies as values.
    """
    # Combine all texts in the column into one large string
    all_text = " ".join(
        df[text_column].dropna()
    ).lower()  # Convert to lowercase for consistent counting

    # Tokenize the text (split into words)
    words = all_text.split()

    # Filter only the words from the provided word list
    filtered_words = [word for word in words if word in word_list]

    # Count the frequencies of the filtered words
    word_frequencies = Counter(filtered_words)

    # Return the result as a dictionary
    return dict(word_frequencies)


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

words = list(set(positive_words + negative_words + exp_words1))


def pre_stats(df, style, date):
    df = df[df["date"] < date]
    mean = df["rating"].mean()
    style_mean = df[df["style"] == style]["rating"].mean()
    num_ratings = len(df)


def is_experienced(user_id, exp_user_ids):
    return user_id in exp_user_ids


def get_top_styles(df_ratings, threshold):
    top_30_styles = df_ratings["style"].value_counts().head(threshold)
