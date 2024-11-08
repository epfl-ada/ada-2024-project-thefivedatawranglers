import pandas as pd
import matplotlib.pyplot as plt

# these are some possibilities of what one could consider
# "word that only experienced beer consumers would use in there beer review"
exp_words0 = ["Ester"]

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
exp_words2 = [
    "Ester",
    "Diacetyl",
    "DMS",
    "resinous",
    "astringent",
    "Effervescent",
    "Tannic",
    "Brettanomyces",
    "lactic",
    "autolysis",
    "Krausen",
]
exp_words4 = [
    "Ester",
    "Diacetyl",
    "DMS",
    "resinous",
    "astringent",
    "Effervescent",
]


def load_data():
    """
    Loads data in pandas dataframes
    :return: these dataframes
    """
    df_rb_ratings = pd.read_csv(
        "C:/Users/nette/Dateistruktur/RWTH/Auslandssemester/ImSemester/CS-401 Applied Data Analysis/Project/P2/Git/ada-2024-project-thefivedatawranglers/src/data/RateBeer/ratings/ratings.csv"
    )
    df_ba_ratings = pd.read_csv(
        "C:/Users/nette/Dateistruktur/RWTH/Auslandssemester/ImSemester/CS-401 Applied Data Analysis/Project/P2/Git/ada-2024-project-thefivedatawranglers/src/data/BeerAdvocate/ratings/BA_ratings.csv"
    )
    return df_rb_ratings, df_ba_ratings


def get_experienced_users(df_ratings, exp_words):
    """
    Filters user_ids to those that are from experienced users, meaning that they've used
    one of the words in the given word list at least once
    :param df_ratings: the ratings we look at
    :param exp_words: the words we consider
    :return:
    """
    regex_pattern = "|".join(exp_words)  # create a regular expression to filter
    df_ratings_exp = df_ratings[
        df_ratings["text"].str.contains(regex_pattern, case=False, na=False)
    ]
    exp_user_ids = df_ratings_exp["user_id"].unique()
    return exp_user_ids


def split_by_experience(df_ratings, exp_user_ids):
    """
    Splits the dataframe in those ratings by experiences users and those by inexperienced
    :param df_ratings:
    :param exp_user_ids:
    :return:
    """
    # we don't need the text attribute in the further analysis and it is very big
    df_ratings_wo_text = df_ratings.drop(columns=["text"])
    # splitting the dataframe via the id list givem
    df_ratings_of_exp = df_ratings_wo_text[
        df_ratings_wo_text["user_id"].isin(exp_user_ids)
    ]
    df_ratings_of_inexp = df_ratings_wo_text[
        ~df_ratings_wo_text["user_id"].isin(exp_user_ids)
    ]
    return df_ratings_of_exp, df_ratings_of_inexp


def calculate_style_distribution(df_ratings_of_exp, df_ratings_of_inexp, top_n=25):
    """
    Calculates the empirical distribution of ratings over the style attribute for both datframes
    :param df_ratings_of_exp: dataframe of experienced users
    :param df_ratings_of_inexp: dataframe of inexperienced users
    :param top_n: the number of beer styles we want to look at
    :return: the plotting dataframe and the most_rates beer styles for other functions
    """
    style_counts_exp = df_ratings_of_exp["style"].value_counts()
    style_counts_inexp = df_ratings_of_inexp["style"].value_counts()

    # 25 most rated beer styles
    most_rated = (style_counts_exp + style_counts_inexp).nlargest(top_n).index

    style_counts_exp = style_counts_exp[most_rated]
    style_counts_inexp = style_counts_inexp[most_rated]

    # scale it to empirical distribution
    plot_df = pd.DataFrame(
        {
            "Experienced": style_counts_exp / style_counts_exp.sum(),
            "Inexperienced": style_counts_inexp / style_counts_inexp.sum(),
        }
    )
    return plot_df, most_rated


def plot_style_distribution(plot_df):
    """
    just plots the distribution
    :param plot_df: the plot df returned by calculate_style_distribution
    :return:
    """
    plot_df.plot(kind="bar", alpha=0.7, figsize=(15, 8))
    plt.xlabel("Style")
    plt.ylabel("Probability")
    plt.title(
        "Distribution of beer styles between experienced / non-experienced users (normalized)"
    )
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def plot_distribution_difference(plot_df):
    """
    Plots the difference in the distribution of ratings over beer styles between experienced / non-experienced users
    :param plot_df: plot_df: the plot df returned by calculate_style_distribution
    :return: Nothing
    """
    # calculate the difference between the two groups
    dif_plot_df = pd.DataFrame(
        {"Difference": plot_df["Experienced"] - plot_df["Inexperienced"]}
    )

    # plotting
    dif_plot_df.plot(kind="bar", alpha=0.7, figsize=(15, 8))
    plt.xlabel("Style")
    plt.ylabel("Probability Difference")
    plt.title(
        "Difference in beer style distribution between experienced and non-experienced users"
    )
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def plot_combined_distribution_and_rating_difference(plot_df, rating_diff_df):
    """
    Plots the difference in the distribution of ratings over beer styles between experienced / non-experienced users
    and also the average rating over beer styles between experienced and non-experienced users
    :param plot_df: plot_df: the plot df returned by calculate_style_distribution
    :param rating_diff_df: rating difference df as generated by calculate_rating_difference
    :return: Nothing
    """
    # calculate the difference between the two groups
    diff_plot_df = pd.DataFrame(
        {"Difference": plot_df["Experienced"] - plot_df["Inexperienced"]}
    )

    # create 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(15, 16), sharex=True)

    # plot for the difference in the count
    diff_plot_df.plot(kind="bar", alpha=0.7, ax=axes[0])
    axes[0].set_xlabel("Style")
    axes[0].set_ylabel("Probability Difference")
    axes[0].set_title(
        "Difference in beer style distribution between experienced and non-experienced users"
    )
    axes[0].tick_params(axis="x", rotation=90)

    # plot for the difference in the average rating
    rating_diff_df.plot(kind="bar", color="orange", alpha=0.7, ax=axes[1])
    axes[1].set_xlabel("Style")
    axes[1].set_ylabel("Rating Difference")
    axes[1].set_title(
        "Difference in average beer style rating between experienced and non-experienced users"
    )
    axes[1].tick_params(axis="x", rotation=90)

    plt.tight_layout()
    plt.show()


def calculate_rating_difference(df_ratings_of_exp, df_ratings_of_inexp, most_rated):
    """Calculate rating difference between experienced and non-experienced users over styles"""
    # average rating per beer style for experienced users
    avg_ratings_exp = (
        df_ratings_of_exp[df_ratings_of_exp["style"].isin(most_rated)]
        .groupby("style")["rating"]
        .mean()
    )

    # average rating per beer style for inexperienced users
    avg_ratings_inexp = (
        df_ratings_of_inexp[df_ratings_of_inexp["style"].isin(most_rated)]
        .groupby("style")["rating"]
        .mean()
    )

    # difference dataframe
    rating_diff_df = pd.DataFrame(
        {"Rating Difference": avg_ratings_exp - avg_ratings_inexp}
    )

    return rating_diff_df
