import seaborn as sns
import matplotlib.pyplot as plt
from src.utils.evaluation_utils import *


def calculate_ratings_by_location(df_users):
    """
    Calculates the number of ratings by each location in the users dataframe.
    :param df_users: the users df
    :return: the summed df
    """
    df_grp_loc = df_users.groupby("location")
    df_sum_rat = df_grp_loc["nbr_ratings"].sum().sort_values(ascending=False)
    return df_sum_rat


def accumulate_us(df_sum_rat):
    """
    Accumulate all the US states to one entry.
    :param df_sum_rat: the result of calculate_ratings_by_location()
    :return: the accumulated df
    """
    df_sum_rat_us = df_sum_rat[df_sum_rat.index.str.contains("United States, ")]
    df_sum_rat_foreign = df_sum_rat[~df_sum_rat.index.str.contains("United States, ")]
    df_sum_rat_foreign.loc["United States"] = df_sum_rat_us.sum()
    return df_sum_rat_foreign


def cutoff_and_sort(df_sum_rat_foreign, cutoff=2000):
    """
    Reduces to only those countries with more than 2000(/cutoff) many ratings.
    Then it sorts the countries by the number of ratings coming from these countries.
    :param df_sum_rat_foreign: result of accumulate_us()
    :param cutoff: the cutoff which countries should still be included
    :return: the resulting df
    """
    df_sum_rat_cutoff = df_sum_rat_foreign[df_sum_rat_foreign > cutoff]  # cutoff
    return df_sum_rat_cutoff.sort_values(ascending=False)  # sort


def plot_location_ratings(df_sum_rat_cutoff):
    """
    Plots the number of ratings from the remaining locations.
    :param df_sum_rat_cutoff: result of cutoff_and_sort
    :return: Nothing
    """
    df_sum_rat_cutoff.plot(kind="bar", color=colors[: len(df_sum_rat_cutoff)])
    plt.xlabel("Location")
    plt.ylabel("Number of ratings")
    plt.title("Number of ratings from users from different locations")
    plt.show()


def merge_users_and_ratings(df_ratings, df_users):
    """
    Joins user and ratings datasets
    :param df_ratings: rating dataset
    :param df_users: user dataset
    :return: joined df
    """
    return df_ratings.merge(df_users, on=["user_name"], how="inner")


def accumulate_us2(df_users, col_name):
    """
    Kind of code duplication tbh. Again we accumulate the US states.
    It was nicer before we split the code into python files and jupyter files, but this way was
    the fastest and maybe also the more readable way (although it hurts to do this twice).
    :param df_users: the users df
    :param col_name: the name of the location column (location or brewery_location)
    :return: accumulated df
    """
    df_users_us = df_users.copy()
    mask = df_users_us[col_name].str.contains("United States, ", na=False)
    df_users_us.loc[mask, col_name] = "United States"
    return df_users_us


def filter_top_countries(df_users_ratings, top_n=50):
    """
    Filters the joined dataframe to only the top 50(/top_n) countries ordered by number of ratings from this country.
    :param df_users_ratings: the joined df
    :param top_n: the threshold
    :return: the filtered df
    """
    top_50 = df_users_ratings["location"].value_counts().head(top_n)
    top_countries = top_50.index
    # we return top_50 as well because we need it later in the plotting with counts
    return df_users_ratings[df_users_ratings["location"].isin(top_countries)], top_50


def avg_rating_by_location(df_users_ratings):
    """
    Calculates the average rating for every country,
    :param df_users_ratings: first return val of filter_top_countries
    :return: The grouped, averaged and sorted df
    """
    return (
        df_users_ratings.groupby("location")["rating"]
        .mean()
        .sort_values(ascending=False)
    )


def plot_mean_rating_by_location(df_plot):
    """
    Plots the mean rating over all remaining countries.
    :param df_plot: result of avg_rating_by_location
    :return: Nothing
    """
    df_plot.plot(kind="bar", color=colors[: len(df_plot)])
    plt.xlabel("Location")
    plt.ylabel("Average rating")
    plt.title("Average rating given by users from different locations")
    plt.show()


def plot_mean_rating_and_rating_count(df_plot, top50, log_scale=True):
    """
    Almost the same function as plot_mean_rating_by_location but this time we also plot the rating count, so we can see,
    although we already filtered the countries, which bars are based on a lot of data. We normally use a log scale here,
    because the US dominates the dataset and even after the filtering there are some countries with just around 2000
    ratings.
    :param df_plot: the result of avg_rating_by_location
    :param top50: the second return val of filter_top_countries
    :param log_scale: boolean whether to plot the count in a log scale or linear
    :return: Nothing
    """
    fig, ax1 = (
        plt.subplots()
    )  # need subplots for barchart and line chart showing number of ratings from that country
    # plotting the bar chart
    df_plot.plot(kind="bar", color=colors[: len(df_plot)], ax=ax1)
    ax1.set_xlabel("Location")
    ax1.set_ylabel("Average grade")
    ax1.set_title("Average grade given from users from diff. locations")
    # plotting the line
    ax2 = ax1.twinx()  # second axis for the number of ratings from that country
    top50_counts = top50[df_plot.index]  # bring top50 in right order
    ax2.plot(
        df_plot.index,
        top50_counts,
        color="red",
        marker="o",
        linestyle="-",
        label="Count",
    )
    ax2.set_ylabel("Count of Ratings")

    if log_scale:
        ax2.set_yscale("log")

    ax2.legend(loc="upper right")
    plt.show()


def merge_ratings_with_breweries(df_users_ratings, df_brew_us):
    """
    Join the df that contains the user and rating information with the brewery df.
    :param df_users_ratings: the joined df of ratings and users
    :param df_brew_us: the brewery df
    :return: the joined df resulting from the two given dfs
    """
    df_merged = df_users_ratings.merge(
        df_brew_us[["brewery_id", "brewery_location"]], on="brewery_id", how="inner"
    )
    # rename the two location columns, so it's clear which one is which
    df_merged.rename(columns={"location": "user_location"}, inplace=True)
    df_merged["foreign"] = df_merged["user_location"] != df_merged["brewery_location"]
    return df_merged


def foreign_beer_stats(df_users_ratings_brew):
    """
    Calculates some interesting statistics about the foreign/domestic beers
    :param df_users_ratings_brew: the result of merge_ratings_with_breweries
    :return:
    """
    # split df in foreign and domestic/own beers
    foreign_beers = df_users_ratings_brew[df_users_ratings_brew["foreign"]]
    own_beers = df_users_ratings_brew[~df_users_ratings_brew["foreign"]]
    # the total amount of ratings
    total_ratings = len(df_users_ratings_brew)
    # ratio of foreign beers in the rating
    foreign_percentage = (len(foreign_beers) / total_ratings) * 100
    # ratio of domestic beers in the rating
    own_percentage = (len(own_beers) / total_ratings) * 100
    return len(foreign_beers), len(own_beers), foreign_percentage, own_percentage


def grouped_counts(df_users_ratings_brew):
    """
    Creating the dataframe for the plot_foreign_vs_own_beer_counts method.
    :param df_users_ratings_brew: the result of merge_ratings_with_breweries.
    :return: the df for plotting
    """
    # first we group by the location of the user (that's going to be our x-axis), and then by the foreign attribute
    # (this will determine the color change in the stacked bar). We count the number of entries with the same attribute
    # combination. Then we use the unstack command to format the results.
    df_grouped_counts = (
        df_users_ratings_brew.groupby(["user_location", "foreign"])
        .size()
        .unstack(fill_value=0)
    )
    df_grouped_counts = df_grouped_counts.div(
        df_grouped_counts.sum(axis=1), axis=0
    )  # scaling the counts so all the bars are of the same size (if you don't the US bar totally dominates)
    return df_grouped_counts


def plot_foreign_vs_own_beer_counts(df_grouped_counts):
    """
    Plots the number of ratings for foreign/domestic beers over the countries.
    :param df_grouped_counts: result of grouped_counts
    :return: Nothing
    """
    df_grouped_counts.plot(kind="bar", stacked=True)
    plt.xlabel("User Location")
    plt.ylabel("Proportion of Ratings")
    plt.title("Proportion of Ratings by User Location (Foreign vs. Non-Foreign Beers)")
    plt.legend(title="Foreign Beer", labels=["Non-Foreign", "Foreign"])
    plt.show()


def change_flag(df_users_ratings_brew):
    """
    Inverts the foreign flag and calls it is_domestic.
    This is basically a compatibility method to make the transition from Sven's to David's code easier.
    :param df_users_ratings_brew: result of merge_ratings_with_breweries
    :return: The changed df.
    """
    df_users_ratings_brew["foreign"] = ~df_users_ratings_brew["foreign"]  # inverting
    df_users_ratings_brew = df_users_ratings_brew.rename(
        columns={"foreign": "is_domestic"}
    )  # renaming

    return df_users_ratings_brew


def avg_scores_domestic_foreign(df_users_ratings_brew):
    """
    Groups by user_location and is_domestic. Then calculates mean rating for both foreign and domestic beers.
    :param df_users_ratings_brew: result of change_flag
    :return: the grouped df.
    """
    return (
        df_users_ratings_brew.groupby(["user_location", "is_domestic"])["rating"]
        .mean()
        .reset_index()
    )


def pivot_average_scores(df_average_scores):
    """
    Creates separate columns for domestic and foreign ratings.
    :param df_average_scores: the result of calculate_average_scores_domestic_vs_foreign
    :return: The new dataframe
    """
    df_pivot = df_average_scores.pivot(
        index="user_location", columns="is_domestic", values="rating"
    )
    df_pivot.columns = [
        "Foreign",
        "Domestic",
    ]
    return df_pivot


def calculate_score_difference(df_pivot):
    """
    Computes the difference between domestic and foreign ratings.
    :param df_pivot: The result of pivot_average_scores.
    :return: The same df but with the new column difference.
    """
    df_pivot["difference"] = (
        df_pivot["Domestic"] - df_pivot["Foreign"]
    )  # creating difference
    return df_pivot.sort_values(by="difference", ascending=False)


def plot_score_difference(df_diff):
    """
    Plots the difference between average domestic and foreign ratings grouped over user location.
    :param df_diff: The result of calculate_score_difference
    :return: Nothing
    """
    df_diff["difference"].plot(
        kind="bar",
        title="Difference Between Average Score for Domestic - Foreign Beers",
        stacked=False,
        figsize=(20, 10),
    )
    plt.xlabel("User Location")
    plt.ylabel("Difference in Average Rating")
    plt.show()


def plot_average_ratings_heatmap(df):
    """
    An exploratory function to see whether user from some specific country like beer from some specific other country
    a lot
    :param df: the ratings joined with user joined with breweries
    :return: Nothing
    """

    # group by the location where the user comes from as well as the location the brewery is located
    # then compute the mean rating for the combination
    pivot_table = (
        df.groupby(["user_location", "brewery_location"])["rating"].mean().unstack()
    )

    # plotting the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot_table,
        annot=False,
        cbar_kws={"label": "Average Rating"},
    )
    plt.title("Average rating based on user and brewery location")
    plt.xlabel("Brewery Location")
    plt.ylabel("User Location")
    plt.show()


def find_best_and_worst_rating_combinations(df, threshold=1000):
    # group by both locations, calculating both the number of ratings in the combination and the avg rating
    average_ratings = (
        df.groupby(["user_location", "brewery_location"])
        .agg(avg_rating=("rating", "mean"), count_ratings=("rating", "size"))
        .reset_index()
    )

    # for a combination to be valid we demand that there are at least 1000(/threshold)
    # many ratings from that combination
    filtered_ratings = average_ratings[average_ratings["count_ratings"] >= threshold]

    # finding the pair with highest and lowest avg rating
    worst_rating = filtered_ratings.loc[filtered_ratings["avg_rating"].idxmin()]
    best_rating = filtered_ratings.loc[filtered_ratings["avg_rating"].idxmax()]

    # print statistics
    print("Combination with the worst average rating::")
    print(worst_rating)

    print("\nCombination with the best average rating:")
    print(best_rating)
