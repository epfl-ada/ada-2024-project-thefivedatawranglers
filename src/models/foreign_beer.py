import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils.evaluation_utils import *
import pandas as pd
from plotly.subplots import make_subplots
from plotly import graph_objects as go
import plotly.express as px
from scipy.stats import t


southern_states = [
    "Alabama",
    "Arkansas",
    "Georgia",
    "Louisiana",
    "Mississippi",
    "North Carolina",
    "South Carolina",
    "Tennessee",
    "Texas",
    "Virginia",
    "Florida",
    "Kentucky",
    "Oklahoma",
]
northern_states = [
    "Connecticut",
    "Delaware",
    "Illinois",
    "Indiana",
    "Iowa",
    "Maine",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "New Hampshire",
    "New Jersey",
    "New York",
    "Ohio",
    "Pennsylvania",
    "Rhode Island",
    "Vermont",
    "Wisconsin",
]


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


def plot_location_ratings(df_sum_rat, cutoff=2000, interactive=True):
    """
    Plots the number of ratings per country, either as an interactive Plotly chart or a static Matplotlib chart.
    :param df_sum_rat: result of accumulate_us() (expects a DataFrame with a 'nbr_ratings' column)
    :param cutoff: Minimum number of ratings for a country to be included
    :param interactive: Boolean flag to indicate whether to use Plotly (True) or Matplotlib (False)
    :return: None
    """
    # ensure the data is in DataFrame format
    df_sum_rat = df_sum_rat.to_frame()
    df_sum_rat.insert(0, "location", df_sum_rat.index)
    df_sum_rat.reset_index(drop=True, inplace=True)
    df_sum_rat.sort_values(ascending=False, by="nbr_ratings", inplace=True)

    # filter by cutoff
    df_sum_rat_cutoff = df_sum_rat[df_sum_rat["nbr_ratings"] > cutoff]

    if interactive:  # if we create it for the website
        rows = 1
        cols = 2
        fig = make_subplots(
            rows=rows,
            cols=cols,
            specs=[[{"type": "xy"}, {"type": "choropleth"}]],
        )

        # creating bar chart (with color-blind-friendly color scheme)
        fig.add_trace(
            go.Bar(
                x=df_sum_rat_cutoff["location"],
                y=df_sum_rat_cutoff["nbr_ratings"],
                marker=dict(color=CB_color_cycle[0]),
            ),
            row=1,
            col=1,
        )
        fig.update_yaxes(
            title_text="Number of reviews in logarithmic scale",
            type="log",
            row=1,
            col=1,
        )

        # creating map chart
        fig.add_trace(
            go.Choropleth(
                locations=df_sum_rat["location"],
                locationmode="country names",
                z=df_sum_rat["nbr_ratings"],
                text="Number of reviews on linear scale",
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            title_text="Number of ratings per country",
        )

        fig.show()
    else:  # if we create it for the GitHub preview of jupyter notebooks
        # extract locations and corresponding ratings
        locations = df_sum_rat_cutoff["location"].values
        ratings = df_sum_rat_cutoff["nbr_ratings"].values

        # create bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(locations, ratings, color=CB_color_cycle[0])
        plt.yscale("log")
        plt.ylabel("Number of reviews (log scale)")
        plt.xlabel("Location")
        plt.title("Number of ratings per country")
        plt.xticks(rotation=90, ha="right")

        plt.tight_layout()
        plt.show()


def merge_users_and_ratings(df_ratings, df_users):
    """
    Joins user and ratings datasets
    :param df_ratings: rating dataset
    :param df_users: user dataset
    :return: joined df
    """
    # using user_name here because the id is ambiguous
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

    # convert input data to a DataFrame and prepare it for plotting
    df_plot = df_plot.to_frame()
    df_plot.insert(0, "location", df_plot.index)
    df_plot.reset_index(drop=True, inplace=True)
    df_plot.sort_values(ascending=False, by="rating", inplace=True)

    # define layout of subplots
    rows = 1
    cols = 2
    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "xy"}, {"type": "choropleth"}]],
    )

    # add bar chart
    fig.add_trace(
        go.Bar(
            x=df_plot["location"],
            y=df_plot["rating"],
            marker=dict(color=CB_color_cycle[0]),
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Average rating (in logarithmic scale)", type="log", row=1, col=1
    )

    # map chart
    fig.add_trace(
        go.Choropleth(
            locations=df_plot["location"],
            locationmode="country names",
            z=df_plot["rating"],
            text="Average rating (on linear scale)",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title_text="Average Rating Per Country",
    )

    fig.show()


def plot_bar_chart(df_plot, save=True, interactive=True):
    """
    plots the mean rating as a bar chart, either interactively (using plotly) or statically (using Matplotlib).
    The second option is give nas well as plotly plots are not visible in the GitHub preview.
    :param df_plot: result of avg_rating_by_location
    :param save: whether we want to save or show the plot
    :param interactive: whether to create an interactive plotly plot or a static matplotlib plot
    :return: Nothing (just plots)
    """

    df_plot = df_plot.to_frame()
    df_plot.insert(0, "location", df_plot.index)
    df_plot.reset_index(drop=True, inplace=True)
    df_plot.sort_values(ascending=False, by="rating", inplace=True)

    if interactive:
        # interactive plotly plot
        fig = go.Figure(
            data=[
                go.Bar(
                    x=df_plot["location"],
                    y=df_plot["rating"],
                    marker=dict(color=CB_color_cycle[0]),
                )
            ]
        )

        fig.update_yaxes(title_text="Average rating", type="log")
        fig.update_layout(
            title_text="Average Rating Per Country (Bar Chart)",
            xaxis=dict(tickangle=90),
        )

        if save:
            fig.write_html("src/plots/bar_chart_avg_rating.html")
        else:
            fig.show()

    else:
        # Static matplotlib plot
        plt.figure(figsize=(10, 6))
        plt.bar(
            df_plot["location"],
            df_plot["rating"],
            color=CB_color_cycle[0],
        )
        plt.xlabel("Location", fontsize=12)
        plt.ylabel("Average Rating", fontsize=12)
        plt.title("Average Rating Per Country", fontsize=14)
        plt.xticks(rotation=90)
        plt.tight_layout()

        if save:
            plt.savefig("src/plots/bar_chart_avg_rating.png")
        else:
            plt.show()


def plot_mean_rating_and_rating_count(
    df_plot, top50, log_scale=True, include_line=True
):
    """
    Almost the same function as plot_mean_rating_by_location but this time we also plot the rating count,
    so we can see, although we already filtered the countries, which bars are based on a lot of data. Also,
    we don't do it interactevily here (GitHub won't let us do this) We normally use a log scale here, because the US
    dominates the dataset and even after the filtering there are some countries with just around 2000 ratings.
    :param df_plot: the result of avg_rating_by_location
    :param top50: the second return val of filter_top_countries
    :param log_scale: boolean whether to plot the count in a log scale or linear
    :param include_line: boolean whether to include the line graph for the count of ratings :return: Nothing
    :returns None (just plots)
    """
    fig, ax1 = plt.subplots(
        figsize=(12, 4), dpi=100
    )  # need subplots for barchart and line chart showing number of ratings from that country

    # plotting the bar chart
    df_plot.plot(kind="bar", color=CB_color_cycle[0], ax=ax1)
    ax1.set_xlabel("Location")
    ax1.set_ylabel("Average grade")
    ax1.set_title("Average grade given from users from diff. locations")

    if include_line:
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
    :return: number of foreign beers, number of domestic beers and also both expressed as relative percentages
    """
    # split df in foreign and domestic/own beers
    foreign_beers = df_users_ratings_brew[df_users_ratings_brew["foreign"]]
    own_beers = df_users_ratings_brew[~df_users_ratings_brew["foreign"]]
    # the total amount of ratings
    total_ratings = len(df_users_ratings_brew)
    # ratio of foreign beers in the rating
    foreign_percentage = ratio_to_percentage(len(foreign_beers) / total_ratings)
    # ratio of domestic beers in the rating
    own_percentage = ratio_to_percentage(len(own_beers) / total_ratings)
    return len(foreign_beers), len(own_beers), foreign_percentage, own_percentage


def plot_foreign_beer_stats(num_foreign, num_domestic):
    """
    Plots a horizontal bar plot showing the proportion of domestic vs foreign beers in ratings.
    This was first created for the website for a more visually pleasing explanation of the relation
    than just showing numbers, but then I thought it might be good for the notebook as well.
    :param num_foreign: number of domestic beers
    :param num_domestic: number of foreign beers
    :return: None (just plots)
    """
    total = num_foreign + num_domestic
    percent_foreign = (num_foreign / total) * 100
    percent_domestic = (num_domestic / total) * 100

    fig, ax = plt.subplots(figsize=(12, 2))

    # horizontal stacked bar plot; one horizontal bar for the left one for the right
    ax.barh(
        0,
        num_domestic,
        color=CB_color_cycle[0],
        label=f"Domestic ({percent_domestic:.1f}%)",
        height=0.5,
        alpha=alpha_val,
    )
    ax.barh(
        0,
        num_foreign,
        left=num_domestic,
        color=CB_color_cycle[1],
        label=f"Foreign ({percent_foreign:.1f}%)",
        height=0.5,
        alpha=alpha_val,
    )

    ax.set_yticks(
        []
    )  # removing y-axis (it's just one stacked bar that we show, so we don't need it)
    ax.set_xlim(0, total)  # scale based on the parameters

    # add a line to show where the 50% mark is
    ax.axvline(
        x=total / 2, color="black", linestyle="--", linewidth=0.8, label="50% Mark"
    )

    # add a small line where the two bars meet
    ax.axvline(x=num_domestic, color="gray", linestyle="--", linewidth=0.8)

    # define offsets for text positions; 2% of total width
    offset = 0.02 * total

    # add labels at the ends of each bar that show the percentage
    ax.text(
        offset,  # at the left of the bar (just not completely left, that looked bad)
        0,
        f"{percent_domestic:.1f}%",
        ha="left",
        va="center",
        fontsize=16,
        color="white",
        fontweight="bold",
    )
    ax.text(
        total
        - offset,  # at the right of the bar (just not completely right, that looked bad)
        0,
        f"{percent_foreign:.1f}%",
        ha="right",
        va="center",
        fontsize=16,
        color="white",
        fontweight="bold",
    )

    ax.set_title("Domestic vs. Foreign", fontsize=14)

    # Remove normal x-axis
    ax.set_xticks([])

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=14)

    plt.tight_layout()
    plt.show()


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
    :return: Nothing (plots stuff)
    """
    fig, ax1 = plt.subplots(figsize=(12, 4), dpi=100)

    df_grouped_counts.plot(kind="bar", stacked=True, ax=ax1)
    plt.xlabel("User Location")
    plt.ylabel("Proportion of Ratings")
    plt.title("Proportion of Ratings by User Location (Foreign vs. Non-Foreign Beers)")
    plt.legend(title="Foreign Beer", labels=["Non-Foreign", "Foreign"])
    plt.show()


def change_flag(df_users_ratings_brew):
    """
    Inverts the foreign flag and calls it is_domestic.
    This is basically a compatibility wrapper to make the transition from Sven's to David's code easier.
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
    Groups by user_location and is_domestic. Then calculates mean, std, and count for both foreign and domestic beers.
    :param df_users_ratings_brew: result of change_flag
    :return: the grouped df with statistics.
    """
    return (
        df_users_ratings_brew.groupby(["user_location", "is_domestic"])["rating"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_rating", "std": "std_dev", "count": "n"})
    )


def pivot_average_scores(df_average_scores):
    """
    Creates separate columns for domestic and foreign ratings along with std and count.
    :param df_average_scores: the result of avg_scores_domestic_foreign_with_stats
    :return: The new dataframe
    """
    df_pivot = df_average_scores.pivot(
        index="user_location",
        columns="is_domestic",
        values=["avg_rating", "std_dev", "n"],
    )
    df_pivot.columns = [
        "Foreign_Avg",
        "Domestic_Avg",
        "Foreign_Std",
        "Domestic_Std",
        "Foreign_n",
        "Domestic_n",
    ]
    return df_pivot


def calculate_score_difference(df_pivot):
    """
    Computes the difference between domestic and foreign ratings and calculates confidence intervals.
    :param df_pivot: The result of pivot_average_scores_with_stats.
    :return: The same df but with the new column difference and CI.
    """
    # Calculate difference in average ratings
    df_pivot["difference"] = df_pivot["Domestic_Avg"] - df_pivot["Foreign_Avg"]

    # Calculate standard error for the difference
    df_pivot["se_diff"] = np.sqrt(
        (df_pivot["Domestic_Std"] ** 2) / df_pivot["Domestic_n"]
        + (df_pivot["Foreign_Std"] ** 2) / df_pivot["Foreign_n"]
    )

    # Calculate 95% confidence intervals (using t-distribution)
    df_pivot["ci_95"] = df_pivot["se_diff"] * t.ppf(
        0.975, df=(df_pivot["Domestic_n"] + df_pivot["Foreign_n"] - 2)
    )

    return df_pivot.sort_values(by="difference", ascending=False)


def plot_score_difference(df_diff):
    """
    Plots the difference between average domestic and foreign ratings grouped over
     user location with confidence intervals.
    :param df_diff: The result of calculate_score_difference_with_ci
    :return: Nothing (plots stuff)
    """
    plt.figure(figsize=(20, 10))

    plt.bar(
        df_diff.index,
        df_diff["difference"],
        yerr=df_diff["ci_95"],
        capsize=5,
        alpha=alpha_val,
        color=CB_color_cycle[0],
        label="Difference",
    )

    plt.title(
        "Difference Between Average Score for Domestic - Foreign Beers",
        fontsize=24,
    )
    plt.xlabel("User Location", fontsize=24)
    plt.ylabel("Difference in Average Rating", fontsize=24)
    plt.xticks(rotation=90, fontsize=20)
    plt.yticks(fontsize=16)

    # there was a gap between the bars and the end of the coordinate system that didn't look well on th website,
    # so I enforce that they start/stop with the coordinate system
    plt.xlim([-0.5, len(df_diff.index) - 0.5])

    plt.legend(fontsize=20)
    plt.tight_layout()

    plt.show()


def plot_average_ratings_heatmap(df):
    """
    An exploratory function to see whether user from some specific country like beer from some specific other country
    a lot
    :param df: the ratings joined with user joined with breweries
    :return: Nothing (plots stuff)
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


def best_and_worst_combinations(df, threshold=1000):
    """
    A method used to find the pair of user-country and brewery-country with the lowest and highest avg score
    :param df: the joined (ratings-user-brewery) dataframe
    :param threshold: We use a threshold because if not we get for both best and worst a combination that is based on
    just one rating. So we say for a combination to be legitimate there need to be at least 1000(/threshold) many
    ratings in that combination.
    :return: Nothing (prints stuff)
    """
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


def filter_to_us_users(df_users):
    """
    This filters the user dataframe to only those entries that come from a US state
    :param df_users: the user dataframe in question
    :return: the filtered dataframe
    """
    df_us_only = df_users.copy()
    mask = df_us_only["location"].str.contains("United States, ", na=False)
    return df_us_only[mask]


def prepare_datasets(df_rb_users, df_ba_users, df_rb_ratings, df_ba_ratings):
    """
    This is the first preparation step for the "inner-US" analysis.
    It joins the users with the ratings datasets and performs some structural changes.
    :param df_rb_users: the RateBeer users dataset
    :param df_ba_users: the BeerAdvocate users dataset
    :param df_rb_ratings: the RateBeer ratings dataset (presumably without the text column)
    :param df_ba_ratings: the BeerAdvocate ratings dataset (presumably without the text column)
    :return: 2 joined and changed dataframes
    """
    # in this analysis we're only interested in users from th US
    df_rb_users_us_only = filter_to_us_users(df_rb_users)
    df_ba_users_us_only = filter_to_us_users(df_ba_users)

    # we drop the nbr_reviews column as it is not present in the RateBeer dataset
    df_ba_users_us_only.drop(columns=["nbr_reviews"], inplace=True)
    # we add a column to specify from which dataset an entry comes
    # I don't think we will actually need this, but it also doesn't hurt
    df_ba_users_us_only["dataset"] = "BeerAdvocate"
    df_rb_users_us_only["dataset"] = "RateBeer"

    # joining with the ratings datasets
    df_ba_users_ratings_us_only = merge_users_and_ratings(
        df_ba_ratings, df_ba_users_us_only
    )
    df_rb_users_ratings_us_only = merge_users_and_ratings(
        df_rb_ratings, df_rb_users_us_only
    )

    # some stats never hurt
    print(
        "Number of ratings from US from BeerAdvocate:", len(df_ba_users_ratings_us_only)
    )
    print("Number of ratings from US from RateBeer:", len(df_rb_users_ratings_us_only))

    return df_rb_users_ratings_us_only, df_ba_users_ratings_us_only


def merge_with_brewery(
    df_rb_users_ratings_us_only, df_ba_users_ratings_us_only, df_rb_brew, df_ba_brew
):
    """
    Merges the already joined (ratings-users) dataset further with the brewery dataset.
    Then concatenates both to create the one dataframe containing all the US-data
     from both BeerAdvocate and BeerAdvocate
    :param df_rb_users_ratings_us_only: first return value of prepare_datasets
    :param df_ba_users_ratings_us_only: second return value of prepare_datasets
    :param df_rb_brew: the RateBeer brewery dataset
    :param df_ba_brew: the BeerAdvocate brewery dataset
    :return: the merged dataframe
    """

    # joining with brewery data
    df_rb_users_ratings_brew_us_only = merge_ratings_with_breweries(
        df_rb_users_ratings_us_only, df_rb_brew
    )
    df_ba_users_ratings_brew_us_only = merge_ratings_with_breweries(
        df_ba_users_ratings_us_only, df_ba_brew
    )

    # concat both dfs
    df_us_only = pd.concat(
        [df_rb_users_ratings_brew_us_only, df_ba_users_ratings_brew_us_only],
        ignore_index=True,
    )
    # stats
    print("Number of ratings from US:", len(df_us_only))

    return df_us_only


def avg_ratings_us(df_us_only):
    """
    Prints the average rating given by US-citizens to beer from the US as well as the average
    rating given by US-citizens to beer that is not from the US
    :param df_us_only: the return val of merge_with_brewery
    :return: the same df given as an input but with the additional flag "is_us_beer"
    """
    mask = df_us_only["brewery_location"].str.contains("United States, ", na=False)
    # the foreign column now only tells, whether the beer comes from the exact same US state or not
    # maybe this will be interesting for future analyses, but I will add a new column called "is_us_beer"
    # containing the information whether the beer comes from the US
    df_us_only["is_us_beer"] = mask

    avg_ratings = df_us_only.groupby("is_us_beer")["rating"].mean()
    print("Avg rating for US beer:", avg_ratings[True])
    print("Avg rating for non-US beer:", avg_ratings[False])

    return df_us_only


def avg_ratings_per_location_us(df_us_only):
    """
    Computes the average ratings for both foreign and US beer for all the US states individually.
    Then it calculates the difference between both in order to later plot these differences.
    :param df_us_only: the return val of avg_ratings_us
    :return: the average rating differences for both foreign and US beer for all the US states
    """
    # we rename the user location to contain just the states name, as every user location is in the US by now
    df_us_only["user_location"] = df_us_only["user_location"].str.replace(
        "United States, ", "", regex=False
    )
    # compute the average ratings for both foreign and US beer for all the US states
    avg_ratings_per_location = (
        df_us_only.groupby(["user_location", "is_us_beer"])["rating"].mean().unstack()
    )
    # calculate the differences
    avg_ratings_per_location["Difference"] = (
        avg_ratings_per_location[True] - avg_ratings_per_location[False]
    )

    return avg_ratings_per_location


def plot_avg_ratings_per_location(avg_ratings_per_location):
    """
    Plots the results of avg_ratings_per_location_us
    :param avg_ratings_per_location: return val of avg_ratings_per_location_us
    :return: Nothing (plots stuff)
    """
    plt.figure(figsize=(10, 6))
    plt.bar(
        avg_ratings_per_location.index,
        avg_ratings_per_location["Difference"],
        color="purple",
    )
    plt.xlabel("User Location")
    plt.ylabel("Difference in Average Rating (US Beer - Non-US Beer)")
    plt.title("Difference in Average Ratings by User Location (US Beer vs Non-US Beer)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def north_south_avg(df_us_only):
    """
    Creates a table for the avg ratings for northern US states and southern US states (and others that are neither)
    for both US beer and Non-US beer
    :param df_us_only: the return val of avg_ratings_us
    :return: the table containing those averages
    """
    # add a column "region" that can be "South" for a southern state where the user comes from
    # "North" for a northern state where the user comes from, or "Other" otherwise
    df_us_only["region"] = df_us_only["user_location"].apply(
        lambda x: (
            "South"
            if x in southern_states
            else ("North" if x in northern_states else "Other")
        )
    )

    # create the average rating for each group
    average_ratings = (
        df_us_only.groupby(["region", "is_us_beer"])["rating"].mean().unstack()
    )
    # until now the columns are called True for is_us_beer=True and False for is_us_beer=False
    # renaming that for better readability
    average_ratings = average_ratings.rename(
        columns={False: "Foreign Beer", True: "US Beer"}
    )

    return average_ratings


def plot_north_south_diffs(avg_ratings_df):
    """

    :param avg_ratings_df: the average ratings dataframe returned by north_south_avg
    :return: None (just plots)
    """
    plt.style.use("ggplot")

    # use the colorblind colors (but just the first two)
    ax = avg_ratings_df.plot(
        kind="bar", figsize=(10, 6), edgecolor="black", color=CB_color_cycle[:2]
    )

    ax.set_title("Average Beer Ratings by Region and Category", fontsize=16)
    ax.set_xlabel("Region", fontsize=16)
    ax.set_ylabel("Average Rating", fontsize=16)

    # customize y-axis to show only values above 3.0 (all the values are larger anyway)
    ax.set_ylim(3.0, None)
    ax.legend(
        title="Beer Type", fontsize=10, title_fontsize=12
    )  # should be 14 generally, but it was overlapping than...

    # add value annotations to each bar
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=16)

    plt.xticks(rotation=0, fontsize=16)
    plt.yticks(fontsize=16)

    plt.tight_layout()
    plt.show()


def filter_usa_duplicates(df_rb, df_ba, cols):
    """
    Here we search for duplicates in the usa ratings in both datasets and remove them from the RateBeer dataset as part
    of the data cleaning process before we to the USA-patriotism analysis
    :param df_rb: the first return val of prepare_datasets
    :param df_ba: the second return val of prepare_datasets
    :param cols: the columns we match in order to say that a duplicate is found
    :return: the cleaned RateBeer dataset
    """
    # we don't want to match rows with NaN values. Especially if there is a NaN-value in the text col, the probability
    # that we find a matching one that is not the same rating is quite high
    df_rb_cleaned = df_rb.dropna(subset=cols)
    df_ba_cleaned = df_ba.dropna(subset=cols)

    # find rows that are matching in the columns "brewery_name", "style", "abv" and "text" (/cols)
    matching_rows = df_rb_cleaned.merge(df_ba_cleaned, on=cols, how="inner")

    # how much did we remove
    print("Number of duplicates: ", len(matching_rows))

    result = df_rb[~df_rb.index.isin(matching_rows.index)]

    return result


def plot_avg_ratings_map(avg_ratings_per_location, save=False):
    """
    Plots an interactive map of the USA showing the difference in average ratings by state.
    The color intensity reflects the difference value, and hovering over a state displays its name and value.

    :param avg_ratings_per_location: DataFrame containing states and the difference in ratings.
    :param save: whether to save the plot as a html for the website
    :return: Nothing (plots an interactive map).
    """

    avg_ratings_per_location = avg_ratings_per_location.reset_index()

    avg_ratings_per_location.rename(
        columns={"user_location": "State", "Difference": "Rating Difference"},
        inplace=True,
    )

    avg_ratings_per_location["State"] = avg_ratings_per_location["State"].map(
        US_STATES_CODES
    )

    print(avg_ratings_per_location.head())

    # create map
    fig = px.choropleth(
        avg_ratings_per_location,
        locations="State",
        locationmode="USA-states",
        color="Rating Difference",  # the color shall express the value
        color_continuous_scale="Viridis",  # on the website we use the Viridis scale for every map - it looked best
        scope="usa",  # Restrict to USA
        title="Difference in Average Ratings by State (US Beer vs Non-US Beer)",
        labels={"Rating Difference": "Diff (US - Non-US)"},  # Label for hover tooltip
    )

    # Customize hover data
    fig.update_traces(hovertemplate="<b>%{location}</b><br>Difference: %{z:.2f}")

    # Show the plot
    fig.show()

    if save:
        fig.write_html("USA_plot_new.html")
