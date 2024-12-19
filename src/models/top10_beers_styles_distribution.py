import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch import unique
from src.utils.evaluation_utils import *
from matplotlib.colors import to_rgba


def top10beer_styles_ratings(
    df_ratings, df_nb_ratings, df_name, experience_threshold=20
):
    # Selecting the wanted columns and creating new df for ratings:
    filtered_ratings_df = pd.DataFrame(
        {
            "user_id": df_ratings["user_id"],
            "user_name": df_ratings["user_name"],
            "ratings": df_ratings["rating"],
            "beer_id": df_ratings["beer_id"],
            "beer_style": df_ratings["style"],
        }
    )

    # And new df for the reviewers to have the number of given ratings per reviewer
    users_df = pd.DataFrame(
        {
            "nb_ratings": df_nb_ratings["nbr_ratings"],
            "user_id": df_nb_ratings["user_id"],
        }
    )

    # Merging the ratings with their respective BA or RB users_df via the 'user_id' column
    filtered_ratings_df = pd.merge(users_df, filtered_ratings_df, on="user_id")

    # Classifying by the number of reviews given per beer style
    valuecount = pd.DataFrame(
        filtered_ratings_df["beer_style"].value_counts().reset_index()
    )
    valuecount.columns = ["beer_style", "count"]

    # Saving the 10 most reviewed beer styles
    top_10_styles = valuecount.head(10)

    # Selecting the rows from the BA or RB ratings that match with the Top_10 BA or RB respectively
    top10_ratings_df = filtered_ratings_df[
        filtered_ratings_df["beer_style"].isin(top_10_styles["beer_style"])
    ]

    # Sharing the Top10_ratings between experienced and new reviewers. The experience_threshold is used as separation

    top10_ratings_df.insert(5, "Experience", "New")

    top10_ratings_df.loc[
        top10_ratings_df["nb_ratings"] >= experience_threshold, "Experience"
    ] = "Experienced"

    top10_ratings_copy_df = top10_ratings_df.copy(deep=True)
    top10_ratings_copy_df["Experience"] = "All"
    top10_ratings_df = pd.concat([top10_ratings_df, top10_ratings_copy_df])

    # adjust alpha value in the sns way
    adjusted_palette = [to_rgba(color, alpha=alpha_val) for color in CB_color_cycle[:3]]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax = sns.boxplot(
        x="beer_style",
        y="ratings",
        data=top10_ratings_df,
        hue="Experience",
        showfliers=False,
        palette=adjusted_palette,
    )
    plt.xticks(rotation=90)
    ax.set_title(f"Top 10 Beer Style Ratings Distribution {df_name}")
    ax.set_xlabel("Beer Style")
    ax.set_ylabel("Ratings")

    # Finding unique reviewers for statistics
    experienced = (
        top10_ratings_df[top10_ratings_df["Experience"] == "Experienced"]
        .user_id.unique()
        .size
    )
    new = (
        top10_ratings_df[top10_ratings_df["Experience"] == "New"].user_id.unique().size
    )
    total = experienced + new

    # Changing to percentages
    percentage_experienced = (experienced / total) * 100
    percentage_new = (new / total) * 100
    percentage_total = (total / total) * 100

    # Relevant statistics
    print(f"{df_name}")
    print(
        "Percentage of Experienced reviewers (>= 50 given reviews):",
        percentage_experienced,
    )
    print("Percentage of New reviewers (<50 given reviews):", percentage_new)
    print("Total:", percentage_total)
    print(50 * "-")
    print("Experienced reviewers (>= 50 given reviews)", experienced)
    print("New reviewers (<50 given reviews):", new)
    print("Total:", total)
