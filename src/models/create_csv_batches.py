from datetime import datetime
from src.models.create_csv import *


# modified version for batch-wise generation
def filter_and_create_stats():
    """
    Creates the user stats, beer stats and filters the ratings to only those surpassing our criteria (for batch version)
    :return: user stats, beer stats, ratings to predict
    """

    ratings_to_predict = filter_beer_ratings(
        df_ba_ratings, user_threshold=20, beer_threshold=50
    )
    print("There are ", len(ratings_to_predict), "ratings worth predicting")

    # shuffling the data just to be sure there is no bias through the order
    ratings_to_predict = ratings_to_predict.sample(frac=1, random_state=42)

    print("Creating features for ratings")
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


# we use a log file so that we don't let hour PC run forever on the
# feature creation but can start computing where we last stopped
def get_last_completed_batch(log_file):
    """
    Gives us the number of the last completed batch
    :param log_file: the file in which we log which batches we already computed
    :return: the number of the batch we last computed
    """
    if not os.path.exists(log_file):
        return 0

    with open(log_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
        if lines:
            last_line = lines[-1].strip()
            try:
                return int(last_line.split(",")[0])
            except (IndexError, ValueError):
                return 0
    return 0


def log_batch_completion(log_file, batch_number):
    """
    logs our progress
    :param log_file: the file in which we log which batches we already computed
    :param batch_number: the number we should append
    :return: None
    """
    with open(log_file, "a", encoding="utf-8") as file:
        completion_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"{batch_number},{completion_time}\n")


def save_csv_in_batches(
    user_stats,
    beer_stats,
    ratings_to_predict,
    log_file="batch_log.txt",
    batch_size=5000,
):
    """
    Saves all the features the NN needs in batche-CSV-files containing
    batch_size(we used 1000 in the end) feature-label pairs each
    :param user_stats: the user stats dataframe
    :param beer_stats: the beer stats dataframe
    :param ratings_to_predict: the ratings to predict for
    :param log_file: the path to the log file which features we already computed
    :param batch_size: the batch size we want to save with
    :return:
    """
    num_batches = (
        len(ratings_to_predict) + batch_size - 1
    ) // batch_size  # Calculate the number of batches

    last_completed_batch = get_last_completed_batch(log_file)

    # for every batch
    for batch_number in range(last_completed_batch + 1, num_batches + 1):
        batch_start = (batch_number - 1) * batch_size
        batch_end = min(batch_start + batch_size, len(ratings_to_predict))

        batch_ratings = ratings_to_predict.iloc[
            batch_start:batch_end
        ]  # get the ratings for this batch
        batch_file_name = f"batch_no_{batch_number}.csv"

        with open(batch_file_name, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            writer.writerow(column_names)

            for idx in tqdm(batch_ratings.index, desc=f"Writing batch {batch_number}"):
                features = get_features(idx, user_stats, beer_stats)  # get features
                writer.writerow(features)  # save features

        log_batch_completion(log_file, batch_number)  # update the log file
        print(f"Batch {batch_number} saved successfully in '{batch_file_name}'.")


"""
------------ Creating the label-feature pairs -------------
user_stats1, beer_stats1, ratings_to_predict1 = filter_and_create_stats()
save_csv_in_batches(
    user_stats1, beer_stats1, ratings_to_predict1, output_csv_file, batch_size=1000
)
"""


def create_foreign_batches():
    """
    Originally I accidentally created the features without the foreign bias feature.
    But of course we wanted to have that as well. So this method creates the foreign
    feature in csv batches of the same batch size as the one we used in the end (1000).
    It creates two versions. One where th US is seen as one state and another one
    where we differentiate between different US-states.
    :return:
    """
    print("Starting execution")
    print("Generating ratings to predict")
    ratings_to_predict = filter_beer_ratings(
        df_ba_ratings, user_threshold=20, beer_threshold=50
    )
    # shuffling the data just to be sure there is no bias through the order
    ratings_to_predict = ratings_to_predict.sample(frac=1, random_state=42)

    print("Loading user and brewery data")
    users_ba_df = pd.read_csv("../../data/BeerAdvocate/users.csv")
    breweries_ba_df = pd.read_csv("../../data/BeerAdvocate/breweries.csv")
    breweries_ba_df.rename(columns={"id": "brewery_id"}, inplace=True)

    print("Creating the foreign features")
    # joining the dataframes
    users_ba_df = users_ba_df.drop_duplicates(subset="user_id", keep="first")
    ratings_to_predict = ratings_to_predict.merge(
        users_ba_df[["user_id", "location"]], on="user_id", how="left"
    )
    ratings_to_predict.rename(columns={"location": "user_location"}, inplace=True)

    breweries_ba_df.drop_duplicates(subset="brewery_id", keep="first")
    ratings_to_predict = ratings_to_predict.merge(
        breweries_ba_df[["brewery_id", "location"]], on="brewery_id", how="left"
    )
    ratings_to_predict.rename(columns={"location": "brewery_location"}, inplace=True)

    # creating the two features (black formats this really weirdly)
    ratings_to_predict["foreign_us_aggr"] = ratings_to_predict.apply(
        lambda row: (
            1
            if (
                row["user_location"] == row["brewery_location"]
                or (
                    "United States" in row["user_location"]
                    and "United States" in row["brewery_location"]
                )
            )
            else 0
        ),
        axis=1,
    )

    ratings_to_predict["foreign_us_split"] = ratings_to_predict.apply(
        lambda row: (1 if (row["user_location"] == row["brewery_location"]) else 0),
        axis=1,
    )

    # creating the output files
    print(
        "Writing the output files"
    )  # again this method is so verbose because it takes a bit
    batch_size = 1000
    for i in tqdm(
        range(230)
    ):  # in the end we created 230,000 feature-label pairs, so it's just hardcoded here
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_df = ratings_to_predict.iloc[start_idx:end_idx]

        file_name = f"foreign_batch_{i + 1}.csv"
        batch_df.to_csv(
            file_name, index=False, columns=["foreign_us_aggr", "foreign_us_split"]
        )

        print(f"Batch {i + 1} saved as: {file_name}")


# originally I forgot to put the foreign feature in there, so here is the data generating for that:
create_foreign_batches()
