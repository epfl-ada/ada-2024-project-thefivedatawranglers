import pandas as pd
from tqdm import tqdm


# Converts text file to a CSV file
def txt_to_csv(txt_file, csv_file):
    file = open(txt_file, "r", encoding="utf-8")

    reviews = []
    current_review = {}

    for line in file.readlines():
        # Adds current review to reviews when the line is empty
        if line.strip() == "":
            if current_review:
                reviews.append(current_review)
                current_review = {}
                # Otherwise the line is split into a key-value pair
        else:
            key, value = line.split(":", 1)
            current_review[key.strip()] = value.strip()

    # Adds the last review if there is no empty line at the end of the file
    if current_review:
        reviews.append(current_review)

    # Creates dataframe with the reviews
    df = pd.DataFrame(reviews)

    # Creates a csv file with the dataframe
    df.to_csv(csv_file, index=False)


# Executed code
"""
txt_to_csv('data/BeerAdvocate/ratings.txt', 'data/Ratings/BA_ratings.csv')
txt_to_csv('data/BeerAdvocate/reviews.txt', 'data/Ratings/BA_reviews.csv')
txt_to_csv('data/RateBeer/ratings.txt', 'data/Ratings/RB_ratings.csv')
txt_to_csv('data/RateBeer/reviews.txt', 'data/Ratings/RB_reviews.csv')
"""


def txt_to_csv2(txt_file, csv_file):
    # Datei mit expliziter Kodierung 'utf-8' öffnen
    with open(txt_file, "r", encoding="utf-8") as file:
        reviews = []
        current_review = {}

        for line in file.readlines():
            if line.strip() == "":
                if current_review:
                    reviews.append(current_review)
                    current_review = {}
            else:
                key, value = line.split(":", 1)
                current_review[key.strip()] = value.strip()

        if current_review:
            reviews.append(current_review)

        df = pd.DataFrame(reviews)
        df.to_csv(csv_file, index=False)


def txt_to_csv_prog(txt_file, csv_file):
    # Datei mit expliziter Kodierung 'utf-8' öffnen
    with open(txt_file, "r", encoding="utf-8") as file:
        reviews = []
        current_review = {}

        # Liest die Zeilen ein und zeigt einen Fortschrittsbalken an
        lines = file.readlines()
        for line in tqdm(lines, desc="Verarbeite Zeilen", unit="Zeile"):
            if line.strip() == "":
                if current_review:
                    reviews.append(current_review)
                    current_review = {}
            else:
                key, value = line.split(":", 1)
                current_review[key.strip()] = value.strip()

        if current_review:
            reviews.append(current_review)

        df = pd.DataFrame(reviews)
        df.to_csv(csv_file, index=False)


txt_to_csv_prog('data/BeerAdvocate/ratings/ratings.txt', 'data/BeerAdvocate/ratings/ratings.csv')
