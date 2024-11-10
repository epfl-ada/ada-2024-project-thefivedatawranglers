from tqdm import tqdm
import pandas as pd


def txt_to_csv_prog(txt_file, csv_file):
    # for some reason I had to add this utf-8 coding
    with open(txt_file, "r", encoding="utf-8") as file:
        reviews = []
        current_review = {}

        # I added a tqdm progress bar because it takes so long...
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
