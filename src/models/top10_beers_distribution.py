import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

experience_threshold = 15 #Can be changed. Defines experience

def top10beers_ratings (df_ratings,df_nb_ratings,df_name):
    #Selecting the wanted columns and creating new df for BeerAdvocate for ratings:
    filtered_ratings_df = pd.DataFrame ({
        'user_id': df_ratings['user_id'],
        'user_name': df_ratings['user_name'],
        'ratings': df_ratings['rating'],
        'beer_id': df_ratings['beer_id'],
        'beer_name': df_ratings['beer_name']
    })

    #And new df for the reviewers to have the number of given ratings per reviewer
    users_df = pd.DataFrame ({
        'nb_ratings': df_nb_ratings['nbr_ratings'],
        'user_id': df_nb_ratings['user_id'],
    })

    #Merging BA ratings with the respective BA users_df using the 'user_id' column
    filtered_ratings_df = pd.merge(users_df, filtered_ratings_df, on='user_id')

    #Classifying by the number of reviews per beer for BA
    valuecount = pd.DataFrame(filtered_ratings_df['beer_name'].value_counts().reset_index())
    valuecount.columns = ['beer_name', 'count']

    #Saving the 10 most reviewed beers
    top_10_beers = valuecount.head(10)

    #Selecting the rows from BA ratings that match with the Top_10 BA respectively
    top10_ratings_df = filtered_ratings_df[filtered_ratings_df['beer_name'].isin(top_10_beers['beer_name'])]

    #Sharing the BA Top10_ratings between experienced and new reviewers. The experience_threshold is used as separation
    #Sharing for BA
    experienced_reviewers = top10_ratings_df[top10_ratings_df['nb_ratings'].apply(lambda x: x >= experience_threshold)]
    new_reviewers = top10_ratings_df[top10_ratings_df['nb_ratings'].apply(lambda x: x < experience_threshold)]

    #Plotting the top 10 most reviewed beers for all (general), experienced and new reviewers (using the experience_threshold to define experience)

    plt.figure(figsize=(6, 6))
    ax = sns.boxplot(x='beer_name', y='ratings', data=top10_ratings_df)
    plt.xticks(rotation=90)  # Rotate x-axis labels for readability
    plt.ylabel('Ratings')
    plt.title(f'Distribution of General Ratings for {df_name}')

    plt.figure(figsize=(6, 6))
    ax = sns.boxplot(x='beer_name', y='ratings', data=experienced_reviewers)
    plt.xticks(rotation=90)  # Rotate x-axis labels for readability
    plt.ylabel('Ratings')
    plt.title(f'Distribution of Experienced Ratings for {df_name}')

    plt.figure(figsize=(6, 6))
    ax = sns.boxplot(x='beer_name', y='ratings', data=new_reviewers)
    plt.xticks(rotation=90)  # Rotate x-axis labels for readability
    plt.ylabel('Ratings')
    plt.title(f'Distribution of New Ratings for {df_name}')

    y_min, y_max = ax.get_ylim()  # Automatically determines y-axis limits
    plt.ylim(y_min, y_max)
    plt.show()