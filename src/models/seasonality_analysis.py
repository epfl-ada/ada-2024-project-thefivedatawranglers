import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_and_head_average_rating_per_month(df):
    """
    Calculate the average rating per month
    :param df: df_rb_ratings
    :return: the average rating per month
    """
    df['month'] = pd.to_datetime(df['date'], unit = 's').dt.month

    # Group by month and calculate the average rating
    monthly_avg_rating = df.groupby('month')['rating'].mean().reset_index()
    print(monthly_avg_rating.head())
    
    ax = monthly_avg_rating.plot.bar(x='month', y='rating')
    ax.set_title("Average Rating per month RateBeers Dataset",)
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Rating")
    plt.show()

def plot_beer_style_ranking_by_amount(df, cutoff = 500, interesting_threshhold = 10):
    """
    Calculate the ranking of beer styles by the amount of reviews per month
    :param df: df_rb_ratings
    :param cutoff: the minimum amount of reviews per style. Default value 500
    :param interesting_threshhold: the minimum difference in max and min to be considered interesting. Default value 10
    """
    # Group by month and style, and count the number of reviews per month and style
    ranked_by_amount_beer_styles_per_season = df.groupby(['month', 'style']).size().reset_index(name='review_count')

    ##  Filter out styles with less than cutoff reviews ---

    size_before_filtering = len(ranked_by_amount_beer_styles_per_season)
    ranked_by_amount_beer_styles_per_season = ranked_by_amount_beer_styles_per_season.groupby('style').filter(lambda x: (x['review_count'] >= cutoff).all())

    print(f'We lost {size_before_filtering - len(ranked_by_amount_beer_styles_per_season)} rows by filtering out styles with less than {cutoff} reviews.')

    # ---

    # Rank the styles within each month
    ranked_by_amount_beer_styles_per_season['rank'] = ranked_by_amount_beer_styles_per_season.groupby('month')['review_count'].rank(ascending=False, method='min')

    # Sort the df by month and rank
    ranked_by_amount_beer_styles_per_season = ranked_by_amount_beer_styles_per_season.sort_values(['month', 'rank'])

    # Pivot the table to get styles as rows, months as columns, with ranks as values
    beer_style_ranking_by_amount = ranked_by_amount_beer_styles_per_season.pivot(index='style', columns='month', values='rank').dropna()

    ## Filtering out styles with low rank change ---

    # Calculate the difference between the max and min review count for each row
    rank_change = beer_style_ranking_by_amount.max(axis=1) - beer_style_ranking_by_amount.min(axis=1)

    styles_with_low_change = rank_change >= interesting_threshhold #Filter 

    beer_style_ranking_by_amount = beer_style_ranking_by_amount[styles_with_low_change]

    # ---

    fig = go.Figure()

    for style in beer_style_ranking_by_amount.index:
        x = beer_style_ranking_by_amount.columns  # Months
        y = beer_style_ranking_by_amount.loc[style]  # Ranks for this style
        fig.add_trace(go.Scatter(
            x=x,  
            y=y,  
            mode='lines+markers',
            name=style,
            hovertemplate=f'{style}'
        ))

    # Reverse y-axis so that rank 1 is at the top
    fig.update_yaxes(autorange='reversed', title='Rank')

    fig.update_xaxes(title='Month', tickvals=list(range(1, 13)),
                    ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    fig.update_layout(
        title='Monthly Rankings of Beer Styles by Review Count',
        legend_title='Beer Style',
        hovermode='closest' 
    )

    fig.show()

def plot_beer_style_ranking_by_avg_score(df, cutoff = 500, interesting_threshhold = 0.1):
    """
    Calculate the average rating per month
    :param df: df_rb_ratings
    :param cutoff: the minimum amount of reviews per style. Default value 500
    :param interesting_threshhold: the minimum difference in rank to be considered interesting. Default value 0.1
    :return: the average rating per month
    """
    # Group by month and style, and count the average rating (average score) per month and style
    ranked_by_avg_score_beer_styles_per_season = df.groupby(['month', 'style'])['rating'].agg(
        avg_score='mean', 
        review_count='count').reset_index()

    ##  Filter out styles with less than cutoff reviews ---

    size_before_filtering = len(ranked_by_avg_score_beer_styles_per_season)
    ranked_by_avg_score_beer_styles_per_season = ranked_by_avg_score_beer_styles_per_season.groupby('style').filter(lambda x: (x['review_count'] >= 500).all())

    # Drop the review_count column
    ranked_by_avg_score_beer_styles_per_season.drop(columns='review_count', axis = 1, inplace=True)

    print(f'We lost {size_before_filtering - len(ranked_by_avg_score_beer_styles_per_season)} rows by filtering out styles with less than {cutoff} reviews.')

    # ---

    # Pivot the table to get styles as rows, months as columns, with ranks as values
    beer_style_ranking_by_avg_score = ranked_by_avg_score_beer_styles_per_season.pivot(index='style', columns='month', values='avg_score').dropna()

    ## Filtering out styles with low rank change ---

    # Calculate the difference between the maximum and minimum score for each row
    rank_change = beer_style_ranking_by_avg_score.max(axis=1) - beer_style_ranking_by_avg_score.min(axis=1)

    styles_with_high_change = rank_change >= interesting_threshhold #Fitler

    # Apply the filter
    beer_style_ranking_by_avg_score = beer_style_ranking_by_avg_score[styles_with_high_change]

    # ---

    fig = go.Figure()

    for style in beer_style_ranking_by_avg_score.index:
        x = beer_style_ranking_by_avg_score.columns  # Months
        y = beer_style_ranking_by_avg_score.loc[style]  # Avg. score for this style
        fig.add_trace(go.Scatter(
            x=x,  
            y=y,  
            mode='lines+markers',
            name=style,
            hovertemplate=f'{style}'
        ))

    # Reverse y-axis so that rank 1 is at the top
    fig.update_yaxes(autorange='reversed', title='Rank')

    fig.update_xaxes(title='Month', tickvals=list(range(1, 13)),
                    ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    fig.update_layout(
        title='Monthly Rankings of Beer Styles by Average Score',
        legend_title='Beer Style',
        hovermode='closest' 
    )

    fig.show()