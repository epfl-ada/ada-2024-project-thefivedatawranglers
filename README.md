# A Review of the Reviewers - Exposing Beer Reviewers’ Biases  
**ADA 2024 — Team TheFiveDataWranglers**

---

## Abstract
Customer reviews of a product is an important source for improving its quality. Companies are now better equipped than ever to evaluate how products are received by their customers and what improvements could be made, thanks to the vast amount of online reviews.
However, reviews can also greatly influence a product's success. Rarely do we buy anything without checking its reviews.
This also applies to the beer market, because taste is something people have always enjoyed discussing – especially after having a drink.
However, as beer reviewers may be biased, we will explore some biases that might be present in the reviews. Identifying these biases is interesting for breweries that aim to brew high-quality beer, as well as for buyers who want to drink the best beer possible and rely on reviews.
Therefore, in our data analysis, we aim to determine which factors related to the reviewers may influence the beer reviews.

---

## Research Questions

We primarily focus on three factors that could influence beer ratings:

1. Does the **beer’s and the user’s location** influence the rating? Are domestic beers rated better than foreign ones in some countries? Do consumers from certain countries rate beers from specific other countries particularly well? Could there be a bias with distance when rating beers?

2. Does the **time of year** the beer is consumed affect the rating? Are there beers that are consumed much more at certain times of the year, and are these beers also rated more favorably during that time?

3. Does a **reviewer’s experience** with beer influence their ratings? Are there perhaps beers or beer styles that appeal to a “beginner” who doesn’t drink much beer, but may not appeal as much to more experienced users? Does the standard for beer increase with experience?

---

## Additional Datasets

- Location data using the ‘Nominatim’ API

---

## Methods

### Data Cleaning

In order to easily analyze the data, we first convert the ratings and reviews txt files to csv files by running the txt_to_csv method in the convertTxtToCSV notebook.
Then, for some of the analyses, individual data cleaning was carried out. For the patriotism analysis, for example, we assigned the location “USA” to all US states, while for the USA internal analysis we filtered duplicate ratings from the two datasets.

### Investigating Locational Biases

To assess the impact of a beer’s origin we need to join the ratings with the users.csv and the brewery.csv. Using the location of the user and the brewery we can tell for every rating whether it’s a “domestic” or “foreign” beer for the reviewer.
Then, we can examine different statistics for these groups. Grouping the beers by the user’s country, allows us to see how these vary by country and identify any countries with significant differences.
Lastly, we can investigate if certain countries have a particularly favorable or unfavorable view of beers from a specific other country by grouping first by the user’s country and then by the country of the beer’s origin.
As a more fine-granular assessment of the locational biases we also consider the distance between the user’s location and the brewery’s location. To be able to process the given countries, we rely on the Web-API Nominatim to convert the given countries to longitude and latitude and subsequently calculate the distances.

### Investigating Seasonal Biases

To investigate biases caused the time of year, we use the date data in the reviews. We use only the month from these dates, as looking at individual days or weeks leads to a fine grained analysis, while we are more interested in trends throughout the entire year.

To analyze the data we use grouping, e.g. grouping by month or grouping by month and beer style. This make it simple to find interesting statistics for the grouped data, such as the amount of reviews per month and beer style. To compare the amount of reviews for different beer styles throughout the year, we rank the reviews by review count, allowing us to filter out beer styles which experience a big change in rank compared to their more stable counterparts, indicating changing popularity.

After confirming that the season has no effect on the average rating in general, we proceed to figure out which beers are more popular per season, think "summery" vs "wintery" beers, and then check if this popularity had any effect on the rating.


### Investigating Experiential Biases

To investigate potential experiential biases we join the ratings data with data on experienced users. We can then plot the relation with the number of reviews submitted and the relative distribution of ratings provided.

Furthermore, we analyse the 10 most reviewed beer types from the two datasets. The goal is to determine if previously identified general tendencies can also be observed on a sample, find if experienced or new reviewers influence ratings the most. Experience here is defined using a threshold of 15 reviews submitted. The results are expressed as boxplots to compare the distribution of ratings for experienced, new and all reviewers.

We can also create a more sophisticated definition of experience based on the words used by the users and compare these two groups using various statistics.

---

## Timeline

### December 2 – 8
- Resume project work, incorporate feedback, create visualizations for findings.
- Present visualizations, brainstorm improvements.
- Choose a Neural Network (NN) architecture for rating prediction, and set up GitHub Pages.

### December 9 – 14
- Present updated visualizations and findings. Assign tasks for writing content and developing the final website. Begin work on these deliverables.
- Train a NN to predict the rating behavior of a next review based on reviewers traits.

### December 14 – 20
- Complete and submit Milestone P3.

---

## Division of Work
Everyone has contributed to the initial data analysis and the writing of the final data story.
Besides this, responsabilities during the project were roughly distributed as follows:
- **Anthony:** Experience Bias
- **Benedikt:** NN, Locational Bias
- **David:** Seasonality Analysis, GitHub Pages Setup, Website Finalization
- **Gabriella:** Experience Bias, Seasonality Bias
- **Sven:** Locational Bias, Language-Based Experience Bias, NN

---

## Website
We present our results and data story on this website:
https://davidholzwarth.github.io/the-five-data-wranglers/
