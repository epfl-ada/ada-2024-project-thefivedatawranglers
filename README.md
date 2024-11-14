# A Review of the Reviewers - Exposing Beer Reviewers’ Biases  
**ADA 2024 — Team TheFiveDataWranglers**

---

## Abstract
Customer reviews of a product is an important source for assessing and improving its quality. Companies are now better equipped than ever to evaluate how products are received by their customers and what improvements could be made, thanks to the vast amount of online reviews available.
However, online reviews can also greatly influence a product's success – or failure. Rarely do we buy anything without checking its reviews.
This also applies to the beer market, perhaps even more so, because taste is something people have always enjoyed discussing – especially after having a drink.
However, as beer reviewers may be biased, we will explore some potential biases that might be present in the reviews. This question is relevant both to breweries that aim to brew high-quality beer and maximize sales, as well as to buyers who want to drink the best beer possible and rely on reviews.
Therefore, in our data analysis, we aim to determine which factors related to the reviewers may influence the beer reviews.

---

## Research Questions

We primarily focus on three factors that could influence beer ratings:

1. Does the **beer’s country of origin and the user’s country** influence the rating? Could it be that in some countries, due to patriotism, domestic beers are rated more favorably than foreign ones? Do consumers from certain countries rate beers from specific other countries particularly poorly or well? Could there be an apparent bias with distance when rating beers?

2. Does the **time of year** the beer is consumed affect the rating? In particular, the season could be of interest. Are there beers that are consumed much more at certain times of the year, and are these beers also rated more favorably during that time? Companies could adjust the output of their beers accordingly, and consumers could base their purchase decisions on this information.

3. Does a **reviewer’s experience** with beer influence their ratings? Are there perhaps beers or beer styles that appeal to a “beginner” who doesn’t drink much beer, but may not appeal as much to more experienced users? Does the standard for beer generally increase with experience? Depending on the results, consumers might consider the ratings of the experience group they identify with.

---

## Additional Datasets

- Location data using the ‘Nominatim’ API

---

## Methods

### Data Cleaning

In order to easily analyze the data, we first need to convert the ratings and reviews txt files to csv files by running the txt_to_csv method in the convertTxtToCSV notebook.
Then, for some of the analyses, individual data cleaning was carried out. For the patriotism analysis, for example, we assigned the location “USA” to all US states, while for the USA internal analysis we filtered duplicate ratings from the two datasets.

### Investigating Locational Biases

To assess the impact of a beer’s origin we need to join the ratings with the users.csv and the brewery.csv. Using the location of the user and the brewery we can tell for every rating whether it’s about a “domestic” or “foreign” beer for the reviewer.
Then, we can examine different statistics for these groups. Grouping the beers by the user’s country, allows us to see how these vary by country and identify any countries with significant differences.
Lastly, we can investigate if certain countries have a particularly favorable or unfavorable view of beers from a specific other country by grouping first by the user’s country and then by the country of the beer’s origin.
As a more fine-granular assessment of the locational biases we also consider the distance between the user’s location and the brewery’s location. To be able to process the given countries, we rely on the Web-API Nominatim to convert the given countries to longitude and latitude and subsequently calculate the distances.

### Investigating Seasonal Biases

To investigate biases caused by the season, or the time of year, we can make use of the date data in the ratings. As looking at individual days or weeks leads to a very fine grained analysis, while we are more interested in trends as the progress throughout an entire year, we have filtered out the just month from this date.

### Investigating Experiential Biases

To investigate potential experiential biases we need to join the data frame containing ratings with the data frame containing the experienced users. We can then plot the relation with the number of reviews given and the relative distribution of ratings provided.

This is followed by a second analysis done on the 10 most reviewed beer types from BeerAdvocate and RateBeer. The goal is to determine if previously identified general tendencies can also be observed on a sample, find if experienced or new reviewers influence ratings the most. Experience here is defined arbitrarily as a threshold number of 15 given reviews per reviewer. The results are expressed as boxplots to compare the distribution of ratings for experienced, new and all reviewers.

We can also create a more sophisticated definition of 'experienced' based on the words used by the users and compare these two groups using various statistics.


---

## Timeline

### December 2 – 8
- **1st Meeting:** Resume project work, incorporate feedback, start creating visualizations for findings.
- **2nd Meeting:** Present visualizations, brainstorm improvements
- Choose a neural network architecture for rating prediction, and set up GitHub Pages.

### December 9 – 14
- **Meeting / Review**: Present updated visualizations and findings. Assign tasks for writing content and developing the final website. Begin work on these deliverables.
- Train a neural network to predict the rating behavior of a next review based on reviewers traits (location, past experience, etc...)

### December 14 – 20
- **Finalization:** Complete and submit Milestone P3.

---

## Division of Work

- **Anthony:** Experience Bias
- **Benedikt:** Neural network training and design, Locational Bias
- **David:** Seasonality Analysis, GitHub Pages setup, website finalization
- **Gabriella:** Experience Bias, Seasonality Bias
- **Sven:** Locational Bias, Language-Based Experience Bias

---
