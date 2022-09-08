Environmental, social, and governance (or ESG) scores plays an increasingly important role in investor’s decisions. According to JP Morgan, 500 Billion dollars of investments flowed into ESG funds in 2021. Yet, ESG data is marked by missing data, unclear reporting standards and biased reporting. For S&P 500 companies, or the 500 leading publicly traded companies in the U.S, Bloomberg ESG ratings, touted as one of the best ESG ratings, only provide a ESG score for 134 of 500 companies, a social and environmental score for 411 companies, and a governance score for 481 companies. Socially conscious investors need more complete data. This repository contains the codes and files, including text scraped 10K forms, to impute missing Bloomberg ESG Governance scores for S&P 500 companies. 

This repository contains the following ipynb files:

-ESG Seeing Through the Smog - overview. 

-get10kurls - gets all URLS of S&P500 10-Ks uploaded to the gov website between STARTDATE and ENDDATE and saves these URLS in urls.txt. URLs are idiosyncratic so this step is critical to laying the ground for text scraping. 

-get10ks - takes URLs and returns as object "info" all text from the 10-K associated with that URL, saving this text to disk.

-generate_master - joins data from Bloomberg, Nasdaq, and SEC to generate a master data file w/ ticker and CIK for S&P 500 companies. 

-predict_gov_score - performs hashingvectorizer on 10K text data; then runs features through a pre-processing pipeline. Conducts dimensionality reduction on features and tests models with CV gridsearch to predict ESG governance scores (and thus impute missing values).

-data_vis - contains exploratory datavis, some plots of which are shown in the overview file.

Complementary csv and txt files are also included in this repository.
