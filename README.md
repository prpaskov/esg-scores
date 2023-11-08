## Context
Environmental, social, and governance (ESG) scores plays an increasingly important role in investors' decisions. According to JP Morgan, 500 billion dollars of investments flowed into ESG funds in 2021. Yet, ESG data is marked by missing data, unclear reporting standards and biased reporting. For S&P 500 companies, or the 500 leading publicly traded companies in the U.S, Bloomberg ESG ratings, touted as one of the best ESG ratings, only provide a ESG score for 134 of 500 companies, a social and environmental score for 411 companies, and a governance score for 481 companies. Socially conscious investors need more complete data. This repository contains the codes and files, including text-scraped 10K forms, to impute missing Bloomberg ESG Governance scores for S&P 500 companies. 

## Methodology
This repo develops the process to use tax filing text to impute missing ESG scores for S&P 500 Companies. Given the size of the 10K reports, the code employs HashVectorizer to obtain partial fits in batches, then merges resulting columns with an additional matrix of numerical features (i.e. number of employees, market cap) and one-hot-encoded categorical features (i.e. sector, state, etc.). These features are run through a pipeline that applies TFIDF on the text sparse matrix; applies standard scaler to all features; runs singular value decomposition on all features; then predicts the Bloomberg Governance Score with a range of models including Ridge and Random Forest. The pipeline applies GridSearchCV for hyperparameter optimization. Code generates plots of predictions and true values on the test and train set.

## Files
This repository contains the following files:
* scrape_10Ks.py - gets all URLS of S&P500 10-Ks uploaded to the gov website between STARTDATE and ENDDATE and saves these URLs to data/url_10k.csv. URLs are idiosyncratic so this step is critical to laying the ground for text scraping. Then takes in URLs and returns as object "info" specified text from the associated 10-K associated and saves this text within data/10K as a txt file.
* gen_and_viz_data.py - joins data from Bloomberg, Nasdaq, and SEC to generate a master data file w/ ticker and CIK for S&P 500 companies. Saves to file. Includes functions to visualize scores and missing data patterns.
* predict_score.py - performs hashingvectorizer on 10K text data, generates feature dataset, runs features through a pre-processing pipeline. Runs CV gridsearch with hyperparameter ranges defined in configs.py. Uses best hyperparameters to predict ESG governance scores (and thus impute missing values). Saves predictions to file.
* configs.py - contains file paths and hyperparameter ranges
* excute.ipynb - runs classes and visualizes Bloomberg data. Insert API key in configs.py file prior to running.

Author - prpaskov 
