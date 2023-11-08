from doctest import DocFileTest
import re
import pandas as pd
import numpy as np
import scipy
import scipy.sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, HashingVectorizer
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.model_selection import GridSearchCV
from configs import Configs 

class predictScore:
    def __init__(self, 
                 score_to_predict: Literal['ESG', 'Governance', 'Social', 'Environmental'] = 'Governance'):
        self.predictConfigs = Configs()
        self.paths = self.predictConfigs.path_dict
        self.score_to_predict = score_to_predict

        self.master_df, self.hashed, self.feature_df = self.generate_pipeline_data()
        self.pipeline = self.generate_pipeline()
        self.train_test_dict = self.get_train_test_dict()
        self.search = self.run_grid_search()
        self.get_predictions()
    
    def generate_pipeline_data(self):
        """
        Generates all data needed for pipeline
        """
        master_df = self.get_master_df()
        hashed = self.get_hashing_vectorizer()
        feature_df = self.get_cat_and_num_features()
        return master_df, hashed, feature_df

    def get_master_df(self)
        """
        Returns a data_dict with master, train, and test.
        """
        df = pd.read_csv(self.paths['master'])
        master = df.dropna(subset=[self.score_to_predict])

        np.random.seed(93)  
        test_mask = np.random.rand(len(master)) < self.predictConfigs.test_size  
        master['test'] = test_mask
        master['train'] = master[~master['test']]
        return master

    def run_hashing_iter(self, i:int):
        """
        Runs one iteration of hashing. Called in a loop in hashing_vectorizer.
        """
        temp = self.master_df[['Symbol', 'CIK']]
        temp['10K'] = ''
        path = self.paths.text_scrape_folder + str(temp['CIK'][i]) + '_' + temp['Symbol'][i]
        CIK = str(temp['CIK'][i])     
        try:
            with open(path, 'r') as f2:
                temp['10K'][i] = f2.read()
        except:
            temp['10K'][0] = ''
        return hv.fit_transform(temp['10K'])

    def get_hashing_vectorizer(self):
        """
        Performs hashing vectorizer and saves npz file to self.paths.hashed
        """
        #perform hashing vectorizer w/ stop words
        STOP_WORDS = STOP_WORDS.difference({'he','his','her','hers'})
        STOP_WORDS = STOP_WORDS.union({'ll', 've'})
        hv = HashingVectorizer(stop_words=STOP_WORDS, ngram_range=self.predictConfigs.ngram_range)

        #kick off hashing first 1 to set sparse matrix frame; then continue with remaining obs
        hashing_master = self.run_hashing_iter(i=0)
        for iter in range(1, len(self.master_df)):
            hashing_temp = self.run_hashing_iter(i=iter)
            hashing_master = scipy.sparse.vstack((hashing_master, hashing_temp))
        
        return hashing_master

    def get_cat_and_num_features(self):
        """
        Runs one hot encoder to get dummies; generates feature dataset with OHE categorical features and numerical features. Assumes the predicted value is Governance.
        """
        ohe = OneHotEncoder(categories='auto')
        feature_arr = ohe.fit_transform(self.master_df[self.predictConfigs.cat_features]).toarray()
        labels = []
        for x in range(len(ohe.categories_)):
            for y in range(len(ohe.categories_[x])):
                labels.append(ohe.categories_[x][y])    
        cat_features = pd.DataFrame(feature_arr, columns=labels)
        num_features = self.master_df[self.predictConfigs.num_features]

        master_features = cat_features.merge(num_features, left_index=True, right_index=True)
        master_features = master_features.fillna(0)
        return master_features

    def generate_pipeline(self):
        """
        Returns pipeline
        """
        tfidfFeatureCount = self.hashed.shape[1] 
        totFeatures = self.hashed.shape[1] + self.feature_df.shape[1]

        tfidf = ColumnTransformer(
            transformers = [
            ('tfidf', TfidfTransformer(), 
            np.arange(tfidfFeatureCount) )],
            remainder='passthrough'
            )

        #Standard Scalar - we can use with_mean=False on tfidf as the sparse matrix mean is ~0 
        standardScale = ColumnTransformer(
            transformers = [
            ('standardScaleSparse', StandardScaler(with_mean=False), np.arange(tfidfFeatureCount) ),
            ('standardScaleDense', StandardScaler(with_mean=False), np.arange(tfidfFeatureCount, totFeatures) )],
            remainder='passthrough'
            )
            
        svd = ColumnTransformer(
            transformers = [
            ('svd', TruncatedSVD(), np.arange(totFeatures) )],
            remainder='passthrough'
            )

        pipe = Pipeline([  
            ('tfidf', tfidf),
            ('standardScale', standardScale),
            ('svd', svd),
            #('ranFor', RandomForestRegressor(n_estimators = 100))
            ('ridge', Ridge()) 
            ])

        return pipe

    def get_train_test_dict(self):
        """
        Combined hashed df and feature df; splits into train/test for X/y and saves to train_test_dict
        """
        tt_dict = {}
        X_train = self.hashed[self.master_df.train] #383 x 1M
        X_test = self.hashed[self.master_df.test] #383 x 1M

        # append the other features on to the end of the X_train matrix 
        tt_dict['X_train'] = scipy.sparse.hstack((X_train, self.feature_df.loc[self.master_df.train].fillna(0).values))
        tt_dict['X_test'] = scipy.sparse.hstack((X_test, self.feature_df.loc[self.master_df.test].fillna(0).values))

        tt_dict['y_train'] = self.master.loc[self.master.train, self.score_to_predict] 
        tt_dict['y_test'] = self.master.loc[self.master.test, self.score_to_predict] 
        return tt_dict

    def run_grid_search(self):
        """
        Performs GridSearchCV with parameters defined in configs. Returns results.
        """
        parameters = {
            'ranFor__max_depth': self.predictConfigs.max_depth,
            'svd__svd__n_components': self.predictConfigs.n_components,
            'ridge__alpha': self.predictConfigs.ridge_alpha
            }

        search = GridSearchCV(self.pipeline, parameters, n_jobs=1, verbose=True)
        search.fit(self.train_test_dict['X_train'], self.train_test_dict['y_train'])

        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print(search.best_params_)
        return search
    
    def get_predictions(self):
        """
        Predicts score using params from CV grid search in run_grid_search.
        Generates data viz for training and testing.
        Outputs predictions to self.paths.prediction_output.
        """

        predictions = self.search.predict(self.train_test_dict['X_train'].tocsr())
        self.plot_predictions(self.train_test_dict['y_train'], predictions, 'Training')

        predictionsTest = self.search.predict(self.train_test_dict['X_test'].tocsr())
        self.plot_predictions(self.train_test_dict['y_test'], predictionsTest, 'Testing')

        print(r2_score(self.train_test_dict['y_test'],predictionsTest))

        X_all = scipy.sparse.hstack((self.hashed, self.features_df.fillna(0).values))
        y_all_pred = self.search.predict(X_all.tocsr())

        final_df = self.master_df.copy()
        final_df['predicted'] = y_all_pred.round()
        preds = final_df[[self.score_to_predict, 'predicted', 'Name']]
        preds.to_csv(self.paths.prediction_output)
    
    def plot_predictions(self, true_values, predictions, train_test: Literal['Training', 'Testing']):
        """
        Plots predictions of train/test set relative to true values (Scatterplot)
        """
        plt.scatter(true_val, predictions)
        plt.title(f'{train_test} Set')
        plt.xlabel('True Values ')
        plt.ylabel('Predictions ')
        plt.axis('equal')
        plt.show()