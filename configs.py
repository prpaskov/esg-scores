import numpy as np

class Configs:
    def __init__(self):
        self.data_path = 'data'
        self.path_dict = {'bloomberg': f'{self.data_path}/ESG_spx_ftse_sx5e.xlsx',
                          'companies': f'{self.data_path}/sp500_companies.csv',
                          'industries': f'{self.data_path}/industries.csv', 
                          'raw_urls': f'{self.data_path}/raw_urls.txt', 
                          'clean_urls': f'{self.data_path}/url_10k.csv',
                          'master':  f'{self.data_path}/master.csv',
                          'hashed': f'{self.data_path}/hashed_data.csv',
                          'text_scrape_folder': f'{self.data_path}/10K',
                          'prediction_output': f'{self.data_path}/predictions.csv'
                         }
        self.test_size = '.2'
        self.ngram_range = (2,1)
        self.max_depth = np.linspace(10, 40, 4)
        self.n_components = [10,15]
        self.ridge_alpha = [1e-1]
        self.cat_features = ['City', 'State', 'Country', 'Sector', 'Industry', 'Exchange']
        self.num_features = ['ESG', 'Social', 'ENERGY_CONSUMPTION_SUB_ISS_SCR', 'RENEWABLE_ENERGY_USE_SUB_ISS_SCR', 'WATER_MANAGEMENT_ISSUE_SCORE', 'WORKFORCE_DIVERSITY_SUB_ISS_SCR', 'Currentprice', 'Marketcap', 'Revenuegrowth', 'Fulltimeemployees']
        self.apiKey = ''
