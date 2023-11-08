class Configs:
    data_path = 'data'
    path_dict = {'bloomberg': f'{self.data_path}/ESG_spx_ftse_sx5e.xlsx',
                        'companies': f'{self.data_path}/sp500_companies.csv',
                        'industries': f'{self.data_path}/industries.csv', 
                        'raw_urls': f'{self.data_path}/raw_urls.txt', 
                        'clean_urls': f'{self.data_path}/url_10k.csv',
                        'master':  f'{self.data_path}/master.csv',
                        'text_scrape_folder': f'{self.data_path}/10K'
                        }
    apiKey = '4288fb37e8c313f6a62599cd625c3036755d3d1419c0496ed6636df16be3890b'