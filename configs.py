class configurations:
    def __init__(self):
        self.data_path = 'data'
        self.path_dict = {'bloomberg': f'{self.data_path}/ESG_spx_ftse_sx5e.xlsx',
                          'companies': f'{self.data_path}/sp500_companies.csv',
                          'industries': f'{self.data_path}/industries.csv'
                         }
        self.apiKey = '4288fb37e8c313f6a62599cd625c3036755d3d1419c0496ed6636df16be3890b'