from sec_api import QueryApi, ExtractorApi
from datetime import datetime
import requests, re, os
import pandas as pd
import numpy as np
from configs import Configs as scrapeConfigs

class scrape_10K_filings:
    """
    This file gets the URLS needed to scrape text from 10-K filings, scrapes the URLS, and saves text output into text_scrape_folder.

    Attributes:
        year_start (str): starting year to extract
        year_end (str): ending year to extract
        quick_run (bool): if True, skips 90% of filings. Only True for testing. Default False.
    """
    def __init__(self, 
                year_start: str, 
                year_end: str, 
                quick_run: bool = False):
        
        self.paths = scrapeConfigs.path_dict
        self.year_start = year_start
        self.year_end = year_end
        self.quick_run = quick_run 
        self.queryApi, self.extractorApi = self.get_query_and_extractor_Api()
    
        self.get_raw_url_set()
        self.get_formatted_url_set()
        self.scrape_records()

    def get_query_and_extractor_Api(self):
        """
        Returns query and extractor API
        """
        query = QueryApi(api_key=Configs.apiKey)
        extractor = ExtractorApi(api_key=Configs.apiKey)
        return query, extractor

    def get_url(self, filing:str):
        """
        Takes in the 'filing' item in the dict returned from self.queryApi.get_filings(base_query)
        Returns a single company-fiscal-year URL for a filing
        """
        for i in filing['documentFormatFiles']:
            if 'description' in i and i['description'] == '10-K':
                try:
                    return i['documentUrl'], filing['periodOfReport'], filing['cik']
                except:
                    import ipdb; ipdb.set_trace()
                    a=1
        return None

    def get_starting_base_query(self): 
        """
        Returns base query. get_uniformed_url_set later fills in base_query['query']['query_string']['query'] (defined here as "Placeholder") then runs the query through self.queryApi.get_filings(base_query).
        """
        base_query = {
            "query": { 
                "query_string": { 
                    "query": "PLACEHOLDER", # set during runtime 
                    "time_zone": "America/New_York"
                } 
            },
            "from": "0",
            "size": "200", 
            "sort": [{ "filedAt": { "order": "desc" } }]
            }
        return base_query
    
    def get_raw_url_set(self): 
        """
        Outputs an raw txt file (saved as paths.raw_urls) with a full URL set for 10-Ks within year_start and year_end.
        """
        base_query = self.get_starting_base_query()

        #overwrites any existing file by default
        os.remove(paths.raw_urls)
        log_file = open(paths.raw_urls, "a")

        for year in range(self.year_start, self.year_end - 1, -1):
            print("starting {year}".format(year=year))

            # a single search universe is represented as a month of the given year
            for month in range(1, 13, 1):
            # get 10-K and 10-K/A filings filed in year and month
            # resulting query example: "formType:\"10-K\" AND filedAt:[2021-01-01 TO 2021-01-31]"
                universe_query = \
                    "formType:\"10-K\" AND ".format(filing_type='10-K') + \
                    "filedAt:[{year}-{month:02d}-01 TO {year}-{month:02d}-31]" \
                    .format(year=year, month=month)
                print(universe_query)

                # set new query universe for year-month combination
                base_query["query"]["query_string"]["query"] = universe_query

                end = 200 if self.quick_run else 10001
                for from_batch in range(0, end, 200):
                # set new "from" starting position of search
                    base_query["from"] = from_batch;

                    response = self.queryApi.get_filings(base_query)

                # no more filings in search universe
                    if len(response["filings"]) == 0:
                        break;

                # for each filing, save the URL pointing to the first file
                # the URL is set in the dict key "documentUrl" of the list "dataFiles"
                    urls_list = list(map(self.get_url, response["filings"]))

                # delete empty URLS
                    urls_list = list(filter(None, urls_list))

                # transform list of URLs into one string by joining all list elements
                # then add new-line character between each element
                    urls_string = ""
                    for i in urls_list:
                        urls_string = urls_string + "\n" + ",".join(i)
                    log_file.write(urls_string)

        log_file.close()
        print('done')
    
    def get_formatted_url_set(self):
        """
        Reads raw URLS from scrapeConfigs.raw_urls
        Reformats
        Outputs formatted URLs to scrapeConfigs.clean_urls
        """
        url = pd.read_csv(paths.raw_urls, header=None)
        url.columns = ['url', 'date', 'CIK']
        url['date'] = url['date'].astype(str)

        #generate most recent indicator
        url['date'] = url['date'].apply(self.convertStrToDate)
        maxUrlDate = url.groupby(['CIK'])['date'].idxmax()
        print(maxUrlDate)
        url['mostrecent'] = 0
        url.loc[maxUrlDate.values,'mostrecent'] = 1
        print(url)
        master = pd.read_csv(paths.master)
        url_10k = url.merge(master, on='CIK')
        url_10k.to_csv(paths.clean_urls) 

    def convertStrToDate(self, string_date:str):
        """
        Takes in string date, returns datetime format
        """
        return datetime.strptime(string_date, '%Y-%m-%d')

    def extract_items(self, filing_url:str):
        """
        Extracts 10K items from a given URL.
        """
        info = []
        items = ["1", "1A", "1B", "2", "3", "4", "5", "6", "7", 
            "7A", "8", "9A", "9B", "10", "11", "12", "13", "14"]
        for item in items:
            try:
                section_text = self.extractorApi.get_section(filing_url=filing_url, 
                                                section=item, 
                                                return_type="text")
                info.append(section_text)    
        except Exception as e:
            print(e)
        return info
    
    def scrape_records(self): 
        """
        Runs through URLs in paths.clean_urls and extracts text from 10Ks via extract_items, then saves the text output to paths.text_scrape_folder.
        """
        data = pd.read_csv(paths.clean_urls)
        data['CIK'] = data['CIK'].astype(str)
        for x in range(len(data)):
            if data['mostrecent'][x] == 1:
                log_file = open(paths.text_scrape_folder+data['CIK'][x]+'_'+data['Symbol'][x], "a")
                info = extract_items(data['url'][x])
                log_file.write(",".join(info))
                log_file.close()