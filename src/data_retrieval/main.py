""" Data loading script. """

import pandas as pd
import os
import logging
from dotenv import load_dotenv
import warnings
import configparser
from company_data_extractor import company_data_extractor
from src.utils import utils


# Register API for Financial Modeling Prep (Financial Statements and Company Fundamentals)
# https://site.financialmodelingprep.com/developer/
# Register API for Federal Reserve Economic Data (For Macroeconomics Data)
# https://fred.stlouisfed.org/docs/api/fred/

warnings.filterwarnings('ignore')

proj_root = utils.get_proj_root()

config = configparser.ConfigParser(interpolation=None)
config.read(proj_root.joinpath('config/data_config.ini'))




def main():

    logger = logging.getLogger(__name__)
    logger.info('Retrieving data')

    load_dotenv('.env')
    API_KEY_FRED = os.environ.get('API_KEY_FRED')
    API_KEY_FMP = os.environ.get('API_KEY_FMP')

    start_year = int(config['year_limits']['start_year'])
    end_year = int(config['year_limits']['start_year'])
    COMPANY_LIST_URL = config['urls']['COMPANY_LIST']

    num_of_years = end_year - start_year + 1

    # Scrap sp500 tickers using pandas datareader
    tables = pd.read_html(COMPANY_LIST_URL)
    ticker_table = tables[0]
    tickers = ticker_table['Symbol'].tolist()

    # Obtain our dataset
    data_extractor = company_data_extractor(API_KEY_FRED, API_KEY_FMP)
    dataset = []
    company_number = 1
    for ticker in tickers[:3]:
        # print(f"{company_number}: Obtaining data for {ticker}")
        logger.info(f'obtaining data for company number {company_number} ({ticker})')
        company_number = company_number + 1
        company_data = data_extractor.get_data(ticker, start_year, end_year, num_of_years)
        if type(company_data).__name__ == "int":
            continue
        dataset.append(company_data)
    dataset = pd.concat(dataset, ignore_index=True)

    # Save data to disk
    raw_data_path = proj_root.joinpath('data/raw/stock_data5.csv')
    dataset.to_csv(raw_data_path, index=False)

    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
