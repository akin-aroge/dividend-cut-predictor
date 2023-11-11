""" Data loading script. """
import pandas as pd
from sklearn.pipeline import Pipeline
import logging
from dotenv import load_dotenv
import warnings
import configparser
import argparse
from src.data_processing import preprocessing as prep
from src.utils import utils


# Register API for Financial Modeling Prep (Financial Statements and Company Fundamentals)
# https://site.financialmodelingprep.com/developer/
# Register API for Federal Reserve Economic Data (For Macroeconomics Data)
# https://fred.stlouisfed.org/docs/api/fred/

warnings.filterwarnings('ignore')

proj_root = utils.get_proj_root()

config = configparser.ConfigParser(interpolation=None)
config.read(proj_root.joinpath('config/data_config.ini'))



def main(label_col_name:str, categorical_col_names:list, collinear_thresh:float=0.98,
         save:bool=True):

    logger = logging.getLogger(__name__)
    logger.info('Preprocessing data...')

    raw_data_path = config['data_paths']['raw_data_path']
    raw_data_path = proj_root.joinpath(raw_data_path)

    raw_data = pd.read_csv(raw_data_path)
    transform_pipeline = Pipeline([
        ('drop_rows_with_NA_in_label', prep.NARoWRemover(cols_to_check=label_col_name)),
        ('drop_rows_with_NA_in_col', prep.NARoWRemover(cols_to_check="dps_growth")),
        ('drop_collinear_columns', prep.CollinearColsRemover(thresh=collinear_thresh, label_col=label_col_name)),
        ('cat_to_ordinal_cols', prep.ColumnsOrdinalEncoder(col_names=categorical_col_names)),
        ('binarize label column', prep.BinarizeCol(col_name=label_col_name, true_val=1)),
        # ('Balance data', prep.SMOTEBalancer(label_col_name=label_col_name, random_state=None))
        ])

    transform_pipeline.fit(raw_data)
    preprocessed_data = transform_pipeline.transform(raw_data)
    logger.info('Preprocessing complete')

    if save:
        preprocessed_data_path = proj_root.joinpath(config['data_paths']['preprocessed_data_path'])
        preprocessed_data.to_csv(preprocessed_data_path, index=False)
        logger.info('Preprocessed data saved')


    return preprocessed_data.head()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    label_col_name = 'dps_change_next_year'
    collinear_thresh = 0.98
    categorical_col_names = ['industry', 'symbol']

    parser =  argparse.ArgumentParser(description="preprocessing parser")
    parser.add_argument("--label_col_name", type=str, default='dps_change_next_year',)
    parser.add_argument("--collinear_thresh", type=float, default=0.98,)
    parser.add_argument("--categorical_cols", type=list, default=['industry', 'symbol'])
    args = parser.parse_args()

    main(label_col_name=args.label_col_name,
         collinear_thresh=args.collinear_thresh,
         categorical_col_names=args.categorical_cols)
