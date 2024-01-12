""" Data extractor class."""

import requests
import pandas as pd
import numpy as np
import yfinance as yf
import logging

import configparser

from src.utils import utils

logger = logging.getLogger(__name__)

proj_root = utils.get_proj_root()

config = configparser.ConfigParser()
config.read(proj_root.joinpath("config/data_config.ini"))

BASE_URL = config["API"]["FMP_BASE_URL"]
FRED_BASE_URL = config["API"]["FRED_BASE_URL"]


class company_data_extractor:
    # Constructor
    def __init__(self, API_KEY_FRED, API_KEY_FMP):
        self.API_KEY_FRED = API_KEY_FRED
        self.API_KEY_FMP = API_KEY_FMP
        self.BASE_URL = BASE_URL

    def get_data(self, company_tick, start_year, end_year, num_of_years):
        company_tick = company_tick
        start_year = start_year - 1
        end_year = end_year + 1
        num_of_years = num_of_years + 2

        # Engineer Dividend Per Share (DPS) data (Dividend related predictors and the target variable)
        response = requests.get(
            f"{self.BASE_URL}/historical-price-full/stock_dividend/{company_tick}?apikey={self.API_KEY_FMP}"
        )
        if response.status_code == 429:
            print("FMP API limit reached")
            return 0
        dividends = pd.DataFrame(response.json()["historical"])

        if dividends.shape == (0, 0):
            dividends = pd.DataFrame(
                {
                    "year": list(range(start_year, end_year + 1)),
                    "adjDividend": [0.0] * num_of_years,
                }
            )
        else:
            dividends["year"] = pd.to_datetime(dividends["date"]).dt.year
            dividends = (
                dividends.groupby("year").agg({"adjDividend": "sum"}).reset_index()
            )
            # Determine the start and end years
            first_year = dividends["year"].min()
            last_year = 2023
            # Create a new DataFrame with all years from start to end
            all_years = pd.DataFrame({"year": list(range(first_year, last_year + 1))})
            # Merge the two DataFrames on the year column and fill missing values with 0.0
            dividends = all_years.merge(dividends, on="year", how="left").fillna(0.0)
            dividends = dividends.loc[
                (dividends["year"] >= start_year) & (dividends["year"] <= end_year)
            ]
        dividends["next_year_dividend"] = dividends["adjDividend"].shift(-1)
        dividends["last_year_dividend"] = dividends["adjDividend"].shift(1)
        conditions = [
            dividends["adjDividend"] == dividends["next_year_dividend"],
            dividends["adjDividend"] > dividends["next_year_dividend"],
            dividends["adjDividend"] < dividends["next_year_dividend"],
        ]
        choices = [0, 1, 2]  # 0: DPS stayed constant
        # 1: DPS decreased
        # 2: DPS increased

        # Create predictor "dps_growth"
        dividends["dps_growth"] = np.where(
            (dividends["last_year_dividend"] == 0) & (dividends["adjDividend"] == 0),
            0,  # If both are 0 then change is 0
            np.where(
                dividends["last_year_dividend"] != 0,
                ((dividends["adjDividend"] / dividends["last_year_dividend"]) - 1)
                * 100,
                999,  # If last year dividend is 0 then return 999
            ),
        )
        # Create the target column 'dps_change' based on the conditions
        dividends["dps_change_next_year"] = np.select(
            conditions, choices, default=np.nan
        )
        # Remove the first and last year since they will be NaN
        dividends = dividends.loc[
            (dividends["year"] > start_year) & (dividends["year"] <= end_year - 1)
        ]
        dividends = dividends[
            ["year", "adjDividend", "dps_growth", "dps_change_next_year"]
        ]

        # Engineer Other Predictors
        predictors = pd.DataFrame({"year": list(range(start_year, end_year + 1))})

        # Company's Industry
        predictors["industry"] = yf.Ticker(company_tick).info.get("industry")

        # Key Financial Ratios
        response = requests.get(
            f"{self.BASE_URL}/ratios/{company_tick}?limit={num_of_years}&apikey={self.API_KEY_FMP}"
        )
        # Check if all data is available
        data_length = len(response.json())
        if data_length != num_of_years:
            # print("Company data not available")
            logger.info(f"data not available for ticker {company_tick}")
            return 0
        financial_ratios = (
            pd.DataFrame(response.json())
            .iloc[:, :]
            .sort_values("date", ascending=True)
            .reset_index(drop=True)
        )
        predictors = pd.concat([predictors, financial_ratios], axis="columns")
        predictors.drop(["date", "period"], axis="columns", inplace=True)

        # Macroeconomics - Federal Interest Rate (Annualized)
        url = (
            f"{FRED_BASE_URL}"
            f"api_key={self.API_KEY_FRED}&"
            f"file_type=json&"
            f'observation_start={str(start_year) + "-01-01"}&observation_end={str(end_year) + "-12-31"}&'
            f"frequency=a"
        )

        response = requests.get(url)
        fed_interest_rates = pd.DataFrame(response.json()["observations"])["value"]
        predictors["interestRate"] = fed_interest_rates.astype("float64")

        predictor_names = list(predictors.columns)
        predictor_names.remove("year")
        predictor_names.remove("industry")
        predictor_names.remove("symbol")
        predictor_names.remove("calendarYear")

        def compute_change(df, predictor_list):
            for predictor in predictor_list:
                # Calculate percentage change
                percentage_change = df[predictor].pct_change() * 100
                # Create new column name
                new_col_name = f"{predictor}_percentage_change"
                # Find the index position of the original predictor column
                original_col_position = df.columns.get_loc(predictor)
                # Insert the new column right after the original predictor column
                df.insert(original_col_position + 1, new_col_name, percentage_change)
            # Replacing inf and NaN values
            df.replace([float("inf"), float("-inf")], 999, inplace=True)
            df.fillna(0, inplace=True)
            return df

        predictors = compute_change(predictors, predictor_names)

        # Combine dividend data with other predictors
        dataset = pd.merge(
            predictors, dividends, left_on="year", right_on="year", how="left"
        )

        # Drop first and last row as they contain nan
        last_row = len(dataset) - 1
        dataset.drop([0, last_row], axis="rows", inplace=True)
        dataset.reset_index(drop=True, inplace=True)

        return dataset
