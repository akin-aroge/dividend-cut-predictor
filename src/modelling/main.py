import logging
import configparser
import argparse
import warnings
from sklearn.pipeline import Pipeline

from src.utils import utils
from src.modelling import training as train
from src.modelling import transforms

warnings.filterwarnings("ignore")


def main(model_name: str, tune_trials=1, balance_data=False, preselect_cols=False):
    logger = logging.getLogger(__name__)
    proj_root = utils.get_proj_root()

    config = configparser.ConfigParser(interpolation=None)
    config.read(proj_root.joinpath("config/data_config.ini"))

    # final_year = int(config["year_limits"]["end_year"])
    inf_year = int(config["year_limits"]["inf_year"])
    final_year = inf_year - 2

    preprocessed_data_rel_path = config["data_paths"]["preprocessed_data_path"]
    preprocessed_data_path = proj_root.joinpath(preprocessed_data_rel_path)
    feature_set_path = proj_root.joinpath(config["modelling_paths"]["optimal_features"])
    model_output_dir = proj_root.joinpath(config["modelling_paths"]["model_output"])

    label_col_name = "dps_change_next_year"

    model_params = config._sections[model_name]

    model_class = train.get_model_class(model_name=model_name)
    model = model_class(**model_params)

    categorical_col_names = ["year", "industry", "symbol"]
    # collinear_thresh = 0.98

    # get data
    preprocessed_data = train.get_training_data(file_path=preprocessed_data_path)

    # split dataset
    training_data, testing_data = train.train_test_split(
        df=preprocessed_data, final_year=final_year
    )

    
    if preselect_cols:
        optimal_col_selector = transforms.OptimalColumnSelector(optimal_cols_path=feature_set_path, 
                                                                label_col_name=label_col_name)

        training_data = optimal_col_selector.fit_transform(training_data)
        testing_data = optimal_col_selector.transform(testing_data)
        logger.info(f"columns pre-selected: {list(training_data.columns)}")

    # categorical_col_names = train.get_categorical_cols(training_data)
    # print('categorical cols: ',categorical_col_names)
    if balance_data:
        # balance data
        cat_col_encoder = transforms.ColumnsOrdinalEncoder(
            col_names=categorical_col_names
        )
        training_data= cat_col_encoder.fit_transform(training_data)
        training_data= transforms.balance_data(
            training_data, label_col_name=label_col_name
        )
        training_data = cat_col_encoder.inverse_transform(training_data)

    pipeline = Pipeline(
        steps=[
            (
                "select_optimal_cols",
                transforms.OptimalColumnSelector(optimal_cols_path=feature_set_path),
            ),
            (
                "cat_to_ordinal_cols",
                transforms.ColumnsOrdinalEncoder(),
            ),
        ]
    )


    model_output_path = model_output_dir.joinpath(model_name + ".pkl")
    trainer = train.ModelTrainer(
        model_class=model,
        transform_pipeline=pipeline,
        training_data=training_data,
        testing_data=testing_data,
        label_col_name=label_col_name,
        model_output_path=model_output_path
    )

    logger.info("==============tuning started=============")
    trainer.tune_model(n_trials=tune_trials)
    # logger.info('tuning completed')
    logger.info("==============training started====================")
    model = trainer.train_model(save_model=True)
    logger.info("==========training completed===============")

    score = trainer.evaluate_model(show_report=True)
    logger.info(f"test score:{score}")
    score = trainer.inf_model(show_report=True)
    logger.info(f"inf score:{score}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser(description="training model")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--tune_trials", type=int, default=1)
    parser.add_argument("--balance_data", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    # args = parser.parse_args(['--model_name', 'logistic_regression', '--tune_trials', '1', '--balance_data'])

    main(
        model_name=args.model_name,
        tune_trials=args.tune_trials,
        balance_data=args.balance_data,
    )
