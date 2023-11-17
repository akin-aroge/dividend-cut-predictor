import logging
import configparser
import argparse
import warnings

from src.utils import utils
from src.modelling import training as train
from src.data_processing import augmentation as aug

warnings.filterwarnings('ignore')


def main(model_name:str, tune_trials=10):
    logger = logging.getLogger(__name__)
    proj_root = utils.get_proj_root()

    config = configparser.ConfigParser(interpolation=None)
    config.read(proj_root.joinpath('config/data_config.ini'))

    final_year = int(config['year_limits']['end_year'])

    training_data_rel_path = config['data_paths']['preprocessed_data_path']
    training_data_path =  proj_root.joinpath(training_data_rel_path)
    feature_set_path = proj_root.joinpath(config['modelling_paths']['optimal_features'])
    model_output_dir = proj_root.joinpath(config['modelling_paths']['model_output'])

    label_col_name = 'dps_change_next_year'
    optimal_features = train.get_features(feature_set_path) 

    model_params = config._sections[model_name]
    
    model_class = train.get_model_class(model_name=model_name)
    model = model_class(**model_params)

    # get data
    training_data = train.get_training_data(file_path=training_data_path)

    # split dataset
    training_data_subset, testing_data_subset = train.train_test_split(df=training_data, final_year=final_year)

    training_data_subset_resampled = aug.balance_data(training_data_subset, label_col_name=label_col_name)

    training_data_subset = training_data_subset_resampled[optimal_features+[label_col_name]]
    testing_data_subset = testing_data_subset[optimal_features+[label_col_name]]

    model_output_path = model_output_dir.joinpath(model_name+'.pkl')
    trainer = train.ModelTrainer(model_class=model,
                                    training_data=training_data_subset,
                                    testing_data=testing_data_subset,
                                    label_col_name=label_col_name,
                                    model_output_path=model_output_path)

    trainer.tune_model(n_trials=tune_trials)
    logger.info('tuning completed')
    model = trainer.train_model(save_model=True)
    logger.info('training completed')

    score = trainer.evaluate_model(show_report=True)
    logger.info(f'test score:{score}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser(description='training model')
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--tune_trials", type=int, default=1)
    args = parser.parse_args()

    main(model_name=args.model_name,
         tune_trials=args.tune_trials)
