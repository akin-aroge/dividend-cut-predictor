import logging
import configparser

from src.utils import utils
from src.modelling import training as train
from src.data_processing import augmentation as aug

logger = logging.getLogger(__name__)
proj_root = utils.get_proj_root()

config = configparser.ConfigParser(interpolation=None)
config.read(proj_root.joinpath('config/data_config.ini'))

final_year = int(config['year_limits']['end_year'])
training_data_rel_path = config['data_paths']['preprocessed_data_path']



training_data_path =  proj_root.joinpath(training_data_rel_path)
training_data_subset_path  = proj_root.joinpath(config['data_paths']['training_subset_path'])
testing_data_subset_path  = proj_root.joinpath(config['data_paths']['testing_subset_path'])
label_col_name = 'dps_change_next_year'



def main():

    
    # get data
    training_data = train.get_training_data(file_path=training_data_path)
    
    # split dataset
    training_data_subset, testing_data_subset = train.train_test_split(df=training_data, final_year=final_year)

    # # Xy split
    # X_train, y_train = train.split_Xy(training_data_subset, label_col_name=label_col_name)

    # balance dataset
    # X_train_oversampled, y_train_oversampled = aug.balance_data(X=X_train, y=y_train)
    training_data_subset_resampled = aug.balance_data(training_data_subset, label_col_name=label_col_name)

    # save test and train data
    train.save_data(training_data_subset_resampled, path=training_data_subset_path)
    train.save_data(testing_data_subset, path=testing_data_subset_path)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()