from imblearn.over_sampling import SMOTE
from src.modelling import training as train
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# def balance_data(X, y, random_state=None):

#     smote = SMOTE(random_state=random_state)
#     X_resample, y_resample = smote.fit_resample(X, y)

#     initial_label_counts = y.value_counts().to_dict()
#     new_label_counts = y_resample.value_counts().to_dict()
#     logger.info(f"balanced data, initial label counts:{initial_label_counts} | new label counts:{new_label_counts}")

#     # resampled_data = 

#     return X_resample, y_resample

def balance_data(df:pd.DataFrame, label_col_name:str, random_state=None):

    X, y = train.split_Xy(df, label_col_name=label_col_name)

    smote = SMOTE(random_state=random_state)
    X_resample, y_resample = smote.fit_resample(X, y)

    initial_label_counts = y.value_counts().to_dict()
    new_label_counts = y_resample.value_counts().to_dict()
    # logger.info(f"balanced data, initial label counts:{initial_label_counts} | new label counts:{new_label_counts}")

    resampled_data = X_resample
    resampled_data[label_col_name] = y_resample

    logger.info(f"balanced data, initial label counts:{initial_label_counts} \
                 and shape: {df.shape} | new label counts:{new_label_counts} \
                    and shape {resampled_data.shape}")

    return resampled_data