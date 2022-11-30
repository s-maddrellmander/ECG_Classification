import pandas as pd
import numpy as np
import wfdb
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

import logging


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


def get_superclasses(Y, agg_df):
    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in set(agg_df.index):
                if y_dic[key] == 100.0:
                    tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    # Apply diagnostic superclass
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)
    Y = Y[Y["diagnostic_superclass"].str.len() != 0]

    # Turn strings to ints
    def to_int(vals):
        # print(inputs[0])
        mapper = {'NORM': 0, 'STTC': 1, 'HYP': 2, 'CD': 3, 'MI': 4}
        return mapper[vals[0]]

    Y["diagnostic_superclass"] = Y["diagnostic_superclass"].apply(to_int)
    return Y


def get_subclasses(Y, agg_df):
    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                if y_dic[key] == 100.0:
                    tmp.append(agg_df.loc[key].diagnostic_subclass)
        return list(set(tmp))

    # Apply diagnostic superclass
    Y['diagnostic_subclass'] = Y.scp_codes.apply(aggregate_diagnostic)
    # import ipdb; ipdb.set_trace()

    Y = Y[Y["diagnostic_subclass"].str.len() != 0]
    # Sub class as one hot encoding
    mlb = MultiLabelBinarizer()
    Y = Y.join(
        pd.DataFrame(mlb.fit_transform(Y['diagnostic_subclass']),
                     columns=mlb.classes_,
                     index=Y.index))
    return Y


def load_dataset(path, sampling_rate):
    # load and convert annotation data
    Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    Y = get_superclasses(Y, agg_df)
    Y = get_subclasses(Y, agg_df)
    # Remove subclasses with few samples

    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, path)
    return X, Y


def get_train_test_split(X, Y):
    # Split data into train and test
    test_fold = 10
    # Train
    X_train = X[np.where(Y.strat_fold != test_fold)]
    y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
    # Test
    X_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
    logging.info(f"Test / Train split is only returning superclass.")
    return X_train, y_train, X_test, y_test


def save_ptbxl_cache(path, X, Y):
    # TODO: Check if the dir exists and make if not
    pd.to_pickle(Y, f"{path}/ptbxl_dataframe.pkl")
    np.save(f"{path}/ptbxl_traces.npy", X)


def load_ptbxl_cache(path):
    # TODO: Check these files exist
    Y = pd.read_pickle(f"{path}/ptbxl_dataframe.pkl")
    X = np.load(f"{path}/ptbxl_traces.npy")
    return X, Y
