import numpy as np
import pandas as pd

from data_loader import get_subclasses, get_superclasses, get_train_test_split


def test_get_superclasses():
    path = 'physionet.org/files/ptb-xl/1.0.3/'
    agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    data = [
        [{
            "NORM": 100.0
        }],
        [{
            "NDT": 100.0
        }],
        [{
            "LVH": 100.0
        }],
        [{
            "NORM": 80.0
        }],
        [{
            "ISCIN": 100.0
        }],
    ]
    Y = pd.DataFrame(data, columns=["scp_codes"])
    Y = get_superclasses(Y, agg_df)

    assert Y["diagnostic_superclass"].min() == 0
    assert Y["diagnostic_superclass"].max() == 2
    assert len(Y) == 4


def test_get_subclasses():
    path = 'physionet.org/files/ptb-xl/1.0.3/'
    agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    data = [
        [{
            'LAFB': 100.0,
            'IVCD': 100.0,
            'SR': 0.0
        }],
        [{
            'NDT': 100.0,
            'PVC': 100.0,
            'VCLVH': 0.0
        }],
        [{
            'NORM': 100.0,
            'ABQRS': 0.0,
            'SR': 0.0
        }],
        [{
            'NORM': 100.0,
            'SR': 0.0
        }],
        [{
            'NORM': 100.0,
            'LVOLT': 0.0,
            'SR': 0.0
        }],
    ]
    Y = pd.DataFrame(data, columns=["scp_codes"])
    Y = get_subclasses(Y, agg_df)
    assert np.all(Y.iloc[0][["IVCD", "LAFB/LPFB"]] == 1)
    assert Y.iloc[1]["STTC"] == 1
    assert Y.iloc[2]["NORM"] == 1
    assert Y.iloc[3]["NORM"] == 1


def test_get_train_test_split():
    dummy = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
    Y = pd.DataFrame(np.array([[0, 0], [0, 0], [0, 0], [1, 10]]),
                     columns=["diagnostic_superclass", "strat_fold"])
    dfX = pd.DataFrame(dummy, columns=['A', 'B', 'C', 'D'])
    X1, y1, X2, y2 = get_train_test_split(dummy, Y)

    assert len(X1) == 3
    assert y2[3] == 1
