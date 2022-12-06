# Copyright Sam Maddrell-Mander 2022

import logging

import numpy as np
import pandas as pd
from pathlib import Path
from os import path


from data_loader import (get_subclasses, get_superclasses,
                         get_train_test_split, load_dataset,
                         save_ptbxl_cache, load_ptbxl_cache)
from data_utils import preprocess_data
from models.simple_classifier import simple_decision_tree_calssifier
from models.simple_net import train_simple_net
from models.conv_net import train_conv_net
from utils import Timer

def main():
    data_path = 'physionet.org/files/ptb-xl/1.0.3/'
    cache_path = "data_cache"
    sampling_rate=100
    agg_df = pd.read_csv(data_path+'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    
    # TODO: check if the cache exists - if not load the dataset properly
    # TODO: Command line arg to load if we want to 
    if not path.isfile(cache_path + "/ptbxl_dataframe.pkl"):
        with Timer("Loading dataset"):
            X, Y = load_dataset(data_path, sampling_rate)
            # If loaded the dataset - save to cache
            save_ptbxl_cache(cache_path, X, Y)
    else:
        with Timer("Loading from cache"):
            X, Y = load_ptbxl_cache(cache_path)
            
        
    with Timer("Getting splits"):
        X_train, y_train, X_test, y_test = get_train_test_split(X, Y)
    
    # Basic classifier 
    # simple_decision_tree_calssifier(X_train, y_train, X_test, y_test)
    with Timer("Creating DataLoader objects"):
        dataloader_train = preprocess_data(X_train, y_train)
        dataloader_test = preprocess_data(X_test, y_test)
    
    # with Timer("Training Simple model"):
    #     train_simple_net(dataloader_train, dataloader_test)
    
    with Timer("Convolutional Net"):
        train_conv_net(dataloader_train, dataloader_test)
        
        

if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    main()