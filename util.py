import pandas as pd
import numpy as np

def prepare_data():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    
    # Treat ints as floats
    train_int_columns = train.select_dtypes(include=['int64']).columns
    train[train_int_columns] = train[train_int_columns].astype(float)
    test_int_columns = test.select_dtypes(include=['int64']).columns
    test[test_int_columns] = test[test_int_columns].astype(float)

    # Replace large negative number with 0
    train.replace(-999999.0, 0.0, inplace=True)
    test.replace(-999999.0, 0.0, inplace=True)
    
    # Remove columns that are constant in train.csv
    const_columns = train.loc[:, (train == train.ix[0]).all()].columns 
    train = train.drop(const_columns, axis=1)
    test = test.drop(const_columns, axis=1)

    # Compute vectors/matrices
    X_train = train.drop(["ID", "TARGET"], axis=1).values
    y_train = train.TARGET.values
    X_test = test.drop(["ID"], axis=1).values
    test_ids = test.ID.astype(int)
    return X_train, y_train, X_test, test_ids

