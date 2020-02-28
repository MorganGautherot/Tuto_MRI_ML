import pandas as pd
import numpy as np

def train_test_val(data_frame_matrix) :
    
    data_frame_matrix_val = pd.DataFrame(data=None, columns=data_frame_matrix.columns)
    data_frame_matrix_val

    data_frame_matrix_test = pd.DataFrame(data=None, columns=data_frame_matrix.columns)
    data_frame_matrix_test

    a25 = data_frame_matrix[data_frame_matrix['age'] < 25].sample(20, random_state=123)
    data_frame_matrix_val = data_frame_matrix_val.append(a25.iloc[:10,])
    data_frame_matrix_test = data_frame_matrix_test.append(a25.iloc[10:,])

    a30 = data_frame_matrix[np.logical_and(data_frame_matrix['age'] >= 25, data_frame_matrix['age'] < 30)].sample(20, random_state=123)
    data_frame_matrix_val = data_frame_matrix_val.append(a30.iloc[:10,])
    data_frame_matrix_test = data_frame_matrix_test.append(a30.iloc[10:,])

    a35 = data_frame_matrix[np.logical_and(data_frame_matrix['age'] >= 30, data_frame_matrix['age'] < 35)].sample(20, random_state=123)
    data_frame_matrix_val = data_frame_matrix_val.append(a35.iloc[:10,])
    data_frame_matrix_test = data_frame_matrix_test.append(a35.iloc[10:,])

    a40 = data_frame_matrix[np.logical_and(data_frame_matrix['age'] >= 35, data_frame_matrix['age'] < 40)].sample(20, random_state=123)
    data_frame_matrix_val = data_frame_matrix_val.append(a40.iloc[:10,])
    data_frame_matrix_test = data_frame_matrix_test.append(a40.iloc[10:,])

    a45 = data_frame_matrix[np.logical_and(data_frame_matrix['age'] >= 40, data_frame_matrix['age'] < 45)].sample(20, random_state=123)
    data_frame_matrix_val = data_frame_matrix_val.append(a45.iloc[:10,])
    data_frame_matrix_test = data_frame_matrix_test.append(a45.iloc[10:,])

    a50 = data_frame_matrix[np.logical_and(data_frame_matrix['age'] >= 45, data_frame_matrix['age'] < 50)].sample(20, random_state=123)
    data_frame_matrix_val = data_frame_matrix_val.append(a50.iloc[:10,])
    data_frame_matrix_test = data_frame_matrix_test.append(a50.iloc[10:,])

    a55 = data_frame_matrix[np.logical_and(data_frame_matrix['age'] >= 50, data_frame_matrix['age'] < 55)].sample(20, random_state=123)
    data_frame_matrix_val = data_frame_matrix_val.append(a55.iloc[:10,])
    data_frame_matrix_test = data_frame_matrix_test.append(a55.iloc[10:,])

    a60 = data_frame_matrix[np.logical_and(data_frame_matrix['age'] >= 55, data_frame_matrix['age'] < 60)].sample(20, random_state=123)
    data_frame_matrix_val = data_frame_matrix_val.append(a60.iloc[:10,])
    data_frame_matrix_test = data_frame_matrix_test.append(a60.iloc[10:,])

    a65 = data_frame_matrix[np.logical_and(data_frame_matrix['age'] >= 60, data_frame_matrix['age'] < 65)].sample(20, random_state=123)
    data_frame_matrix_val = data_frame_matrix_val.append(a65.iloc[:10,])
    data_frame_matrix_test = data_frame_matrix_test.append(a65.iloc[10:,])

    a70 = data_frame_matrix[np.logical_and(data_frame_matrix['age'] >= 65, data_frame_matrix['age'] < 70)].sample(20, random_state=123)
    data_frame_matrix_val = data_frame_matrix_val.append(a70.iloc[:10,])
    data_frame_matrix_test = data_frame_matrix_test.append(a70.iloc[10:,])

    a75 = data_frame_matrix[data_frame_matrix['age'] >= 70].sample(20, random_state=123)
    data_frame_matrix_val = data_frame_matrix_val.append(a75.iloc[:10,])
    data_frame_matrix_test = data_frame_matrix_test.append(a75.iloc[10:,])
    
    data_frame_matrix_train = data_frame_matrix.drop(data_frame_matrix_test.index)
    data_frame_matrix_train = data_frame_matrix_train.drop(data_frame_matrix_val.index)

    return data_frame_matrix_train, data_frame_matrix_test, data_frame_matrix_val
