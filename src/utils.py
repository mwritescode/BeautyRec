import numpy as np

def train_test_split(data, test_prc=0.25, random_seeed=42):
    data_size = len(data)
    test_size = int(data_size * test_prc)
    np.random.seed(random_seeed)
    test_indexes = np.random.randint(low=0, 
                                    high=data_size, 
                                    size=test_size)
    test_split = data[data.index.isin(test_indexes)].copy().reset_index(drop=True)
    train_split = data[~data.index.isin(test_indexes)].copy().reset_index(drop=True)
    return train_split, test_split