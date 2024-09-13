import numpy as np
import random

def split(valid_size: float, test_size: float, feats, labels):
    """
    train test split a set of param graph features. 

    Args:
    - valid_size and test_size: floats, proportions. train size = 1 - valid_size - test_size
    - feats: 
    """
    n = len(feats)
    indices = np.arange(n)
    random.shuffle(indices)
    num_valid = int(n * valid_size)
    num_test = int(n * test_size)
    num_train = int(n - (num_valid + num_test))
    assert num_train > 0, 'be nice and leave something to train on'

    train_indices = indices[:num_train]
    valid_indices = indices[num_train : num_train + num_valid]
    test_indices = indices[num_train + num_valid :]

    feats_train = [feats[i] for i in train_indices]
    feats_valid = [feats[i] for i in valid_indices]
    feats_test = [feats[i] for i in test_indices]

    labels_train = [labels[i] for i in train_indices]
    labels_valid = [labels[i] for i in valid_indices]
    labels_test = [labels[i] for i in test_indices]

    train_set = [feats_train, labels_train]
    valid_set = [feats_valid, labels_valid]
    test_set = [feats_test, labels_test]

    return train_set, valid_set, test_set
