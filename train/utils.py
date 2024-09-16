import numpy as np
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# def split(valid_size: float, test_size: float, feats, labels):
#     """
#     train test split a set of param graph features.

#     Args:
#     - valid_size and test_size: floats, proportions. train size = 1 - valid_size - test_size
#     - feats:
#     """
#     n = len(feats)
#     indices = np.arange(n)
#     random.shuffle(indices)
#     num_valid = int(n * valid_size)
#     num_test = int(n * test_size)
#     num_train = int(n - (num_valid + num_test))
#     assert num_train > 0, "be nice and leave something to train on"

#     train_indices = indices[:num_train]
#     valid_indices = indices[num_train : num_train + num_valid]
#     test_indices = indices[num_train + num_valid :]

#     feats_train = [feats[i] for i in train_indices]
#     feats_valid = [feats[i] for i in valid_indices]
#     feats_test = [feats[i] for i in test_indices]

#     labels_train = [labels[i] for i in train_indices]
#     labels_valid = [labels[i] for i in valid_indices]
#     labels_test = [labels[i] for i in test_indices]

#     train_set = [feats_train, labels_train]
#     valid_set = [feats_valid, labels_valid]
#     test_set = [feats_test, labels_test]

#     return train_set, valid_set, test_set


def split(
    features, labels, test_size=0.1, valid_size=None
) -> tuple[tuple[list, list], tuple[list, list], tuple[list, list]]:
    """
    Split features and labels into train, test, and valid subsets.

    Args:
    - features, labels (list): List of features and labels.
    - test_size (float): Proportion of the data to include in the test set.
    - valid_size (float, optional): Proportion of the data to include in the validation set. Defaults to None.

    Returns:
    - 3-tuple containing 2-tuples of train, test, and valid subsets of features and labels.
    """

    valid_size = valid_size or test_size

    features, labels = shuffle(features, labels)

    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=test_size, random_state=0
    )
    train_features, valid_features, train_labels, valid_labels = train_test_split(
        train_features, train_labels, test_size=valid_size, random_state=0
    )

    return (
        (train_features, train_labels),
        (test_features, test_labels),
        (valid_features, valid_labels),
    )
