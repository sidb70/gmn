from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def split(
    features: list, labels: list, test_size=0.1, valid_size=None
) -> tuple[tuple[list, list], tuple[list, list], tuple[list, list]]:
    """
    Split features and labels into train, test, and valid subsets.

    Args:
    - features (list), labels (list): Lists of features and labels.
    - test_size (float): Proportion of the data to include in the test set.
    - valid_size (float, optional): Proportion of the train data to include in the validation set.
        Defaults to test_size

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
