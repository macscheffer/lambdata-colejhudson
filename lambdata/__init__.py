"""
    Description
    -----------
    An example python package
"""

import pandas as pd
import numpy as np

def hasna(dataframe):
    """
        Parameters
        ----------
        dataframe: pandas.DataFrame
            Non-empty DataFrame

        Returns
        -------
        mask: pandas.Series, shape = [n_columns]
            A series of booleans indicating whether the
            corresponding column contains any NaN values

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> import lambdata
        >>> foo = pd.DataFrame([[1, 2], [3, 4], [5, np.nan]])
        >>> lambdata.hasna(foo)
        0    False
        1     True
        dtype: bool
    """
    mask = dataframe.isna().sum().apply(lambda nancount: nancount > 0)

    return mask

def confusion_matrix(true, pred):
    """
        Parameters
        ----------
        true: numpy.ndarray, shape = [n]
            Actual classes
        pred: numpy.ndarray, shape = [n]
            Predicted classes

        Returns
        -------
        mat: numpy.ndarray, shape = [2, 2]
            Confusion matrix

        Examples
        --------
        >>> import lambdata
        >>> true = ["foo", "foo", "bar", "foo", "bar"]
        >>> pred = ["foo", "bar", "foo", "foo", "bar"]
        >>> lambdata.confusion_matrix(true, pred)
        array([[2, 1],
               [1, 1]])
    """
    true = pd.Series(true)
    pred = pd.Series(pred)

    classes = true.unique()

    err_msg = """
        good god man, this is for demonstrations purposes only.
        For the love of all that is holy, use scikit-learn instead.

        >>> from sklearn.metrics import confusion_matrix
    """
    assert classes.size == 2, err_msg

    pos, neg = classes

    true_positives = pred[(true == pos) & (pred == pos)].size
    false_positives = pred[(true == pos) & (pred == neg)].size
    true_negatives = pred[(true == neg) & (pred == neg)].size
    false_negatives = pred[(true == neg) & (pred == pos)].size

    mat = np.array([[true_positives, false_positives], [false_negatives, true_negatives]])

    return mat
