import pandas as pd
import numpy as np

def hasna(df):
    """
        >>> import lambdata
        >>> foo = pd.DataFrame([[1, 2], [3, 4], [5, np.nan]])
        >>> lambdata.hasna(foo)
        0    False
        1     True
        dtype: bool
    """
    return df.isna().sum().apply(lambda nancount: nancount > 0)

def confusion_matrix(true, pred):
    """
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
    assert classes.size == 2, "good god man, this is for demonstrations purposes only, for the love of all that is good, use scikit-learn instead."

    pos, neg = classes 

    tp = pred[(true == pos) & (pred == pos)].size
    fp = pred[(true == pos) & (pred == neg)].size
    tn = pred[(true == neg) & (pred == neg)].size
    fn = pred[(true == neg) & (pred == pos)].size

    return np.array([[tp, fp], [fn, tn]])
