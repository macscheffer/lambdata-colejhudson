"""
    Description
    -----------
    Extensions to the sklearn package
"""

from sklearn import base

def nop(*_args, **_kwargs):
    """
    Does nothing.
    """
    pass

class TransformerWrapper(base.TransformerMixin):
    """
    Uses the given hooks to wrap the fit and transform methods of
    the given object.

    Parameters
    ----------
    transformer: Class
        Transformer object supplied with a fit and transform method
    fit_hook: Callable
        Function called before invoking the fit method of the given
        object
    transform_hook: Callable
        Function called before invoking the transform method of the
        given object

    Examples
    --------
    >>> from sklearn import preprocessing
    >>> from sklearn import datasets
    >>> from lambdata import misc
    >>> import pandas as pd

    >>> wine_dataset = datasets.load_wine()
    >>> wine = pd.DataFrame(wine_dataset.data, columns=wine_dataset.feature_names)
    >>> scaler = preprocessing.StandardScaler()

    >>> x_shape = None
    >>> def fit_hook(X, y):
            global x_shape
            print('Hooked!')
            x_shape = X.shape

    >>> hooked = misc.sklearn.TransformerWrapper(transformer=scaler, fit_hook=fit_hook)

    >>> print(x_shape)
    None
    >>> scaled = hooked.fit_transform(wine_dataset.data)
    Hooked!
    >>> print(x_shape)
    (178, 13)
    """

    def __init__(self, transformer=None, fit_hook=nop, transform_hook=nop):
        if not isinstance(transformer, type(transformer)):
            self.transformer = transformer()
        else:
            self.transformer = transformer

        self.fit_hook = fit_hook
        self.transform_hook = transform_hook

    def fit(self, X, y=None):
        self.fit_hook(X, y)

        if self.transformer is not None:
            self.transformer.fit(X, y)

        return self

    def transform(self, X, y=None):
        self.transform_hook(X, y)

        if self.transformer is not None:
            return self.transformer.transform(X, y)
        
        return X

class LoggingTransformer(TransformerWrapper):
    """
    Parameters
    ----------
    fit_message: str
        String to be printed before the fit method
        of the wrapped object is called
    transform_messager: str
        String to be printed before the transform method
        of the wrapped object is called

    Examples
    --------
    >>> from lambdata import misc
    >>> logger = misc.sklearn.LoggingTransformer(
    ...     fit_message="fit!",
    ...     transform_message="transformed!"
    ... )
    >>> logger.fit_transform([1,2,3])
    fit!
    transformed!
    [1,2,3]
    """

    def __init__(self, fit_message='', transform_message='', **kwargs):
        def fit_hook(*_args):
            print(fit_message)

        def transform_hook(*_args):
            print(transform_message)

        super(LoggingTransformer, self).__init__(fit_hook=fit_hook, transform_hook=transform_hook, **kwargs)
