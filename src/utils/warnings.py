import warnings
import pandas as pd
from functools import wraps

def suppress_pandas_warnings(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            warnings.simplefilter("ignore", category=pd.errors.SettingWithCopyWarning)
            return func(*args, **kwargs)
    return wrapper
