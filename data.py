import numpy as np
from sklearn.externals import joblib

DATA_DIR = "data"


def load_data(fname):
    data = np.genfromtxt(
        fname,
        delimiter=" ",
        dtype=[("name", "S50"), ("race", "S1")],
        converters={0: lambda s: s.upper()},
    )
    return (data["name"], data["race"])


def serialize_model(fname, clf):
    return joblib.dump(clf, f"{DATA_DIR}/{fname}.joblib")


def deserialize_model(fname):
    return joblib.load(f"{DATA_DIR}/{fname}.joblib")
