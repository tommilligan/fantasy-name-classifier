import sys

import numpy as np

from data import load_data, deserialize_model
from features import featurize


def test_model(clf, data):
    X, y = data
    score = np.mean(clf.predict(X) == y)
    return score


def main():
    names, races = load_data(sys.argv[1])
    X = featurize(names)
    y = races

    clf = deserialize_model("model")
    score = test_model(clf, (X, y))
    print(score)


if __name__ == "__main__":
    main()
