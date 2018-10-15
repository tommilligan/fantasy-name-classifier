from collections import defaultdict

import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier


from data import load_data, serialize_model
from features import featurize
from test import test_model

RANDOM_STATE = 0


def main():
    names, races = load_data("train.txt")
    X = featurize(names)
    y = races

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.33, random_state=RANDOM_STATE
    )
    clf = RandomForestClassifier(
        n_estimators=100, min_samples_split=2, random_state=RANDOM_STATE
    )

    clf.fit(Xtr, ytr)
    serialize_model("model", clf)

    score = test_model(clf, (Xte, yte))
    print(score)
    top_features = np.argsort(clf.feature_importances_)[::-1][:10]
    print(top_features)


if __name__ == "__main__":
    main()
