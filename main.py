import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from data import names, races


def name_features(name):
    arr = np.zeros(26)
    for c in name:
        arr[c - ord("A")] += 1
    return arr


featurize = np.vectorize(name_features, otypes=[np.ndarray])
X = np.array(featurize(names).tolist())
y = races

for x in range(5):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.33)
    clf = RandomForestClassifier(n_estimators=100, min_samples_split=2)
    clf.fit(Xtr, ytr)
    print(np.mean(clf.predict(Xte) == yte))
