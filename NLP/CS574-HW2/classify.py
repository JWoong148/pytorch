from sklearn import metrics
from sklearn.linear_model import LogisticRegression

def train_classifier(X, y):
    """Train a classifier using the given training data.
    Trains a logistic regression on the input data with default parameters.
    """
    cls = LogisticRegression()
    cls.fit(X, y)
    return cls

def evaluate(X, yt, cls):
    """Evaluated a classifier on the given labeled data using accuracy."""
    yp = cls.predict(X)
    acc = metrics.accuracy_score(yt, yp)
    print("  Accuracy", acc)