

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split as TTS
from sklearn.naive_bayes import GaussianNB

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = TTS(X, y, test_size = 0.5, random_state=13)


gnb_clf = GaussianNB()

y_pred = gnb_clf.fit(X_train, y_train).predict(X_test)

mislabeled_pts = (y_test != y_pred).sum()
print(f"Mislabeled points out of {X.shape[0]}: {mislabeled_pts}")

# eulers_number = lambda n: (1 + (1 / n)) ** n

# for i in range(100, 120): print( eulers_number(i))