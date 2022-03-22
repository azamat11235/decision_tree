import os

from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor

import numpy as np

class PotentialTransformer:
    """
    A potential transformer.

    This class is used to convert the potential's 2d matrix to 1d vector of features.
    """
    
    def _h(self, img):
        argmin = img.argmin()
        m = img[argmin//img.shape[1], argmin%img.shape[1]]
        row = img[argmin//img.shape[1]]
        m = argmin%img.shape[1] + row[row==m].shape[0]//2
        c = img.shape[1]//2
        if m > c:
            for i in range(256):
                if i <= 256-m+c-1:
                    img[:, i] = img[:, i+m-c]
                else:
                    img[:, i] = img[:, -i]
        elif m < c:
            for i in range(255, -1, -1):
                if i-(c-m) >= 0:
                    img[:, i] = img[:, i-(c-m)]
                else:
                    img[:, i] = img[:, -i]

    def _v(self, img):
        argmin = img.argmin()
        m = img[argmin//img.shape[1], argmin%img.shape[1]]
        col = img[:, argmin%img.shape[1]]
        m = argmin//img.shape[1] + col[col==m].shape[0]//2
        c = img.shape[0]//2
        if m > c:
            for i in range(256):
                if i <= 256-m+c-1:
                    img[i, :] = img[i+m-c, :]
                else:
                    img[i, :] = img[-i, :]

        elif m < c:
            for i in range(255, -1, -1):
                if i-(c-m) >= 0:
                    img[i, :] = img[i-(c-m), :]
                else:
                    img[i, :] = img[-i, :]
                    
    def fit(self, x, y):
        """
        Build the transformer on the training set.
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: trained transformer
        """
        return self

    def fit_transform(self, x, y):
        """
        Build the transformer on the training set and return the transformed dataset (1d vectors).
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: transformed potentials (list of 1d vectors)
        """
        return self.transform(x)

    def transform(self, x):
        """
        Transform the list of potential's 2d matrices with the trained transformer.
        :param x: list of potential's 2d matrices
        :return: transformed potentials (list of 1d vectors)
        """
        for xi in x:
            self._h(xi)
            self._v(xi)
        return x.reshape((x.shape[0], -1))

def load_dataset(data_dir):
    """
    Read potential dataset.

    This function reads dataset stored in the folder and returns three lists
    :param data_dir: the path to the potential dataset
    :return:
    files -- the list of file names
    np.array(X) -- the list of potential matrices (in the same order as in files)
    np.array(Y) -- the list of target value (in the same order as in files)
    """
    files, X, Y = [], [], []
    for file in os.listdir(data_dir):
        potential = np.load(os.path.join(data_dir, file))
        files.append(file)
        X.append(potential["data"])
        Y.append(potential["target"])
    return files, np.array(X), np.array(Y)

def train_model_and_predict(train_dir, test_dir):
    _, X_train, Y_train = load_dataset(train_dir)
    test_files, X_test, _ = load_dataset(test_dir)
    # it's suggested to modify only the following line of this function
    regressor = Pipeline([('vectorizer', PotentialTransformer()), ('decision_tree', ExtraTreesRegressor(criterion='mae', n_estimators=2600, max_features=X_train.shape[1]//8, random_state=42))])
    regressor.fit(X_train, Y_train)
    predictions = regressor.predict(X_test)
    return {file: value for file, value in zip(test_files, predictions)}


