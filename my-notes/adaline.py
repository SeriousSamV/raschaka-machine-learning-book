import numpy as np


# noinspection PyPep8Naming
class AdalineGD:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.losses_ = []
        self.b_ = np.float_(0.)
        self.w_ = None
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdalineGD":
        """
        Fits the Adaline Gradient Descent model to the provided dataset.

        This method adjusts the weights (`self.w_`) and bias (`self.b_`)
        of the Adaline model using gradient descent. The optimization
        aims to minimize the Mean Squared Error (MSE) loss over a
        specified number of iterations (`self.n_iter`). The loss values
        for each iteration are stored in the `self.losses_` attribute.

        Parameters
        ----------
        X : np.ndarray, shape = [n_examples, n_features]
            Training vectors, where `n_examples` is the number of examples
            and `n_features` si the number of features.
        y : np.ndarray, shape = [n_examples]
            The target values corresponding to the input dataset `X`.

        Returns
        -------
        self : AdalineGD
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.losses_ = []
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            # below is equivalent to:
            # ```python
            # for w_j in range(self.w_.shape[0]):
            #     self.w_[w_j] += self.eta * ((2. * (X[:, w_j] * errors)).mean())
            # ```
            self.w_ += self.eta * 2. * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2. * errors.mean()
            loss = (errors ** 2).mean()
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    @staticmethod
    def activation(X):
        """Linear activation function"""
        return X

    def predict(self, X):
        """Return class label after unit step on values from `net_input`"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, -1)
