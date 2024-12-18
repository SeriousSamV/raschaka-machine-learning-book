from typing import Optional

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


# noinspection PyPep8Naming
class AdalineSGD:
    """
    Implements an Adaptive Linear Neuron (Adaline) classifier using stochastic gradient
    descent (SGD).

    The AdalineSGD class provides functionality for training and predicting with an
    adaptive linear neuron classifier. The classifier minimizes the cost function
    using stochastic gradient descent for optimization. This implementation
    includes adjustable learning rate, number of iterations, random state for
    reproducibility, and optional shuffling of the training data to improve learning
    performance. It also supports incremental learning via partial fitting.

    Attributes
    ----------
    eta : float
        Learning rate for weight updates (default: 0.01).
    n_iter : int
        Number of passes over the training dataset (default: 10).
    shuffle : bool
        Whether to shuffle the training data before each epoch (default: True).
    random_state : int | None
        Determines the random number generation for shuffling and weight
        initialization (default: None).
    losses_ : List[float]
        A record of the average loss of the model in each iteration.
    w_initialized : bool
        Indicator of whether the weights have been initialized.
    w_ : 1d-array
        Weight vector for the features after the training process.
    b_ : float
        Bias unit value after completing the training process.
    rgen : RandomState
        NumPy random state instance used for random number generation and
        reproducibility.
    """

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.losses_ = []
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state: Optional[int] = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdalineSGD":
        """
        Fits the Adaline Stochastic Gradient Descent (SGD) model to the training data.
        This function trains the model by iteratively updating the weights based on
        randomly shuffled data for a specified number of iterations. The average loss
        per epoch is computed and stored to monitor the learning progress.

        Parameters
        ----------
        X : np.ndarray, shape = [n_samples, n_features]
            Feature matrix of shape (n_samples, n_features), where n_samples is the
            number of samples and n_features is the number of features.
        y : np.ndarray, shape = [n_samples]
            Target vector of shape (n_samples,), representing the class labels.

        Returns
        -------
        self : AdalineSGD
            Fitted instance of the `AdalineSGD` class.
        """
        self._initialize_weights(X.shape[1])
        self.losses_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = []
            for xi, target in zip(X, y):
                losses.append(self._update_weights(xi, target))
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)
        return self

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> "AdalineSGD":
        """
            Update the model weights incrementally using a subset of the training data.

            This method performs a partial fit on the provided training data such that
            the weights are updated iteratively for individual samples or small batches.
            If the weights have not been initialized previously, they will be initialized
            before updating.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The input data used for training the model. Each row represents a sample,
                and each column represents a feature.

            y : array-like, shape (n_samples,)
                The target values corresponding to the input data X. Each entry is the
                target value associated with the respective input sample in X.

            Returns
            -------
            object
                Returns the instance of the model itself after performing partial fitting
                for method chaining.
            """
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float_(0.)
        self.w_initialized = True

    def _update_weights(self, xi: np.ndarray, target: float) -> float:
        """
            Updates the weights of the model using the given input and target values.

            This operation adjusts the weights (`self.w_`) and bias (`self.b_`) of the
            perceptron based on the calculated error between the predicted value and
            the target value. Additionally, the loss, computed as the squared error,
            is returned for tracking purposes.

            Parameters
            ----------
            xi : array-like of shape (n_features,)
                Input features for the current sample used to update the weights.

            target : float
                Target output value corresponding to the input sample.

            Returns
            -------
            loss : float
                Squared error computed as the square of the difference between the
                target value and the predicted output.
            """
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_ += self.eta * 2. * xi * error
        self.b_ += self.eta * 2. * error
        loss = error ** 2
        return loss

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    @staticmethod
    def activation(X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, -1)
