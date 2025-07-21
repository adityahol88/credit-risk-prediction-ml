import numpy as np


class SVM:
    def __init__(self, kernel="linear", C=1.0, max_iter=10000, degree=3, gamma=1.0):
        # Parameters:
        #     kernel (str): Kernel type ('linear', 'poly', 'rbf').
        #     C (float): Regularization parameter.
        #     max_iter (int): Maximum number of iterations.
        #     degree (int): Degree for polynomial kernel.
        #     gamma (float): Gamma for RBF kernel.
        self.kernel = self._get_kernel(kernel, degree, gamma)
        self.C = C
        self.max_iter = max_iter

    def _get_kernel(self, kernel, degree, gamma):
        if kernel == "linear":
            return lambda x, y: np.dot(x, y.T)
        elif kernel == "poly":
            return lambda x, y: (np.dot(x, y.T) + 1) ** degree
        elif kernel == "rbf":
            return lambda x, y: np.exp(
                -gamma
                * (
                    np.sum(x**2, axis=1)[:, np.newaxis]
                    + np.sum(y**2, axis=1)
                    - 2 * np.dot(x, y.T)
                )
            )
        else:
            raise ValueError("Unsupported kernel type.")

    def _restrict_to_box(self, t, v0, u):
        # Restricting lambda updates to the box constraints [0, C].
        t = (np.clip(v0 + t * u, 0, self.C) - v0)[1] / u[1]
        return (np.clip(v0 + t * u, 0, self.C) - v0)[0] / u[0]

    def fit(self, X, y):

        # Convert labels to {-1, 1}
        self.X = X
        self.y = 2 * y - 1
        n_samples = len(y)

        self.lambdas = np.zeros(n_samples, dtype=float)
        self.K = self.kernel(self.X, self.X) * self.y[:, np.newaxis] * self.y

        for _ in range(self.max_iter):
            idx1, idx2 = np.random.choice(n_samples, size=2, replace=False)
            Q = self.K[[[idx1, idx1], [idx2, idx2]], [[idx1, idx2], [idx1, idx2]]]
            v0 = self.lambdas[[idx1, idx2]]
            k0 = 1 - np.sum(self.lambdas * self.K[[idx1, idx2]], axis=1)
            u = np.array([-self.y[idx2], self.y[idx1]])
            t_max = np.dot(k0, u) / (np.dot(np.dot(Q, u), u) + 1e-15)
            self.lambdas[[idx1, idx2]] = v0 + u * self._restrict_to_box(t_max, v0, u)

        support_vectors = self.lambdas > 1e-5
        self.b = np.mean(
            self.y[support_vectors]
            - np.sum(
                self.kernel(self.X[support_vectors], self.X) * self.lambdas * self.y,
                axis=1,
            )
        )

    def decision_function(self, X):
        return np.sum(self.kernel(X, self.X) * self.lambdas * self.y, axis=1) + self.b

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)
