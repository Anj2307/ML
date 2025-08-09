import numpy as np

class LinearRegressionmatrices:
    def __init__(self):
        self.theta = None

    def fit(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        self.theta = np.linalg.inv(X_b.T @ X_b) @ (X_b.T @ Y)

    def predict(self, X):
        if self.theta is None:
            raise ValueError("Model not trained yet")
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.theta

    def get_pranams(self):
        if self.theta is None:
            raise ValueError("Model not trained yet")
        intercept = self.theta[0]
        coefficients = self.theta[1:]
        print(f"Intercept = {intercept}, Coefficients = {coefficients}")


# Sample run
X = [1, 2, 3, 4, 5]
y = np.array([1, 2, 1.3, 3.75, 2.5])

model = LinearRegressionmatrices()
model.fit(X, y)
model.get_pranams()
