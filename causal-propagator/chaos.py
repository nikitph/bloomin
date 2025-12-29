import numpy as np

def generate_logistic_map(n_steps, x0=0.5, r=3.99):
    """
    A simple chaotic system: The Logistic Map.
    x_{n+1} = r * x_n * (1 - x_n)
    """
    x = np.zeros(n_steps)
    x[0] = x0
    for i in range(1, n_steps):
        x[i] = r * x[i-1] * (1 - x[i-1])
    return x

class ShadowForecaster:
    """
    The L1 Shadow: Local Extrapolation / Polynomial Fitting.
    Sensitive to the 'Butterfly Effect'.
    """
    def __init__(self, degree=3):
        self.degree = degree
        self.coeffs = None

    def fit(self, data):
        # We try to fit a curve to the recent history
        x = np.arange(len(data))
        self.coeffs = np.polyfit(x, data, self.degree)

    def predict(self, horizon):
        # We extrapolate the polynomial into the future
        x_future = np.arange(len(self.coeffs) + 1, len(self.coeffs) + 1 + horizon)
        # Note: Polynomials explode at infinity - the definition of an unstable forecast
        return np.polyval(self.coeffs, x_future)
