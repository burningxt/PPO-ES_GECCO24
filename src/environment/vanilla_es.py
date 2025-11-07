import numpy as np
from src.config.config import POP_SIZE

class ES:
    def __init__(self, dim, sigma_0=0.5, lambda_=POP_SIZE):
        self.dim = dim
        self.sigma = sigma_0
        self.lambda_ = lambda_

        self.mu = self.lambda_ // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)


        self.B = np.eye(dim)  # B defines the coordinate system
        self.D = np.ones(dim)  # Diagonal D defines the scaling

        # Initialize mean
        self.xmean = np.zeros(dim)

    def ask(self):
        # Generate and return lambda_ offspring
        arz = np.random.randn(self.lambda_, self.dim)
        # 
        return self.xmean + self.sigma * (self.B @ (self.D * arz).T).T

        # this returns a NumPy array of shape (lambda, dim)
        # represents the offsprint candidate solutions (each row is one vector in the search space)

    def tell(self, arx, fitnesses):
        # Update the state of the CMA-ES based on evaluated solutions
        sorted_idx = np.argsort(fitnesses)
        self.xmean = arx[sorted_idx[:self.mu]].T @ self.weights

