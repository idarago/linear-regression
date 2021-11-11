import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class GibbsSampler:
    """
    Gibbs sampler for Bayesian Linear Regression
    Given data y ~ x, we find the distribution of the parameters beta_0, beta_1 of linear regression.
    We update the parameters beta_0, beta_1 according to the conditional distributions.
    """
    def __init__(self, x,y, n_iters = 1000):
        self.x = x
        self.y = y
        self.N = len(y)
        self.n_iters = n_iters

        # Initializing the hyperparameters
        # --------------------------------
        # beta_0 ~ N(mu_0, 1/tau_0)
        # beta_1 ~ N(mu_1, 1/tau_1)
        # y = beta_0 + beta_1 x + e
        # e ~ N(0, 1/tau)
        # tau ~ Gamma(alpha, beta)
        self.beta_0 = 0
        self.beta_1 = 0
        self.mu_0 = 0
        self.mu_1 = 0
        self.tau_0 = 1 
        self.tau_1 = 1
        self.tau = 2
        self.alpha = 2
        self.beta = 1

        # Store all the values of beta_0, beta_1, tau as we update them
        self.trace = np.zeros((self.n_iters,3))

        for _ in range(self.n_iters):
            self.sample_beta_0()
            self.sample_beta_1()
            self.sample_tau() 
            self.trace[_,:] = np.array((self.beta_0, self.beta_1, self.tau))

        self.trace = pd.DataFrame(self.trace)
        self.trace.columns = ["beta_0", "beta_1", "tau"]

    # Update the beta_0 parameter
    def sample_beta_0(self):
        precision = self.tau_0 + self.tau * self.N
        mean = self.tau_0 * self.mu_0 + self.tau * np.sum(self.y - self.beta_1 * self.x)
        mean /= precision
        self.beta_0 = np.random.normal(mean, 1/np.sqrt(precision))
    
    # Update the beta_1 parameter
    def sample_beta_1(self):
        precision = self.tau_1 + self.tau * np.sum(self.x * self.x)
        mean = self.tau_1 * self.mu_1 + self.tau * np.sum((self.y - self.beta_0) * self.x)
        mean /= precision
        self.beta_1 = np.random.normal(mean, 1 / np.sqrt(precision))

    # Update the tau parameter
    def sample_tau(self):
        self.alpha +=  self.N // 2
        residuals = self.y - self.beta_0 - self.beta_1 * self.x
        self.beta += np.sum(residuals * residuals) // 2
        self.tau = np.random.gamma(self.alpha, 1/self.beta)

    def get_trace(self):
        return self.trace

def simulate_linear_data(N, beta_0, beta_1, eps_sigma_sq):
    """
    Simulate a toy dataset using a noisy linear process
    N: number of data points to simulate
    beta_0: intercept
    beta_1: slope
    """
    # Create a Pandas DataFrame with column "x" containing
    # N uniformly sampled values between 0.0 and 1.0
    df = pd.DataFrame({"x":np.random.choice(list(map(lambda x : float(x)/100.0, np.arange(N))), size=N, replace=False)})
    # Use a linear model (y ~ beta_0 + beta_1 * x  + epsilon) to generate a column "y"
    eps_mean = 0.0
    df["y"] = beta_0 + beta_1 * df["x"] + np.random.RandomState(42).normal(eps_mean, eps_sigma_sq, N)
    return df

def bayesianRegression(df, n_iters=1000):
    """
    Bayesian Linear Regression implemented using Gibbs sampling.
    """
    gibbs = GibbsSampler(df["x"], df["y"], n_iters)
    trace = gibbs.get_trace()
    return [gibbs.beta_0, gibbs.beta_1, gibbs.tau, n_iters, trace]

def traceplot(trace, n_iters):
    """
    Plots the histogram of the posterior distributions obtained by our linear regression.
    """
    traceplot = trace.plot()
    traceplot.set_xlabel("Iteration")
    traceplot.set_ylabel("Parameter value")
    plt.show()

    trace_burnt = trace[n_iters - 500: n_iters]
    hist_plot = trace_burnt.hist(bins = 30, layout = (1,3))
    plt.show()

if __name__ == "__main__":
    # These are our "true" parameters
    beta_0 = -1.0 # Intercept
    beta_1 = 2.0  # Slope
    tau = 1.0     # Precision = 1 / Standard Deviation
    eps_sigma_sq = 1 / tau
    N = 400

    # Simulate "linear" data using the above parameters
    df = simulate_linear_data(N, beta_0, beta_1, eps_sigma_sq)
    
    sns.regplot(x="x",y="y", data=df, scatter_kws={"color":"black"}, line_kws={"color":"red"})
    plt.show()

    estimated_beta_0, estimated_beta_1, estimated_tau, n_iters, trace = bayesianRegression(df,10000)
    print(estimated_beta_0)
    print(estimated_beta_1)
    print(estimated_tau)
    
    traceplot(trace, n_iters)

