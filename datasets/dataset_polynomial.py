import time

import numpy as np
from sklearn.linear_model import LinearRegression

def MinMaxScaler(data):
    """Min Max normalizer.

    Args:
      - datasets: original datasets

    Returns:
      - norm_data: normalized datasets
    """
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data

def generate_polynomial_dataset(n_samples=1000, n_points=320, param_dists=None, x_range=(-1, 1), noise_std=0.1):
    """
    Generate a cubic function y = ax^3 + bx^2 + cx + d 的 synthetic dataset

    Args:
        n_samples (int): How many curves are generated (each curve corresponds to a set of parameters a, b, c, d)
        n_points (int): How many points are sampled on each curve
        param_dists (dict): The distribution of each parameter, for example:
            {
                "a": lambda n: np.random.normal(0, 1, size=n),
                "b": lambda n: np.random.uniform(-1, 1, size=n),
                "c": lambda n: np.random.normal(0, 0.5, size=n),
                "d": lambda n: np.random.uniform(-2, 2, size=n),
            }
        x_range (tuple): x range of values
        noise_std (float): Add noise standard deviation

    Returns:
        dict: {
            "X": shape (n_samples, n_points),
            "Y": shape (n_samples, n_points),
            "params": (n_samples, 4) The real parameters [a, b, c, d]
        }
    """

    #torch.Size([256, 18, 6]) torch.Size([256, 36, 6])
    if param_dists is None:
        param_dists = {
            "a": lambda n: np.random.normal(-1, 0.5, size=n),
            "b": lambda n: np.random.normal(4, 1, size=n),
            "c": lambda n: np.random.normal(2, 1, size=n),
            "d": lambda n: np.random.normal(-2, 0.5, size=n),
        }

    a = param_dists["a"](n_samples)
    b = param_dists["b"](n_samples)
    c = param_dists["c"](n_samples)
    d = param_dists["d"](n_samples)

    params = np.stack([a, b, c, d], axis=1)

    X = np.linspace(x_range[0], x_range[1], n_points)

    X = np.tile(X, (n_samples, 1))

    Y = a[:, None] * X**3 + b[:, None] * X**2 + c[:, None] * X + d[:, None]


    return {"X": X, "Y": Y, "params": params}

def fit_polynomial_params(X, Y):
    """
    拟合三次函数参数 (a, b, c, d)
    """
    n_samples, n_points = X.shape
    fitted_params = []

    for i in range(n_samples):
        Xi = X[i][:, None]
        Phi = np.concatenate([Xi**3, Xi**2, Xi, np.ones_like(Xi)], axis=1)
        model = LinearRegression(fit_intercept=False).fit(Phi, Y[i])
        fitted_params.append(model.coef_)

    return np.array(fitted_params)

def evaluate_distribution_shift(true_params, fitted_params):
    """
 KL divergence
    """
    results = {}
    param_names = ["a", "b", "c", "d"]

    def _kl_from_samples(p_samples, q_samples, num_bins=50, eps=1e-12):
        # Using a common histogram binning to approximate two distributions
        min_val = min(p_samples.min(), q_samples.min())
        max_val = max(p_samples.max(), q_samples.max())
        if min_val == max_val:
            return 0.0, 0.0, 0.0

        bins = np.linspace(min_val, max_val, num_bins + 1)
        p_hist, _ = np.histogram(p_samples, bins=bins, density=False)
        q_hist, _ = np.histogram(q_samples, bins=bins, density=False)

        # Smooth to avoid zero probability and normalize to probability distribution
        p_prob = p_hist.astype(float) + eps
        q_prob = q_hist.astype(float) + eps
        p_prob /= p_prob.sum()
        q_prob /= q_prob.sum()

        kl_pq = float(np.sum(p_prob * np.log(p_prob / q_prob)))
        kl_qp = float(np.sum(q_prob * np.log(q_prob / p_prob)))
        js = 0.5 * (kl_pq + kl_qp)
        return kl_pq, kl_qp, js

    for i, name in enumerate(param_names):
        kl_pq, kl_qp, js = _kl_from_samples(true_params[:, i], fitted_params[:, i])
        results[name] = {"KL_P||Q": kl_pq, "KL_Q||P": kl_qp, "JS_div": js}

    return results

import matplotlib.pyplot as plt

def plot_examples(data, n_show=5):
    """
 Randomly select a few curves and draw them
    Args:
        data: generate_polynomial_dataset 返回的 dict
        n_show: How many curves are displayed
    """
    X = data["X"]
    Y = data["Y"]
    params = data["params"]

    idx = np.random.choice(len(X), size=n_show, replace=False)

    plt.figure(figsize=(8, 5))
    for i in idx:
        plt.plot(X[i], Y[i], label=f"a={params[i,0]:.2f}, b={params[i,1]:.2f}, c={params[i,2]:.2f}, d={params[i,3]:.2f}")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Sample cubic functions")
    plt.legend()
    plt.grid(True)
    plt.savefig("sample_cubic_functions.png")



