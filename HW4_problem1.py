"""
MAE 3403 - HW 4 - Problem 1

Assumptions:
-rock diameters are spherical
-pre-sieved rocks follow log normal distribution
-distribution is truncated between Dmin and Dmax
-11 samples of N=100 generated from truncated distribution
-CLI reports sample means, variances, and statistics of sampling mean
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
import math

# ---------------------------
# Log-normal PDF
# ---------------------------

def lognormal_pdf(D, mu, sigma):
    if D <= 0:
        return 0.0
    return (1.0 / (D * sigma * math.sqrt(2.0 * math.pi))) * \
           math.exp(-((math.log(D) - mu) ** 2) / (2.0 * sigma ** 2))


# ---------------------------
# Log-normal CDF using quad
# ---------------------------

def lognormal_cdf(x, mu, sigma):
    if x <= 0:
        return 0.0
    result, _ = quad(lognormal_pdf, 0, x, args=(mu, sigma))
    return result


# ---------------------------
# Truncated PDF
# ---------------------------

def truncated_pdf(D, mu, sigma, Fmin, Fmax):
    if D <= 0:
        return 0.0
    return lognormal_pdf(D, mu, sigma) / (Fmax - Fmin)


# ---------------------------
# Truncated CDF using quad
# ---------------------------

def truncated_cdf(D, mu, sigma, Dmin, Fmin, Fmax):
# CDF of truncated distribution is 0 below Dmin,
# 1 above Dmax (if caller limited D to [Dmin, Dmax])
    if D <= Dmin:
        return 0.0
    integral, _ = quad(truncated_pdf, Dmin, D,
                       args=(mu, sigma, Fmin, Fmax))
    return integral


# ---------------------------
# Compute sample mean and variance
# ---------------------------

def sample_stats(sample):
    arr = np.array(sample)
    mean = np.mean(arr)
# Sample variance with N-1 in denominator
    var = np.var(arr, ddof=1)
    return mean, var


# ---------------------------
# Generate one sample using inverse transform (via fsolve)
# ---------------------------

def generate_sample(mu, sigma, Dmin, Dmax, Fmin, Fmax, N=100):
    sample = []
    for _ in range(N):
        u = np.random.rand()  # uniform(0,1)

# Solve truncated_cdf(D) = u
        func = lambda D: truncated_cdf(D[0], mu, sigma, Dmin, Fmin, Fmax) - u

        D_guess = (Dmin + Dmax) / 2.0
        D_solution = fsolve(func, D_guess)[0]

# Clamp to [Dmin, Dmax] in case of tiny numerical drift
        D_solution = max(min(D_solution, Dmax), Dmin)

        sample.append(D_solution)
    return sample


# -----------------------------
# Main program
# -----------------------------

def main():
    print("\nIndustrial Gravel Production Simulation\n")

# Default values
    mu_default = math.log(2.0)
    sigma_default = 1.0
    Dmax_default = 1.0
    Dmin_default = 3.0 / 8.0

    # User input
    mu_in = input(f"Mean of ln(D)? (default {mu_default:.3f}): ")
    mu = mu_default if mu_in.strip() == "" else float(mu_in)

    sigma_in = input(f"Std dev of ln(D)? (default {sigma_default}): ")
    sigma = sigma_default if sigma_in.strip() == "" else float(sigma_in)

    Dmax_in = input(f"Large screen aperture Dmax? (default {Dmax_default}): ")
    Dmax = Dmax_default if Dmax_in.strip() == "" else float(Dmax_in)

    Dmin_in = input(f"Small screen aperture Dmin? (default {Dmin_default:.3f}): ")
    Dmin = Dmin_default if Dmin_in.strip() == "" else float(Dmin_in)

# Precompute truncation constants in terms of original lognormal CDF
    Fmax = lognormal_cdf(Dmax, mu, sigma)
    Fmin = lognormal_cdf(Dmin, mu, sigma)

    sample_means = []

    print("\nGenerating 11 samples of 100 rocks each...\n")

    for i in range(11):
        sample = generate_sample(mu, sigma, Dmin, Dmax,
                                 Fmin, Fmax, N=100)

        mean, var = sample_stats(sample)
        sample_means.append(mean)

        print(f"Sample {i + 1}:")
        print(f"  Mean = {mean:.4f}")
        print(f"  Variance = {var:.6f}\n")

# Statistics of sampling mean (over the 11 sample means)
    mean_of_means = np.mean(sample_means)
    var_of_means = np.var(sample_means, ddof=1)

    print("Statistics of Sampling Means:")
    print(f"  Mean of sampling mean = {mean_of_means:.4f}")
    print(f"  Variance of sampling mean = {var_of_means:.6f}")


if __name__ == "__main__":
    main()
