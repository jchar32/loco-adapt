import numpy as np
from scipy import optimize


def exponential(x, a, b, c):
    return a * np.exp(-b * x) + c


def inverse_exponential(y, a, b, c):
    return -1 * np.log((y - c) / a) / b


def linear(x, m, b):
    return m * x + b


def fit_exp(x, y):
    popt, pcov, infodict, mesg, ier = optimize.curve_fit(
        exponential, x, y, full_output=True, p0=[1, 1e-6, 1]
    )
    return popt, pcov, infodict, mesg


def fit_lin(x, y):
    popt, pcov, infodict, mesg, ier = optimize.curve_fit(linear, x, y, full_output=True)
    return popt, pcov, infodict, mesg


def rsquared(y, predy):
    ss_res = np.sum((y - predy) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


def rad_of_curve(xdata, ydata):
    yd = derivative(xdata, ydata)
    ydd = derivative(xdata, yd)
    return ((1 + yd**2)**1.5) / np.abs(ydd)


def derivative(x, y):
    return np.gradient(y, x)
