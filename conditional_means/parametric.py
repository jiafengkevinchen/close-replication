import pyfixest as pf
import numpy as np
import scipy
import pandas as pd


def parametric_conditional_moments(estimates, standard_errors):
    """
    Model the E[theta | sigma] and Var(theta | sigma) parametrically
    via E[theta | sigma] = linear in log(sigma)
    and Var[theta | sigma] = exp( linear in log(sigma) ).
    Estimate via least squares.
    """
    x = np.log10(standard_errors)
    fit = pf.feols("estimates ~ 1 + x", data=pd.DataFrame({"estimates": estimates, "x": x}))
    conditional_mean = fit.predict()
    resid_sq_minus_s2 = fit.resid() ** 2 - standard_errors**2

    def objective(params):
        return resid_sq_minus_s2 - np.exp(params[0] + params[1] * x)

    result = scipy.optimize.least_squares(objective, x0=np.array([0, 0]), loss="linear")
    a, b = result["x"]

    conditional_var = np.exp(a + b * x)
    conditional_std = np.sqrt(conditional_var)

    return conditional_mean, conditional_std
