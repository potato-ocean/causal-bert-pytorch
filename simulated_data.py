import numpy as np
import pandas as pd 
from scipy.stats import norm, bernoulli

def simulation(n, z, b_1, gamma, binary_outcome=True):
    # n is the number of data
    # z is our confounder here to input
    # t is the treatment to input 
    # pi_z is hard coded by original paper
    # b_1 parameter in the equation which controls level of confoudning 
    # gamma parameter in the equation 
    # binary_outcome= true for binary false for continuous

    y_0 = np.zeros(n)
    y_1 = np.zeros(n)

    if z == 1:
        pi_z = 0.27
    else:
        pi_z = 0.07

    for i in range(n):
        if binary_outcome:
            y_1 = bernoulli.rvs(norm.cdf(0.25 * 1 + b_1 * (pi_z - 0.2)))
            y_0 = bernoulli.rvs(norm.cdf(0.25 * 0 + b_1 * (pi_z - 0.2)))
        else:
            epsilon = np.random.normal(0, gamma)
            y_0 = 0 + b_1 * (pi_z - 0.5) + epsilon
            y_1 = 1 + b_1 * (pi_z - 0.5) + epsilon

    data = pd.DataFrame({
        'simulated_y1': y_1,
        'simulated_y0': y_0
    })
    return data
