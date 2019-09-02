"""
Shared tools for simulation
"""


import numpy as np
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve


# Needs to be in global scope here rather than in abc.py
# or it would not be accessible by abc_model()
PARAMS = {}


def amax(lst):
    """
    return max value by deviation from 0

    :param lst iterable: list of values to find max in
    :returns: max value by deviation from 0
    """
    return max([abs(x) for x in lst])


def get_time_axis(t_end, t_points):
    t_range = (0, t_end)
    return np.linspace(t_range[0], t_range[1], t_points)


def simulate_pde(f_initial, f_rate, f_noise, t_end, t_points, x_view, x_points, convolve_method='np'):
    """
    Simulate evolution of parameter probability density with PDE formulation.

    :param f_initial f(x): density function of initial density (integral 1)
    :param f_rate f(x): rate function
    :param f_noise f(x): density function of noise (integral 1)
    :param t_end float:
    :param t_points int: time resolution (total samples)
    :param x_view (float, float): range of parameter values in return density array
    :param x_points int: parameter resolution (total samples)
    :returns: (time axis [np array], parameter axis [np array], density over time [2d np array])
    """

    if convolve_method == 'np':
        convolve_method = np.convolve
    elif convolve_method == 'fft':
        convolve_method = fftconvolve
    else:
        print("Unknown convolution method:", convolve_method)
        exit(0)

    def lr(x):
        # utility function for 'lenght' of (start, end) tuple
        return x[1] - x[0]

    # simulate for a wider range of x to avoid edge effects
    # symmetry around 0 is required for convolution
    x_range = (-amax(x_view)*2, amax(x_view)*2)
    t_range = (0, t_end)

    h = lr(x_range)/x_points
    k = lr(t_range)/t_points

    x_points_full = int(x_points*lr(x_range)/lr(x_view))

    x = np.linspace(x_range[0], x_range[1], x_points_full)
    t = get_time_axis(t_end, t_points)

    mesh = np.zeros((x_points_full, t_points))

    # setup initial parameters
    mesh[:, 0] = f_initial(x)
    mesh[:, 0] /= simps(mesh[:, 0], x=x) # normalize

    rate = f_rate(x)
    noise = f_noise(x)

    # solve
    for i in range(1, t_points):
        dxh = convolve_method(rate*mesh[:, i-1],
                          noise, mode='same') / (x_points_full/lr(x_range)) - \
                          mesh[:, i-1] * simps(rate*mesh[:, i-1], x=x)
        xh = mesh[:, i-1] + dxh*k/2
        dx = convolve_method(rate*xh, noise, mode='same') / (x_points_full/lr(x_range)) - \
                         xh * simps(rate*xh, x=x)
        mesh[:, i] = mesh[:, i-1] + dx*k

        # normalize to integral 1 (probability density)
        # is this neccessary? should it be here?
        mesh[:, i] /= simps(mesh[:, i], x=x)

    # return relevant section of mesh
    x_ix0 = int((x_view[0] - x_range[0])/lr(x_range) * x_points_full)
    x_ix1 = int((x_view[1] - x_range[1])/lr(x_range) * x_points_full)

    assert x_points_full + x_ix1 - x_ix0 == x_points

    return t, x[x_ix0:x_ix1], mesh[x_ix0:x_ix1, :]


def get_stationary_distribution(f_rate, f_noise, x_view, x_points, iters=100):
    """
    Get the stationary distribution with a particular rate and noise function.

    :param f_rate f(x): rate function
    :param f_noise f(x): density function of noise (integral 1)
    :param x_view (float, float): range of parameter values in return density array
    :param x_points int: parameter resolution (total samples)
    :returns: (parameter axis [np array], probability density [np array])
    """

    def lr(x):
        # utility function for 'lenght' of (start, end) tuple
        return x[1] - x[0]

    x_range = (-amax(x_view)*2, amax(x_view)*2)
    x_points_full = int(x_points*lr(x_range)/lr(x_view))
    x = np.linspace(x_range[0], x_range[1], x_points_full)

    rate = f_rate(x)
    noise = f_noise(x)

    # We can start the iteration from any distribution
    stationary = np.ones(x.size)

    for __ in range(iters):
        stationary = stationary*rate
        stationary = np.convolve(stationary, noise, mode='same')
        stationary /= simps(stationary, x=x)

    x_ix0 = int((x_view[0] - x_range[0])/lr(x_range) * x_points_full)
    x_ix1 = int((x_view[1] - x_range[1])/lr(x_range) * x_points_full)

    assert x_points_full + x_ix1 - x_ix0 == x_points

    return x[x_ix0:x_ix1], stationary[x_ix0:x_ix1]


def get_stationary_distribution_function(f_rate, f_noise, x_view, x_points, iters=100):
    extended_view = [x*2 for x in x_view]
    x, y = get_stationary_distribution(f_rate, f_noise, extended_view, x_points, iters=iters)
    return interp1d(x, y, kind='cubic', bounds_error=False, fill_value=0.0)


def get_child_distribution(parameter_density, f_rate, f_noise, x_view):
    """
    Given a parameter distribution, get the child distribution.
    """

    def lr(x):
        # utility function for 'lenght' of (start, end) tuple
        return x[1] - x[0]

    # symmetry around 0 is required for convolution
    x_range = (-amax(x_view)*2, amax(x_view)*2)
    x_points_full = int(parameter_density.size*lr(x_range)/lr(x_view))
    x_ix0 = int((x_view[0] - x_range[0])/lr(x_range) * x_points_full)
    x_ix1 = int((x_view[1] - x_range[1])/lr(x_range) * x_points_full)
    x = np.linspace(x_range[0], x_range[1], x_points_full)

    parameter_density_full = np.zeros(x.shape)
    parameter_density_full[x_ix0:x_ix1] = parameter_density

    noise = f_noise(x)
    rate = f_rate(x)

    child_density = np.convolve(rate*parameter_density_full, noise, mode='same')
    child_density /= simps(child_density, x=x)

    return child_density[x_ix0:x_ix1]
