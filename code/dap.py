"""
Use Approximate Bayesian Computation (ABC) to parametrize the rate function
given a hypothetical experiment timeline.
"""


import csv
from timeit import default_timer as timer

import click
import numpy as np
from scipy.interpolate import PchipInterpolator as pchip
import toml

import simtools


class Rate:
    def __init__(self, s, c, w, u, m):
        """
        :param s float: shape parameter in R
        :param c float: center parameter in (0, 1)
        :param w float: width between function ends in R
        :param u float: mode of function in R
        :param m float: function maximum in in R > 0
        """
        self.u = u
        self.w = w
        self.m = m
        self.c = c
        self.a = s*c
        self.b = s - self.a
        self.factor = self.a**self.a * self.b**self.b * (self.a + self.b)**(-self.a - self.b)

    def __call__(self, x):
        y = (x/self.w - self.u/self.w + self.c)**self.a * (1 - (x/self.w - self.u/self.w + self.c))**self.b
        y = self.m * y / self.factor
        y[x <= self.u - self.c*self.w] = 0
        y[x >= self.u - (self.c - 1)*self.w] = 0
        return y


class Noise:
    def __init__(self, s):
        """
        :param s float: standard deviation of normal distribution
        """
        self.s = s

    def __call__(self, x):
        return 1/np.sqrt(2*np.pi*self.s**2) * \
               np.exp(-x**2/(2*self.s**2))


class SBP:
    # Synchronises a death with every birth, thus keeping the
    # number of particles constant
    def __init__(self, particles, rate_function, noise_function):
        # list of floating point numbers specifying
        # internal parameter x for each extant particle
        self.particles = np.array(particles)
        # function that takes self.particles and returns
        # the growth rate of each particle
        self.rate_function = rate_function
        # funciton that returns a value that is added
        # to x for a newly born particle
        self.noise_function = noise_function

        # Initial setup
        self.t = 0
        self.rates = rate_function(self.particles)
        self.birth_rate = np.sum(self.rates)

    def simulate(self, increment_time):
        end_time = self.t + increment_time
        while self.t < end_time:
            # because of synchronization birth rate is the only important rate
            total_rate = self.birth_rate
            print(total_rate/self.particles.size)

            # increment time dependent on total rate
            self.t += np.random.exponential(1/total_rate)

            # replicate random particle
            # normalize rates so that we can use it as probabilities to
            # select the dividing particle
            selection_probabilities = self.rates / self.birth_rate
            select_particle = np.random.choice(self.particles.size, 1, p=selection_probabilities)
            new_particle = self.particles[select_particle] + self.noise_function()
            new_rate = self.rate_function(np.array(new_particle))[0]
            self.particles = np.append(self.particles, new_particle)
            self.rates = np.append(self.rates, new_rate)
            self.birth_rate += new_rate

            # kill random particle
            select_particle = np.random.choice(self.particles.size)
            self.particles = np.delete(self.particles, select_particle)
            self.birth_rate -= self.rates[select_particle]
            self.rates = np.delete(self.rates, select_particle)


@click.group()
def main():
    """
    Click construct
    """
    pass


@main.command()
@click.option('-p', '--paramfile', type=click.Path())
def mpi_rate_test(paramfile):

# def noise_rv():
#     return np.random.normal(0, NOISE_SIGMA)
    simtools.PARAMS = toml.load(paramfile)

    f_rate = Rate(simtools.PARAMS['mpi_rate_function_shape'],
                  simtools.PARAMS['mpi_rate_function_center'],
                  simtools.PARAMS['mpi_rate_function_width'],
                  simtools.PARAMS['optimum_normal'],
                  simtools.PARAMS['mpi_rate_function_max'])
                  # simtools.PARAMS['mpi_rate_function_max']*simtools.PARAMS['mpi_rate_function_ratio'])

    # f_noise = Noise(simtools.PARAMS['mpi_noise_function_sigma'])
    f_noise = lambda: np.random.normal(0, simtools.PARAMS['mpi_noise_function_sigma'])

    sbp = SBP(np.zeros(1000), f_rate, f_noise)
    sbp.simulate(30)


if __name__ == '__main__':
    main()
