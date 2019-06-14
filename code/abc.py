"""
Use Approximate Bayesian Computation (ABC) to parametrize the rate function
given a hypothetical experiment timeline.
"""


import csv
import time

import click
import numpy as np
from pyabc import (ABCSMC, Distribution, RV)
from pyabc.populationstrategy import AdaptivePopulationSize
from pyabc.sampler import MulticoreEvalParallelSampler
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



class Observation:
    """
    A stochastic process that describes an observation
    """
    def __init__(self):
        pass

    def __str__(self):
        return str(self.obs)

    def parse_observations(self, obsfile_up, obsfile_down):
        """
        An observation file holds a probability density function
        specified by a mean and sigma at a number of time coordinates.
        The sigma are used for a weighted least squares of the means from simulation.

        :param obsfile_up path: path to csv of observations for parameter increase
        :param obsfile_down path: path to csv of observations for parameter decrease
        """

        self.obs = {
            'up': {'t': [], 'x': [], 's': []},
            'down': {'t': [], 'x': [], 's': []}
        }
        with open(obsfile_up, 'r') as obs_up:
            rdr = csv.DictReader(obs_up)
            for line in rdr:
                self.obs['up']['t'].append(float(line['time']))
                self.obs['up']['x'].append(float(line['param']))
                self.obs['up']['s'].append(float(line['stdev']))
        with open(obsfile_down, 'r') as obs_down:
            rdr = csv.DictReader(obs_down)
            for line in rdr:
                self.obs['down']['t'].append(float(line['time']))
                self.obs['down']['x'].append(float(line['param']))
                self.obs['down']['s'].append(float(line['stdev']))

        self.interpolators = {
            'up': {
                'x': pchip(self.obs['up']['t'], self.obs['up']['x'], extrapolate=True),
                's': pchip(self.obs['up']['t'], self.obs['up']['s'], extrapolate=True)
            },
            'down': {
                'x': pchip(self.obs['down']['t'], self.obs['down']['x'], extrapolate=True),
                's': pchip(self.obs['down']['t'], self.obs['down']['s'], extrapolate=True)
            }
        }

    def get_instance(self, time_up, time_down):
        """
        Get means and sigmas at specified times.

        :param time np.array: time axis for realization of up trend
        :param time np.array: time axis for realization of down trend
        :returns: {'up': [up example]}
        """

        instance = {}

        for time, obs_set in zip([time_up, time_down], ['up', 'down']):
            obs_t = np.array(self.obs[obs_set]['t'])
            obs_s = np.array(self.obs[obs_set]['s'])
            obs_x = np.array(self.obs[obs_set]['x'])
            # are we outside observations?
            # if so, add data to improve interpolation and warn user
            if time[0] < obs_t[0]:
                print('Warning: requesting observation interpolation outside observations')
                obs_t = np.insert(obs_t, 0, time[0])
                obs_x = np.insert(obs_x, 0, obs_x[0])
                obs_s = np.insert(obs_s, 0, obs_s[0])
            if time[0] > obs_t[0]:
                print('Warning: requesting observation interpolation outside observations')
                obs_t = np.append(obs_t, time[0])
                obs_x = np.append(obs_x, obs_x[0])
                obs_s = np.append(obs_s, obs_s[0])

            instance['x_' + obs_set] = self.interpolators[obs_set]['x'](time)
            instance['s_' + obs_set] = self.interpolators[obs_set]['s'](time)

        return instance


def abc_model(params):
    """
    Model for abc computation
    """
    # Simulation proceeds as follows:
    # 1. Find the initial distribution given a certain rate function
    # 2. Simulate rise of parameter with pde
    # 3. Find stationary distribution after rise
    # 4. Simulate decay with pde

    sim = {}
    f_noise = Noise(params['n'])
    f_rate_up = Rate(params['s'], params['c'], params['w'],
                     simtools.PARAMS['optimum_treatment'], 1)
    f_rate_down = Rate(params['s'], params['c'], params['w'],
                       simtools.PARAMS['optimum_normal'], 1)

    f_initial = simtools.get_stationary_distribution_function(
        f_rate_down,
        f_noise,
        simtools.PARAMS['parameter_range'],
        simtools.PARAMS['parameter_points']
    )

    time_axis, parameter_axis, parameters = simtools.simulate_pde(
        f_initial,
        f_rate_up,
        f_noise,
        simtools.PARAMS['time_range_up'][1],
        simtools.PARAMS['time_points_up'],
        simtools.PARAMS['parameter_range'],
        simtools.PARAMS['parameter_points']
    )

    sim['x_up'] = np.mean(parameters, axis=1)
    sim['s_up'] = np.zeros(sim['x_up'].size)

    f_initial = simtools.get_stationary_distribution_function(
        f_rate_up,
        f_noise,
        simtools.PARAMS['parameter_range'],
        simtools.PARAMS['parameter_points']
    )

    time_axis, parameter_axis, parameters = simtools.simulate_pde(
        f_initial,
        f_rate_down,
        f_noise,
        simtools.PARAMS['time_range_up'][1],
        simtools.PARAMS['time_points_up'],
        simtools.PARAMS['parameter_range'],
        simtools.PARAMS['parameter_points']
    )

    sim['x_down'] = np.mean(parameters, axis=1)
    sim['s_down'] = np.zeros(sim['x_up'].size)

    print(sim)

    return sim


def abc_distance(obs1, obs2):
    """
    Weighted rmsd between two dataset.
    """

    print(obs1)
    print(obs2)

    total = 0

    # Identify PDE data which has s=0 by definition
    for obs_set in ['up', 'down']:

        obs = {'up': {}, 'down': {}}
        pde = {'up': {}, 'down': {}}

        if np.any(obs1['s_up']):
            pde['x'] = obs2['x_' + obs_set]
            # pde['s'] = obs2['s_' + obs_set]
            obs['x'] = obs1['x_' + obs_set]
            obs['s'] = obs1['s_' + obs_set]
        else:
            pde['x'] = obs1['x_' + obs_set]
            # pde['s'] = obs2['s_' + obs_set]
            obs['x'] = obs2['x_' + obs_set]
            obs['s'] = obs2['s_' + obs_set]

        total += np.sum((pde['x'] - obs['x'])**2 / obs['s'])

    return total


def abc_setup():
    """
    Create abc model
    """

    abc_prior_dict = {
        's': RV("uniform", 0, 100),
        'c': RV("uniform", 0, 1),
        'w': RV("uniform", 0, 10),
        'n': RV("uniform", 0, 10)
    }

    abc_priors = Distribution(abc_prior_dict)

    abc = ABCSMC(abc_model, abc_priors, abc_distance,
                 population_size=AdaptivePopulationSize(simtools.PARAMS['abc_initial_population_size'], 0.15),
                 sampler=MulticoreEvalParallelSampler(simtools.PARAMS['abc_parallel_simulations']))

    return abc


@click.group()
def main():
    """
    Click construct
    """
    pass


@main.command()
@click.option('-p', '--paramfile', type=click.Path())
@click.option('-u', '--obsfile-up', type=click.Path())
@click.option('-d', '--obsfile-down', type=click.Path())
@click.option('-b', '--dbfile', type=click.Path())
def parametrize(paramfile, obsfile_up, obsfile_down, dbfile):

    np.set_printoptions(threshold=10)

    simtools.PARAMS = toml.load(paramfile)
    print('Simulation parameters:', simtools.PARAMS)

    observation = Observation()
    observation.parse_observations(obsfile_up, obsfile_down)
    print('Observation:', observation)

    observation = observation.get_instance(
        simtools.get_time_axis(simtools.PARAMS['time_range_up'][1], simtools.PARAMS['time_points_up']),
        simtools.get_time_axis(simtools.PARAMS['time_range_down'][1], simtools.PARAMS['time_points_down']))
    print('Observation instance:', observation)

    abc = abc_setup()
    db_path = 'sqlite:///' + dbfile
    print('Saving database in:', db_path)

    print('Constructing ABC')
    abc.new(db_path, observation)
    print('Running ABC')
    abc.run(minimum_epsilon=simtools.PARAMS['abc_min_epsilon'],
            max_nr_populations=simtools.PARAMS['abc_max_populations'],
            min_acceptance_rate=simtools.PARAMS['abc_min_acceptance'])

if __name__ == '__main__':
    main()
