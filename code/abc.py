"""
Use Approximate Bayesian Computation (ABC) to parametrize the rate function
given a hypothetical experiment timeline.
"""


import csv

import click
import numpy as np
from pyabc import (ABCSMC, Distribution, RV)
from pyabc.populationstrategy import AdaptivePopulationSize
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
        print(self.w, self.u, self.a, self.b, self.factor, self.m)
        y = (x*self.w - self.u + self.c)**self.a * (1 - (x*self.w - self.u + self.c))**self.b
        # print(y)
        print(self.u - self.c)
        print(1 - self.c + self.u)
        y = self.m * y / self.factor
        y[x <= self.u - self.c] = 0
        y[x >= 1 - self.c + self.u] = 0
        return y


def rate(x):
    tmp = np.maximum((x - RATE_MU + BEST_X)**ALPHA*(1 - (x - RATE_MU + BEST_X))**BETA, 0) \
          / (ALPHA**ALPHA*BETA**BETA*(ALPHA + BETA)**(-ALPHA - BETA)) \
          * MAX_RATE
    tmp[x <= -BEST_X + RATE_MU] = 0
    tmp[x >= 1 - BEST_X + RATE_MU] = 0
    return tmp


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

        instance = {'up': {}, 'down': {}}

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

            instance[obs_set]['x'] = self.interpolators[obs_set]['x'](time)
            instance[obs_set]['s'] = self.interpolators[obs_set]['s'](time)

        return instance


OBS = Observation()
PARAMS = {}


def abc_model(params):
    pass


def abc_distance(obs1, obs2):
    """
    Weighted rmsd between two dataset.
    """

    # Identify PDE data which has s=0 by definition
    if np.any(obs1['s']):
        pde = obs2
        obs = obs1
    else:
        pde = obs1
        obs = obs2

    return np.sum((pde['x'] - obs['x'])**2 / obs['s'])


def abc_setup(observation):
    pass


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

    OBS.parse_observations(obsfile_up, obsfile_down)
    print('Observation:', OBS)

    PARAMS = toml.load(paramfile)
    print('Simulation parameters:', PARAMS)

    observation = OBS.get_instance(
        simtools.get_time_axis(PARAMS['time_end_up'], PARAMS['time_points_up']),
        simtools.get_time_axis(PARAMS['time_end_down'], PARAMS['time_points_down']))

    abc = abc_setup(observation)
    db_path = 'sqlite:///' + dbfile
    print('Saving database in:', db_path)

    # print('Constructing ABC')
    # abc.new(db_path, observed)
    # print('Running ABC')
    # abc.run(minimum_epsilon=min_epsilon,
    #         max_nr_populations=max_populations,
    #         min_acceptance_rate=min_acceptance)

    # def __init__(self, s, c, w, u, m):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots()
    x = np.linspace(-4, 4, 1000)
    rf = Rate(10, 0.2, 1, 0, 1)
    y = rf(x)
    axs.plot(x, y)
    plt.show()



if __name__ == '__main__':
    main()
