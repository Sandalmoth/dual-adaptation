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
    """
    Model for abc computation
    """
    pass


def abc_distance(obs1, obs2):
    """
    Weighted rmsd between two dataset.
    """

    total = 0

    # Identify PDE data which has s=0 by definition
    for obs_set in ['up', 'down']:
        if np.any(obs1['s']):
            pde[obs_set] = obs2
            obs[obs_set] = obs1
        else:
            pde[obs_set] = obs1
            obs[obs_set] = obs2

        total += np.sum((pde['x'] - obs['x'])**2 * obs['s'])

    return total

def abc_setup():
    """
    Create abc model
    """

    abc_prior_dict = {
        'a': RV("uniform", 0, 100),
        'b': RV("uniform", 0, 100),
        'w': RV("uniform", 0, 10)
    }

    abc = ABCSMC(abc_model, abc_prior_dict, abc_distance,
                 population_size=AdaptivePopulationSize(500, 0.15))

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
def parametrize(paramfile, obsfile_up, obsfile_down):

    OBS.parse_observations(obsfile_up, obsfile_down)
    print('Observation:', OBS)

    PARAMS = toml.load(paramfile)
    print('Simulation parameters:', PARAMS)

    observation = OBS.get_instance(
        simtools.get_time_axis(PARAMS['time_end_up'], PARAMS['time_points_up']),
        simtools.get_time_axis(PARAMS['time_end_down'], PARAMS['time_points_down']))

    print(observation)

    abc = abc_setup()
    db_path = 'sqlite:///' + dbfile
    print('Saving database in:', db_path, file=sys.stderr)


if __name__ == '__main__':
    main()
