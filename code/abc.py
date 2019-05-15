"""
Use Approximate Bayesian Computation (ABC) to parametrize the rate function
given a hypothetical experiment timeline.
"""


import csv

import click
import numpy as np
from pyabc import (ABCSMC, Distribution, RV)
from pyabc.populationstrategy import AdaptivePopulationSize

import simtools


def parse_observations(obsfile_up, obsfile_down):
    obs = {'up': {'t': [], 'x': []},
           'down': {'t': [], 'x': []}
    }
    with open(obsfile_up, 'r') as obs_up:
        rdr = csv.DictReader(obs_up)
        for line in rdr:
            obs['up']['t'].append(line['time'])
            obs['up']['x'].append(line['param'])
    with open(obsfile_down, 'r') as obs_down:
        rdr = csv.DictReader(obs_down)
        for line in rdr:
            obs['down']['t'].append(line['time'])
            obs['down']['x'].append(line['param'])


def interpolate_observation(time, obs):
    pass


def abc_model(params):
    pass


def abc_distance(obs1, obs2):
    pass


def abc_setup():
    pass


@click.group()
def main():
    """
    Click construct
    """
    pass


@main.command()
@click.option('-u', '--obsfile-up', type=click.Path())
@click.option('-d', '--obsfile-down', type=click.Path())
def parametrize(obsfile_up, obsfile_down):
    pass


if __name__ == '__main__':
    main()
