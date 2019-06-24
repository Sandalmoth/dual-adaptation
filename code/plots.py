"""
All kinds of plotting
- abcdiag: abc diagnostics
"""


import csv

import click
import h5py
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from pyabc import History
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


@click.group()
def main():
    """
    main construct for click to function in self-contained file
    """
    pass


@main.command()
@click.option('-p', '--paramfile', type=click.Path())
@click.option('-u', '--obsfile-up', type=click.Path())
@click.option('-d', '--obsfile-down', type=click.Path())
@click.option('-b', '--dbfile', type=click.Path())
@click.option('--save', type=click.Path(), default=None)
@click.option('-i', '--history-id', type=int, default=1)
def abcdiag(paramfile, obsfile_up, obsfile_down, dbfile, save, history_id):
    """
    Diagnostic plots for examining how abc fitting worked
    """
    db_path = 'sqlite:///' + dbfile
    abc_history = History(db_path)
    abc_history.id = history_id

    simtools.PARAMS = toml.load(paramfile)

    if save is not None:
        pdf_out = PdfPages(save)

    ### ABC SIMULATION PARAMETERS ###

    fig, axs = plt.subplots(nrows=3, sharex=True)

    t_axis = list(range(abc_history.max_t + 1))
    populations = abc_history.get_all_populations()
    populations = populations[populations.t >= 0]

    axs[0].plot(t_axis, populations['particles'])
    axs[1].plot(t_axis, populations['epsilon'])
    axs[2].plot(t_axis, populations['samples'])

    axs[0].set_title('ABC parameters per generation')
    axs[0].set_ylabel('Particles')
    axs[1].set_ylabel('Epsilon')
    axs[2].set_ylabel('Samples')
    axs[-1].set_xlabel('Generation (t)')

    fig.set_size_inches(8, 5)

    if save is not None:
        pdf_out.savefig()
    else:
        plt.show()

    # PLOT SHOWING PARAMETERS WITH CONFIDENCE OVER GENERATIONS ###

    fig, axs = plt.subplots(nrows=4, sharex=True)

    t_axis = np.arange(abc_history.max_t + 1)
    quartile1 = []
    medians = []
    quartile3 = []
    parameters = ['s', 'c', 'w', 'n']

    for generation in t_axis:
        abc_data, __ = abc_history.get_distribution(m=0, t=generation)
        data = [abc_data[x] for x in parameters]
        t_quartile1, t_medians, t_quartile3 = np.percentile(
            data, [25, 50, 75], axis=1
        )
        quartile1.append(t_quartile1)
        medians.append(t_medians)
        quartile3.append(t_quartile3)

    quartile1 = np.array(quartile1)
    medians = np.array(medians)
    quartile3 = np.array(quartile3)

    for i, param in enumerate(parameters):
        axs[i].plot(t_axis, medians[:, i])
        axs[i].fill_between(t_axis, quartile1[:, i], quartile3[:, i],
                            alpha=0.3, color='gray')
        axs[i].set_ylabel(param)

    axs[-1].set_xlabel('Generation (t)')

    fig.set_size_inches(8, 8)

    if save is not None:
        pdf_out.savefig()
    else:
        plt.show()


    if save is not None:
        pdf_out.close()



@main.command()
@click.option('-p', '--paramfile', type=click.Path())
@click.option('-u', '--obsfile-up', type=click.Path())
@click.option('-d', '--obsfile-down', type=click.Path())
@click.option('-b', '--dbfile', type=click.Path())
@click.option('--save', type=click.Path(), default=None)
@click.option('-i', '--history-id', type=int, default=1)
def abcfit(paramfile, obsfile_up, obsfile_down, dbfile, save, history_id):
    """
    Plots showing off the fit from abc
    """
    db_path = 'sqlite:///' + dbfile
    abc_history = History(db_path)
    abc_history.id = history_id

    simtools.PARAMS = toml.load(paramfile)

    if save is not None:
        pdf_out = PdfPages(save)


    ### PLOT OF RATE AND NOISE FUNCTION ###
    abc_data, __ = abc_history.get_distribution(m=0,
                                                t=abc_history.max_t)

    parameters = ['s', 'c', 'w', 'n']
    params = {k: np.median(abc_data[k]) for k in parameters}

    f_rate = Rate(params['s'], params['c'], params['w'], 0, 1)
    f_noise = Noise(params['n'])

    x_width = simtools.PARAMS['parameter_range'][1] - \
              simtools.PARAMS['parameter_range'][0]
    x_axis = np.linspace(-x_width/2, x_width/2, simtools.PARAMS['parameter_points'])

    fig, axs = plt.subplots(ncols=2)
    axs[0].plot(x_axis, f_rate(x_axis))
    axs[1].plot(x_axis, f_noise(x_axis))

    fig.set_size_inches(8, 5)
    plt.tight_layout()

    if save is not None:
        pdf_out.savefig()
    else:
        plt.show()

    ### HEATMAP OF RISE AND FALL WITH MEAN AND OBSERVATION ###

    fig, axs = plt.subplots(nrows=2)

    sim = {}
    f_rate_up = Rate(params['s'], params['c'], params['w'],
                     simtools.PARAMS['optimum_treatment'], 1)
    f_rate_down = Rate(params['s'], params['c'], params['w'],
                       simtools.PARAMS['optimum_normal'], 1)

    parameter_range = simtools.PARAMS['parameter_range'][1] - \
                      simtools.PARAMS['parameter_range'][0]

    observation = Observation()
    observation.parse_observations(obsfile_up, obsfile_down)
    obs = observation.get_instance(
        simtools.get_time_axis(simtools.PARAMS['time_range_up'][1],
                               simtools.PARAMS['time_points_up']),
        simtools.get_time_axis(simtools.PARAMS['time_range_down'][1],
                               simtools.PARAMS['time_points_down'])
    )

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

    sim['x_up'] = np.array(
        [np.sum(parameters[:, i]*parameter_axis) / \
         parameter_axis.size*parameter_range \
         for i in range(parameters.shape[1])]
    )
    axs[0].plot(sim['x_up'], time_axis, color='k',
                linewidth=1.0)
    axs[0].imshow(
        np.transpose(parameters),
        aspect=parameter_range/simtools.PARAMS['time_range_up'][1],
        extent=[np.min(parameter_axis), np.max(parameter_axis), 0,
                simtools.PARAMS['time_range_up'][1]],
        cmap=cm.cubehelix,
        origin='lower'
    )
    axs[0].plot(obs['x_up'], time_axis, linewidth=1.0, color='r')

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
        simtools.PARAMS['time_range_down'][1],
        simtools.PARAMS['time_points_down'],
        simtools.PARAMS['parameter_range'],
        simtools.PARAMS['parameter_points']
    )

    sim['x_down'] = np.array(
        [np.sum(parameters[:, i]*parameter_axis) / \
         parameter_axis.size*parameter_range \
         for i in range(parameters.shape[1])]
    )
    axs[1].plot(sim['x_down'], time_axis, color='k',
                linewidth=1.0)
    axs[1].imshow(
        np.transpose(parameters),
        aspect=parameter_range/simtools.PARAMS['time_range_down'][1],
        extent=[np.min(parameter_axis), np.max(parameter_axis), 0,
                simtools.PARAMS['time_range_down'][1]],
        cmap=cm.cubehelix,
        origin='lower'
    )
    axs[1].plot(obs['x_down'], time_axis, linewidth=1.0, color='r')

    fig.set_size_inches(5, 8)
    plt.tight_layout()

    if save is not None:
        pdf_out.savefig()
    else:
        plt.show()


    if save is not None:
        pdf_out.close()


@main.command()
@click.option('-p', '--paramfile', type=click.Path())
@click.option('-b', '--dbfile', type=click.Path())
@click.option('-o', '--outfile', type=click.Path())
@click.option('-i', '--history-id', type=int, default=1)
def generate_dataset(paramfile, dbfile, outfile, history_id):
    """
    Generate a field using the pde for further c++ mpi simulation
    """

    db_path = 'sqlite:///' + dbfile
    abc_history = History(db_path)
    abc_history.id = history_id

    simtools.PARAMS = toml.load(paramfile)

    abc_data, __ = abc_history.get_distribution(m=0, t=abc_history.max_t)

    parameters = ['s', 'c', 'w', 'n']
    params = {k: np.median(abc_data[k]) for k in parameters}

    f_rate = Rate(params['s'], params['c'], params['w'], 0, 1)
    f_noise = Noise(params['n'])
    simtools.PARAMS = toml.load(paramfile)

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

    time_axis, parameter_axis, parameter_density = simtools.simulate_pde(
        f_initial,
        f_rate_up,
        f_noise,
        simtools.PARAMS['time_range_up'][1],
        simtools.PARAMS['time_points_up'],
        simtools.PARAMS['parameter_range'],
        simtools.PARAMS['parameter_points']
    )

    # write parameter density hdf5
    out = h5py.File(outfile, 'w')
    gp_pd = out.create_group('parameter_density')
    gp_pd['time_axis'] = time_axis
    gp_pd['parameter_axis'] = parameter_axis
    gp_pd['parameter_density'] = parameter_density

    # write rate function data to simulation config toml
    simtools.PARAMS['mpi_noise_function_sigma'] = params['n']
    simtools.PARAMS['mpi_rate_function_width'] = params['w']
    simtools.PARAMS['mpi_rate_function_center'] = params['c']
    simtools.PARAMS['mpi_rate_function_shape'] = params['s']

    with open(paramfile, 'w') as params_toml:
        toml.dump(simtools.PARAMS, params_toml)


if __name__ == '__main__':
    main()
