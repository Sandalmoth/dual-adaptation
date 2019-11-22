"""
All kinds of plotting
- abcdiag: abc diagnostics
"""


import copy
import csv

import click
import h5py
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from pyabc import History
from scipy.integrate import simps
from scipy.interpolate import PchipInterpolator as pchip
import toml

import simtools

# import cm_xml_to_matplotlib as cmx

# BUOR = cmx.make_cmap('blue-orange-div.xml')



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
    Plotting and data generation tools
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

    fig, axs = plt.subplots(nrows=6, ncols=2)

    t_axis = np.arange(abc_history.max_t + 1)
    quartile1 = []
    medians = []
    quartile3 = []
    parameters = ['s', 'c', 'w', 'n', 'm', 'r']

    for i, generation in enumerate(t_axis):
        abc_data, __ = abc_history.get_distribution(m=0, t=generation)
        data = [abc_data[x] for x in parameters]
        t_quartile1, t_medians, t_quartile3 = np.percentile(
            data, [25, 50, 75], axis=1
        )
        quartile1.append(t_quartile1)
        medians.append(t_medians)
        quartile3.append(t_quartile3)

        last_distro = data
        if i == 0:
            first_distro = data

    quartile1 = np.array(quartile1)
    medians = np.array(medians)
    quartile3 = np.array(quartile3)

    for i, param in enumerate(parameters):
        axs[i][0].plot(t_axis, medians[:, i])
        axs[i][0].fill_between(t_axis, quartile1[:, i], quartile3[:, i],
                            alpha=0.3, color='gray')
        axs[i][0].set_ylabel(param)

        axs[i][1].hist(first_distro[i], bins=32, density=True)
        axs[i][1].hist(last_distro[i], bins=32, density=True)

    axs[-1][0].set_xlabel('Generation (t)')

    fig.set_size_inches(8, 8)

    plt.tight_layout()

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


    ### PLOT OF RATE###
    abc_data, __ = abc_history.get_distribution(m=0,
                                                t=abc_history.max_t)

    parameters = ['s', 'c', 'w', 'n', 'm', 'r']
    params = {k: np.median(abc_data[k]) for k in parameters}

    f_rate_1 = Rate(params['s'], params['c'], params['w'], simtools.PARAMS['optimum_normal'], params['m'])
    f_rate_2 = Rate(params['s'], params['c'], params['w'], simtools.PARAMS['optimum_treatment'], params['m']*params['r'])
    f_noise = Noise(params['n'])

    # x_width = simtools.PARAMS['parameter_range'][1] - \
    #           simtools.PARAMS['parameter_range'][0]
    # x_axis = np.linspace(-x_width/2, x_width/2, simtools.PARAMS['parameter_points'])
    x_axis = np.linspace(*simtools.PARAMS['parameter_range'], simtools.PARAMS['parameter_points'])

    fig, axs = plt.subplots()
    axs.plot(x_axis, f_rate_1(x_axis), color='k', linestyle='-', linewidth='1.0', label='Mutant or untreated normal cell')
    axs.plot(x_axis, f_rate_2(x_axis), color='k', linestyle='--', linewidth='1.0', label='Normal cell with treatment')

    axs.legend(frameon=False)
    axs.set_xlabel('$x$')
    axs.set_ylabel('$\lambda(x)$')
    axs.set_ylim(axs.get_ylim()[0], axs.get_ylim()[1]*1.2)

    fig.set_size_inches(3.8, 3.8)
    plt.tight_layout()

    if save is not None:
        pdf_out.savefig()
    else:
        plt.show()

    ### HEATMAP OF RISE AND FALL WITH MEAN AND OBSERVATION ###

    fig, axs = plt.subplots(nrows=2)

    sim = {}
    f_rate_up = Rate(params['s'], params['c'], params['w'],
                     simtools.PARAMS['optimum_treatment'], params['m']*params['r'])
    f_rate_down = Rate(params['s'], params['c'], params['w'],
                       simtools.PARAMS['optimum_normal'], params['m'])

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
        simtools.PARAMS['parameter_points'],
        simtools.PARAMS['abc_convolution_method']
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
        cmap=cm.viridis,
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
        cmap=cm.viridis,
        origin='lower'
    )
    axs[1].plot(obs['x_down'], time_axis, linewidth=1.0, color='r')

    fig.set_size_inches(5, 8)
    plt.tight_layout()

    if save is not None:
        pdf_out.savefig()
    else:
        plt.show()

    ### HEATMAP OF RISE AND FALL WITH MEAN AND OBSERVATION ###
    ### HORIZONTAL NICE VERSION ###

    fig, axs = plt.subplots(ncols=2, sharey=True)

    sim = {}
    f_rate_up = Rate(params['s'], params['c'], params['w'],
                     simtools.PARAMS['optimum_treatment'], params['m']*params['r'])
    f_rate_down = Rate(params['s'], params['c'], params['w'],
                       simtools.PARAMS['optimum_normal'], params['m'])

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
        simtools.PARAMS['parameter_points'],
        simtools.PARAMS['abc_convolution_method']
    )

    sim['x_up'] = np.array(
        [np.sum(parameters[:, i]*parameter_axis) / \
         parameter_axis.size*parameter_range \
         for i in range(parameters.shape[1])]
    )
    axs[0].plot(time_axis, sim['x_up'], color='k',
                linewidth=1.0)
    axs[0].imshow(
        parameters,
        aspect=simtools.PARAMS['time_range_up'][1]/parameter_range,
        extent=[0, simtools.PARAMS['time_range_up'][1],
                np.min(parameter_axis), np.max(parameter_axis)],
        cmap=cm.magma,
        origin='lower'
    )
    axs[0].plot(time_axis, obs['x_up'], linewidth=1.0, color='k',
                linestyle='--')

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
    axs[1].plot(time_axis, sim['x_down'], color='k',
                linewidth=1.0, label="Mean (Simulated)")
    img = axs[1].imshow(
        parameters,
        aspect=simtools.PARAMS['time_range_up'][1]/parameter_range,
        extent=[0, simtools.PARAMS['time_range_down'][1],
                np.min(parameter_axis), np.max(parameter_axis)],
        cmap=cm.magma,
        origin='lower'
    )
    axs[1].plot(time_axis, obs['x_down'], linewidth=1.0, color='k',
                label="Mean (Hypothetical)", linestyle='--')

    cbr = fig.colorbar(img, ax=axs[1], fraction=0.046, pad=0.04)
    cbr.set_label('Parameter density', labelpad=-15)
    cbr.set_ticks([np.min(parameters), np.max(parameters)])
    cbr.set_ticklabels(['Low', 'High'])

    axs[0].set_ylabel('$x$')
    axs[0].set_xlabel('Time')
    axs[1].set_xlabel('Time')

    axs[1].legend(loc='center left', bbox_to_anchor=(1.6, 0.5), frameon=False)

    fig.set_size_inches(6.2, 2.5)
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
def generate_dataset_mpi(paramfile, dbfile, outfile, history_id):
    """
    Generate a field using the pde for further c++ mpi simulation
    """

    db_path = 'sqlite:///' + dbfile
    abc_history = History(db_path)
    abc_history.id = history_id

    simtools.PARAMS = toml.load(paramfile)

    abc_data, __ = abc_history.get_distribution(m=0, t=abc_history.max_t)

    parameters = ['s', 'c', 'w', 'n', 'm', 'r']
    params = {k: np.median(abc_data[k]) for k in parameters}

    f_noise = Noise(params['n'])
    simtools.PARAMS = toml.load(paramfile)

    f_rate_up = Rate(params['s'], params['c'], params['w'],
                     simtools.PARAMS['optimum_treatment'], params['m']*params['r'])
    f_rate_down = Rate(params['s'], params['c'], params['w'],
                       simtools.PARAMS['optimum_normal'], params['m'])

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

    # find the child distribution at each point in time
    child_density = np.zeros(shape=parameter_density.shape)
    for i in range(parameter_density.shape[1]):
        child_density[:, i] = simtools.get_child_distribution(parameter_density[:, i],
                                                              f_rate_up, f_noise,
                                                              simtools.PARAMS['parameter_range'])

    # find growth rate at each point in time
    growth_rate = np.zeros(shape=time_axis.shape)
    for i in range(parameter_density.shape[1]):
        growth_rate[i] = simps(parameter_density[:, i]*f_rate_up(parameter_axis), x=parameter_axis)

    # write parameter density hdf5
    out = h5py.File(outfile, 'w')
    gp_pd = out.create_group('parameter_density')
    gp_pd['time_axis'] = time_axis
    gp_pd['parameter_axis'] = parameter_axis
    # gp_pd['parameter_density'] = parameter_density
    gp_pd['parameter_density'] = child_density
    gp_pd['growth_rate'] = growth_rate

    # write rate function data to simulation config toml
    simtools.PARAMS['mpi_noise_function_sigma'] = params['n']
    simtools.PARAMS['mpi_rate_function_width'] = params['w']
    simtools.PARAMS['mpi_rate_function_center'] = params['c']
    simtools.PARAMS['mpi_rate_function_shape'] = params['s']
    simtools.PARAMS['mpi_rate_function_max'] = params['m']
    simtools.PARAMS['mpi_rate_function_ratio'] = params['r']
    simtools.PARAMS['mpi_death_rate'] = growth_rate[-1]

    with open(paramfile, 'w') as params_toml:
        toml.dump(simtools.PARAMS, params_toml)


@main.command()
@click.option('-i', '--infile', type=click.Path())
@click.option('--save', type=click.Path(), default=None)
def plot_dataset(infile, save):
    """
    Plots for examining input to mpi simulator
    """

    def lr(x):
        return abs(x[-1] - x[0])

    data = h5py.File(infile, 'r')
    gp_pd = data['parameter_density']

    if save is not None:
        pdf_out = PdfPages(save)

    parameter_density = np.array(gp_pd['parameter_density'])
    parameter_axis = np.array(gp_pd['parameter_axis'])
    time_axis = np.array(gp_pd['time_axis'])

    # child density plot
    fig, axs = plt.subplots()
    fig.set_size_inches(4, 4)
    img = axs.imshow(
        np.transpose(parameter_density),
        extent=(np.min(parameter_axis), np.max(parameter_axis),
                np.min(time_axis), np.max(time_axis)),
        aspect=lr(parameter_axis)/lr(time_axis),
        cmap=cm.viridis,
        origin='lower'
    )

    cbr = fig.colorbar(img, ax=axs, fraction=0.046, pad=0.04)
    cbr.set_label('Parameter density', labelpad=-15)
    cbr.set_ticks([np.min(parameter_density), np.max(parameter_density)])
    cbr.set_ticklabels(['Low', 'High'])

    axs.set_ylabel('Time')
    axs.set_xlabel('Parameter')
    axs.grid()

    plt.tight_layout()

    if save is not None:
        pdf_out.savefig()
    else:
        plt.show()

    # child density first vs last
    fig, axs = plt.subplots()
    fig.set_size_inches(4, 3)
    axs.plot(parameter_axis, parameter_density[:, 0], color='k', linewidth=1.0, label='t = 0')
    axs.plot(parameter_axis, parameter_density[:, -1], color='k', linewidth=1.0, linestyle='--',
             label='t = ' + str(time_axis[-1]))
    axs.set_xlabel('Time')
    axs.set_ylabel('Parameter density')
    axs.legend()

    plt.tight_layout()

    if save is not None:
        pdf_out.savefig()
    else:
        plt.show()

    # growth rate over time
    growth_rate = np.array(gp_pd['growth_rate'])
    fig, axs = plt.subplots()
    fig.set_size_inches(4, 3)
    axs.plot(time_axis, growth_rate, color='k', linewidth=1.0)
    axs.set_xlabel('Time')
    axs.set_ylabel('Growth rate')

    plt.tight_layout()

    if save is not None:
        pdf_out.savefig()
    else:
        plt.show()


    if save is not None:
        pdf_out.close()


def moving_mean(vector, window):
    """
    Calculate moving mean of array-like object
    Reduces window size near edges
    """
    extent = (window - 1) / 2
    average = []
    for i, __ in enumerate(vector):
        local_extent = extent
        while not (i - local_extent >= 0 and i + local_extent + 1 <= len(vector)):
            local_extent -= 1
        imin = int(i - local_extent) if i - local_extent > 0 else 0
        imax = int(i + local_extent + 1) if i + local_extent + 1 < len(vector) else len(vector)
        sample = sorted(vector[imin:imax])
        average.append(sum(sample) / len(sample))
    return np.array(average)


@main.command()
@click.option('-p', '--paramfile', type=click.Path())
@click.option('-i', '--infile', type=click.Path())
@click.option('-o', '--outfile', type=click.Path())
@click.option('--save', type=click.Path(), default=None)
def mpiout(paramfile, infile, outfile, save):

    data = h5py.File(outfile, 'r')
    gp_result = data['result']

    indata = h5py.File(infile, 'r')
    gp_input = indata['parameter_density']

    simtools.PARAMS = toml.load(paramfile)

    if save is not None:
        pdf_out = PdfPages(save)


    # escape probability as a function of time of mutation
    fig, axs = plt.subplots()

    time_axis = simtools.get_time_axis(simtools.PARAMS['time_range_up'][1],
                                       simtools.PARAMS['time_points_up'])

    escaped_sum = np.sum(gp_result['escaped'], axis=0) / \
                  simtools.PARAMS['mpi_simulations_per_time_point']

    axs.plot(time_axis, escaped_sum, color='lightgrey', linewidth='0.5')
    axs.plot(time_axis, moving_mean(escaped_sum, 101), color='k', linewidth='1.0')
    axs.set_xlabel('Time of mutation')
    axs.set_ylabel('Probability of a mutant reaching ' + \
                   str(simtools.PARAMS['mpi_max_population_size']) + ' cells')

    if save is not None:
        pdf_out.savefig()
    else:
        plt.show()


    # mutation vulnerability as a function of time of mutation
    fig, axs = plt.subplots()

    time_axis = simtools.get_time_axis(simtools.PARAMS['time_range_up'][1],
                                       simtools.PARAMS['time_points_up'])

    escaped_sum = np.sum(gp_result['escaped'], axis=0) / \
                  simtools.PARAMS['mpi_simulations_per_time_point']

    growth_rate = gp_input['growth_rate']

    axs.plot(time_axis, escaped_sum*growth_rate, color='lightgrey', linewidth='0.5')
    axs.plot(time_axis, moving_mean(escaped_sum*growth_rate, 101), color='k',
             linewidth='1.0', label='Mutation risk')
    axs_cum = axs.twinx()
    axs_cum.plot(time_axis, np.cumsum(escaped_sum*growth_rate), color='k',
                 linestyle='--', linewidth='1.0')
    # empty curve drawn on first axis for legend purposes
    axs.plot([], [], color='k',
             linewidth='1.0', linestyle='--', label='Cumulative mutation risk')
    axs.set_xlabel('Time of mutation')

    axs.set_ylim(0, axs.get_ylim()[1])
    axs.set_yticks([0])
    axs_cum.set_ylim(0, axs_cum.get_ylim()[1])
    axs_cum.set_yticks([0])

    axs.legend()

    if save is not None:
        pdf_out.savefig()
    else:
        plt.show()


    # plot of growth rate, escape probability and mutation vulnerability all in one
    fig, axs = plt.subplots()

    time_axis = simtools.get_time_axis(simtools.PARAMS['time_range_up'][1],
                                       simtools.PARAMS['time_points_up'])

    escaped_sum = np.sum(gp_result['escaped'], axis=0) / \
                  simtools.PARAMS['mpi_simulations_per_time_point']

    growth_rate = gp_input['growth_rate']

    axs.plot(time_axis, escaped_sum, color='orange', linewidth='0.5', alpha=0.5)
    axs.plot(time_axis, moving_mean(escaped_sum, 101), color='orange', linewidth='1.0')
    axs_rate = axs.twinx()
    axs_rate.plot(time_axis, growth_rate, color='blue', linewidth=1.0)
    axs_risk = axs.twinx()
    axs_risk.plot(time_axis, escaped_sum*growth_rate, color='lightgrey', linewidth='0.5')
    axs_risk.plot(time_axis, moving_mean(escaped_sum*growth_rate, 101), color='k',
             linewidth='1.0', label='Mutation risk')
    axs_cum = axs.twinx()
    axs_cum.plot(time_axis, np.cumsum(escaped_sum*growth_rate), color='k',
                 linestyle='--', linewidth='1.0')

    # empty curves drawn on first axis for legend purposes
    axs.plot([], [], color='orange',
             linewidth='1.0', linestyle='-', label='Probability of reaching ' + str(simtools.PARAMS['mpi_max_population_size']) + ' cells')
    axs.plot([], [], color='blue',
             linewidth='1.0', linestyle='-', label='Normal cell average growth rate')
    axs.plot([], [], color='k',
             linewidth='1.0', linestyle='-', label='Mutation risk')
    axs.plot([], [], color='k',
             linewidth='1.0', linestyle='--', label='Cumulative mutation risk')

    axs.set_ylabel('Probability of a mutant reaching ' + \
                   str(simtools.PARAMS['mpi_max_population_size']) + ' cells')
    axs_rate.set_ylabel('Normal cell growth rate')
    axs.set_xlabel('Time of mutation')

    axs.set_ylim(0, axs.get_ylim()[1])
    axs_rate.set_ylim(0, axs_rate.get_ylim()[1])
    axs_risk.set_ylim(0, axs_risk.get_ylim()[1])
    axs_risk.set_yticks([0])
    axs_cum.set_ylim(0, axs_cum.get_ylim()[1])
    axs_cum.set_yticks([0])

    axs.legend(loc='lower right', frameon=False)

    if save is not None:
        pdf_out.savefig()
    else:
        plt.show()


    # plot of growth rate, escape probability and mutation vulnerability all in one
    # small multiples version
    # fig, axs = plt.subplots(nrows=3)
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(7, 4)
    gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
    axs = []
    axs.append(fig.add_subplot(gs[:, 0]))
    axs.append(fig.add_subplot(gs[0, 1]))
    axs.append(fig.add_subplot(gs[1, 1]))

    time_axis = simtools.get_time_axis(simtools.PARAMS['time_range_up'][1],
                                       simtools.PARAMS['time_points_up'])

    escaped_sum = np.sum(gp_result['escaped'], axis=0) / \
                  simtools.PARAMS['mpi_simulations_per_time_point']

    growth_rate = gp_input['growth_rate']

    axs[0].plot(time_axis, escaped_sum, color='orange', linewidth='0.4', alpha=0.5)
    axs[0].plot(time_axis, moving_mean(escaped_sum, 101), color='orange', linewidth='1.0')
    axs_rate = axs[0].twinx()
    axs_rate.plot(time_axis, growth_rate, color='blue', linewidth=1.0)
    axs[1].plot(time_axis, escaped_sum*growth_rate, color='lightgrey', linewidth='0.5')
    axs[1].plot(time_axis, moving_mean(escaped_sum*growth_rate, 101), color='k',
             linewidth='1.0', label='Mutation risk')
    axs[2].plot(time_axis, np.cumsum(escaped_sum*growth_rate), color='k',
                 linestyle='-', linewidth='1.0')

    # empty curves drawn on first axis for legend purposes
    axs[0].plot([], [], color='orange',
             linewidth='1.0', linestyle='-', label='Probability of reaching ' + str(simtools.PARAMS['mpi_max_population_size']) + ' cells')
    axs[0].plot([], [], color='blue',
             linewidth='1.0', linestyle='-', label='Normal cell average growth rate')
    # axs.plot([], [], color='k',
             # linewidth='1.0', linestyle='-', label='Mutation risk')
    # axs.plot([], [], color='k',
             # linewidth='1.0', linestyle='--', label='Cumulative mutation risk')

    axs[0].set_ylabel('Probability of a new mutant reaching ' + \
                   str(simtools.PARAMS['mpi_max_population_size']) + ' cells')
    axs_rate.set_ylabel('Normal cell growth rate')
    axs[1].set_ylabel('Mutation risk')
    axs[2].set_ylabel('Cumulative risk')
    for i in range(3):
        axs[i].set_xlabel('Time')

    axs[0].set_ylim(0, axs[0].get_ylim()[1])
    axs_rate.set_ylim(0, axs_rate.get_ylim()[1])
    axs[1].set_ylim(0, axs[1].get_ylim()[1])
    axs[1].set_yticks([0])
    axs[2].set_ylim(0, axs[2].get_ylim()[1])
    axs[2].set_yticks([0])

    axs[0].tick_params(axis='y', colors='orange')
    axs_rate.tick_params(axis='y', colors='blue')
    # axs[0].legend(frameon=False)

    if save is not None:
        pdf_out.savefig()
    else:
        plt.show()


    # plot of growth rate, escape probability and mutation vulnerability all in one

    # death time and escape time distribution as a function of time of mutation
    fig, axs = plt.subplots(ncols=2)
    fig.set_size_inches(6, 3)

    escaped = np.array(gp_result['escaped'])
    time = np.array(gp_result['time'])

    quantiles_death = [[], [], [], []]
    quantiles_escaped = [[], [], [], []]

    colors = ['grey', 'black', 'grey', 'lightgrey']

    for i in range(escaped.shape[1]):
        death_times = time[:, i][escaped[:, i] == 0]
        escaped_times = time[:, i][escaped[:, i] == 1]
        q_death = np.percentile(death_times, (25, 50, 75, 95))
        q_escaped = np.percentile(escaped_times, (25, 50, 75, 95)) if escaped_times.size != 0 else (None, None, None, None)
        for j in range(4):
            quantiles_death[j].append(q_death[j])
            quantiles_escaped[j].append(q_escaped[j])

    for i, color in enumerate(colors):
        axs[0].plot(
            simtools.get_time_axis(simtools.PARAMS['time_range_up'][1],
                                   simtools.PARAMS['time_points_up']),
            quantiles_death[i],
            linewidth=1.0,
            color=color
        )

    for i, color in enumerate(colors):
        axs[1].plot(
            simtools.get_time_axis(simtools.PARAMS['time_range_up'][1],
                                   simtools.PARAMS['time_points_up']),
            quantiles_escaped[i],
            linewidth=1.0,
            color=color
        )

    axs[0].set_xlabel('Time of mutation')
    axs[1].set_xlabel('Time of mutation')
    axs[0].set_ylabel('Time of death')
    axs[1].set_ylabel('Time of escape')

    plt.tight_layout()

    if save is not None:
        pdf_out.savefig()
    else:
        plt.show()


    # histogram of aggregate death/escape time distributions
    fig, axs = plt.subplots(ncols=2)
    fig.set_size_inches(6, 3)

    axs[0].hist(time[escaped == 0], color='lightgrey',
                range=(0, np.percentile(time[escaped == 0], 99)), bins=100,
                density=True)
    axs[1].hist(time[escaped == 1], color='lightgrey',
                range=(0, np.percentile(time[escaped == 1], 99) if escaped_times.size != 0 else 1), bins=100,
                density=True)

    x0 = np.linspace(0, np.percentile(time[escaped == 0], 99), 100)
    death_rate = simtools.PARAMS['mpi_death_rate']
    axs[0].plot(x0, death_rate*np.exp(-death_rate*x0), color='k', linewidth=1.0,
                label='Exponential dist.\n$\lambda$ = Death rate')
    axs[0].legend()
    axs[0].set_xlabel('Time of death')
    axs[1].set_xlabel('Time of escape')
    axs[0].set_ylabel('Probability density')
    axs[1].set_ylabel('Probability density')

    plt.tight_layout()

    if save is not None:
        pdf_out.savefig()
    else:
        plt.show()


    # histogram of max # cells in populations that did not escape
    # fig, axs = plt.subplots()
    # fig.set_size_inches(4, 4)

    # max_cells = np.array(gp_result['max_cells'])

    # axs.hist(max_cells[escaped == 0], color='k',
    #             range=(0.5, 5.5), bins=5)
    # axs.set_xlabel('Maximum number of cells achieved')
    # axs.set_ylabel('Frequency')

    # plt.tight_layout()

    # if save is not None:
    #     pdf_out.savefig()
    # else:
    #     plt.show()


    # histogram of first parameter in dead/escaped lines
    fig, axs = plt.subplots(ncols=2)
    fig.set_size_inches(6, 3)

    first_parameter = np.array(gp_result['first_parameter'])

    axs[0].hist(first_parameter[escaped == 0], color='lightgrey',
                bins=100, density=True)
    axs[1].hist(first_parameter[escaped == 1], color='lightgrey',
                bins=100, density=True)

    f_rate_down = Rate(
        simtools.PARAMS['mpi_rate_function_shape'],
        simtools.PARAMS['mpi_rate_function_center'],
        simtools.PARAMS['mpi_rate_function_width'],
        simtools.PARAMS['optimum_normal'], 1)

    x0 = np.linspace(axs[0].get_xlim()[0], axs[0].get_xlim()[1], 1000)
    axs[0].plot(x0, f_rate_down(x0)*axs[0].get_ylim()[1], color='k', linewidth=1.0, label='Rate function')
    x1 = np.linspace(axs[1].get_xlim()[0], axs[1].get_xlim()[1], 1000)
    axs[1].plot(x1, f_rate_down(x1)*axs[1].get_ylim()[1], color='k', linewidth=1.0, label='Rate function')

    axs[1].legend()

    for i in range(2):
        axs[i].set_xlabel('Parameter of first cell')
        axs[i].set_ylabel('Probability density')

    axs[0].set_title('Mutants that did not survive')
    axs[1].set_title('Mutants that reached ' + \
                     str(simtools.PARAMS['mpi_max_population_size']) + ' cells')

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
def generate_dataset_verify(paramfile, dbfile, outfile, history_id):
    """
    Generate start and end distribution for c++ mpi verification simulations
    """

    db_path = 'sqlite:///' + dbfile
    abc_history = History(db_path)
    abc_history.id = history_id

    simtools.PARAMS = toml.load(paramfile)

    abc_data, __ = abc_history.get_distribution(m=0, t=abc_history.max_t)

    parameters = ['s', 'c', 'w', 'n', 'm', 'r']
    params = {k: np.median(abc_data[k]) for k in parameters}

    f_noise = Noise(params['n'])
    simtools.PARAMS = toml.load(paramfile)

    f_rate_up = Rate(params['s'], params['c'], params['w'],
                     simtools.PARAMS['optimum_treatment'], params['m']*params['r'])
    f_rate_down = Rate(params['s'], params['c'], params['w'],
                       simtools.PARAMS['optimum_normal'], params['m'])

    f_initial_up = simtools.get_stationary_distribution_function(
        f_rate_down,
        f_noise,
        simtools.PARAMS['parameter_range'],
        simtools.PARAMS['parameter_points']
    )

    f_initial_down = simtools.get_stationary_distribution_function(
        f_rate_up,
        f_noise,
        simtools.PARAMS['parameter_range'],
        simtools.PARAMS['parameter_points']
    )


    time_axis_up, parameter_axis_up, __ = simtools.simulate_pde(
        f_initial_up,
        f_rate_up,
        f_noise,
        simtools.PARAMS['time_range_up'][1],
        simtools.PARAMS['time_points_up'],
        simtools.PARAMS['parameter_range'],
        simtools.PARAMS['parameter_points']
    )

    time_axis_down, parameter_axis_down, __ = simtools.simulate_pde(
        f_initial_down,
        f_rate_down,
        f_noise,
        simtools.PARAMS['time_range_down'][1],
        simtools.PARAMS['time_points_down'],
        simtools.PARAMS['parameter_range'],
        simtools.PARAMS['parameter_points']
    )

    assert all(parameter_axis_up == parameter_axis_down)

    # write parameter density hdf5
    out = h5py.File(outfile, 'w')
    gp_pd = out.create_group('parameter_density')
    gp_pd['time_axis_up'] = time_axis_up
    gp_pd['time_axis_down'] = time_axis_down
    gp_pd['parameter_axis'] = parameter_axis_up
    gp_pd['parameter_density_up'] = f_initial_up(parameter_axis_up)
    gp_pd['parameter_density_down'] = f_initial_down(parameter_axis_up)

    # write rate function data to simulation config toml
    simtools.PARAMS['mpi_noise_function_sigma'] = params['n']
    simtools.PARAMS['mpi_rate_function_width'] = params['w']
    simtools.PARAMS['mpi_rate_function_center'] = params['c']
    simtools.PARAMS['mpi_rate_function_shape'] = params['s']
    simtools.PARAMS['mpi_rate_function_max'] = params['m']
    simtools.PARAMS['mpi_rate_function_ratio'] = params['r']

    with open(paramfile, 'w') as params_toml:
        toml.dump(simtools.PARAMS, params_toml)


@main.command()
@click.option('-p', '--paramfile', type=click.Path())
@click.option('-i', '--infile', type=click.Path())
@click.option('-o', '--outfile', type=click.Path())
@click.option('--save', type=click.Path(), default=None)
def verification_plots(paramfile, infile, outfile, save):
    """
    plot comparing exact verification data to pde solution
    """

    if save is not None:
        pdf_out = PdfPages(save)

    simtools.PARAMS = toml.load(paramfile)

    params = {}

    params['n'] = simtools.PARAMS['mpi_noise_function_sigma']
    params['w'] = simtools.PARAMS['mpi_rate_function_width']
    params['c'] = simtools.PARAMS['mpi_rate_function_center']
    params['s'] = simtools.PARAMS['mpi_rate_function_shape']
    params['m'] = simtools.PARAMS['mpi_rate_function_max']
    params['r'] = simtools.PARAMS['mpi_rate_function_ratio']

    f_noise = Noise(params['n'])

    f_rate_up = Rate(params['s'], params['c'], params['w'],
                     simtools.PARAMS['optimum_treatment'], params['m']*params['r'])
    f_rate_down = Rate(params['s'], params['c'], params['w'],
                       simtools.PARAMS['optimum_normal'], params['m'])

    f_initial_up = simtools.get_stationary_distribution_function(
        f_rate_down,
        f_noise,
        simtools.PARAMS['parameter_range'],
        simtools.PARAMS['parameter_points']
    )

    f_initial_down = simtools.get_stationary_distribution_function(
        f_rate_up,
        f_noise,
        simtools.PARAMS['parameter_range'],
        simtools.PARAMS['parameter_points']
    )

    time_axis_up, parameter_axis_up, parameter_density_up = simtools.simulate_pde(
        f_initial_up,
        f_rate_up,
        f_noise,
        simtools.PARAMS['time_range_up'][1],
        simtools.PARAMS['time_points_up'],
        simtools.PARAMS['parameter_range'],
        simtools.PARAMS['parameter_points'],
        convolve_method='fft'
    )

    time_axis_down, parameter_axis_down, parameter_density_down = simtools.simulate_pde(
        f_initial_down,
        f_rate_down,
        f_noise,
        simtools.PARAMS['time_range_down'][1],
        simtools.PARAMS['time_points_down'],
        simtools.PARAMS['parameter_range'],
        simtools.PARAMS['parameter_points'],
        convolve_method='fft'
    )

    def lr(x):
        return x[1] - x[0]

    data = h5py.File(outfile, 'r')
    gp_result = data['result']

    inpt = h5py.File(infile, 'r')
    gp_input = inpt['parameter_density']

    # plot of expected density (pde)
    fig, axs = plt.subplots(ncols=2, nrows=2)
    fig.set_size_inches(6, 5)

    img = axs[0][0].imshow(
        np.transpose(parameter_density_up),
        extent=(np.min(parameter_axis_up), np.max(parameter_axis_up),
                np.min(time_axis_up), np.max(time_axis_up)),
        aspect=lr(parameter_axis_up)/lr(time_axis_up),
        cmap=cm.viridis,
        origin='lower'
    )

    # cbr = fig.colorbar(img, ax=axs, fraction=0.046, pad=0.04)
    # cbr.set_label('Parameter density', labelpad=-15)
    # cbr.set_ticks([np.min(parameter_density_up), np.max(parameter_density_up)])
    # cbr.set_ticklabels(['Low', 'High'])

    axs[0][0].set_ylabel('Time')
    axs[0][0].set_xlabel('Parameter')
    # axs[0].grid()

    img = axs[0][1].imshow(
        np.transpose(parameter_density_down),
        extent=(np.min(parameter_axis_down), np.max(parameter_axis_down),
                np.min(time_axis_down), np.max(time_axis_down)),
        aspect=lr(parameter_axis_down)/lr(time_axis_down),
        cmap=cm.viridis,
        origin='lower'
    )

    # cbr = fig.colorbar(img, ax=axs, fraction=0.046, pad=0.04)
    # cbr.set_label('Parameter density', labelpad=-15)
    # cbr.set_ticks([np.min(parameter_density_down), np.max(parameter_density_down)])
    # cbr.set_ticklabels(['Low', 'High'])

    axs[0][1].set_ylabel('Time')
    axs[0][1].set_xlabel('Parameter')
    # axs[0][1].grid()

    axs[1][0].plot(gp_input['parameter_axis'][:], gp_input['parameter_density_up'][:],
                label='Starting density (up)', color='k', linewidth=1.0)
    axs[1][1].plot(gp_input['parameter_axis'][:], gp_input['parameter_density_down'][:],
                label='Starting density (down)', color='k', linewidth=1.0)

    axs[1][0].set_xlabel('Parameter ($x$)')
    axs[1][0].set_ylabel('Parameter density (up)')
    axs[1][1].set_xlabel('Parameter ($x$)')
    axs[1][1].set_ylabel('Parameter density (down)')

    plt.tight_layout()

    if save is not None:
        pdf_out.savefig()
    else:
        plt.show()

    # define some shorthand names for upcoming calculations
    parameter_range = lr(simtools.PARAMS['parameter_range'])
    time_points_up = simtools.PARAMS['time_points_up']
    pdu = parameter_density_up
    pau = parameter_axis_up
    time_points_down = simtools.PARAMS['time_points_down']
    pdd = parameter_density_down
    pad = parameter_axis_down

    # plot of mean over time pde vs exact
    fig, axs = plt.subplots(ncols=2, nrows=2)
    fig.set_size_inches(6, 6)

    for i in range(simtools.PARAMS['mpi_statics_number_of_simulations']):
        axs[0][0].plot(
            gp_input['time_axis_up'][:], gp_result['mean_up'][:, i], color='k', linewidth=1.0, alpha=0.2)
        axs[1][0].plot(
            gp_input['time_axis_down'][:], gp_result['mean_down'][:, i], color='k', linewidth=1.0, alpha=0.2)
    axs[0][0].plot(time_axis_up, [np.sum(pdu[:, i]*pau)/pau.size*parameter_range
                                  for i in range(time_points_up)],
                   color='r', linewidth=2.0)
    axs[1][0].plot(time_axis_down, [np.sum(pdd[:, i]*pad)/pad.size*parameter_range
                                    for i in range(time_points_down)],
                   color='r', linewidth=2.0)

    for i in range(2):
        axs[i][0].set_xlabel('Time [days]')
        axs[i][0].set_ylabel('Mean $x$')

    for i in range(simtools.PARAMS['mpi_statics_number_of_simulations']):
        axs[0][1].plot(
            gp_input['time_axis_up'][:], gp_result['stdev_up'][:, i], color='k', linewidth=1.0, alpha=0.2)
        axs[1][1].plot(
            gp_input['time_axis_down'][:], gp_result['stdev_down'][:, i], color='k', linewidth=1.0, alpha=0.2)
    axs[0][1].plot(
        time_axis_up,
        [np.sqrt(np.sum(pdu[:, i]*pau**2)/pau.size*parameter_range - \
                 (np.sum(pdu[:, i]*pau)/pau.size*parameter_range)**2)
         for i in range(time_points_up)],
        color='r', linewidth=2.0)
    axs[1][1].plot(
        time_axis_down,
        [np.sqrt(np.sum(pdd[:, i]*pad**2)/pad.size*parameter_range - \
                 (np.sum(pdd[:, i]*pad)/pad.size*parameter_range)**2)
         for i in range(time_points_down)],
        color='r', linewidth=2.0)


    for i in range(2):
        axs[i][1].set_xlabel('Time [days]')
        axs[i][1].set_ylabel('Standard deviation of $x$')

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
def generate_dataset_holiday(paramfile, dbfile, outfile, history_id):
    """
    Generate a field using the pde for further c++ mpi simulation
    """

    db_path = 'sqlite:///' + dbfile
    abc_history = History(db_path)
    abc_history.id = history_id

    simtools.PARAMS = toml.load(paramfile)

    abc_data, __ = abc_history.get_distribution(m=0, t=abc_history.max_t)

    parameters = ['s', 'c', 'w', 'n', 'm', 'r']
    params = {k: np.median(abc_data[k]) for k in parameters}

    f_noise = Noise(params['n'])
    simtools.PARAMS = toml.load(paramfile)

    f_rate_up = Rate(params['s'], params['c'], params['w'],
                     simtools.PARAMS['optimum_treatment'], params['m']*params['r'])
    f_rate_down = Rate(params['s'], params['c'], params['w'],
                       simtools.PARAMS['optimum_normal'], params['m'])

    f_initial = simtools.get_stationary_distribution_function(
        f_rate_down,
        f_noise,
        simtools.PARAMS['parameter_range'],
        simtools.PARAMS['parameter_points']
    )

    time_points_full = int(simtools.PARAMS['time_points_up']* \
                           simtools.PARAMS['holiday_time_up_factor'])
    time_step = simtools.PARAMS['time_range_up'][1]*simtools.PARAMS['holiday_time_up_factor']/ \
                time_points_full

    # set up outfile
    out = h5py.File(outfile, 'w')
    gp_pd = out.create_group('parameter_density')
    gp_pd.create_dataset('time_axis', (1, time_points_full),
                         maxshape=(None, time_points_full))
    gp_pd.create_dataset('parameter_density', (1, simtools.PARAMS['parameter_points'], time_points_full),
                         maxshape=(None, simtools.PARAMS['parameter_points'], time_points_full))
    gp_pd.create_dataset('growth_rate', (1, time_points_full),
                         maxshape=(None, time_points_full))

    holiday_times = []
    n_trials = 0
    capacity = 1

    # lead simulation can be shared
    time_axis_lead, parameter_axis_lead, parameter_density_lead = simtools.simulate_pde(
        f_initial,
        f_rate_up,
        f_noise,
        simtools.PARAMS['time_range_up'][1]*simtools.PARAMS['holiday_time_up_factor'],
        time_points_full,
        simtools.PARAMS['parameter_range'],
        simtools.PARAMS['parameter_points']
    )

    for i in range(*simtools.PARAMS['holiday_start_range']):
        for j in range(*simtools.PARAMS['holiday_duration_range']):
            if i + j > time_points_full - 1:
                continue

            print(i, j)
            n_trials += 1

            holiday_times.append((i, j))

            lead_length = i
            holiday_length = j + 1
            tail_length = time_points_full - i - j + 1
            tail_length = max(0, tail_length)

            time_range_lead = simtools.PARAMS['time_range_up'][1]*lead_length/ \
                              simtools.PARAMS['time_points_up']
            time_range_holiday = simtools.PARAMS['time_range_up'][1]*holiday_length/ \
                                 simtools.PARAMS['time_points_up']
            time_range_tail = simtools.PARAMS['time_range_up'][1]*tail_length/ \
                              simtools.PARAMS['time_points_up']

            time_axis_holiday, parameter_axis_holiday, parameter_density_holiday = simtools.simulate_pde(
                simtools.distribution_to_function(parameter_axis_lead, parameter_density_lead[:, lead_length]),
                f_rate_down,
                f_noise,
                time_range_holiday,
                holiday_length,
                simtools.PARAMS['parameter_range'],
                simtools.PARAMS['parameter_points']
            )
            time_axis_tail, parameter_axis_tail, parameter_density_tail = simtools.simulate_pde(
                simtools.distribution_to_function(parameter_axis_holiday, parameter_density_holiday[:, -1]),
                f_rate_up,
                f_noise,
                time_range_tail,
                tail_length,
                simtools.PARAMS['parameter_range'],
                simtools.PARAMS['parameter_points']
            )

            growth_rate_lead = np.zeros(shape=time_axis_lead.shape)
            for k in range(lead_length):
                growth_rate_lead[k] = simps(parameter_density_lead[:, k]*f_rate_up(parameter_axis_lead),
                                            x=parameter_axis_lead)
            growth_rate_holiday = np.zeros(shape=time_axis_holiday.shape)
            for k in range(parameter_density_holiday.shape[1]):
                growth_rate_holiday[k] = simps(parameter_density_holiday[:, k]*f_rate_down(parameter_axis_holiday),
                                               x=parameter_axis_holiday)
            growth_rate_tail = np.zeros(shape=time_axis_tail.shape)
            for k in range(parameter_density_tail.shape[1]):
                growth_rate_tail[k] = simps(parameter_density_tail[:, k]*f_rate_up(parameter_axis_tail),
                                            x=parameter_axis_tail)

            child_density_lead = np.zeros(shape=(parameter_density_lead.shape[0], lead_length))
            for k in range(lead_length):
                child_density_lead[:, k] = simtools.get_child_distribution(parameter_density_lead[:, k],
                                                                           f_rate_up, f_noise,
                                                                           simtools.PARAMS['parameter_range'])
            child_density_holiday = np.zeros(shape=parameter_density_holiday.shape)
            for k in range(parameter_density_holiday.shape[1]):
                child_density_holiday[:, k] = simtools.get_child_distribution(parameter_density_holiday[:, k],
                                                                              f_rate_down, f_noise,
                                                                              simtools.PARAMS['parameter_range'])
            child_density_tail = np.zeros(shape=parameter_density_tail.shape)
            for k in range(parameter_density_tail.shape[1]):
                child_density_tail[:, k] = simtools.get_child_distribution(parameter_density_tail[:, k],
                                                                           f_rate_up, f_noise,
                                                                           simtools.PARAMS['parameter_range'])

            time_axis = np.concatenate([time_axis_lead[:lead_length],
                                        time_axis_holiday[:-1] + time_range_lead - time_step,
                                        time_axis_tail[:-1] + time_range_lead + time_range_holiday - time_step*2])
            # time_axis2 = simtools.get_time_axis(simtools.PARAMS['time_range_up'][1]* \
            #                                    simtools.PARAMS['holiday_time_up_factor'], time_points_full) # same for all
            parameter_axis = parameter_axis_lead # same for all
            parameter_density = np.concatenate([parameter_density_lead[:, :lead_length],
                                                parameter_density_holiday[:, :-1],
                                                parameter_density_tail[:, :-1]], axis=1)
            growth_rate = np.concatenate([growth_rate_lead[:lead_length],
                                          growth_rate_holiday[:-1],
                                          growth_rate_tail[:-1]])
            child_density = np.concatenate([child_density_lead[:, :lead_length],
                                            child_density_holiday[:, :-1],
                                            child_density_tail[:, :-1]], axis=1)

            if n_trials > capacity:
                gp_pd['time_axis'].resize(gp_pd['time_axis'].shape[0] * 2, 0)
                gp_pd['parameter_density'].resize(gp_pd['parameter_density'].shape[0] * 2, 0)
                gp_pd['growth_rate'].resize(gp_pd['growth_rate'].shape[0] * 2, 0)
                capacity *= 2
            gp_pd['time_axis'][n_trials - 1] = time_axis
            gp_pd['parameter_density'][n_trials - 1] = child_density
            gp_pd['growth_rate'][n_trials - 1] = growth_rate

    gp_pd['time_axis'].resize(n_trials, 0)
    gp_pd['parameter_density'].resize(n_trials, 0)
    gp_pd['growth_rate'].resize(n_trials, 0)


    # gp_pd['time_axis'] = time_axis
    gp_pd['parameter_axis'] = parameter_axis
    gp_pd['holiday_parameters'] = holiday_times
    # gp_pd['parameter_density'] = parameter_density
    # gp_pd['parameter_density'] = child_density
    # gp_pd['growth_rate'] = growth_rate

    # # write rate function data to simulation config toml
    simtools.PARAMS['mpi_noise_function_sigma'] = params['n']
    simtools.PARAMS['mpi_rate_function_width'] = params['w']
    simtools.PARAMS['mpi_rate_function_center'] = params['c']
    simtools.PARAMS['mpi_rate_function_shape'] = params['s']
    simtools.PARAMS['mpi_rate_function_max'] = params['m']
    simtools.PARAMS['mpi_rate_function_ratio'] = params['r']
    simtools.PARAMS['mpi_death_rate'] = growth_rate[-1]

    # simulation needs to know number of timelines
    simtools.PARAMS['mpi_holiday_timelines'] = len(holiday_times)

    with open(paramfile, 'w') as params_toml:
        toml.dump(simtools.PARAMS, params_toml)


@main.command()
@click.option('-p', '--paramfile', type=click.Path())
@click.option('-i', '--infile', type=click.Path())
@click.option('--save', type=click.Path(), default=None)
def plot_dataset_holiday(paramfile, infile, save):
    """
    Plots for examining input to drug holiday simulator
    """

    simtools.PARAMS = toml.load(paramfile)

    def lr(x):
        return abs(x[-1] - x[0])

    data = h5py.File(infile, 'r')
    gp_pd = data['parameter_density']

    if save is not None:
        pdf_out = PdfPages(save)

    # growth rate over time
    fig, axs = plt.subplots()

    growth_rate = np.array(gp_pd['growth_rate'])
    time_axis = np.array(gp_pd['time_axis'])

    rate_range = np.max(growth_rate) - np.min(growth_rate)

    for i in range(growth_rate.shape[0]):
        plt.plot(time_axis[i, :], growth_rate[i, :] + i*rate_range*1.2,
                 color='k', linewidth=0.5)

    axs.set_xlabel('Time [days]')
    axs.set_yticks([])

    fig.set_size_inches(4, 8)
    plt.tight_layout()

    if save is not None:
        pdf_out.savefig()
    else:
        plt.show()

    # cumulative growth heatmap
    fig, axs = plt.subplots()
    fig.set_size_inches(6, 4)

    ts_start_axis = np.array(sorted(set(gp_pd['holiday_parameters'][:, 0])))
    ts_duration_axis = np.array(sorted(set(gp_pd['holiday_parameters'][:, 1])))
    start_axis = ts_start_axis \
                 /(simtools.PARAMS['time_points_up']) \
                 *simtools.PARAMS['time_range_up'][1]
    duration_axis = ts_duration_axis \
                 /(simtools.PARAMS['time_points_up']) \
                 *simtools.PARAMS['time_range_up'][1]

    coordinates = [(np.where(ts_start_axis==x)[0][0],
                    np.where(ts_duration_axis==y)[0][0])
                   for x, y in gp_pd['holiday_parameters'][:, ]]
    cumulative_map = np.empty(shape=(start_axis.size, duration_axis.size))
    cumulative_map[:] = np.nan

    for i in range(gp_pd['parameter_density'].shape[0]):
        time_axis = gp_pd['time_axis'][i, :]
        growth_rate = gp_pd['growth_rate'][i, :]

        cumulative_map[coordinates[i]] = np.sum(growth_rate)

    print(cumulative_map)
    print(np.nanmax(cumulative_map))
    print(np.nanmin(cumulative_map))
    print(np.where(cumulative_map == np.nanmax(cumulative_map)))
    print(np.where(cumulative_map == np.nanmin(cumulative_map)))
    cum_min = np.where(cumulative_map == np.nanmin(cumulative_map))
    print(ts_start_axis[cum_min[0]])
    print(ts_duration_axis[cum_min[1]])

    print(cumulative_map[0, :])
    print(cumulative_map[:, 0])
    zero_effect = np.mean(cumulative_map[:, 0])

    print("zero_effect", zero_effect, np.std(cumulative_map[:, 0]))
    effect_range = max(abs(np.min(cumulative_map)), abs(np.max(cumulative_map)))

    img = axs.imshow(
        np.transpose(cumulative_map),
        extent=(np.min(start_axis), np.max(start_axis),
                np.min(duration_axis), np.max(duration_axis)),
        aspect=lr(start_axis)/lr(duration_axis),
        cmap=cm.RdBu_r,
        origin='lower',
        vmin=effect_range - (effect_range - zero_effect)*2,
        vmax=effect_range
    )

    cbr = fig.colorbar(img, ax=axs, fraction=0.046, pad=0.04)
    cbr.set_label('Average divisions per surviving cell')

    axs.set_xlabel('Holiday start day')
    axs.set_ylabel('Holiday duration [days]')

    if save is not None:
        pdf_out.savefig()
    else:
        plt.show()


    # holiday effect (if repeated)
    fig, axs = plt.subplots()

    ts_start_axis = np.array(sorted(set(gp_pd['holiday_parameters'][:, 0])))
    ts_duration_axis = np.array(sorted(set(gp_pd['holiday_parameters'][:, 1])))
    start_axis = ts_start_axis \
                 /(simtools.PARAMS['time_points_up']) \
                 *simtools.PARAMS['time_range_up'][1]
    duration_axis = ts_duration_axis \
                 /(simtools.PARAMS['time_points_up']) \
                 *simtools.PARAMS['time_range_up'][1]

    coordinates = [(np.where(ts_start_axis==x)[0][0],
                    np.where(ts_duration_axis==y)[0][0])
                   for x, y in gp_pd['holiday_parameters'][:, ]]
    cumulative_map = np.empty(shape=(start_axis.size, duration_axis.size))
    cumulative_map[:] = np.nan

    for i in range(gp_pd['parameter_density'].shape[0]):
        time_axis = gp_pd['time_axis'][i, :]
        growth_rate = gp_pd['growth_rate'][i, :]

        cumulative_map[coordinates[i]] = np.sum(growth_rate)

    print(cumulative_map)
    print(np.nanmax(cumulative_map))
    print(np.nanmin(cumulative_map))
    print(np.where(cumulative_map == np.nanmax(cumulative_map)))
    print(np.where(cumulative_map == np.nanmin(cumulative_map)))
    cum_min = np.where(cumulative_map == np.nanmin(cumulative_map))
    print(ts_start_axis[cum_min[0]])
    print(ts_duration_axis[cum_min[1]])

    zero_effect = np.mean(cumulative_map[0, :])

    print("zero_effect", zero_effect, np.std(cumulative_map[:, 0]))

    cumulative_map = zero_effect - cumulative_map

    effect_range = max(abs(np.min(cumulative_map)), abs(np.max(cumulative_map)))

    img = axs.imshow(
        np.transpose(cumulative_map),
        extent=(np.min(start_axis), np.max(start_axis),
                np.min(duration_axis), np.max(duration_axis)),
        aspect=lr(start_axis)/lr(duration_axis),
        cmap=cm.BrBG,
        origin='lower',
        vmin=-effect_range,
        vmax=effect_range
    )

    cbr = fig.colorbar(img, ax=axs, fraction=0.046, pad=0.04)
    # cbr.set_ticklabels(['Low', 'High'])


    if save is not None:
        pdf_out.savefig()
    else:
        plt.show()


    # # mean child density over time
    # fig, axs = plt.subplots()

    # child_density = np.array(gp_pd['parameter_density'])
    # parameter_axis = np.array(gp_pd['parameter_axis'])
    # time_axis = np.array(gp_pd['time_axis'])
    # time_points = time_axis.shape[0]
    # parameter_range = np.max(parameter_axis) - np.min(parameter_axis)
    # print(child_density.shape)
    # print(time_points)

    # average_child_density = \
    #     np.array([[np.sum(parameter_axis*child_density[i, :, j]/ \
    #                       parameter_axis.size*parameter_range)
    #                for i in range(time_points)]
    #               for j in range(child_density.shape[2])])

    # print(average_child_density.shape)

    # density_range = np.max(average_child_density) - np.min(average_child_density)

    # for i in range(time_axis.shape[0]):
    #     print(time_axis[i, :].shape, average_child_density[:, i].shape)
    #     plt.plot(time_axis[i, :], average_child_density[:, i] + i*density_range*1.2,
    #              color='k', linewidth=0.5)

    # if save is not None:
    #     pdf_out.savefig()
    # else:
    #     plt.show()

    # # mean child density#  over time heatmap
    # # fig, axs = plt.subplots()

    # # child_density = np.array(gp_pd['parameter_density'])
    # # parameter_axis = np.array(gp_pd['parameter_axis'])
    # # time_axis = np.mean(np.array(gp_pd['time_axis']), axis=0)
    # # time_points = time_axis.shape[0]
    # # parameter_range = np.max(parameter_axis) - np.min(parameter_axis)
    # # print(child_density.shape)
    # # print(time_points)

    # # average_child_density = \
    # #     np.array([[np.sum(parameter_axis*child_density[j, :, i]/ \
    # #                       parameter_axis.size*parameter_range)
    # #                for i in range(time_points)]
    # #               for j in range(child_density.shape[0])])

    # # fig.set_size_inches(4, 4)
    # # img = axs.imshow(
    # #     np.transpose(average_child_density),
    # #     extent=(np.min(parameter_axis), np.max(parameter_axis),
    # #             np.min(time_axis), np.max(time_axis)),
    # #     aspect=lr(parameter_axis)/lr(time_axis),
    # #     vmin=np.min(parameter_axis), vmax=np.max(parameter_axis),
    # #     cmap=cm.viridis,
    # #     origin='lower'
    # # )

    # # if save is not None:
    # #     pdf_out.savefig()
    # # else:
    # #     plt.show()

    # # time axis homogenaeity
    # fig, axs = plt.subplots()
    # time_axis = np.array(gp_pd['time_axis'])
    # average_time_axis = np.mean(np.array(gp_pd['time_axis']), axis=0)
    # for i in range(time_axis.shape[0]):
    #     axs.plot(time_axis[i, :] - average_time_axis, alpha=0.5, linewidth=1.0, color='k')

    # if save is not None:
    #     pdf_out.savefig()
    # else:
        # plt.show()

    if save is not None:
        pdf_out.close()


@main.command()
@click.option('-p', '--paramfile', type=click.Path())
@click.option('-i', '--infile', type=click.Path())
@click.option('-o', '--outfile', type=click.Path())
@click.option('-t', '--interfile', type=click.Path(), default=None)
def process_holiday(paramfile, infile, outfile, interfile):

    data = h5py.File(outfile, 'r')
    gp_result = data['result']

    indata = h5py.File(infile, 'r')
    gp_input = indata['parameter_density']
    parameter_density = gp_input['parameter_density']

    simtools.PARAMS = toml.load(paramfile)

    def lr(x):
        return x[1] - x[0]

    ts_start_axis = np.array(sorted(set(gp_input['holiday_parameters'][:, 0])))
    ts_duration_axis = np.array(sorted(set(gp_input['holiday_parameters'][:, 1])))
    start_axis = ts_start_axis \
                 /(simtools.PARAMS['time_points_up']) \
                 *simtools.PARAMS['time_range_up'][1]
    duration_axis = ts_duration_axis \
                 /(simtools.PARAMS['time_points_up']) \
                 *simtools.PARAMS['time_range_up'][1]

    coordinates = [(np.where(ts_start_axis==x)[0][0],
                    np.where(ts_duration_axis==y)[0][0])
                   for x, y in gp_input['holiday_parameters'][:, ]]
    cumulative_map = np.zeros(shape=(start_axis.size, duration_axis.size))

    for i in range(parameter_density.shape[0]):
        print(i, parameter_density.shape[0])
        growth_rate = gp_input['growth_rate'][i, :]
        escaped_sum = np.sum(gp_result['escaped'][:, :, i], axis=0) / \
                  simtools.PARAMS['mpi_holiday_simulations_per_timeline']

        cumulative_map[coordinates[i]] = np.sum(escaped_sum*growth_rate)

    print(cumulative_map)

    inter = h5py.File(interfile, 'w')
    gp_proc = inter.create_group('processed_output')
    # gp_proc.create_dataset('cumulative_risk', cumulative_map.shape)
    gp_proc['cumulative_risk'] = cumulative_map


@main.command()
@click.option('-p', '--paramfile', type=click.Path())
@click.option('-i', '--infile', type=click.Path())
@click.option('-o', '--outfile', type=click.Path())
@click.option('-t', '--interfile', type=click.Path(), default=None)
@click.option('--save', type=click.Path(), default=None)
def plot_processed_holiday(paramfile, infile, outfile, interfile, save):

    data = h5py.File(outfile, 'r')
    gp_result = data['result']

    indata = h5py.File(infile, 'r')
    gp_input = indata['parameter_density']

    inter = h5py.File(interfile, 'r')
    gp_proc = inter['processed_output']

    simtools.PARAMS = toml.load(paramfile)

    if save is not None:
        pdf_out = PdfPages(save)

    # heatmap

    fig, axs = plt.subplots()

    def lr(x):
        return x[-1] - x[0]

    ts_start_axis = np.array(sorted(set(gp_input['holiday_parameters'][:, 0])))
    ts_duration_axis = np.array(sorted(set(gp_input['holiday_parameters'][:, 1])))
    start_axis = ts_start_axis \
                 /(simtools.PARAMS['time_points_up']) \
                 *simtools.PARAMS['time_range_up'][1]
    duration_axis = ts_duration_axis \
                 /(simtools.PARAMS['time_points_up']) \
                 *simtools.PARAMS['time_range_up'][1]

    coordinates = [(np.where(ts_start_axis==x)[0][0],
                    np.where(ts_duration_axis==y)[0][0])
                   for x, y in gp_input['holiday_parameters'][:, ]]

    cumulative_map = np.array(gp_proc['cumulative_risk'])

    img = axs.imshow(
        np.transpose(cumulative_map),
        extent=(np.min(start_axis), np.max(start_axis),
                np.min(duration_axis), np.max(duration_axis)),
        aspect=lr(start_axis)/lr(duration_axis),
        cmap=cm.magma_r,
        origin='lower'
    )

    print(lr(start_axis), lr(duration_axis))

    cbr = fig.colorbar(img, ax=axs, fraction=0.046, pad=0.04)
    cbr.set_label('Cumulative mutation risk [multiples of baseline]')
    max_c = np.max(cumulative_map)/np.min(cumulative_map)
    cbr.set_ticks([(x + 1)*np.min(cumulative_map) for x in range(int(max_c + 1))])
    cbr.set_ticklabels([(x + 1) for x in range(int(max_c + 1))])

    axs.set_xlabel('Holiday start day')
    axs.set_ylabel('Holiday duration [days]')

    plt.tight_layout()

    if save is not None:
        pdf_out.savefig()
    else:
        plt.show()

    # linearity comparison

    final_risk = np.array(gp_proc['cumulative_risk'][-1, :])

    cumulative_growth = np.empty(shape=(start_axis.size, duration_axis.size))
    cumulative_growth[:] = np.nan

    for i in range(gp_input['parameter_density'].shape[0]):
        time_axis = gp_input['time_axis'][i, :]
        growth_rate = gp_input['growth_rate'][i, :]

        cumulative_growth[coordinates[i]] = np.sum(growth_rate)

    cum_min = np.where(cumulative_growth == np.nanmin(cumulative_growth))
    zero_effect = np.mean(cumulative_growth[:, 0])

    effect_range = max(abs(np.min(cumulative_growth)), abs(np.max(cumulative_growth)))

    final_rate = cumulative_growth[-1, :]

    fig, axs = plt.subplots()
    ax2 = axs.twinx()

    axs.plot(duration_axis, final_rate, color='blue')
    ax2.plot(duration_axis, final_risk, color='orange')

    axs.set_xlabel('Holiday duration [days]')
    axs.set_ylabel('Average divisions per surviving cell')
    ax2.set_ylabel('Cumulative mutation risk')

    if save is not None:
        pdf_out.savefig()
    else:
        plt.show()

    if save is not None:
        pdf_out.close()



@main.command()
@click.option('-p', '--paramfile', type=click.Path())
@click.option('-i', '--infile', type=click.Path())
@click.option('-o', '--outfile', type=click.Path())
@click.option('--save', type=click.Path(), default=None)
def holiday_plots(paramfile, infile, outfile, save):

    data = h5py.File(outfile, 'r')
    gp_result = data['result']

    indata = h5py.File(infile, 'r')
    gp_input = indata['parameter_density']

    simtools.PARAMS = toml.load(paramfile)

    if save is not None:
        pdf_out = PdfPages(save)


    # escape probability as a function of time of mutation
    fig, axs = plt.subplots()

    parameter_density = gp_input['parameter_density']

    for i in range(parameter_density.shape[0]):
        time_axis = gp_input['time_axis'][i, :]
        escaped_sum = np.sum(gp_result['escaped'][:, :, i], axis=0) / \
                  simtools.PARAMS['mpi_holiday_simulations_per_timeline']

        axs.plot(time_axis, escaped_sum + i/20,
                 color='lightgrey', linewidth='0.5', zorder=1, alpha=0.5)
        axs.plot(time_axis, moving_mean(escaped_sum, 101) + i/20,
                 color='k', linewidth='0.5', zorder=2)
    axs.set_xlabel('Time of mutation')
    axs.set_ylabel('Probability of a mutant reaching ' + \
                   str(simtools.PARAMS['mpi_max_population_size']) + ' cells')

    if save is not None:
        pdf_out.savefig()
    else:
        plt.show()


    # mutation vulnerability as a function of time of mutation
    fig, axs = plt.subplots()

    for i in range(parameter_density.shape[0]):
        time_axis = gp_input['time_axis'][i, :]
        growth_rate = gp_input['growth_rate'][i, :]
        escaped_sum = np.sum(gp_result['escaped'][:, :, i], axis=0) / \
                  simtools.PARAMS['mpi_holiday_simulations_per_timeline']

        axs.plot(time_axis, escaped_sum*growth_rate + i/2000,
                 color='lightgrey', linewidth='0.5', zorder=1, alpha=0.5)
        axs.plot(time_axis, moving_mean(escaped_sum*growth_rate, 101) + i/20,
                 color='k', linewidth='0.5', zorder=2)
    axs.set_xlabel('Time of mutation')

    if save is not None:
        pdf_out.savefig()
    else:
        plt.show()


    # cumulative mutation vulnerability as a function of time of mutation
    fig, axs = plt.subplots()

    for i in range(parameter_density.shape[0]):
        time_axis = gp_input['time_axis'][i, :]
        growth_rate = gp_input['growth_rate'][i, :]
        escaped_sum = np.sum(gp_result['escaped'][:, :, i], axis=0) / \
                  simtools.PARAMS['mpi_holiday_simulations_per_timeline']

        axs.plot(time_axis, np.cumsum(escaped_sum*growth_rate) + i/20,
                 color='k', linewidth='0.5', zorder=1)
    axs.set_xlabel('Time of mutation')

    if save is not None:
        pdf_out.savefig()
    else:
        plt.show()

    # cumulative mutation vulnerability heatmap
    fig, axs = plt.subplots()

    def lr(x):
        return x[1] - x[0]

    ts_start_axis = np.array(sorted(set(gp_input['holiday_parameters'][:, 0])))
    ts_duration_axis = np.array(sorted(set(gp_input['holiday_parameters'][:, 1])))
    start_axis = ts_start_axis \
                 /(simtools.PARAMS['time_points_up']) \
                 *simtools.PARAMS['time_range_up'][1]
    duration_axis = ts_duration_axis \
                 /(simtools.PARAMS['time_points_up']) \
                 *simtools.PARAMS['time_range_up'][1]

    coordinates = [(np.where(ts_start_axis==x)[0][0],
                    np.where(ts_duration_axis==y)[0][0])
                   for x, y in gp_input['holiday_parameters'][:, ]]
    cumulative_map = np.zeros(shape=(start_axis.size, duration_axis.size))

    for i in range(parameter_density.shape[0]):
        time_axis = gp_input['time_axis'][i, :]
        growth_rate = gp_input['growth_rate'][i, :]
        escaped_sum = np.sum(gp_result['escaped'][:, :, i], axis=0) / \
                  simtools.PARAMS['mpi_holiday_simulations_per_timeline']

        cumulative_map[coordinates[i]] = np.sum(escaped_sum*growth_rate)

    img = axs.imshow(
        np.transpose(cumulative_map),
        extent=(np.min(start_axis), np.max(start_axis),
                np.min(duration_axis), np.max(duration_axis)),
        aspect=lr(start_axis)/lr(duration_axis),
        cmap=cm.viridis,
        origin='lower'
    )

    if save is not None:
        pdf_out.savefig()
    else:
        plt.show()

    # cumulative mutation vulnerability heatmap
    # masking the top mask_amount of numbers
    mask_amount = 50 # in %

    fig, axs = plt.subplots()

    mask_limit = np.percentile(cumulative_map, 100 - mask_amount)
    masked_cumulative_map = copy.deepcopy(cumulative_map)
    masked_cumulative_map[cumulative_map > mask_limit] = None

    img = axs.imshow(
        np.transpose(masked_cumulative_map),
        extent=(np.min(start_axis), np.max(start_axis),
                np.min(duration_axis), np.max(duration_axis)),
        aspect=lr(start_axis)/lr(duration_axis),
        cmap=cm.viridis,
        origin='lower'
    )

    if save is not None:
        pdf_out.savefig()
    else:
        plt.show()


    # fig, axs = plt.subplots()

    # time_axis = simtools.get_time_axis(simtools.PARAMS['time_range_up'][1],
    #                                    simtools.PARAMS['time_points_up'])

    # escaped_sum = np.sum(gp_result['escaped'], axis=0) / \
    #               simtools.PARAMS['mpi_simulations_per_time_point']

    # growth_rate = gp_input['growth_rate']

    # axs.plot(time_axis, escaped_sum*growth_rate, color='lightgrey', linewidth='0.5')
    # axs.plot(time_axis, moving_mean(escaped_sum*growth_rate, 101), color='k',
    #          linewidth='1.0', label='Mutation risk')
    # axs_cum = axs.twinx()
    # axs_cum.plot(time_axis, np.cumsum(escaped_sum*growth_rate), color='k',
    #              linestyle='--', linewidth='1.0')
    # # empty curve drawn on first axis for legend purposes
    # axs.plot([], [], color='k',
    #          linewidth='1.0', linestyle='--', label='Cumulative mutation risk')
    # axs.set_xlabel('Time of mutation')

    # axs.set_ylim(0, axs.get_ylim()[1])
    # axs.set_yticks([0])
    # axs_cum.set_ylim(0, axs_cum.get_ylim()[1])
    # axs_cum.set_yticks([0])

    # axs.legend()

    # if save is not None:
    #     pdf_out.savefig()
    # else:
    #     plt.show()


    # # plot of growth rate, escape probability and mutation vulnerability all in one
    # fig, axs = plt.subplots()

    # time_axis = simtools.get_time_axis(simtools.PARAMS['time_range_up'][1],
    #                                    simtools.PARAMS['time_points_up'])

    # escaped_sum = np.sum(gp_result['escaped'], axis=0) / \
    #               simtools.PARAMS['mpi_simulations_per_time_point']

    # growth_rate = gp_input['growth_rate']

    # axs.plot(time_axis, escaped_sum, color='orange', linewidth='0.5', alpha=0.5)
    # axs.plot(time_axis, moving_mean(escaped_sum, 101), color='orange', linewidth='1.0')
    # axs_rate = axs.twinx()
    # axs_rate.plot(time_axis, growth_rate, color='blue', linewidth=1.0)
    # axs_risk = axs.twinx()
    # axs_risk.plot(time_axis, escaped_sum*growth_rate, color='lightgrey', linewidth='0.5')
    # axs_risk.plot(time_axis, moving_mean(escaped_sum*growth_rate, 101), color='k',
    #          linewidth='1.0', label='Mutation risk')
    # axs_cum = axs.twinx()
    # axs_cum.plot(time_axis, np.cumsum(escaped_sum*growth_rate), color='k',
    #              linestyle='--', linewidth='1.0')

    # # empty curves drawn on first axis for legend purposes
    # axs.plot([], [], color='orange',
    #          linewidth='1.0', linestyle='-', label='Probability of reaching ' + str(simtools.PARAMS['mpi_max_population_size']) + ' cells')
    # axs.plot([], [], color='blue',
    #          linewidth='1.0', linestyle='-', label='Normal cell average growth rate')
    # axs.plot([], [], color='k',
    #          linewidth='1.0', linestyle='-', label='Mutation risk')
    # axs.plot([], [], color='k',
    #          linewidth='1.0', linestyle='--', label='Cumulative mutation risk')

    # axs.set_ylabel('Probability of a mutant reaching ' + \
    #                str(simtools.PARAMS['mpi_max_population_size']) + ' cells')
    # axs_rate.set_ylabel('Normal cell growth rate')
    # axs.set_xlabel('Time of mutation')

    # axs.set_ylim(0, axs.get_ylim()[1])
    # axs_rate.set_ylim(0, axs_rate.get_ylim()[1])
    # axs_risk.set_ylim(0, axs_risk.get_ylim()[1])
    # axs_risk.set_yticks([0])
    # axs_cum.set_ylim(0, axs_cum.get_ylim()[1])
    # axs_cum.set_yticks([0])

    # axs.legend(loc='lower right', frameon=False)

    # if save is not None:
    #     pdf_out.savefig()
    # else:
    #     plt.show()


    # # plot of growth rate, escape probability and mutation vulnerability all in one

    # # death time and escape time distribution as a function of time of mutation
    # fig, axs = plt.subplots(ncols=2)
    # fig.set_size_inches(6, 3)

    # escaped = np.array(gp_result['escaped'])
    # time = np.array(gp_result['time'])

    # quantiles_death = [[], [], [], []]
    # quantiles_escaped = [[], [], [], []]

    # colors = ['grey', 'black', 'grey', 'lightgrey']

    # for i in range(escaped.shape[1]):
    #     death_times = time[:, i][escaped[:, i] == 0]
    #     escaped_times = time[:, i][escaped[:, i] == 1]
    #     q_death = np.percentile(death_times, (25, 50, 75, 95))
    #     q_escaped = np.percentile(escaped_times, (25, 50, 75, 95)) if escaped_times.size != 0 else (None, None, None, None)
    #     for j in range(4):
    #         quantiles_death[j].append(q_death[j])
    #         quantiles_escaped[j].append(q_escaped[j])

    # for i, color in enumerate(colors):
    #     axs[0].plot(
    #         simtools.get_time_axis(simtools.PARAMS['time_range_up'][1],
    #                                simtools.PARAMS['time_points_up']),
    #         quantiles_death[i],
    #         linewidth=1.0,
    #         color=color
    #     )

    # for i, color in enumerate(colors):
    #     axs[1].plot(
    #         simtools.get_time_axis(simtools.PARAMS['time_range_up'][1],
    #                                simtools.PARAMS['time_points_up']),
    #         quantiles_escaped[i],
    #         linewidth=1.0,
    #         color=color
    #     )

    # axs[0].set_xlabel('Time of mutation')
    # axs[1].set_xlabel('Time of mutation')
    # axs[0].set_ylabel('Time of death')
    # axs[1].set_ylabel('Time of escape')

    # plt.tight_layout()

    # if save is not None:
    #     pdf_out.savefig()
    # else:
    #     plt.show()


    # # histogram of aggregate death/escape time distributions
    # fig, axs = plt.subplots(ncols=2)
    # fig.set_size_inches(6, 3)

    # axs[0].hist(time[escaped == 0], color='lightgrey',
    #             range=(0, np.percentile(time[escaped == 0], 99)), bins=100,
    #             density=True)
    # axs[1].hist(time[escaped == 1], color='lightgrey',
    #             range=(0, np.percentile(time[escaped == 1], 99) if escaped_times.size != 0 else 1), bins=100,
    #             density=True)

    # x0 = np.linspace(0, np.percentile(time[escaped == 0], 99), 100)
    # death_rate = simtools.PARAMS['mpi_death_rate']
    # axs[0].plot(x0, death_rate*np.exp(-death_rate*x0), color='k', linewidth=1.0,
    #             label='Exponential dist.\n$\lambda$ = Death rate')
    # axs[0].legend()
    # axs[0].set_xlabel('Time of death')
    # axs[1].set_xlabel('Time of escape')
    # axs[0].set_ylabel('Probability density')
    # axs[1].set_ylabel('Probability density')

    # plt.tight_layout()

    # if save is not None:
    #     pdf_out.savefig()
    # else:
    #     plt.show()


    # # histogram of first parameter in dead/escaped lines
    # fig, axs = plt.subplots(ncols=2)
    # fig.set_size_inches(6, 3)

    # first_parameter = np.array(gp_result['first_parameter'])

    # axs[0].hist(first_parameter[escaped == 0], color='lightgrey',
    #             bins=100, density=True)
    # axs[1].hist(first_parameter[escaped == 1], color='lightgrey',
    #             bins=100, density=True)

    # f_rate_down = Rate(
    #     simtools.PARAMS['mpi_rate_function_shape'],
    #     simtools.PARAMS['mpi_rate_function_center'],
    #     simtools.PARAMS['mpi_rate_function_width'],
    #     simtools.PARAMS['optimum_normal'], 1)

    # x0 = np.linspace(axs[0].get_xlim()[0], axs[0].get_xlim()[1], 1000)
    # axs[0].plot(x0, f_rate_down(x0)*axs[0].get_ylim()[1], color='k', linewidth=1.0, label='Rate function')
    # x1 = np.linspace(axs[1].get_xlim()[0], axs[1].get_xlim()[1], 1000)
    # axs[1].plot(x1, f_rate_down(x1)*axs[1].get_ylim()[1], color='k', linewidth=1.0, label='Rate function')

    # axs[1].legend()

    # for i in range(2):
    #     axs[i].set_xlabel('Parameter of first cell')
    #     axs[i].set_ylabel('Probability density')

    # axs[0].set_title('Mutants that did not survive')
    # axs[1].set_title('Mutants that reached ' + \
    #                  str(simtools.PARAMS['mpi_max_population_size']) + ' cells')

    # plt.tight_layout()

    # if save is not None:
    #     pdf_out.savefig()
    # else:
    #     plt.show()


    if save is not None:
        pdf_out.close()



if __name__ == '__main__':
    main()
