
from timeit import default_timer as timer

import csv
import click
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import PchipInterpolator as pchip
from pyabc import History
import toml


import simtools


COLOR_CYCLE = []
STYLE_CYCLE = ['-', '--', '..', '-.']


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


@click.group()
def main():
    pass


@main.command()
def stationary_distribution():
    rf = Rate(50, 0.2, 0.008, 0.5, 1)
    nf = Noise(0.5)
    sdf = simtools.get_stationary_distribution_function(
        rf,
        nf,
        [-2, 3],
        1000
    )
    fig, axs = plt.subplots()
    x = np.linspace(-2, 3, 1000)
    y = sdf(x)
    axs.plot(x, y)
    plt.show()

@main.command()
def pde_scaling():
    rf = Rate(50, 0.2, 0.008, 0.5, 1)
    nf = Noise(0.01)
    sdf = simtools.get_stationary_distribution_function(
        rf,
        nf,
        [-2, 3],
        1000
    )
    fig, axs = plt.subplots()
    for method in ['np', 'fft']:
        x = [100, 200, 500, 1000, 2000]
        y = []
        for i in x:
            print(i)
            start = timer()
            simtools.simulate_pde(
                sdf,
                rf,
                nf,
                10,
                1000,
                [-2, 3],
                i,
                method
            )
            y.append(timer() - start)
        print(x, y)
        axs.plot(x, y)
        # axs.set_yscale('log')
    plt.show()
    fig, axs = plt.subplots()
    for method in ['np', 'fft']:
        x = [100, 200, 500, 1000, 2000, 4000]
        y = []
        means = []
        sigmas = []
        for i in x:
            print(i)
            start = timer()
            simtools.simulate_pde(
                sdf,
                rf,
                nf,
                10,
                i,
                [-2, 3],
                1000,
                method
            )
            y.append(timer() - start)
        print(x, y)
        axs.plot(x, y)
        # axs.set_yscale('log')
    plt.show()


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
@click.option('-x', '--indices', type=int, multiple=True)
def holiday_input_plots(paramfile, infile, indices):

    indata = h5py.File(infile, 'r')
    gp_input = indata['parameter_density']

    simtools.PARAMS = toml.load(paramfile)

    print('Index list')
    print('index', 'start', 'duration', sep='\t')
    for i, ix in enumerate(gp_input['holiday_parameters']):
        print(i, *ix, sep='\t')


    fig, axs = plt.subplots()
    fig.set_size_inches(6, 8)

    for i, style in zip(indices, STYLE_CYCLE):
        time_axis = gp_input['time_axis'][i, :]
        growth_rate = gp_input['growth_rate'][i, :]
        holiday_params = gp_input['holiday_parameters'][i, :]

        axs.plot(time_axis, growth_rate,
                 color='k', linewidth='1', linestyle=style)

        axs.axvline(time_axis[holiday_params[0]],
                    linewidth=0.5, color='lightgrey')
        axs.axvline(time_axis[holiday_params[0] + holiday_params[1]],
                    linewidth=0.5, color='lightgrey')

        rate_before = growth_rate[holiday_params[0] - 1]
        rate_return = np.argmax(
            growth_rate[(holiday_params[0] + holiday_params[1]):] >= rate_before
        )

        pts_to_days = simtools.PARAMS['time_points_up'] \
                      /simtools.PARAMS['time_range_up'][1]

        print(rate_before, rate_return,
              growth_rate[rate_return + holiday_params[0] + holiday_params[1]])
        print(holiday_params[0]/pts_to_days,
              holiday_params[1]/pts_to_days,
              rate_return/pts_to_days)
        axs.axvline(time_axis[rate_return + holiday_params[0] + holiday_params[1]],
                    linewidth=0.5, color='blue')

    axs.set_xlabel('Time')
    axs.set_xlabel('Growth rate')

    plt.tight_layout()
    plt.show()

    # small multiples version
    if len(indices) > 1:
        fig, axs = plt.subplots(nrows=len(indices))
        fig.set_size_inches(4, 2*len(indices))

        for i, ix in enumerate(indices):
            time_axis = gp_input['time_axis'][ix, :]
            growth_rate = gp_input['growth_rate'][ix, :]
            holiday_params = gp_input['holiday_parameters'][ix, :]

            axs[i].plot(time_axis, growth_rate,
                        color='k', linewidth='1', linestyle='-')

            # axs[i].axvline(time_axis[holiday_params[0]],
            #                linewidth=0.5, color='lightgrey')
            # axs[i].axvline(time_axis[holiday_params[0] + holiday_params[1]],
            #                linewidth=0.5, color='lightgrey')

            rate_before = growth_rate[holiday_params[0] - 1]
            rate_return = np.argmax(
                growth_rate[(holiday_params[0] + holiday_params[1]):] >= rate_before
            )

            pts_to_days = simtools.PARAMS['time_points_up'] \
                          /simtools.PARAMS['time_range_up'][1]

            print(rate_before, rate_return,
                  growth_rate[rate_return + holiday_params[0] + holiday_params[1]])
            print(holiday_params[0]/pts_to_days,
                  holiday_params[1]/pts_to_days,
                  rate_return/pts_to_days)

            # axs[i].axvline(time_axis[rate_return + holiday_params[0] + holiday_params[1]],
            #                linewidth=0.5, color='blue')

            holiday_start = holiday_params[0]/pts_to_days
            holiday_end = (holiday_params[0] + holiday_params[1])/pts_to_days
            effect_end = (rate_return + holiday_params[0] + holiday_params[1])/pts_to_days

            wt_area = plt.Polygon([(holiday_start, axs[i].get_ylim()[0]),
                                   (holiday_end, axs[i].get_ylim()[0]),
                                   (holiday_end, axs[i].get_ylim()[1]),
                                   (holiday_start, axs[i].get_ylim()[1])],
                                  color='lightgrey', zorder=-3)
            axs[i].add_patch(wt_area)

            axs[i].text((holiday_start + holiday_end)/2, axs[i].get_ylim()[1]*1.1,
                        'Treatment holiday',
                        color='grey', size=8,
                        verticalalignment='center',
                        horizontalalignment='center')

            axs[i].axvline(effect_end, linewidth=1.0, linestyle='--', color='k')

            axs[i].text(effect_end + axs[i].get_xlim()[1]*0.01,
                        axs[i].get_ylim()[0] + axs[i].get_ylim()[1]*0.1,
                        'End of holiday\naftereffects',
                        color='k', size=8,
                        verticalalignment='center',
                        horizontalalignment='left')

            axs[i].set_xlabel('Time')
            axs[i].set_ylabel('Growth rate')

        plt.tight_layout()
        plt.show()


@main.command()
@click.option('-p', '--paramfile', type=click.Path())
@click.option('-i', '--infile', type=click.Path())
@click.option('-o', '--outfile', type=click.Path())
@click.option('-x', '--indices', type=int, multiple=True)
def holiday_plots(paramfile, infile, outfile, indices):

    data = h5py.File(outfile, 'r')
    gp_result = data['result']

    indata = h5py.File(infile, 'r')
    gp_input = indata['parameter_density']

    simtools.PARAMS = toml.load(paramfile)

    fig, axs = plt.subplots(nrows=3)
    fig.set_size_inches(6, 8)

    print('Index list')
    print('index', 'start', 'duration', sep='\t')
    for i, ix in enumerate(gp_input['holiday_parameters']):
        print(i, *ix, sep='\t')

    parameter_density = gp_input['parameter_density']

    for i in indices:
        time_axis = gp_input['time_axis'][i, :]
        escaped_sum = np.sum(gp_result['escaped'][:, :, i], axis=0) / \
                  simtools.PARAMS['mpi_holiday_simulations_per_timeline']

        axs[0].plot(time_axis, escaped_sum,
                 color='lightgrey', linewidth='0.5', zorder=1, alpha=0.5)
        axs[0].plot(time_axis, moving_mean(escaped_sum, 101),
                 color='k', linewidth='0.5', zorder=2)
    axs[0].set_xlabel('Time of mutation')
    axs[0].set_ylabel('Probability of a mutant reaching ' + \
                   str(simtools.PARAMS['mpi_max_population_size']) + ' cells')

    for i in indices:
        time_axis = gp_input['time_axis'][i, :]
        growth_rate = gp_input['growth_rate'][i, :]
        escaped_sum = np.sum(gp_result['escaped'][:, :, i], axis=0) / \
                  simtools.PARAMS['mpi_holiday_simulations_per_timeline']

        axs[1].plot(time_axis, escaped_sum*growth_rate,
                 color='lightgrey', linewidth='0.5', zorder=1, alpha=0.5)
        axs[1].plot(time_axis, moving_mean(escaped_sum*growth_rate, 101),
                 color='k', linewidth='0.5', zorder=2)
    axs[1].set_xlabel('Time')
    axs[1].set_xlabel('Mutation risk')

    for i in indices:
        time_axis = gp_input['time_axis'][i, :]
        growth_rate = gp_input['growth_rate'][i, :]

        axs[2].plot(time_axis, growth_rate,
                 color='k', linewidth='0.5', zorder=2)
    axs[2].set_xlabel('Time')
    axs[2].set_xlabel('Growth rate')

    plt.tight_layout()
    plt.show()


    fig, axs = plt.subplots(ncols=len(indices))
    fig.set_size_inches(3*len(indices), 4)

    for j, i in enumerate(indices):
        axs[j].imshow(
            gp_input['parameter_density'][i, :, :]
        )

    plt.tight_layout()
    plt.show()

    # fig, axs = plt.subplots(nrows=len(indices) + 1, ncols=3)
    # fig.set_size_inches(6, 3*(len(indices) + 1))

    # first_parameter = gp_result['first_parameter']
    # escaped = gp_result['escaped']
    # print(first_parameter.shape)

    # for j, i in enumerate(indices):
    #     axs[j][0].hist(first_parameter[:, :512, i][escaped[:, :512, i] == 1],
    #                    bins=100, color='lightgrey', density=True)
    #     axs[j][1].hist(first_parameter[:, :512, i][escaped[:, :512, i] == 0],
    #                    bins=100, color='lightgrey', density=True)
    #     axs[j][2].hist(first_parameter[:, :512, i].flatten(),
    #                    bins=100, color='lightgrey', density=True)
    #     axs[-1][0].hist(first_parameter[:, :512, i][escaped[:, :512, i] == 1],
    #                     bins=100, density=True, alpha=0.3)
    #     axs[-1][1].hist(first_parameter[:, :512, i][escaped[:, :512, i] == 0],
    #                     bins=100, density=True, alpha=0.3, label=i)
    #     axs[-1][2].hist(first_parameter[:, :512, i].flatten(),
    #                     bins=100, density=True, alpha=0.3)

    # axs[-1][1].legend()

    # plt.tight_layout()
    # plt.show()

    # print the cumulative risk for a single start slice as a function of duration
    risk = []
    slice_ixs = [i for i, ix in enumerate(gp_input['holiday_parameters']) if ix[0] == 685]
    for ix in slice_ixs:
        print(ix)
        growth_rate = gp_input['growth_rate'][ix, :]
        escaped_sum = np.sum(gp_result['escaped'][:, :, ix], axis=0) / \
                  simtools.PARAMS['mpi_holiday_simulations_per_timeline']
        risk.append(np.sum(growth_rate*escaped_sum))
    fig, axs = plt.subplots(ncols=2)
    duration_axis = [gp_input['holiday_parameters'][ix, 1] for ix in slice_ixs]
    print(duration_axis)
    axs[0].plot(duration_axis, risk, color='k')
    axs[1].plot(gp_input['growth_rate'][slice_ixs[-1], :], color='k', linestyle='--')
    plt.show()

    # time axis homogenaeity
    fig, axs = plt.subplots()
    time_axis = np.array(gp_input['time_axis'])
    average_time_axis = np.mean(np.array(gp_input['time_axis']), axis=0)
    for i in indices:
        axs.plot(time_axis[i, :] - average_time_axis, alpha=0.5, linewidth=1.0, label=i)

    axs.legend()
    plt.show()


@main.command()
@click.option('-p', '--paramfile', type=click.Path())
@click.option('-b', '--dbfile', type=click.Path())
@click.option('-i', '--history-id', type=int, default=1)
def print_abcfit_rate_data(paramfile, dbfile, history_id):
    """
    Plots showing off the fit from abc
    """
    db_path = 'sqlite:///' + dbfile
    abc_history = History(db_path)
    abc_history.id = history_id

    simtools.PARAMS = toml.load(paramfile)

    ### PLOT OF RATE###
    abc_data, __ = abc_history.get_distribution(m=0,
                                                t=abc_history.max_t)

    parameters = ['s', 'c', 'w', 'n', 'm', 'r']
    params = {k: np.median(abc_data[k]) for k in parameters}

    f_rate_1 = Rate(params['s'], params['c'], params['w'], simtools.PARAMS['optimum_normal'], params['m'])
    f_rate_2 = Rate(params['s'], params['c'], params['w'], simtools.PARAMS['optimum_treatment'], params['m']*params['r'])
    f_noise = Noise(params['n'])

    x_axis = np.linspace(*simtools.PARAMS['parameter_range'], simtools.PARAMS['parameter_points'])

    print('x\trate1\trate2')
    for x, r1, r2 in zip(x_axis, f_rate_1(x_axis), f_rate_2(x_axis)):
        print(x, r1, r2, sep='\t')


@main.command()
@click.option('-c', '--csvfiles', type=click.Path(), multiple=True)
def plot_abcfit_rate_data(csvfiles):

    styles = ['-', '--']
    colours = ['k', 'r']

    fig, axs = plt.subplots()
    i = 0
    for csvfile in csvfiles:
        with open(csvfile, 'r') as in_csv:
            rdr = csv.DictReader(in_csv, dialect='excel-tab')
            x = []
            r1 = []
            r2 = []
            for l in rdr:
                x.append(l['x'])
                r1.append(l['rate1'])
                r2.append(l['rate2'])
            x = [float(z) for z in x]
            r1 = [float(z) for z in r1]
            r2 = [float(z) for z in r2]
            print(x, r1, r2)
            if i == 0:
                axs.plot(x, r1, linestyle=styles[0], color=colours[i], label='Mutant or untreated normal cell')
                axs.plot(x, r2, linestyle=styles[1], color=colours[i], label='Normal cell with treatment')
            else:
                axs.plot(x, r1, linestyle=styles[0], color=colours[i])
                axs.plot(x, r2, linestyle=styles[1], color=colours[i])
        i += 1

    axs.text(4.5, 1.2, 'Slow loss', color='k')
    axs.text(1.5, 0.5, 'Rapid loss', color='r')

    axs.set_xlabel('$x$')
    axs.set_ylabel('$\lambda(x)$')
    axs.set_xlim(0, 6)
    axs.set_ylim(axs.get_ylim()[0], axs.get_ylim()[1]*1.2)
    axs.legend(frameon=False)

    fig.set_size_inches(3.8, 3.8)
    plt.tight_layout()
    plt.show()



@main.command()
@click.option('-p', '--paramfiles', type=click.Path(), multiple=True)
@click.option('-i', '--infiles', type=click.Path(), multiple=True)
@click.option('-o', '--outfiles', type=click.Path(), multiple=True)
@click.option('--save', type=click.Path(), default=None)
def mpi_small_multiples_plot(paramfiles, infiles, outfiles, save):
    # plot of growth rate, escape probability and mutation vulnerability all in one
    # small multiples version
    # for multiple datasets

    if save is not None:
        pdf_out = PdfPages(save)

    linestyles = ['-', '--']

    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(7, 4)
    gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
    axs = []
    axs.append(fig.add_subplot(gs[:, 0]))
    axs.append(fig.add_subplot(gs[0, 1]))
    axs.append(fig.add_subplot(gs[1, 1]))
    axs_rate = axs[0].twinx()

    for i, __ in enumerate(infiles):

        outfile = outfiles[i]
        infile = infiles[i]
        paramfile = paramfiles[i]
        linestyle = linestyles[i]

        data = h5py.File(outfile, 'r')
        gp_result = data['result']

        indata = h5py.File(infile, 'r')
        gp_input = indata['parameter_density']

        simtools.PARAMS = toml.load(paramfile)

        time_axis = simtools.get_time_axis(simtools.PARAMS['time_range_up'][1],
                                          simtools.PARAMS['time_points_up'])

        escaped_sum = np.sum(gp_result['escaped'], axis=0) / \
                      simtools.PARAMS['mpi_simulations_per_time_point']

        # convert to log-odds
        escaped_sum = np.log(escaped_sum/(1 - escaped_sum))

        growth_rate = gp_input['growth_rate']

        axs[0].plot(time_axis, escaped_sum, color='orange', linewidth='0.4', alpha=0.5)
        axs[0].plot(time_axis, moving_mean(escaped_sum, 101), color='orange', linewidth=1.0, linestyle=linestyle)
        axs_rate.plot(time_axis, growth_rate, color='blue', linewidth=1.0, linestyle=linestyle)
        axs[1].plot(time_axis, escaped_sum*growth_rate, color='lightgrey', linewidth=0.5)
        axs[1].plot(time_axis, moving_mean(escaped_sum*growth_rate, 101), color='k',
                    linewidth=1.0, label='Mutation risk', linestyle=linestyle)
        axs[2].plot(time_axis, np.cumsum(escaped_sum*growth_rate), color='k',
                    linewidth=1.0, linestyle=linestyle)

        # empty curves drawn on first axis for legend purposes
        axs[0].plot([], [], color='orange',
                    linewidth='1.0', linestyle='-', label='Probability of reaching ' + str(simtools.PARAMS['mpi_max_population_size']) + ' cells')
        axs[0].plot([], [], color='blue',
                    linewidth='1.0', linestyle='-', label='Normal cell average growth rate')
        # axs.plot([], [], color='k',
        # linewidth='1.0', linestyle='-', label='Mutation risk')
        # axs.plot([], [], color='k',
        # linewidth='1.0', linestyle='--', label='Cumulative mutation risk')

        axs[0].set_ylabel('Log-odds that new mutant grows to ' + \
                          str(simtools.PARAMS['mpi_max_population_size']) + ' cells')
        axs_rate.set_ylabel('Normal cell growth rate')
        axs[1].set_ylabel('Mutation risk')
        axs[2].set_ylabel('Cumulative risk')
        for i in range(3):
            axs[i].set_xlabel('Time')

        # axs[0].set_ylim(0, axs[0].get_ylim()[1])
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


def hpdi(data, width=0.89):
    """
    calculate the hpdi for a set of samples
    """
    n_width = int(np.ceil(len(data)*width))
    # print(n_width)
    if n_width == 1:
        return [data[0], data[0]]
    if n_width == 2:
        return sorted(data)
    if n_width == len(data):
        return([min(data), max(data)])
    data_s = sorted(data)
    hpdis = []
    for i, a in enumerate(data_s):
        j = i + n_width
        if j >= len(data_s):
            continue
        b = data_s[j]
        hpdis.append([b - a, a, b])
    hpdis = sorted(hpdis, key=lambda x: x[0])
    # print(hpdis)
    return [hpdis[0][1], hpdis[0][2]]


@main.command()
# @click.option('-p', '--paramfile', type=click.Path())
# @click.option('-u', '--obsfile-up', type=click.Path())
# @click.option('-d', '--obsfile-down', type=click.Path())
@click.option('-b', '--dbfile', type=click.Path())
@click.option('-i', '--history-id', type=int, default=1)
def abchpdi(dbfile, history_id):
    """
    Diagnostic plots for examining how abc fitting worked
    """
    db_path = 'sqlite:///' + dbfile
    abc_history = History(db_path)
    abc_history.id = history_id

    # simtools.PARAMS = toml.load(paramfile)
    parameters = ['s', 'c', 'w', 'n', 'm', 'r']

    abc_data, __ = abc_history.get_distribution(m=0, t=abc_history.max_t)
    data = {x: abc_data[x] for x in parameters}
    # print(list(data['m']))
    hpdis = {x: hpdi(list(data[x])) for x in data}
    means = {x: np.mean(list(data[x])) for x in data}
    stds = {x: np.std(list(data[x])) for x in data}
    covs = {x: np.std(list(data[x]))/np.mean(list(data[x])) for x in data}
    # t_quartile1, t_medians, t_quartile3 = np.percentile(
    #     data, [25, 50, 75], axis=1
    # )
    for k, v in hpdis.items():
        print(k, v)
    for k, v in means.items():
        print(k, v)
    for k, v in stds.items():
        print(k, v)
    for k, v in covs.items():
        print("cov", k, np.round(v, 3))





if __name__ == '__main__':
    main()
