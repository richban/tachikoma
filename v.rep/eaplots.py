import matplotlib as mpl
from scipy.interpolate.interpolate import interp1d

mpl.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np

def _set_plot_params(title, ratio):
    # Optionally fix the aspect ratio
    if ratio: plt.figure(figsize=plt.figaspect(ratio))

    mpl.style.use('seaborn-dark-palette')

    if title: plt.title(title)

def _save_or_show(save):
    if save:
        plt.savefig(save)
    else:
        plt.show()

    exit()

def plot_single_run(gen, fit_mins, fit_avgs, fit_maxs, title=None, ratio=None, save=None):
    _set_plot_params(title, ratio)

    line1 = plt.plot(gen, fit_mins, 'C1:', label="Minimum Fitness")
    line2 = plt.plot(gen, fit_avgs, "C2-", label="Average Fitness")
    line3 = plt.plot(gen, fit_maxs, "C3:", label="Max Fitness")

    lns = line1 + line2 + line3
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc="lower right")

    _save_or_show(save)

def runs_avg_std(run):
    set = np.array(run)
    return np.average(run, axis=0), np.std(run, axis=0)

def plot_multiple_runs(gen, fit_mins, fit_avgs, fit_maxs, title=None, ratio=None, save=None):
    gen = gen[0]

    _set_plot_params(title, ratio)
    ax = plt.gca()

    # Find averages and standard deviations over a set of run
    mins_avg, mins_std = runs_avg_std(fit_mins)
    avgs_avg, avgs_std = runs_avg_std(fit_avgs)
    maxs_avg, maxs_std = runs_avg_std(fit_maxs)

    # Find interpolations
    gen_interp = np.linspace(gen[0], gen[-1], 1000)
    mins_avg = interp1d(gen, mins_avg, kind='cubic')(gen_interp)
    mins_std = interp1d(gen, mins_std, kind='cubic')(gen_interp)
    avgs_avg = interp1d(gen, avgs_avg, kind='cubic')(gen_interp)
    avgs_std = interp1d(gen, avgs_std, kind='cubic')(gen_interp)
    maxs_avg = interp1d(gen, maxs_avg, kind='cubic')(gen_interp)
    maxs_std = interp1d(gen, maxs_std, kind='cubic')(gen_interp)

    # Plot standard deviations
    min_l = mins_avg - mins_std
    min_h = mins_avg + mins_std
    ax.fill_between(gen_interp, min_l, min_h, facecolor='C1', alpha=0.25, interpolate=True)

    avg_l = avgs_avg - avgs_std
    avg_h = avgs_avg + avgs_std
    ax.fill_between(gen_interp, avg_l, avg_h, facecolor='C2', alpha=0.25, interpolate=True)

    max_l = maxs_avg - maxs_std
    max_h = maxs_avg + maxs_std
    ax.fill_between(gen_interp, max_l, max_h, facecolor='C3', alpha=0.25, interpolate=True)

    mins = plt.plot(gen_interp, mins_avg, 'C1:', label="Minimum Fitnesses")
    avgs = plt.plot(gen_interp, avgs_avg, "C2-", label="Average Fitnesses")
    maxs = plt.plot(gen_interp, maxs_avg, "C3:", label="Max Fitnesses")

    plt.ylabel('Fitness')
    plt.xlabel('Generation')

    lns = mins + avgs + maxs
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc="upper right")

    _save_or_show(save)

    # elif backend == SEABORN:
    #     sns.set(style="darkgrid")
    #     # Plot the response with standard error
    #     sns.tsplot(data=fit_avgs, time="Generation", value="Final Distance to Goal")