# imports
from .load import load
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from types import SimpleNamespace

from bokeh.plotting import figure, output_file, show
import bokeh


def visualize(filename, input_channel=0, input_channel2=1,
              stim_channel=2, stim_channel2=None,
              plot_mean=True,
              plot_separate_trials=False):
    """Simple function to visualize individual physiological recordings.

    Analog signals for up to two input channels and two stim channels
    are plotted. By default, all trial traces are plotted in the same plot
    in light grey lines, while a mean trace is plotted in black.

    Optionally, each trial can be plotted in a separate figure
    (plot_separate_trials).

    Parameters
    ------------
    filename : str
        The filename to visualize. Must be in same folder.

    input_channel : int
        Index of the primary input channel to visualize.
        - The default value of 0 typically gives voltage in current-clamp,
          and current in voltage-clamp.

    input_channel2 : int
        Index of the secondary input channel to visualize.
        - The default value of 1 typically gives current in current-clamp,
          and voltage in voltage-clamp.

    stim_channel : int or None
        Index of the first stim channel to visualize.
        - The default value of 2 typically gives the first stim channel
          (non-input-channel).
        - Use None to deactivate

    stim_channel2 : int or None
        Index of the second stim channel to visualize. Optional.
        - Set to a value if more than one stimulation type is used
          (eg two optic fibers, etc.)
        - Use None to deactivate.

    plot_mean : bool
        If True, plots a trial-averaged trace in black on top of the
        individual trial traces.

    plot_separate_trials : bool
        If True, plots all trials as separate figures with the trial
        number listed.
    """

    # Load data and initialize variables
    # ----------------
    if stim_channel is None:
        plot_stim = False
    elif stim_channel is not None:
        plot_stim = True

    if stim_channel2 is None and plot_stim is True:
        stim_channel2 = stim_channel

    d = load([filename], input_channel=input_channel,
             stim_channel=stim_channel)
    d2 = load([filename], input_channel=input_channel2,
              stim_channel=stim_channel2)

    # Plotting of all trials together
    # ------------
    if plot_separate_trials is False:
        fig = plt.figure()

        # Set up figure depending on whether stim is needed or not
        if plot_stim is True:
            spec = gs.GridSpec(nrows=3, ncols=1, figure=fig,
                               height_ratios=[1, 0.5, 0.5])
            ax_stim = fig.add_subplot(spec[2, 0])
            ax_ana1 = fig.add_subplot(spec[0, 0], sharex=ax_stim)
            ax_ana2 = fig.add_subplot(spec[1, 0], sharex=ax_stim)
        elif plot_stim is False:
            spec = gs.GridSpec(nrows=2, ncols=1, figure=fig,
                               height_ratios=[1, 0.5])
            ax_ana2 = fig.add_subplot(spec[1, 0])
            ax_ana1 = fig.add_subplot(spec[0, 0], sharex=ax_ana2)

        # Figure out number of trials
        n_trials = d.analog_signals[0].shape[0]

        # plot all traces
        for trial in range(n_trials):
            ax_ana1.plot(d.times[0], d.analog_signals[0][trial, :],
                         color=[0.5, 0.5, 0.5], linewidth=0.5)
            ax_ana2.plot(d2.times[0], d2.analog_signals[0][trial, :],
                         color=[0.5, 0.5, 0.5], linewidth=0.5)

        if plot_stim is True:
            ax_stim.plot(d.times[0], np.array(d.stim_signals[0]),
                         color='b', linewidth=0.5)
            ax_stim.plot(d2.times[0], np.array(d2.stim_signals[0]),
                         color='r', linewidth=0.5)
            ax_stim.set_ylabel('stim')
            ax_stim.set_xlabel('time (s)')

        # Calculate and plot mean, if asked for
        if plot_mean is True:
            _ana_smoothed = np.zeros_like(d.analog_signals[0][0, :])

            for trial in range(n_trials):
                _ana_smoothed += d.analog_signals[0][trial, :]

            _ana_smoothed /= n_trials

            ax_ana1.plot(d.times[0], _ana_smoothed,
                         color='k', linewidth=2)

        # label axes, etc.
        ax_ana1.set_ylabel('ch0')
        ax_ana2.set_ylabel('ch1')

        plt.show()
        plt.close(fig)
        del fig

    # Plotting of all trials separately
    # ---------------
    elif plot_separate_trials is True:
        n_trials = d.analog_signals[0].shape[0]
        figs = np.ndarray(n_trials, dtype=object)
        axs = np.ndarray(n_trials, dtype=object)

        for trial in range(n_trials):
            figs[trial] = plt.figure()
            axs[trial] = SimpleNamespace()

            if plot_stim is True:
                # Set up plots
                _spec = gs.GridSpec(nrows=3, ncols=1, figure=figs[trial],
                                    height_ratios=[1, 0.5, 0.5])

                axs[trial].stim = figs[trial].add_subplot(_spec[2, 0])
                axs[trial].ana1 = figs[trial].add_subplot(
                    _spec[0, 0], sharex=axs[trial].stim)
                axs[trial].ana2 = figs[trial].add_subplot(
                    _spec[1, 0], sharex=axs[trial].stim)

                # Plot stims
                axs[trial].stim.plot(d.times[0], np.array(d.stim_signals[0]),
                                     color='b', linewidth=0.5)
                axs[trial].stim.plot(d2.times[0], np.array(d2.stim_signals[0]),
                                     color='r', linewidth=0.5)
                axs[trial].stim.set_ylabel('stim')
                axs[trial].stim.set_xlabel('time (s)')

            elif plot_stim is False:
                _spec = gs.GridSpec(nrows=2, ncols=1, figure=figs[trial],
                                    height_ratios=[1, 0.5])

                axs[trial].ana2 = figs[trial].add_subplot(
                    _spec[1, 0])
                axs[trial].ana1 = figs[trial].add_subplot(
                    _spec[0, 0], sharex=axs[trial].ana2)

            # Plot analog channels
            axs[trial].ana1.plot(d.times[0], d.analog_signals[0][trial, :],
                                 color=[0.5, 0.5, 0.5], linewidth=0.5)
            axs[trial].ana2.plot(d2.times[0], d2.analog_signals[0][trial, :],
                                 color=[0.5, 0.5, 0.5], linewidth=0.5)

            figs[trial].suptitle(f'trial {trial}')
            axs[trial].ana1.set_ylabel('ch0')
            axs[trial].ana2.set_ylabel('ch1')

        plt.show()

        for fig in figs:
            plt.close(fig)
        del figs

    del d
    del d2

    plt.clf()

    return

    #plot_events compares two synwrapper attributes (eg group1.ampli,
    # group2.ampli)

    #plot_statwrapper compares two stat files on attributes. For ex, to plot means +- standard error:
    #stat1 = syn.get_stats(group1.ampli), stat2 = syn.get_stats(group2.ampli), syn.plot_statwrapper(stat1, stat2, ind = 0, err_ind = 2)


def plot_events(postsynaptic_events_1, postsynaptic_events_2, name1 = 'Group 1', name2 = 'Group 2',
                          hz = ' ', ylabel = False, ind = 0, err_ind = 1, ylimmin = True,
                          by_neuron = False, pool = False):

    if type(hz) is not str:
        hz = str(hz)


    stats_postsynaptic_events_1 = get_stats(postsynaptic_events_1)
    stats_postsynaptic_events_2 = get_stats(postsynaptic_events_2)

#    if pool is False:

    x_1 = range(1, len(stats_postsynaptic_events_1) + 1)
    x_2 = range(1, len(stats_postsynaptic_events_2) + 1)


    if ylabel  is False:
        ylabel = 'Normalized current amplitude'

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x_1, stats_postsynaptic_events_1[:, ind], color = 'g', linewidth = 2)
    plt.plot(x_2, stats_postsynaptic_events_2[:, ind],color = 'r', linewidth = 2)

    ax.fill_between(x_1, stats_postsynaptic_events_1[:, ind] - stats_postsynaptic_events_1[:, err_ind],
           stats_postsynaptic_events_1[:,ind] + stats_postsynaptic_events_1[:,err_ind], alpha=0.2, facecolor='g', linewidth = 0)
    ax.fill_between(x_2, stats_postsynaptic_events_2[:,ind] - stats_postsynaptic_events_2[:, err_ind],
           stats_postsynaptic_events_2[:,ind] + stats_postsynaptic_events_2[:, err_ind],alpha=0.2,facecolor='r', linewidth = 0)

    ax.set_xlabel('Stimulation number (' + hz + ')')
    ax.set_ylabel(ylabel)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.legend([name1, name2],frameon = False , loc = 1)

    ylim_curr = ax.get_ylim()
    if ylimmin is True:
        ax.set_ylim([0, ylim_curr[1]])

    name = 'stimtrain_' + name1 + '_' + name2 + '_' + hz + '_' + ylabel + '.jpg'
    plt.savefig(name, dpi = 800)


# ** plot_statwrappers :
def plot_statwrappers(stats_postsynaptic_events_1, stats_postsynaptic_events_2, name1 = 'Group 1', name2 = 'Group 2',
                          hz = ' ', ylabel = False, xlabel = False, ind = 0, err_ind = 1, ylimmin = False,
                          by_neuron = False, save = True):

    if type(hz) is not str:
        hz = str(hz)

    x_1 = range(1, len(stats_postsynaptic_events_1) + 1)
    x_2 = range(1, len(stats_postsynaptic_events_2) + 1)

    if ylabel  is False:
        ylabel = 'Normalized current amplitude'

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x_1, stats_postsynaptic_events_1[:, ind], color = 'g', linewidth = 2)
    plt.plot(x_2, stats_postsynaptic_events_2[:, ind],color = 'r', linewidth = 2)

    ax.fill_between(x_1, stats_postsynaptic_events_1[:, ind] - stats_postsynaptic_events_1[:, err_ind],
           stats_postsynaptic_events_1[:,ind] + stats_postsynaptic_events_1[:,err_ind], alpha=0.2, facecolor='g', linewidth = 0)
    ax.fill_between(x_2, stats_postsynaptic_events_2[:,ind] - stats_postsynaptic_events_2[:, err_ind],
           stats_postsynaptic_events_2[:,ind] + stats_postsynaptic_events_2[:, err_ind],alpha=0.2,facecolor='r', linewidth = 0)

    if xlabel is False:
        ax.set_xlabel('Stimulation number (' + hz + ')')
    else:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.legend([name1, name2],frameon = False , loc = 1)

    ylim_curr = ax.get_ylim()
    if ylimmin is True:
        ax.set_ylim([0, ylim_curr[1]])

    if save is True:
        name = 'stimtrain_' + name1 + '_' + name2 + '_' + hz + '_' + ylabel + '.jpg'
        plt.savefig(name, dpi = 800)



#---------TOOLS TO VISUALIZE SPONTANEOUS EVENTS-------#

# ** visualize spontaneous events :
# *** plot_identified_events :
def plot_identified_events(wrap, neuron = 0, trial = 0, start = 0, end = 10,  deriv = False):
    start_ind = np.int32(start * 5000)
    end_ind = np.int32(end * 5000)

    times = wrap.times[neuron][start_ind:end_ind]

    mask_for_ana = np.ones(len(wrap.analog_signals[neuron][trial,:]))
    mask_for_ana_allfinds = np.ones(len(wrap.analog_signals[neuron][trial,:]))

    mask_for_ana_allfinds[np.int32(wrap.stim_on[neuron][trial][:] + 25)] = 0
    mask_for_ana[np.int32(wrap.ampli[neuron][trial, :, 1].compressed())] = 0

    if deriv is False:
        ana_masked = np.ma.filled(np.ma.array(wrap.analog_signals[neuron][trial,:], mask = mask_for_ana), fill_value = np.nan)
        ana_verysmooth = sp.signal.savgol_filter(wrap.analog_signals[neuron][trial, :], 1001, 4)

        ana_allfinds_masked = np.ma.array(wrap.analog_signals[neuron][trial,:], mask = mask_for_ana_allfinds)


    elif deriv is True:
        ana_masked = np.ma.filled(np.ma.array(np.gradient(wrap.analog_signals[neuron][trial,:]), mask = mask_for_ana), fill_value = np.nan)
        ana_verysmooth = sp.signal.savgol_filter(np.gradient(wrap.analog_signals[neuron][trial, :]), 1001, 4)

        ana_allfinds_masked = np.ma.array(np.gradient(wrap.analog_signals[neuron][trial,:]), mask = mask_for_ana_allfinds)

    bokeh.io.curdoc().clear()
    output_file("identified_events.html", title = 'Identified mEPSCs')
    p = figure(title="Identified mEPSCs",
    plot_width = 1000, plot_height = 600)

    if deriv is False:
        full_signal = np.double(wrap.analog_signals[neuron][trial, start_ind:end_ind])

        p.line(times[0::10], full_signal[0::10], line_width=0.5)
        p.circle(times, ana_masked, color = "firebrick", size = 4)
        #p.circle(times, ana_allfinds_masked[start_ind:end_ind], color = (0, 1, 0), size = 2)



    elif deriv is True:

        plt.plot(np.gradient(wrap.analog_signals[neuron][trial, start_ind:end_ind]), color = [0.2, 0.2, 0.8], alpha = 0.5)
        plt.plot(ana_masked[start_ind:end_ind], '.r', linewidth = 5, color = [0.8, 0.2, 0.2])
        plt.plot(ana_allfinds_masked[start_ind:end_ind], '.r', linewidth = 5, color = [0.6, 0.2, 0.5], alpha = 0.3)

        plt.plot(ana_verysmooth[start_ind:end_ind], color = [0, 0.4, 0], alpha = 0.8)


    show(p)
    #output_file("MPL_plot.html", title = "Identified events")

    #mpld3.show(fig)
    #mpld3.save_html(fig, 'figure')
