# imports
from .load import load_all_channels
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.widgets import Button
import matplotlib.style as mplstyle

from types import SimpleNamespace

from bokeh.plotting import figure, output_file, show
import bokeh


class preview(object):

    def __init__(self, filename,
                 plot_mean=False,
                 plot_separate_trials=False,
                 _color_reg=[0.65, 0.65, 0.65],
                 _color_bold=[0.0118, 0.443, 0.612],
                 _color_mean=[0.980, 0.259, 0.141],
                 _subsampling=5):
        """Simple class to preview individual physiological recordings.

        Analog signals for up to two input channels and two stim channels
        are plotted. By default, all trial traces are plotted in the same plot
        in light grey lines, while a mean trace is plotted in black.

        Optionally, each trial can be plotted in a separate figure
        (plot_separate_trials).

        Parameters
        ------------
        filename : str
            The filename to visualize. Must be in same folder.

        plot_mean : bool
            If True, plots a trial-averaged trace in black on top of the
            individual trial traces.

        plot_separate_trials : bool
            If True, plots all trials as separate figures with the trial
            number listed.
        """
        # Load data and initialize variables
        self.fname = filename
        self._color_reg = _color_reg
        self._color_bold = _color_bold
        self._color_mean = _color_mean
        self._subsampling = _subsampling

        # plt.ion()
        mplstyle.use('fast')
        mpl.rcParams["path.simplify_threshold"] = 1.0
        mpl.rcParams["axes.spines.top"] = False
        mpl.rcParams["axes.spines.right"] = False

        self.rec = load_all_channels(filename)
        self.n_sigs = self.rec.sig.shape[0]
        self.n_trials = self.rec.sig[0].shape[0]

        if plot_separate_trials is False:
            # Setup figure, gridspec and axs
            self.fig = plt.figure(constrained_layout=True)
            n_rows = self.n_sigs + 1
            height_ratios = np.ones(n_rows)
            height_ratios[0] = 2  # First signal is taller
            height_ratios[-1] = 0.2

            spec = gs.GridSpec(nrows=n_rows, ncols=15,
                               figure=self.fig,
                               height_ratios=height_ratios)

            self.ax = []
            self.ax.append(self.fig.add_subplot(spec[0, :]))

            for sig in range(1, self.n_sigs):
                self.ax.append(self.fig.add_subplot(spec[sig, :],
                                                    sharex=self.ax[0]))

            # plot analog signals
            self.lines = np.empty(self.n_sigs, dtype=np.ndarray)
            self.lines_mean = np.empty(self.n_sigs, dtype=np.ndarray)
            self.mean = np.empty(self.n_sigs, dtype=np.ndarray)

            for sig in range(self.n_sigs):
                self.lines[sig] = np.empty(self.n_trials, dtype=object)
                self.ax[sig].set_ylabel(f'{self.rec.units[sig]}')

                for trial in range(self.n_trials):
                    self.lines[sig][trial] = self.ax[sig].plot(
                        self.rec.t[sig],
                        self.rec.sig[sig][trial, :],
                        color=self._color_reg,
                        linewidth=0.5,
                        markevery=self._subsampling)

                self.mean[sig] = np.mean(self.rec.sig[sig], axis=0)

            self.ax[-1].set_xlabel('time (s)')

            # buttons and callback
            self._curr_trial = 0
            self._mean_plotted = False

            ax_but_next = self.fig.add_subplot(
                spec[-1, 12:15])
            ax_but_next.set_zorder(10000)

            ax_but_prev = self.fig.add_subplot(
                spec[-1, 9:12])
            ax_but_prev.set_zorder(10000)

            ax_but_mean = self.fig.add_subplot(
                spec[-1, 7:9])

            self.buttons = SimpleNamespace()

            self.buttons.next = Button(ax_but_next, 'Next')
            self.buttons.next.on_clicked(self.on_next)

            self.buttons.prev = Button(ax_but_prev, 'Prev')
            self.buttons.prev.on_clicked(self.on_prev)

            self.buttons.mean = Button(ax_but_mean, 'Avg.')
            self.buttons.mean.on_clicked(self.on_mean)

            self.bold_trial(self._curr_trial)

            # add text
            self.text = self.fig.text(0.05, 0.03,
                                      s=f'file: {self.fname} | ' +
                                      f'trial: {self._curr_trial}',
                                      fontweight='bold',
                                      fontsize='medium',
                                      color=[0, 0, 0])
            plt.show()

        elif plot_separate_trials is True:
            self.figs = np.ndarray(self.n_trials, dtype=object)
            self.axs = np.ndarray(self.n_trials, dtype=object)

            for trial in range(self.n_trials):
                self.figs[trial] = plt.figure()
                self.axs[trial] = np.ndarray(self.n_sigs, dtype=object)

            height_ratios = np.ones(self.n_sigs)
            height_ratios[0] = 2

            spec = gs.GridSpec(nrows=self.n_sigs, ncols=1,
                               height_ratios=height_ratios)

            # plot analog signals
            for trial in range(self.n_trials):
                # Make axes for this trial's figure
                self.axs[trial][0] = self.figs[trial].add_subplot(
                    spec[0, 0])
                for sig in range(1, self.n_sigs):
                    self.axs[trial][sig] = self.figs[trial].add_subplot(
                        spec[sig, 0], sharex=self.axs[trial][0])

                # Plot on all axes
                for sig in range(self.n_sigs):
                    self.axs[trial][sig].plot(
                        self.rec.t[sig],
                        self.rec.sig[sig][trial, :],
                        color=self._color_reg, linewidth=0.5)

                    self.axs[trial][sig].set_xlabel('time (s)')
                    self.axs[trial][sig].set_ylabel(
                        f'{self.rec.units[sig]}')

            plt.show()

        return

    def update_curr_trial_next(self):
        if self._curr_trial >= self.n_trials - 1:
            self._curr_trial = 0
        else:
            self._curr_trial += 1

    def update_curr_trial_prev(self):
        if self._curr_trial <= 0:
            self._curr_trial = self.n_trials - 1
        else:
            self._curr_trial -= 1

    def on_next(self, event):
        self.unbold_trial(self._curr_trial)
        self.update_curr_trial_next()
        self.bold_trial(self._curr_trial)

        self.update_trialtext()
        self.fig.canvas.draw_idle()

    def on_prev(self, event):
        self.unbold_trial(self._curr_trial)
        self.update_curr_trial_prev()
        self.bold_trial(self._curr_trial)

        self.update_trialtext()
        self.fig.canvas.draw_idle()

    def on_mean(self, event):
        if self._mean_plotted is False:
            for sig in range(self.n_sigs):
                self.lines_mean[sig] = self.ax[sig].plot(
                    self.rec.t[0],
                    self.mean[sig],
                    color=self._color_mean,
                    linewidth=1.5)
            self._mean_plotted = True

        elif self._mean_plotted is True:
            for sig in range(self.n_sigs):
                self.lines_mean[sig][0].remove()
            self._mean_plotted = False

        self.fig.canvas.draw_idle()

    def bold_trial(self, trial):
        n_sigs = self.rec.sig.shape[0]

        for sig in range(n_sigs):
            self.lines[sig][trial][0].remove()
            self.lines[sig][trial] = self.ax[sig].plot(
                self.rec.t[sig],
                self.rec.sig[sig][trial, :],
                color=self._color_bold,
                linewidth=0.8,
                markevery=self._subsampling)

    def unbold_trial(self, trial):
        n_sigs = self.rec.sig.shape[0]

        for sig in range(n_sigs):
            self.lines[sig][trial][0].remove()
            self.lines[sig][trial] = self.ax[sig].plot(
                self.rec.t[sig],
                self.rec.sig[sig][trial, :],
                color=self._color_reg,
                linewidth=0.5,
                markevery=self._subsampling)

    def update_trialtext(self):
        self.text.set_text(f'file: {self.fname} | ' +
                           f'trial: {self._curr_trial}')


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
