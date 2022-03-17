# imports
from .load import load_all_channels
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.widgets import Button
import matplotlib.style as mplstyle

from types import SimpleNamespace


def preview(fname):
    """Simple function to interactively preview a single .abf file.
    Loads the file and plots the signals for all channels.

    Highlighted trials can be advanced with the Next and Prev buttons,
    while the Avg. button toggles a mean trace across all trials.

    Parameters
    ------------
    fname : str
        The name of the .abf file to preview.

    Returns
    ------------
    p : PreviewFile class instance
        An instance of the PreviewFile class.

    """

    p = PreviewFile()
    p.plot()

    return p


class PreviewFile(object):

    def __init__(self, fname,
                 figsize=(10, 7),
                 _color_reg=[0.65, 0.65, 0.65],
                 _color_bold=[0.0118, 0.443, 0.612],
                 _color_mean=[0.980, 0.259, 0.141]):
        """Simple class to interactively preview a single .abf file.
        Loads the file and plots the signals for all channels.

        Highlighted trials can be advanced with the Next and Prev buttons,
        while the Avg. button toggles a mean trace across all trials.

        Parameters
        ------------
        filename : str
            The filename to visualize. Must be in same folder.

        """
        # Load data and initialize variables
        self.fname = fname
        self._color_reg = _color_reg
        self._color_bold = _color_bold
        self._color_mean = _color_mean
        self.figsize = figsize

        plt.ion()
        mplstyle.use('fast')
        mpl.rcParams["path.simplify_threshold"] = 1.0
        mpl.rcParams["axes.spines.top"] = False
        mpl.rcParams["axes.spines.right"] = False

        # load and print summary of neuron
        self.rec = load_all_channels(fname)

        self.n_sigs = self.rec.sig.shape[0]
        self.n_trials = self.rec.sig[0].shape[0]

        print(f'file: {self.fname}')
        print(f'trials: {self.n_trials}')
        print(f'dur: {self.rec.t[0][-1]:.2f}s')

        for ind_sig in range(self.n_sigs):
            _sig_max = np.max(self.rec.sig[ind_sig])
            _sig_min = np.min(self.rec.sig[ind_sig])
            print(f'sig {ind_sig} ({self.rec.units[ind_sig]})')
            print(f'\trange: {_sig_max:.2f} to {_sig_min:.1f}')

    def plot(self):
        # Setup figure, gridspec and axs
        self.fig = plt.figure(figsize=self.figsize,
                              constrained_layout=True)
        n_rows = self.n_sigs + 1
        height_ratios = np.ones(n_rows)
        height_ratios[0] = 4  # First signal is taller
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
                    linewidth=0.5)

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
        self.fig.canvas.flush_events()

    def on_prev(self, event):
        self.unbold_trial(self._curr_trial)
        self.update_curr_trial_prev()
        self.bold_trial(self._curr_trial)

        self.update_trialtext()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

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
                linewidth=0.8)

    def unbold_trial(self, trial):
        n_sigs = self.rec.sig.shape[0]

        for sig in range(n_sigs):
            self.lines[sig][trial][0].remove()
            self.lines[sig][trial] = self.ax[sig].plot(
                self.rec.t[sig],
                self.rec.sig[sig][trial, :],
                color=self._color_reg,
                linewidth=0.5)

    def update_trialtext(self):
        self.text.set_text(f'file: {self.fname} | ' +
                           f'trial: {self._curr_trial}')
