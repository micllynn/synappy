# imports
from .core_fns import find_spontaneous_events, find_stim_events
from .helper_fns import find_last, _get_median_filtered
import numpy as np
import scipy.signal as sp_signal
import scipy.optimize as sp_opt
import scipy.integrate as sp_integrate
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.widgets import Button

from types import SimpleNamespace


class EphysObject(object):
    def __init__(self):
        """The EphysObject class is the primary class in synappy for loading
        and working with electrophysiological data.

        It is created by calling the synappy.load() function, which loads a
        dataset of recordings.

        Class methods provide convenient ways to search for stimulus-
        triggered or spontaneous events, as well as to quantify event stats
        such as amplitude, decay, integral. Some class methods can also
        visualize traces and plot summary statistics.

        The following provides a brief tutorial of working with EphysObjects:
        >>> import synappy as syn
        >>> d = syn.load(['ex_file_1.abf', 'ex_file_2.abf'])
            Loading 1 recordings...
                    File: 17511006.abf
                            50 trials
                            pA
                            3.0s
            Added .sig
            Added .sig_stim
            Added .t
        >>> d.add_events()
            Event counts:
                    Neuron  0 :  1  event(s)
            Added .events
        """
        pass

    def add_events(self,
                   event_type='stim',
                   stim_thresh=2,
                   **kwargs_spont):
        """
        Add event times, either triggered by a stimulus channel or
        spontaneously occurring, and add their indices to
        an EphysObject instance. Used by methods to quantify stimulus-triggered
        event statistics.

        Stores results in self.events, which has the format:
            events[neuron][trial][event_index]

        Parameters
        -------------
        event_type : str
            Indicates whether the event is stimulus-triggered
            (event_type='stim') or spontaneous (event_type='spont').

        stim_thresh : float
            The stimulus onset threshold, typically in volts for TTL pulses
            (although this can vary depending on the type of stimulus signal.)

        **kwargs_spont : dict
            Keyword arguments for spontaneous event detection,
            passed to find_spontaneous(). See .find_spontaneous() for
            more details.

            All possible kwargs and their recommended values are:
                {spont_filtsize=1001,
                spont_threshampli=3,
                spont_threshderiv=-1.2,
                spont_thresh_pos_deriv=0.1,
                savgol_polynomial=3}

        Example
        ------------
        >>> d = syn.load(['ex_file.abf'])
        >>> d.add_events(stim_thresh=2)

        Attributes added to class instance
        ------------
        self.events : np.ndarray
            An array of stimulus/spontaneous event onset indices
            for each neuron.
            Indices correspond to the .sig or .t attributes.
            - if event_type='stim':
                .events[neuron][trial][stim]
            - if event_type='spont':
                .events[neuron][trial][spontaneous_event]
        """
        self.event_type = event_type

        if event_type == 'stim':
            self.events = find_stim_events(self, thresh=stim_thresh)
        elif event_type == 'spont':
            self.events = find_spontaneous_events(self, kwargs_spont)

        print('Added .events')

        return

    def add_event_times(self, t_events, neur='all'):
        """
        Manually specify a list of event times and add them to
        an EphysObject instance. Used by methods to quantify stimulus-triggered
        event statistics.

        Stores results in self.events, which has the format:
            events[neuron][trial][event_index]

        Parameters
        -------------
        t_events : list
            A list of event times.

        neur : 'all' or int
            If neur is 'all', then all neurons are updated with a shared set
            of event times specified by t_events.
            If neur is int, then only neur is updated with these times.

        Example
        ------------
        Example 1: all neurons
        >>> d = syn.load(['ex_file.abf'])
        >>> d.add_event_times(t_events=[0.1, 0.2, 0.5], neur='all')

        >>> d = syn.load(['ex_file.abf'])
        >>> d.add_event_times(t_events=[0.5, 0.7, 0.8], neur=2)

        Attributes added to class instance
        ------------
        self.events : np.ndarray
            An array of event onset indices
            for each neuron.
            Indices correspond to the .sig or .t attributes.
        """
        self.event_type = 'stim'

        n_neurs = len(self.sig)
        self.events = np.empty(n_neurs, dtype=np.ndarray)

        if neur == 'all':
            for neur in range(n_neurs):
                # convert from t to inds
                _ind_events = np.empty_like(t_events, dtype=np.int)
                for ind, t_event in enumerate(t_events):
                    _ind_events[ind] = self._from_t_to_ind(t_event, neur,
                                                           t_format='s')
                # update self.events
                n_trials = self.sig[neur].shape[0]
                self.events[neur] = np.empty(n_trials, dtype=np.ndarray)
                for trial in range(n_trials):
                    self.events[neur][trial] = _ind_events

        elif type(neur) == int:
            # convert from t to inds
            _ind_events = np.empty_like(t_events, dtype=np.int)
            for ind, t_event in enumerate(t_events):
                _ind_events[ind] = self._from_t_to_ind(t_event, neur,
                                                       t_format='s')
            # update self.events
            n_trials = self.sig[neur].shape[0]
            self.events[neur] = np.empty(n_trials, dtype=np.ndarray)
            for trial in range(n_trials):
                self.events[neur][trial] = _ind_events

        return

    def remove_events(self, neur, event_inds):
        n_trials = len(self.events[neur])
        for trial in range(n_trials):
            self.events[neur][trial] = np.delete(self.events[neur][trial],
                                                 event_inds)
        return

    def add_ampli(self, event_sign='pos',
                  t_baseline_lower=20, t_baseline_upper=0.2,
                  t_event_lower=5, t_event_upper=30,
                  t_savgol_filt=2,
                  latency_method='max_ampli',
                  x_sd=0):
        """
        Computes pre-event baseline values, event amplitudes,
        and event latencies (computed in a number of ways).

        This requires a self.events attribute, created by calling
        the method self.add_events(). For each stimulus, a baseline
        signal is calculated between t_baseline_lower and
        t_baseline_upper before the stimulus onset. Next, a
        maximum event amplitude is calculated. Finally, the event
        latency (time to the peak amplitude, or alternately
        other latency metrics) is computed.

        These values are stored as the following attributes in the
        EphysObject instance:
            .ampli
            .baseline
            .latency

        Parameters
        -------------
        event_sign : str
            The sign of the events. Can either be 'pos',
            reflecting EPSPs/IPSCs in IC/VC, or 'neg',
            reflecting IPSPs/EPSCs in IC/VC.

        t_baseline_lower : float
            The time before stimuli, in ms, from which to
            start computing a pre-event baseline.

        t_baseline_upper : float
            The time before stimuli, in ms, from which to
            stop computing a pre-event baseline.

        t_event_lower : float
            The time after stimuli, in ms, from which to
            start searching for events.

        t_event_upper : float
            The time after stimuli, in ms, from which to
            stop searching for events.

        t_savgol_filt : int
            Width of savgol filter applied to data, in ms,
            before computing maximum amplitude.

        latency_method : str
            The method used to calculate latency from stimulus to
            event. Possible methods are:
            - 'max_ampli': Time from stimulus to maximum amplitude.
            - 'max_slope': Time from stimulus to maximum first deriv.
            - 'baseline_plus_4sd': Time from stimulus to time where
                signal exceeds baseline levels + 4 standard deviations.
            - 'baseline_plus_x_sd' : Time from stimulus to time where signal
                exceeds baseline + x sd.
            - '80_20_line': Computes the times where the signal reaches 20%
                and 80% of the maximum amplitude, then draws a straight line
                between these and determines where this line first intersects
                the signal. The time from the stimulus to this point gives
                the latency. (Provides a best guess of when the signal starts
                to rise.)

        Attributes added to class instance
        ------------
        .ampli : SimpleNamespace
            .ampli.data[neuron][trial, event]
                Baseline-subtracted maximum amplitude data
                (in pA or mV).
            .ampli.inds[neuron][trial, event]
                Indices in .sig of maximum amplitudes.
            .ampli.params
                SimpleNamespace storing key params from .add_ampli() method
                related to amplitude.
        .baseline : SimpleNamespace
            .baseline.mean[neuron][trial, event]
                Mean baseline values (in pA or mV).
            .baseline.std[neuron][trial, event]
                Standard deviation of baseline values (in pA or mV).
            .baseline.inds_start[neuron][trial, event]
                Indices of the start of baseline period in .sig
            .baseline.inds_stop[neuron][trial, event]
                Indices of the end of baseline period in .sig
            .baseline.params
                SimpleNamespace storing key params from .add_ampli() method
                related to baseline.
        .latency : SimpleNamespace
            .latency.data[neuron][trial, event]
                Event latency from stimulus onset (sec).
            .latency.inds[neuron][trial, event]
                Indices in .sig of event latency.
            .latency.params
                SimpleNamespace storing key params from .add_ampli() method
                related to latency.
        """
        self.__check_for_events()

        num_neurons = len(self.sig)

        if event_sign == 'pos' or event_sign == 'up':
            self.event_sign = 'pos'
        if event_sign == 'neg' or event_sign == 'down':
            self.event_sign = 'neg'

        # Define new vars to store event amplitude and latency
        template_arr_neurs = np.empty(num_neurons, dtype=np.ndarray)

        self.baseline = SimpleNamespace(mean=template_arr_neurs.copy(),
                                        std=template_arr_neurs.copy(),
                                        inds_start=template_arr_neurs.copy(),
                                        inds_stop=template_arr_neurs.copy(),
                                        params=SimpleNamespace())
        self.baseline.params.t_baseline_lower = t_baseline_lower
        self.baseline.params.t_baseline_upper = t_baseline_upper

        self.ampli = SimpleNamespace(data=template_arr_neurs.copy(),
                                     inds=template_arr_neurs.copy(),
                                     params=SimpleNamespace())
        self.ampli.params.t_event_lower = t_event_lower
        self.ampli.params.t_event_upper = t_event_upper

        self.latency = SimpleNamespace(data=template_arr_neurs.copy(),
                                       inds=template_arr_neurs.copy(),
                                       params=SimpleNamespace())
        self.latency.params.latency_method = latency_method

        self._all_stats = [self.baseline.mean, self.baseline.std,
                           self.baseline.inds_start, self.baseline.inds_stop,
                           self.ampli.data, self.ampli.inds,
                           self.latency.data, self.latency.inds]

        for neuron in range(num_neurons):
            n_trials = len(self.sig[neuron])

            # preinitialize vars and arrays for this neuron
            # -----------
            ind_savgol_filt = self._from_t_to_ind(t_savgol_filt, neuron) + 1

            ind_rel_bl_lower = self._from_t_to_ind(t_baseline_lower, neuron)
            ind_rel_bl_upper = self._from_t_to_ind(t_baseline_upper, neuron)
            ind_rel_event_lower = self._from_t_to_ind(t_event_lower, neuron)
            ind_rel_event_upper = self._from_t_to_ind(t_event_upper, neuron)

            # condition for spontaneous events:
            if self.event_type == 'spont':
                max_events = 0
                for trial in range(n_trials):
                    if len(self.events[neuron][trial]) > max_events:
                        max_events = len(self.events[neuron][trial])
            elif self.event_type == 'stim':
                max_events = len(self.events[neuron][0])

            # initialize arrays in .baseline, .ampli, .latency:
            template_arr_neurs = np.empty((n_trials, max_events),
                                          dtype=np.ndarray)
            for _arr in self._all_stats:
                _arr[neuron] = template_arr_neurs.copy()

            for trial in range(n_trials):
                n_events = len(self.events[neuron][trial])

                # if spontaneous events, mask non-events in this trial
                to_mask = np.arange(max_events-1, n_events-1, -1)
                for _arr in self._all_stats:
                    _arr[neuron][trial, to_mask] = 0

                for event in range(n_events):
                    # Calculate event baseline mean and stdev
                    # ------------
                    _ind_bl_start = int(
                        self.events[neuron][trial][event]
                        - ind_rel_bl_lower)
                    _ind_bl_stop = int(
                        self.events[neuron][trial][event]
                        - ind_rel_bl_upper)
                    _ind_eventsearch_start = int(
                        self.events[neuron][trial][event]
                        + ind_rel_event_lower)
                    _ind_eventsearch_stop = int(
                        self.events[neuron][trial][event]
                        + ind_rel_event_upper)

                    self.baseline.inds_start[neuron][
                        trial, event] = _ind_bl_stop
                    self.baseline.inds_stop[neuron][
                        trial, event] = _ind_bl_start

                    # Calculate event baseline
                    self.baseline.mean[neuron][trial, event] = np.mean(
                        self.sig[neuron][
                            trial, _ind_bl_start:_ind_bl_stop])
                    self.baseline.std[neuron][trial, event] = np.std(
                        self.sig[neuron][
                            trial, _ind_bl_start:_ind_bl_stop])

                    # Calculate event max amplitude and its index
                    # -------------
                    _sig_event = self.sig[neuron][
                        trial, _ind_eventsearch_start:_ind_eventsearch_stop]
                    _sig_event_ind_start = _ind_eventsearch_start
                    _sig_event_filt = sp_signal.savgol_filter(
                        _sig_event, ind_savgol_filt, 3)

                    if self.event_sign == 'pos':
                        _sig_event_ind_maxampli = np.argmax(
                            _sig_event_filt, axis=-1)
                    elif self.event_sign == 'neg':
                        _sig_event_ind_maxampli = np.argmin(
                            _sig_event_filt, axis=-1)
                    _ind_maxampli = _sig_event_ind_maxampli \
                        + _sig_event_ind_start

                    # index of max amplitude
                    self.ampli.inds[neuron][trial, event] = \
                        _ind_maxampli

                    # event amplitude
                    self.ampli.data[neuron][trial, event] \
                        = self.sig[neuron][
                            trial,
                            self.ampli.inds[neuron][trial, event]]

                    # subtract baseline from amplitude
                    self.ampli.data[neuron][trial, event] \
                        -= self.baseline.mean[neuron][trial, event]

                    # Calculate event onset latencies
                    # ------------------------------
                    if latency_method == 'max_ampli':
                        self.latency.data[neuron][trial, event] \
                            = self.t[neuron][
                                self.ampli.inds[neuron][trial, event]] \
                            - self.t[neuron][
                                self.events[neuron][trial][event]]
                        self.latency.inds[neuron][trial, event] \
                            = self.ampli.inds[neuron][trial, event]

                    elif latency_method == 'max_slope':
                        # calc
                        _sig_event_filt_grad = np.gradient(
                            _sig_event_filt[
                                0:np.max([_sig_event_ind_maxampli, 2]).astype(
                                    np.int)])
                        _ind_lat = np.argmax(np.abs(_sig_event_filt_grad))

                        # store
                        self.latency.inds[neuron][trial, event] \
                            = _ind_lat + _sig_event_ind_start
                        self.latency.data[neuron][trial, event] \
                            = self.t[neuron][
                                int(_ind_lat - _sig_event_ind_start)]

                    elif latency_method == 'baseline_plus_4sd':
                        # calc
                        if self.event_sign == 'pos':
                            _thresh = self.baseline.mean[neuron][trial, event]\
                                + 4 * self.baseline.std[neuron][trial, event]
                        elif self.event_sign == 'neg':
                            _thresh = self.baseline.mean[neuron][trial, event]\
                                - 4 * self.baseline.std[neuron][trial, event]
                        try:
                            if self.event_sign == 'pos':
                                _ind_lat = np.where(
                                    _sig_event_filt > _thresh)[0][0]
                            if self.event_sign == 'neg':
                                _ind_lat = np.where(
                                    _sig_event_filt < _thresh)[0][0]
                        except IndexError:  # If no crossings are found (fail)
                            _ind_lat = 0

                        # store
                        self.latency.inds[neuron][trial, event] \
                            = _ind_lat + _sig_event_ind_start
                        self.latency.data[neuron][trial, event] \
                            = self.t[neuron][
                                _ind_lat + _sig_event_ind_start] \
                            - self.t[neuron][_sig_event_ind_start]

                    elif latency_method == 'baseline_plus_x_sd':
                        # calc
                        if self.event_sign == 'pos':
                            _thresh = self.baseline.mean[neuron][trial, event]\
                                + x_sd * self.baseline.std[neuron][
                                    trial, event]
                        elif self.event_sign == 'neg':
                            _thresh = self.baseline.mean[neuron][trial, event]\
                                - x_sd * self.baseline.std[neuron][
                                    trial, event]
                        try:
                            if self.event_sign == 'pos':
                                _ind_lat = np.where(
                                    _sig_event_filt > _thresh)[0][0]
                            if self.event_sign == 'neg':
                                _ind_lat = np.where(
                                    _sig_event_filt < _thresh)[0][0]
                        except IndexError:  # If no crossings are found (fail)
                            _ind_lat = 0

                        # store
                        self.latency.inds[neuron][trial, event] \
                            = _ind_lat + _sig_event_ind_start
                        self.latency.data[neuron][trial, event] \
                            = self.t[neuron][
                                _ind_lat + _sig_event_ind_start] \
                            - self.t[neuron][_sig_event_ind_start]

                    elif latency_method == '80_20_line':
                        # calc
                        if self.event_sign == 'pos':
                            val_80pc = 0.8 * (
                                self.ampli.data[neuron][trial, event])\
                                + self.baseline.mean[neuron][trial, event]
                            val_20pc = 0.2 * (
                                self.ampli.data[neuron][trial, event])\
                                + self.baseline.mean[neuron][trial, event]
                        if self.event_sign == 'neg':
                            val_80pc = 0.8 * (
                                self.ampli.data[neuron][trial, event])\
                                - self.baseline.mean[neuron][trial, event]
                            val_20pc = 0.2 * (
                                self.ampli.data[neuron][trial, event])\
                                - self.baseline.mean[neuron][trial, event]

                        arr_80pc = val_80pc * np.ones(len(
                                _sig_event_filt[0:_sig_event_ind_maxampli]))
                        arr_20pc = val_20pc * np.ones(len(
                                _sig_event_filt[0:_sig_event_ind_maxampli]))

                        diff_80pc = (_sig_event[0:_sig_event_ind_maxampli]
                                     - arr_80pc) > 0
                        diff_20pc = (_sig_event[0:_sig_event_ind_maxampli]
                                     - arr_20pc) > 0

                        if self.event_sign == 'pos':
                            ind_80cross = find_last(diff_80pc, val=0)
                            ind_20cross = find_last(diff_20pc, val=0)
                        elif self.event_sign == 'neg':
                            ind_80cross = find_last(diff_80pc, val=1)
                            ind_20cross = find_last(diff_20pc, val=1)

                        if ind_20cross > ind_80cross or ind_80cross == 0:
                            ind_80cross = int(1)
                            ind_20cross = int(0)

                        val_80cross = _sig_event_filt[ind_80cross]
                        val_20cross = _sig_event_filt[ind_20cross]

                        slope_8020_line = (val_80cross - val_20cross) \
                            / (ind_80cross - ind_20cross)

                        vals_8020_line = np.zeros(
                            len(_sig_event_filt[0:ind_80cross + 1]))
                        vals_8020_line = [(val_80cross - (ind_80cross - i)
                                           * slope_8020_line)
                                          for i in range(ind_80cross)]

                        vals_baseline = self.baseline.mean[
                            neuron][trial, event] * np.ones(len(
                                _sig_event_filt[0:ind_80cross]))
                        diff_sq_8020_line = (
                            (vals_baseline - vals_8020_line)**2
                            + (_sig_event[0:ind_80cross] - vals_8020_line)**2)
                        _ind_lat = np.argmin(diff_sq_8020_line)

                        # store
                        self.latency.inds[neuron][trial, event] \
                            = _ind_lat + _sig_event_ind_start
                        self.latency.data[neuron][trial, event] \
                            = self.t[neuron][
                                _ind_lat + _sig_event_ind_start] \
                            - self.t[neuron][_sig_event_ind_start]

                    else:
                        raise Exception('The specified latency_method ' +
                                        'does not exist. Please ' +
                                        'choose a valid string.')

        # self.height is a legacy attribute
        self.height = self.ampli

        print('Added .ampli \nAdded .latency \nAdded .baseline')

        return

    def add_ampli_norm(self):
        """Adds normalized amplitude measurement to the class instance as
        .ampli_norm. Amplitudes are normalized to the mean ampli for each
        stimulus delivered to each neuron.

        .ampli must be an existing attribute, through the .add_ampli() method.

        Attributes added to class instance
        ------------
        self.ampli : SimpleNamespace
            .ampli_norm.data[neuron][trial, event]
                Baseline-subtracted normalized max amplitude data
                (in pA or mV).
            .ampli_norm.inds[neuron][trial, event]
                Indices in .sig of normalized max amplitudes.
        """
        self.__check_for_events()
        self.__check_for_ampli()

        num_neurons = len(self.ampli.data)
        self.ampli_norm = SimpleNamespace(
            data=np.empty(num_neurons, dtype=np.ndarray),
            inds=np.empty(num_neurons, dtype=np.ndarray))

        for neuron in range(num_neurons):
            _mean_amplis = np.mean(self.ampli.data[neuron], axis=0)

            self.ampli_norm.data[neuron] = self.ampli.data[neuron].copy() \
                / _mean_amplis
            self.ampli_norm.inds[neuron] = self.ampli.inds[neuron].copy()

        print('Added .ampli_norm')

        return

    def add_failure_sorting(self, thresh=False, _invert_sorting=False):
        """
        Adds trial-by-trial success/failure sorting to the events, where
        failures are stored as masked elements in the following attributes:
        .ampli, .latency, .baseline. A copy of the mask is stored in .mask.

        Failures are determined in one of two ways.
            1. If thresh=False, the method uses a dynamic threshold
            (baseline mean + 3*S.D.) for failures/successes.
            2. If thresh is defined, it specifies the baseline-subtracted
            amplitude threshold for failures/successes.

        Parameters
        ---------------
        thresh : bool, float or list
            Determines how the threshold for event failures and successes is
            calculated.

            - If thresh = False, the threshold for event failures is
            automatically specified as the mean +- 3*S.D. for each trace.
            - Otherwise, the threshold is determined by the user
            for all neurons, as a minimum amplitude from baseline.
            (if type(thresh) == float), or for each neuron individually
            (if type(thresh) == list)

        _invert_sorting : bool
            If True, keeps failures and masks successes. Can be useful
            in some cases.

        Attributes added to class instance
        ------------
        The .data and .inds attributes of .ampli, .latency, .baseline are
        np.ma.arrays, with the masked elements corresponding to failures.
        Also, the following attributes are added:

        .mask[neuron][trial, event]
            A boolean mask structrue in which True values denote failures
            and False values denote successes.
        .fail_rate[neuron][stim]
            An array of the fractional failure rate of each stimulus
            presentation in each neuron.
            - For example, if stimulus 3 delivered to neuron 1 evoked
            suprathreshold responses in 15/20 trials (where thresh defines
            threshold), self.fail_rate[1][3] = 0.75.
        """
        self.__check_for_events()
        self.__check_for_ampli()

        n_neurons = len(self.ampli.data)

        # initialize arrays
        self.mask = np.empty(n_neurons, dtype=np.ndarray)
        self.mask_params = SimpleNamespace(thresh=thresh,
                                           _invert_sorting=_invert_sorting)
        self.fail_rate = np.empty(n_neurons, dtype=np.ndarray)

        if thresh is False:
            dynamic_thresholding = True
        elif type(thresh) == list:
            dynamic_thresholding = False
        elif type(thresh) == float or type(thresh) == int:
            dynamic_thresholding = False
            thresh = np.ones(n_neurons) * thresh

        for neuron in range(n_neurons):
            # compare threshold with amplitudes to make mask
            if dynamic_thresholding is True:
                _thresh = 4 * self.baseline.std[neuron][:, :]
            else:
                _thresh = thresh[neuron] * np.ones_like(
                    self.ampli.data[neuron].shape[0])

            _diff = np.abs(self.ampli.data[neuron]) - _thresh

            if _invert_sorting is False:
                self.mask[neuron] = _diff < 0
            elif _invert_sorting is True:
                self.mask[neuron] = _diff > 0

            # failure rate calc
            n_stims = self.ampli.data[neuron].shape[1]
            self.fail_rate[neuron] = np.empty(n_stims)
            for stim in range(n_stims):
                self.fail_rate[neuron][stim] = np.sum(
                    self.mask[neuron], axis=0) \
                    / self.mask[neuron].shape[0]

            self._propagate_mask()

        print('Added .mask for success/failure event sorting')
        print('Updated all attributes with .mask')

        return

    def add_decay(self, t_prestim=0, t_poststim=10, plotting=False,
                  fn='monoexp'):
        """Fits each post-synaptic event with an exponential decay fuction
        and stores the fitted parameters in self.decay.

        Decay equation variables correspond to the fitted variables for
        the equation used (see the kwarg fn for more info).
        - monoexponential decay: lambda1, b.
        - biexponential decay: lambda1, lambda2, vstart2, b.

        Parameters
        ------------
        t_prestim : float
            Time before stimulus, in ms, to include in signal
            used to compute decay.

        t_poststim : float
            Time after stimulus, in ms, to include in signal
            used to compute decay.

        plotting : bool
            Whether to plot examples of decay fits (True) or not (False).

        fn : str
            Exponential decay function to use.
            - 'monoexp': y = e^(-t * lambda1) + b
            - 'biexp_normalized_plusb': y = e^(-t * lambda1)
                + vstart * e^(-t / lambda2) + b

            (In all cases, the more traditional decay tau can be computed
            as tau= 1/lambda).

        Attributes added
        ------------
        .decay : SimpleNamespace
            .decay.vars[neuron][trial, stim, decay_var]
                Fitted variables for monoexponential decay.
                - If fn='monoexp', then
                    decay_var=0 : lambda1
                    decay_var=1 : b
                - If fn='biexp_normalized_plusb', then
                    decay_var=0 : lambda1
                    decay_var=1 : lambda2
                    decay_var=2 : vstart2
                    decay_var=3 : b

            .decay.covari[neuron][trial, stim, decay_param]
                Covariance matrices for fitted variables,
                as documented in .decay.vars

            .decay.params
                Parameters related to the decay fitting.
        """
        self.__check_for_events()
        self.__check_for_ampli()

        n_neurons = len(self.ampli.data)

        self.decay = SimpleNamespace(vars=np.empty(n_neurons,
                                                   dtype=np.ndarray),
                                     covari=np.empty(n_neurons,
                                                     dtype=np.ndarray),
                                     params=SimpleNamespace())
        self.decay.params.fn = fn
        self.decay.params.t_prestim = t_prestim
        self.decay.params.t_poststim = t_poststim

        def biexp(time_x, lambda1, lambda2, vstart2, b):
            y = np.exp(time_x * (-1) * lambda1) \
                + vstart2 * np.exp(time_x * (-1) * lambda2) + b
            return y

        def monoexp(time_x, lambda1, b):
            y = np.exp(time_x * (-1) * lambda1) + b
            return y

        if fn == 'monoexp':
            n_vars = 2
            vars_guess = [100, 0]
        elif fn == 'biexp':
            n_vars = 4
            vars_guess = [100, 100, 1, 0]

        for neuron in range(n_neurons):
            n_trials = self.ampli.data[neuron].shape[0]
            n_stims = self.ampli.data[neuron].shape[1]

            ind_prestim = self._from_t_to_ind(t_prestim, neuron)
            ind_poststim = self._from_t_to_ind(t_poststim, neuron)

            if type(self.ampli.data[neuron]) is np.ma.core.MaskedArray:
                self.decay.vars[neuron] = np.ma.array(np.empty(
                    [n_trials, n_stims, n_vars], dtype=np.ndarray))
                self.decay.vars[neuron].mask = self.ampli.data[neuron].mask
                self.decay.covari[neuron] = np.ma.array(np.empty(
                    [n_trials, n_stims, n_vars], dtype=np.ndarray))
                self.decay.covari[neuron].mask = self.ampli.data[neuron].mask

            else:
                self.decay.vars[neuron] = np.empty(
                    [n_trials, n_stims, n_vars], dtype=np.ndarray)
                self.decay.covari[neuron] = np.empty(
                    [n_trials, n_stims, n_vars], dtype=np.ndarray)

            for trial in range(n_trials):
                for stim in range(n_stims):

                    if (type(self.ampli.data[neuron])
                            == np.ma.core.MaskedArray and
                            self.ampli.data[neuron][trial, stim] is not
                            np.ma.masked):

                        event_ind_min = self.ampli.inds[neuron][
                            trial, stim] - ind_prestim
                        event_ind_max = event_ind_min + ind_poststim

                        postsynaptic_curve = self.sig[neuron][
                            trial, event_ind_min:event_ind_max] \
                            - self.baseline.mean[neuron][trial, stim]
                        postsynaptic_curve /= np.mean(postsynaptic_curve[0:2])

                        times_forfit = self.t[neuron][0:ind_poststim]

                        try:
                            [popt, pcov] = sp_opt.curve_fit(
                                monoexp, times_forfit,
                                postsynaptic_curve, p0=vars_guess)

                        except RuntimeError:
                            popt = np.ones(n_vars) * 10000
                            pcov = 10000

                        except ValueError:
                            print(postsynaptic_curve, 'neuron: ', neuron,
                                  'trial: ', trial, 'stim: ', stim)

                        self.decay.vars[neuron][trial, stim, :] = popt[:]

                    elif (type(self.ampli.data[neuron])
                          == np.ma.core.MaskedArray and
                          self.ampli.data[neuron][trial, stim] is
                          np.ma.masked):

                        self.decay.vars[neuron][trial, stim, :] = np.ones(
                            n_vars) * 10000
                        self.decay.vars[neuron][trial, stim, :].mask = np.ones(
                            n_vars, dtype=np.bool)

                    elif (type(self.ampli.data[neuron]) is not
                          np.ma.core.MaskedArray):
                        event_ind_min = self.ampli.inds[neuron][
                            trial, stim] - ind_prestim
                        event_ind_max = event_ind_min + ind_poststim

                        postsynaptic_curve = self.sig[neuron][
                            trial, event_ind_min:event_ind_max] \
                            - self.baseline.mean[neuron][trial, stim]
                        postsynaptic_curve /= postsynaptic_curve[0]

                        times_forfit = self.t[neuron][0:ind_poststim]

                        try:
                            [popt, pcov] = sp_opt.curve_fit(
                                monoexp, times_forfit,
                                postsynaptic_curve, p0=vars_guess)

                        except RuntimeError:
                            popt = np.ones(n_vars) * 10000

                        self.decay.vars[neuron][trial, stim, :] = popt[:]

            if plotting is True:
                if type(self.ampli.data[neuron]) == np.ma.core.MaskedArray:
                    first_nonmasked_trial = (np.argwhere(
                        self.ampli.inds[neuron][:, 0].mask is False)
                                             [0][0])

                    postsynaptic_curve = (self.sig[neuron][
                        first_nonmasked_trial,
                        self.ampli.inds[neuron]
                        [first_nonmasked_trial, 0] - ind_prestim:
                        self.ampli.inds[neuron]
                        [first_nonmasked_trial, 0] + ind_poststim]
                        - self.baseline.mean[neuron][first_nonmasked_trial, 0])

                    y_fitted = (self.ampli.data[neuron]
                                [first_nonmasked_trial, 0]
                                * monoexp(
                                    self.t[neuron][0:ind_poststim
                                                   + ind_prestim],
                                    self.decay.vars[neuron][
                                        first_nonmasked_trial, 0, 0],
                                    self.decay.vars[neuron][
                                        first_nonmasked_trial, 0, 1]))

                    plt.figure()
                    plt.plot(self.t[neuron][0:ind_poststim + ind_prestim],
                             y_fitted, 'r')
                    plt.plot(self.t[neuron][0:ind_poststim + ind_prestim],
                             postsynaptic_curve, 'b')

                elif type(self.ampli.data[neuron]) is np.ndarray:
                    postsynaptic_curve = (self.sig[neuron]
                                          [0, self.ampli.inds[neuron]
                                           [0, 0] - ind_prestim:
                                           self.ampli.inds[neuron][0, 0]
                                           + ind_poststim]
                                          - self.baseline.mean[neuron][0, 0])
                    y_fitted = (self.ampli.data[neuron][0, 0]
                                * monoexp(
                                    self.t[neuron][0:ind_poststim
                                                   + ind_prestim],
                                    self.decay.vars[neuron][0, 0, 0],
                                    self.decay.vars[neuron][0, 0, 1]))
                    plt.figure()
                    plt.plot(self.t[neuron][0:ind_poststim + ind_prestim],
                             y_fitted, 'r')
                    plt.plot(self.t[neuron][0:ind_poststim + ind_prestim],
                             postsynaptic_curve, 'b')

            # convert from lambda to tau
            fitted_ones = np.ones([self.decay.vars[neuron][:, :, 0].shape[0],
                                   self.decay.vars[neuron][:, :, 0].shape[1]])

            if type(self.ampli.data[neuron]) is np.ma.core.MaskedArray:
                self.decay.vars[neuron][:, :, 0] = fitted_ones / np.array(
                    self.decay.vars[neuron][:, :, 0])
                self.decay.vars[neuron][:, :, 0] = _get_median_filtered(
                    self.decay.vars[neuron][:, :, 0], threshold=10)
                fittedvarmask = np.ma.mask_or(
                    self.ampli.data[neuron][:, :].mask,
                    self.decay.vars[neuron][:, :, 0].mask)
                self.decay.vars[neuron][:, :, 0].mask = fittedvarmask

            else:
                self.decay.vars[neuron][:, :, 0] = (
                    fitted_ones / self.decay.vars[neuron][:, :, 0])
                self.decay.vars[neuron][:, :, 0] = _get_median_filtered(
                    self.decay.vars[neuron][:, :, 0], threshold=10)

        plt.show()
        print('Added .decay')

        return

    def add_integral(self, t_integral=200, cdf_bins=100):
        """
        Compute the integral for each post-synaptic event.

        Parameters
        -----------------
        t_integral : float
            The total post-stimulus time to integrate, in milliseconds.

        cdf_bins : int
            Number of bins for the cumulative integral

        Attributes added
        ------------
        .integral : SimpleNamespace
            .integral.data[neuron][trial, event]
                Integral values for each event (in pA*sec or mV*sec).
            .integral.inds_start[neuron][trial, event]
                Indices of the start of integral period in .sig
            .integral.inds_stop[neuron][trial, event]
                Indices of the end of integral period in .sig
            .integral.cdf[neuron][trial, event]
                Cumulative distribution of the integral over time
                for each event.
            .integral.params
                SimpleNamespace storing key params from .add_integral() method
                related to integral.
        """
        self.__check_for_events()
        self.__check_for_ampli()

        n_neurons = len(self.sig)

        self.integral = SimpleNamespace(
            data=np.empty(n_neurons, dtype=np.ndarray),
            inds_start=np.empty(n_neurons, dtype=np.ndarray),
            inds_stop=np.empty(n_neurons, dtype=np.ndarray),
            cdf=np.empty(n_neurons, dtype=np.ndarray),
            params=SimpleNamespace(t_integral=t_integral,
                                   cdf_bins=cdf_bins))

        for neuron in range(n_neurons):
            n_trials = self.ampli.data[neuron].shape[0]
            n_stims = self.ampli.data[neuron].shape[1]
            self.integral.data[neuron] = np.zeros([n_trials, n_stims])
            self.integral.cdf[neuron] = np.zeros(
                [n_trials, n_stims, int(cdf_bins)])
            self.integral.inds_start[neuron] = np.zeros(
                [n_trials, n_stims], dtype=int)
            self.integral.inds_stop[neuron] = np.zeros(
                [n_trials, n_stims], dtype=int)

            for trial in range(n_trials):
                for stim in range(n_stims):
                    # calc integral
                    ind_start = int(self.events[neuron][stim])
                    ind_stop = self._from_t_to_ind(t_integral, neuron) \
                        + ind_start

                    self.integral.inds_start[neuron][trial, stim] = ind_start
                    self.integral.inds_stop[neuron][trial, stim] = ind_stop

                    _sig_event = (
                        self.sig[neuron][trial, ind_start:ind_stop]
                        - self.baseline.mean[neuron][trial, stim])
                    _t_event = self.t[neuron][ind_start:ind_stop]

                    self.integral.data[neuron][trial, stim] \
                        = sp_integrate.trapz(_sig_event, _t_event)

                    # calc cdf
                    for nbin in range(cdf_bins):
                        _curr_cdf_frac = (nbin+1) / cdf_bins

                        ind_stop = int((self._from_t_to_ind(t_integral, neuron)
                                       + ind_start) * _curr_cdf_frac)

                        _sig_event = (self.sig[neuron][
                            trial, ind_start:ind_stop]
                                    - self.baseline.mean[neuron][trial, stim])
                        _t_event = self.t[neuron][ind_start:ind_stop]

                        self.integral.cdf[neuron][trial, stim, nbin] = (
                            sp_integrate.trapz(_sig_event, _t_event)
                            / self.integral.data[neuron][trial, stim])

        self._all_stats.append(self.integral.data)
        self._all_stats.append(self.integral.inds_start)
        self._all_stats.append(self.integral.inds_stop)

        print('Added .integral')

        return

    def add_all(self, kwargs_add_ampli={'event_sign': 'pos'},
                kwargs_add_integral={},
                kwargs_add_decay={},
                kwargs_mask_unclamped_aps=False,
                kwargs_add_failure_sorting=False):
        """
        Convenience method which takes an initialized EphysObject with
        a .events attribute (stimulus onsets), and quantifies
        several postsynaptic event attributes by calling the following
        class methods:

            - amplitude (.ampli, .norm_ampli, .baseline,
            .latency) by calling the .add_ampli() method.
            - integral (.norm_integral, .norm_cdf_integral)
            by calling the .add_integral() method
            - decays (.decay) by calling the .add_decay() method.

        The following methods are optionally called by setting the
        associated kwargs to a dictionary, instead of the default of False:
            - masking unclamped action potentials through the
            .mask_unclamped_aps() method.
            - successes and failure sorting (.mask, used to create an
            array maksing failures within .ampli, .baseline, .latency) through
            the .add_failure_sorting() method

        Parameters
        -----------------
        kwargs_add_ampli : dict
            Dictionary of keyword arguments to be passed to the .add_ampli()
            method. The docstring for .add_ampli() contains more details.
            - event_sign can be 'pos' or 'neg' depending on the polarity of
            responses under investigation.

        kwargs_add_integral : dict
            Dictionary of keyword arguments to be passed to the .add_integral()
            method. The docstring for .add_integral() contains more details.

        kwargs_add_decay : dict
            Dictionary of keyword arguments to be passed to the .add_decay()
            method. The docstring for .add_decay() contains more details.

        kwargs_mask_unclamped_aps : bool or dict
            If False, does not mask unclamped action potentials with the
            .mask_unclamped_aps() method.
            If a dict, masks unclamped aps using the dict as kwargs to
            .mask_unclamped_aps().

        kwargs_add_failure_sorting : bool or dict
            If False, does not add success/failure sorting of events.
            If a dict, adds suc/fail sorting to events, using the
            dict as kwargs to .add_failure_sorting().

        Attributes added
        ------------
        See .add_ampli(), .add_integral(), .add_decay(),
        .mask_unclamped_aps() and .add_failure_sorting() for more info on the
        particular attributes added.
        """

        self.add_ampli(**kwargs_add_ampli)
        self.add_integral(**kwargs_add_integral)
        self.add_decay(**kwargs_add_decay)

        if type(kwargs_mask_unclamped_aps) is dict:
            self.mask_unclamped_aps(**kwargs_mask_unclamped_aps)

        if type(kwargs_add_failure_sorting) is dict:
            self.add_failure_sorting(**kwargs_add_failure_sorting)

        return

    def preview(self, neur, attrs=None, **kwargs):
        """Plots the analog signals from a given neuron, and a list of attributes.

        Parameters
        ------------------
        neur : ind
            The index of the neuron to preview.

        attr : None or list
            If not None, a list of attribute strings within the EphysObj class
            instance to preview. These strings can be:
                'ampli'
                'latency'
                'baseline'
        """

        p = PreviewEphysObject(self, neur, attrs, **kwargs)
        p.plot()

        return

    def get_stats(self, arr_name, pooling_index=0, mask='suc'):
        """Computes statistics over all neurons for a given postsynaptic event
        attribute (eg ampli, latency).

        Parameters
        ------------------
        arr_name : str
            Name of the event attribute to get stats for. Can be 'ampli',
            'latency', 'decay', etc.
        pooling_index : int
            Index of event attribute to pool over. Typically 0, but
            provided as an option since several attributes have multiple
            dimensions that denote different parameters.
        mask : str
            Passed to self.pool_stat_across_neurs(), can be 'suc' or 'all'
            to denote whether to pool stats across only successful events,
            or all events.

        Returns
        -----------------
        stats : np.ndarray
            An array of statistics of the form [n_stim, 5]. The second
            dimension is as follows: 0:mean, 1:stdev, 2:sem, 3:suc_rate_mean,
            4:suc_rate_stdev
        """
        postsynaptic_event = self.__getattribute__(arr_name)

        stimpooled_postsynaptic_events = self.pool_stat_across_neurs(
            postsynaptic_event, pooling_index, mask=mask)
        if type(postsynaptic_event[0]) is np.ma.core.MaskedArray:
            success_rate = self.special_sucrate(self)
            num_stims = len(stimpooled_postsynaptic_events)
            stats = np.zeros([num_stims, 5])

            for stim in range(num_stims):
                num_nonmasked_stims = len(
                    stimpooled_postsynaptic_events[stim, :]) \
                    - np.ma.count_masked(
                        stimpooled_postsynaptic_events[stim, :])

                stats[stim, 0] = np.mean(
                    stimpooled_postsynaptic_events[stim, :])
                stats[stim, 1] = np.std(
                    stimpooled_postsynaptic_events[stim, :])
                stats[stim, 2] = np.std(
                    stimpooled_postsynaptic_events[stim, :]) \
                    / np.sqrt(num_nonmasked_stims)
                stats[stim, 3] = success_rate[stim, 0]
                stats[stim, 4] = success_rate[stim, 1]

        else:
            num_stims = len(stimpooled_postsynaptic_events)
            stats = np.zeros([num_stims, 3])

            for stim in range(num_stims):
                num_nonmasked_stims = len(
                    stimpooled_postsynaptic_events[stim, :]) \
                    - np.ma.count_masked(
                        stimpooled_postsynaptic_events[stim, :])

                stats[stim, 0] = np.mean(
                    stimpooled_postsynaptic_events[stim, :])
                stats[stim, 1] = np.std(
                    stimpooled_postsynaptic_events[stim, :])
                stats[stim, 2] = np.std(
                    stimpooled_postsynaptic_events[stim, :]) \
                    / np.sqrt(num_nonmasked_stims)

        return (stats)

    def plot_attribute(self, arr_name,
                       ylim=False, by_neuron=False,
                       ind=0, hist=False):
        """
        Plots a particular postsynaptic event attribute, like event
        amplitude or latency, across all neurons or stimuli.

        Parameters
        -----------------
        arr_name : str
            The name of the attribute. Can be 'ampli', 'ampli_norm',
            'baseline', 'decay', 'latency'.

        ylim : bool or list
            If not False, this two-element list specifies the y-limits
            (lower and upper) of the plot.)

        by_neuron : bool
            If True, each neuron's data is plotted separately.
            If False, the per-neuron data is combined.

        ind : int
            The index of the array to plot.

        hist : bool
            Whether to plot a histogram (True) or a dot plot (False)
        """

        arr = self.__getattribute__(arr_name)
        num_neurons = len(arr)

        if 'ampli' == arr_name:
            yax = 'Event amplitude (pA)'
        elif 'ampli_norm' == arr_name:
            yax = 'Normalized event amplitude'
        elif 'baseline' == arr_name:
            yax = 'Baseline holding current (pA)'
        elif 'decay' == arr_name:
            yax = 'tau (s)'
            hist = True
        elif 'latency' == arr_name:
            yax = 'Latency (s)'

        if hist is False:
            for neuron in range(num_neurons):
                print('\nNeuron: ', neuron)
                x_1 = range(1, len(arr[neuron][0, :, 0]) + 1)
                num_trials = len(arr[neuron][:, 0, 0])

                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                for trial in range(num_trials):
                    ratio_thistrial = trial / (num_trials)
                    red_thistrial = 1 / (1 + np.exp(
                        -5 * (ratio_thistrial - 0.5)))
                    color_thistrial = [red_thistrial, 0.2, 1 - red_thistrial]
                    if type(arr[neuron]) is np.ma.core.MaskedArray:
                        ax.plot(x_1,
                                arr[neuron][trial, :, ind].filled(np.nan),
                                '.', color=color_thistrial, alpha=0.6)
                        ma = arr[neuron][trial, :, ind].mask
                        inv_ma = np.logical_not(ma)
                        new_pse = np.ma.array(np.array(
                            arr[neuron][trial, :, ind]), mask=inv_ma)
                        ax.plot(x_1,
                                new_pse.filled(np.nan),
                                '.', color='0.7', alpha=0.6)
                    else:
                        ax.plot(x_1,
                                arr[neuron][trial, :, ind],
                                '.', color=color_thistrial, alpha=0.6)

                ax.set_xlabel('Stimulation number')
                ax.set_ylabel(yax)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticks_position('bottom')
                xlim_curr = ax.get_xlim()
                ylim_curr = ax.get_ylim()
                ax.set_xlim([xlim_curr[0], xlim_curr[1] + 1])
                if arr_name == 'latency' or arr_name == 'ampli_norm':
                    ax.set_ylim([0, ylim_curr[1]])

                if ylim is not False:
                    ax.set_ylim(ylim)

                plt.show()

        elif hist is True:
            for neuron in range(num_neurons):
                print('\nNeuron: ', neuron)
                num_trials = len(arr[neuron][:, 0, 0])
                array_to_plot = arr[neuron][:, :, ind]

                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                if type(arr[neuron]) is np.ma.core.MaskedArray:
                    histpool_thisneuron = array_to_plot.compressed()
                    ax.hist(histpool_thisneuron,
                            bins=30,
                            facecolor=[0.2, 0.4, 0.8],
                            normed=True,
                            alpha=0.6,
                            linewidth=0.5)

                else:
                    histpool_thisneuron = array_to_plot.flatten()
                    ax.hist(histpool_thisneuron,
                            bins=30,
                            facecolor=[0.2, 0.4, 0.8],
                            normed=True,
                            alpha=0.6,
                            linewidth=0.5)

                ax.set_xlabel('Decay (tau) (s)')
                ax.set_ylabel('Number')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticks_position('bottom')
                xlim_curr = ax.get_xlim()

                plt.show()

    def filter_sig_savgol(self, t_savgol_filt=1, polyorder=3):
        """Applies a specified filter to the raw signal for all
        neurons. (Reccommended to call this method before adding
        ampli, etc.)

        Parameters
        ---------------
        t_savgol_filt : float
            Window length of savgol filter, in seconds.

        polyorder : int
            Polynomial order for savgol filter.

        Attributes added
        -------------
        self.sig : filtered signal
        """
        print(f'Filtering signal...')

        n_neurs = len(self.sig)
        for neur in range(n_neurs):
            print(f'\tNeuron {neur}')
            ind_savgol_filt = self._from_t_to_ind(t_savgol_filt,
                                                  neur,
                                                  t_format='s') + 1
            n_trials = self.sig[neur].shape[0]

            for trial in range(n_trials):
                _sig = self.sig[neur][trial, :]
                _sig_filt = sp_signal.savgol_filter(_sig,
                                                    ind_savgol_filt,
                                                    polyorder=polyorder)
                self.sig[neur][trial, :] = _sig_filt
            
        return

    def filter_sig_butterworth(self, butter_n, butter_wn):
        """Applies a specified filter to the raw signal for all
        neurons. (Reccommended to call this method before adding
        ampli, etc.)

        Parameters
        ---------------
        butter_n : int
            Order of the butterworth filter
        butter_wn : float
            Critical frequency, expressed as fraction of the Nyquist frequency.

        Attributes added
        -------------
        self.sig : filtered signal
        self._sig_raw : original unfiltered signal for reference
        """
        self._sig_raw = self.sig

        print('Filtering signal...')

        n_neurs = len(self.sig)
        for neur in range(n_neurs):
            print(f'\tNeuron {neur}')
            n_trials = self.sig[neur].shape[0]
            for trial in range(n_trials):
                sos = sp_signal.butter(butter_n, butter_wn, output='sos')
                _sig = self.sig[neur][trial, :]
                _sig_filt = sp_signal.sosfiltfilt(sos, _sig)
                self.sig[neur][trial, :] = _sig_filt

        print(f'added .sig (Butterworth filter applied: order {butter_n} and '
              f'critical frequency {butter_wn} of Nyquist)')

        return

    def mask_unclamped_aps(self, thresh=5):
        """Removes unclamped action potentials from a voltage-clamp recording
        by masking the appropriate entries in the self.ampli, self.latency and
        self.ampli_norm attributes.

        Parameters
        ---------------
        thresh : float
            Amplitude threshold for removing action potentials, expressed as
            fold difference from the normalized (mean) amplitude for that
            cell/stim.
            - For example, if thresh=5 and the mean event amplitude was 50pA,
            any events larger than 250pA would be considered unclamped aps and
            masked.
        """
        num_neurons = len(self.ampli_norm.data)

        for neuron in range(num_neurons):
            to_replace = np.argwhere(
                self.ampli_norm.data[neuron][:, :, 0] > thresh)

            self.ampli.data[neuron][to_replace[:, 0],
                                    to_replace[:, 1], :] = np.nan
            self.ampli.data[neuron].mask[to_replace[:, 0],
                                         to_replace[:, 1], :] = True

            self.latency.data[neuron][to_replace[:, 0],
                                      to_replace[:, 1], :] = np.nan
            self.latency.data[neuron].mask[to_replace[:, 0],
                                           to_replace[:, 1], :] = True

            try:
                self.ampli_norm.data[neuron][to_replace[:, 0],
                                             to_replace[:, 1], :] = np.nan
                self.ampli_norm.data[neuron].mask[to_replace[:, 0],
                                                  to_replace[:, 1], :] = True
            except:
                pass

            self.mask[neuron] = self.ampli.data[neuron].mask

        self._propagate_mask()

        print('Masked APs')
        return

    def _from_t_to_ind(self, t, neur, t_format='ms'):
        """Converts a time, in ms, to an index, using a
        known sampling rate.

        Parameters
        ------------
        t : float
            Time in either seconds or milliseconds.

        neur : ind
            Index of current neuron. Used to fetch sampling rate
            (self._sampling_rate[neur]).

        t_format : str
            Whether time is in milliseconds 'ms' or seconds 's'

        Returns
        -----------
        """
        if t_format == 'ms':
            t /= 1000

        ind = int(t * self._sampling_rate[neur])

        return ind

    def _get_median_filtered(signal, threshold=3):
        """Internal function passed to .add_decay()
        """
        if type(signal) is not np.ma.core.MaskedArray:
            signal = signal.copy()
            difference = np.abs(signal - np.median(signal))
            median_difference = np.median(difference)
            if median_difference == 0:
                s = 0
            else:
                s = difference / float(median_difference)

            mask = s > threshold
            mask_2 = signal < 0
            signal[mask] = 0
            signal[mask_2] = 0

        else:
            original_mask = signal.mask

            signal = np.array(signal.copy())
            difference = np.abs(signal - np.median(signal))
            median_difference = np.median(difference)
            if median_difference == 0:
                s = 0
            else:
                s = difference / float(median_difference)
            mask = s > threshold
            signal[mask] = np.median(signal)

            mask = s > threshold
            mask_2 = signal < 0
            signal[mask] = 0
            signal[mask_2] = 0

            combined_mask_1 = np.ma.mask_or(mask, mask_2)
            combined_mask_2 = np.ma.mask_or(combined_mask_1, original_mask)

            signal = np.ma.array(signal, mask=combined_mask_2)

        return signal

    def _propagate_mask(self):
        """Internal function which takes an updated mask of successes/failures
        stored as self.mask, and uses it to update .ampli, .ampli_norm,
        .ampli_fails, .latency, .baseline, .decay.

        (everything stored in ._all_stats is updated).

        Called by self.remove_unclamped_aps to propagate a changed mask
        across all class instance attributes.
        """
        n_neurons = self.ampli.data.shape[0]

        for stat in self._all_stats:
            for neuron in range(n_neurons):
                # check that this is a masked array
                if type(stat[neuron]) == np.ndarray:
                    stat[neuron] = np.ma.array(stat[neuron])

                _updated_mask = np.logical_or(stat[neuron].mask,
                                              self.mask[neuron])

                # update stat
                stat[neuron].mask = _updated_mask

        return

    def __check_for_events(self):
        """Checks whether the .events attribute exists, and if it does not,
        returns an Exception.

        Called before most class methods.
        """
        if hasattr(self, 'events'):
            pass
        else:
            raise Exception('No .events attribute exists. It must be first ' +
                            'added using the class method .add_events().')

        return

    def __check_for_ampli(self):
        """Checks whether the .ampli attribute exists, and if it does not,
        returns an Exception.

        Called before some class methods that require .ampli.
        """
        if hasattr(self, 'ampli'):
            pass
        else:
            raise Exception('No .ampli attribute exists. It must be first ' +
                            'added using the class method .add_ampli().')

        return


class PreviewEphysObject(object):
    def __init__(self, ephysobj, neur,
                 attrs=None,
                 plt_sig_raw=False,
                 figsize=(10, 7),
                 _color_reg=[0.65, 0.65, 0.65],
                 _color_bold=[0.0118, 0.443, 0.612],
                 _color_mean=[0.980, 0.259, 0.141],
                 _color_ampli_bl=[0.98, 0.26, 0.14],
                 _color_lat=[0.99, 0.78, 0.08],
                 _color_integ=[0.145, 0.639, 0.436],
                 _color_failure=[0.3, 0.3, 0.3]):
        """Simple class that takes an EphysObject and neuron index as input
        and plots signals and a given set of event attributes
        (eg amplitude, baseline, etc.)

        Parameters
        ------------
        ephysobj : EphysObject
            The EphysObject instance to plot.

        neur : ind
            Index of neuron within EphysObject to plot.

        attrs : None or list
            If None, no data attribute is plotted.
            Otherwise, attrs should be a list of strings
            corresponding to attributes of the EphysObject instance:
                'ampli'
                'latency'
                'baseline'
                'decay'
                'integral'

        plt_sig_raw : bool
            Whether to plot the original signal or not. Only applies if a
            filtered signal has been calculated using .filter_sig_* to replace
            .sig. This argument then plots ._sig_raw as well as the
            filtered .sig.
        """
        # Load data and initialize variables
        self._color_reg = _color_reg
        self._color_bold = _color_bold
        self._color_mean = _color_mean

        self._color_ampli_bl = _color_ampli_bl
        self._color_lat = _color_lat
        self._color_integ = _color_integ
        self._color_failure = _color_failure

        self.neur = neur
        self.figsize = figsize
        self.ephysobj = ephysobj
        self._dtype = type(self.ephysobj.ampli.data[self.neur])

        self.attrs = attrs
        self.n_attrs = len(attrs)
        self.attr_data = np.empty(self.n_attrs, dtype=np.ndarray)
        self.attr_inds = np.empty(self.n_attrs, dtype=np.ndarray)
        self.attr_lineobjs = np.empty(self.n_attrs, dtype=np.ndarray)

        self.n_stims = self.ephysobj.events[self.neur][0].shape[0]
        for ind_attr, attr in enumerate(self.attrs):
            self.attr_lineobjs[ind_attr] = np.empty(self.n_attrs,
                                                    dtype=np.ndarray)
            for stim in range(self.n_stims):
                self.attr_lineobjs[ind_attr] = np.empty(
                    self.n_stims, dtype=np.ndarray)

        plt.ion()
        mpl.style.use('fast')
        mpl.rcParams["path.simplify_threshold"] = 1.0
        mpl.rcParams["axes.spines.top"] = False
        mpl.rcParams["axes.spines.right"] = False

        self.n_trials = ephysobj.sig[neur].shape[0]

    def plot(self):
        # Setup figure, gridspec and axs
        self.fig = plt.figure(figsize=self.figsize,
                              constrained_layout=True)
        height_ratios = [4, 1, 0.2]

        spec = gs.GridSpec(nrows=3, ncols=15,
                           figure=self.fig,
                           height_ratios=height_ratios)

        self.ax = []
        self.ax.append(self.fig.add_subplot(spec[0, :]))
        self.ax.append(self.fig.add_subplot(spec[1, :],
                                            sharex=self.ax[0]))

        # setup lines and temporary variables
        self.lines = np.empty(2, dtype=np.ndarray)
        self.lines_mean = np.empty(2, dtype=np.ndarray)
        self.mean = np.empty(2, dtype=np.ndarray)

        # .sig (main analog signal)
        self.lines[0] = np.empty(self.n_trials, dtype=object)
        self.ax[0].set_ylabel(f'{self.ephysobj.sig_units}')

        for trial in range(self.n_trials):
            self.lines[0][trial] = self.ax[0].plot(
                self.ephysobj.t[self.neur],
                self.ephysobj.sig[self.neur][trial, :],
                color=self._color_reg,
                linewidth=0.5)

            self.mean[0] = np.mean(self.ephysobj.sig[self.neur],
                                   axis=0)

        # .sig_stim (signal for stimulus channel)
        self.lines[1] = np.empty(self.n_trials, dtype=object)
        self.ax[1].set_ylabel('stim')

        for trial in range(self.n_trials):
            self.lines[1][trial] = self.ax[1].plot(
                self.ephysobj.t[self.neur],
                self.ephysobj.sig_stim[self.neur],
                color=self._color_reg,
                linewidth=0.5)

            self.mean[1] = self.ephysobj.sig_stim[self.neur]

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
        self.add_attrs_for_trial()

        # add text
        self.text = self.fig.text(0.05, 0.03,
                                  s=f'neur: {self.neur} | ' +
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
        self.remove_attrs_for_trial()

        self.update_curr_trial_next()

        self.bold_trial(self._curr_trial)
        self.add_attrs_for_trial()

        self.update_trialtext()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def on_prev(self, event):
        self.unbold_trial(self._curr_trial)
        self.remove_attrs_for_trial()

        self.update_curr_trial_prev()

        self.bold_trial(self._curr_trial)
        self.add_attrs_for_trial()

        self.update_trialtext()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def on_mean(self, event):
        if self._mean_plotted is False:
            # main signal
            self.lines_mean[0] = self.ax[0].plot(
                self.ephysobj.t[self.neur],
                self.mean[0],
                color=self._color_mean,
                linewidth=1.5)
            # stim signal
            self.lines_mean[1] = self.ax[1].plot(
                self.ephysobj.t[self.neur],
                self.mean[1],
                color=self._color_mean,
                linewidth=1.5)
            self._mean_plotted = True

        elif self._mean_plotted is True:
            for sig in range(2):
                self.lines_mean[sig][0].remove()
            self._mean_plotted = False

        self.fig.canvas.draw_idle()

    def bold_trial(self, trial):
        # main signal
        self.lines[0][trial][0].remove()
        self.lines[0][trial] = self.ax[0].plot(
            self.ephysobj.t[self.neur],
            self.ephysobj.sig[self.neur][trial, :],
            color=self._color_bold,
            linewidth=0.8)

    def unbold_trial(self, trial):
        # main signal
        self.lines[0][trial][0].remove()
        self.lines[0][trial] = self.ax[0].plot(
            self.ephysobj.t[self.neur],
            self.ephysobj.sig[self.neur][trial, :],
            color=self._color_reg,
            linewidth=0.5)

    def add_attrs_for_trial(self):
        for ind, attr in enumerate(self.attrs):
            if attr == 'ampli':
                self._plot_ampli(ind)
            elif attr == 'baseline':
                self._plot_baseline(ind)
            elif attr == 'latency':
                self._plot_latency(ind)
            elif attr == 'decay':
                self._plot_decay(ind)
            elif attr == 'integral':
                self._plot_integral(ind)

    def remove_attrs_for_trial(self):
        for ind, attr in enumerate(self.attrs):
            self._unplot(ind)

    def update_trialtext(self):
        self.text.set_text(f'neur: {self.neur} | ' +
                           f'trial: {self._curr_trial}')

    def _plot_ampli(self, ind_attr):
        for stim in range(self.n_stims):
            _curr_element = self.ephysobj.ampli.data[self.neur][
                self._curr_trial, stim]

            if self._dtype == np.ndarray:
                _ind_ampli = self.ephysobj.ampli.inds[self.neur][
                    self._curr_trial, stim]
                _color = self._color_ampli_bl
            elif self._dtype == np.ma.MaskedArray:
                _ind_ampli = self.ephysobj.ampli.inds[self.neur].data[
                    self._curr_trial, stim]
                if np.ma.is_masked(_curr_element) is True:
                    _color = self._color_failure
                elif np.ma.is_masked(_curr_element) is False:
                    _color = self._color_ampli_bl

            _t_ampli = self.ephysobj.t[self.neur][
                _ind_ampli]
            _sig_ampli = self.ephysobj.sig[self.neur][
                self._curr_trial, _ind_ampli]

            self.attr_lineobjs[ind_attr][stim] = self.ax[0].plot(
                _t_ampli, _sig_ampli,
                '.', color=_color, markersize=10,
                zorder=150)

    def _plot_baseline(self, ind_attr):
        for stim in range(self.n_stims):
            _curr_element = self.ephysobj.baseline.mean[self.neur][
                self._curr_trial, stim]

            if self._dtype == np.ndarray:
                _color = self._color_ampli_bl
                _ind_bl_start = self.ephysobj.baseline.inds_start[self.neur][
                    self._curr_trial, stim]
                _ind_bl_stop = self.ephysobj.baseline.inds_stop[self.neur][
                    self._curr_trial, stim]
                _mean_bl = self.ephysobj.baseline.mean[self.neur][
                    self._curr_trial, stim]

            elif self._dtype == np.ma.MaskedArray:
                _ind_bl_start = self.ephysobj.baseline.inds_start[
                    self.neur].data[self._curr_trial, stim]
                _ind_bl_stop = self.ephysobj.baseline.inds_stop[
                    self.neur].data[self._curr_trial, stim]
                _mean_bl = self.ephysobj.baseline.mean[self.neur].data[
                    self._curr_trial, stim]

                if np.ma.is_masked(_curr_element) is True:
                    _color = self._color_failure
                elif np.ma.is_masked(_curr_element) is False:
                    _color = self._color_ampli_bl

            _t_start = self.ephysobj.t[self.neur][
                _ind_bl_start]
            _t_stop = self.ephysobj.t[self.neur][
                _ind_bl_stop]


            self.attr_lineobjs[ind_attr][stim] = self.ax[0].plot(
                [_t_start, _t_stop], [_mean_bl, _mean_bl],
                color=_color, linewidth=2,
                zorder=150)

    def _plot_latency(self, ind_attr):
        for stim in range(self.n_stims):
            _curr_element = self.ephysobj.latency.data[self.neur][
                self._curr_trial, stim]

            if self._dtype == np.ndarray:
                _ind_lat = self.ephysobj.latency.inds[self.neur][
                    self._curr_trial, stim]
                _color = self._color_lat

            elif self._dtype == np.ma.MaskedArray:
                _ind_lat = self.ephysobj.latency.inds[self.neur].data[
                    self._curr_trial, stim]
                if np.ma.is_masked(_curr_element) is True:
                    _color = self._color_failure
                elif np.ma.is_masked(_curr_element) is False:
                    _color = self._color_lat

            _t_ampli = self.ephysobj.t[self.neur][
                _ind_lat]
            _sig_ampli = self.ephysobj.sig[self.neur][
                self._curr_trial, _ind_lat]

            self.attr_lineobjs[ind_attr][stim] = self.ax[0].plot(
                _t_ampli, _sig_ampli,
                '.', color=_color, markersize=6,
                zorder=150)

    def _plot_integral(self, ind_attr):
        for stim in range(self.n_stims):
            _curr_element = self.ephysobj.integral.data[self.neur][
                self._curr_trial, stim]

            if self._dtype == np.ndarray:
                _ind_integ_start = self.ephysobj.integral.inds_start[
                    self.neur][self._curr_trial, stim]
                _ind_integ_stop = self.ephysobj.integral.inds_stop[
                    self.neur][self._curr_trial, stim]
                _mean_bl = self.ephysobj.baseline.mean[self.neur][
                    self._curr_trial, stim]
                _color = self._color_integ

            elif self._dtype == np.ma.MaskedArray:
                _ind_integ_start = self.ephysobj.integral.inds_start[
                    self.neur].data[self._curr_trial, stim]
                _ind_integ_stop = self.ephysobj.integral.inds_stop[
                    self.neur].data[self._curr_trial, stim]
                _mean_bl = self.ephysobj.baseline.mean[self.neur].data[
                    self._curr_trial, stim]

                if np.ma.is_masked(_curr_element) is True:
                    _color = self._color_failure
                elif np.ma.is_masked(_curr_element) is False:
                    _color = self._color_integ

            _t_vec = self.ephysobj.t[self.neur][
                _ind_integ_start:_ind_integ_stop]

            _inds_in_bl_line = _ind_integ_stop - _ind_integ_start
            _mean_bl_line = np.ones(_inds_in_bl_line) * _mean_bl

            _sig_line = self.ephysobj.sig[self.neur][
                self._curr_trial, _ind_integ_start:_ind_integ_stop]

            self.attr_lineobjs[ind_attr][stim] = self.ax[0].fill_between(
                _t_vec, _mean_bl_line, _sig_line,
                alpha=0.5, color=_color, interpolate=True,
                zorder=100)

    def _unplot(self, ind_attr):
        for stim in range(self.n_stims):
            _line_obj = self.attr_lineobjs[ind_attr][stim]
            if 'matplotlib' in str(type(_line_obj)):
                _line_obj.remove()
            elif type(_line_obj) == list:
                _line_obj[0].remove()
        return
