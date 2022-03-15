# imports
from .core_fns import find_spontaneous, find_stims
from .helper_fns import find_last, _get_median_filtered
import numpy as np
import scipy.signal as sp_signal
import scipy.optimize as sp_opt
import scipy.integrate as sp_integrate
import matplotlib.pyplot as plt


class AttributeArray(np.ndarray):
    def __new__(cls, data, inds=None):
        """An np.ndarray of data which can be assigned attributes.

        Parameters
        -------------
        data : np.ndarray
            A numpy array of data to transform into an InfoArray.
        inds : np.ndarray
            A numpy array represenitng indexes for each datapoint,
            to store as self.inds.
        """
        arr = np.asarray(data).view(cls)

        if inds is not None:
            arr.inds = inds

        return arr


class EphysObject(object):
    def __init__(self):
        """The EphysObject class is the primary class in synappy for loading
        and working with electrophysiological classes.

        It is created by calling the synappy.load() function, which loads a
        dataset of recordings.

        Class methods provide convenient ways to search for stimulus-
        triggered or spontaneous events, as well as to quantify event stats
        such as amplitude, decay, integral. Some class methods can also
        visualize traces and plot summary statistics.
        """
        pass

    def add_stim_on(self,
                    event_type='stim',
                    stim_thresh=2,
                    **kwargs_spont):
        """
        Find stimulus onsets from a stimulus channel, and add their indices to
        an EphysObject instance. Used by methods to quantify stimulus-triggered
        event statistics.

        Stores results in self.stim_on, which has the format:
            stim_on[neuron][trial][event_index]

        Parameters
        -------------
        event_type : str
            Indicates whether the event is stimulus-triggered
            (event_type='stim') or spontaneous (event_type='spontaneous').

        stim_thresh : float
            The stimulus onset threshold, typically in volts for TTL pulses
            (although this can vary depending on the type of stimulus signal.)

        **kwargs_spont : dict
            Keyword arguments for spontaneous event detection,
            passed to find_spontaneous().

            All possible kwargs and their recommended values are:
                spont_filtsize=1001,
                spont_threshampli=3,
                spont_threshderiv=-1.2,
                spont_thresh_pos_deriv=0.1,
                savgol_polynomial=3}

        Example
        ------------
        >>> d = syn.load(['ex_file.abf'])
        >>> d.add_stim_on(stim_thresh=2)

        Attributes added to class instance
        ------------
        self.stim_on : np.ndarray
            An array of stimulus/spontaneous event onset indices
            for each neuron.
            - if event_type='stim':
                .stim_on[neuron][trial][stim]
            - if event_type='spontaneous':
                .stim_on[neuron][trial][spontaneous_event]
        """

        if event_type == 'stim':
            self.stim_on = find_stims(self, thresh=stim_thresh)
        elif event_type == 'spontaneous':
            self.stim_on = find_spontaneous(self, kwargs_spont)

        print('\nAdded stim_on (event_type = ', event_type, ')')

        return

    def add_ampli(self, event_sign='pos',
                  t_baseline_lower=4, t_baseline_upper=0.2,
                  t_event_lower=5, t_event_upper=30,
                  smoothing_width=None,
                  latency_method='max_ampli'):
        """
        Computes pre-event baseline values, event amplitudes,
        and event latencies (computed in a number of ways).

        This requires a self.stim_on attribute, created by calling
        the method self.add_stim_on(). For each stimulus, a baseline
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

        smoothing_width : None or int
            Width of savgol filter applied to data, in inds,
            before computing maximum amplitude.
            If None, defaults to minimal smoothing (3 inds).

        latency_method : str
            The method used to calculate latency from stimulus to
            event. Possible methods are:
            - 'max_ampli': Time from stimulus to maximum amplitude.
            - 'max_slope': Time from stimulus to maximum first deriv.
            - 'baseline_plus_4sd': Time from stimulus to time where
                signal exceeds baseline levels + 4 standard deviations.
            - '80_20_line': Computes the times where the signal reaches 20%
                and 80% of the maximum amplitude, then draws a straight line
                between these and determines where this line first intersects
                the signal. The time from the stimulus to this point gives
                the latency. (Provides a best guess of when the signal starts
                to rise.)

        Attributes added to class instance
        ------------
        self.ampli : np.ndarray
            An array of amplitudes and associated metrics.
                .ampli[neuron][trial, stim, ampli_stat], where
                    ampli_stat=0: max amplitude (pA/mV);
                    ampli_stat=1: index of max amplitude;
                    ampli_stat=2: time from stim onset to max amplitude.
        self.baseline : np.ndarray
            An array of baselines and associated metrics.
                .baseline[neuron][trial, stim, bl_stat], where
                    bl_stat=0: mean of baseline before stimulus (pA/mV);
                    bl_stat=1: st. dev. of baseline before stim (pA/mV)
        self.latency : np.ndarray
            An array of latencies and associated metrics. latency_method
            specifies the way of calculating latency (eg to max ampli,
            max deriv., first time crossing mean+4*sd, etc.)
                .latency[neuron][trial, stim, lat_stat], where
                    lat_stat=0: latency from stim onset (s);
                    lat_stat=1: index of latency;
        """
        num_neurons = len(self.analog_signals)

        # Define new vars to store event amplitude and latency
        baseline = np.empty(num_neurons, dtype=np.ndarray)
        ampli = np.empty(num_neurons, dtype=np.ndarray)
        latency = np.empty(num_neurons, dtype=np.ndarray)

        # Determine direction of postsynaptic events
        if event_sign == 'pos' or event_sign == 'up':
            event_direction = 1
        if event_sign == 'neg' or event_sign == 'down':
            event_direction = -1

        # extract event statistics for each neuron
        for neuron in range(num_neurons):
            # Define vars for this neuron
            num_trials = len(self.analog_signals[neuron])
            sample_rate = np.int32(np.round(1 / (self.times[neuron][1]
                                                 - self.times[neuron][0])))

            # compute sample-rate-dependent variables
            if smoothing_width is None:
                smoothing_width = 2
                savgol_width = np.int32(
                    smoothing_width * sample_rate / 1000) + 1
            else:
                savgol_width = np.int32(
                    smoothing_width * sample_rate / 1000) + 1

            abs_base_lower = np.int32(t_baseline_lower * sample_rate / 1000)
            abs_base_upper = np.int32(t_baseline_upper * sample_rate / 1000)
            abs_pse_lower = np.int32(t_event_lower * sample_rate / 1000)
            abs_pse_upper = np.int32(t_event_upper * sample_rate / 1000)

            max_stims = 0
            for trial in range(num_trials):
                if len(self.stim_on[neuron][trial]) > max_stims:
                    max_stims = len(self.stim_on[neuron][trial])

            baseline[neuron] = np.empty([num_trials, max_stims, 2],
                                        dtype=np.ndarray)
            ampli[neuron] = np.empty([num_trials, max_stims, 4],
                                     dtype=np.ndarray)
            latency[neuron] = np.empty([num_trials, max_stims, 2],
                                       dtype=np.ndarray)

            for trial in range(num_trials):
                num_stims = len(self.stim_on[neuron][trial])

                to_mask = np.arange(max_stims-1, num_stims-1, -1)
                baseline[neuron][trial, to_mask] = 0
                ampli[neuron][trial, to_mask] = 0
                latency[neuron][trial, to_mask] = 0

                for stim in range(num_stims):
                    # Calculate inds for this instance
                    _base_lower = np.int32(
                        self.stim_on[neuron][trial][stim] - abs_base_lower)
                    _base_upper = np.int32(
                        self.stim_on[neuron][trial][stim] - abs_base_upper)
                    _pse_lower = np.int32(
                        self.stim_on[neuron][trial][stim] + abs_pse_lower)
                    _pse_upper = np.int32(
                        self.stim_on[neuron][trial][stim] + abs_pse_upper)

                    # Calculate event baseline
                    baseline[neuron][trial, stim, 0] = np.mean(
                        self.analog_signals[neuron][
                            trial, _base_lower:
                            _base_upper])
                    baseline[neuron][trial, stim, 1] = np.std(
                        self.analog_signals[neuron][
                            trial, _base_lower:
                            _base_upper])

                    # Calculate event amplitude
                    _analog = self.analog_signals[neuron][
                        trial, _pse_lower:_pse_upper]
                    _analog_sm = sp_signal.savgol_filter(
                        _analog, savgol_width, 3)

                    # calculate max event ampli [stim,0] and its index [stim,1]
                    if event_direction == 1:
                        ampli[neuron][trial, stim, 1] \
                            = np.argmax(_analog_sm, axis=-1)
                    elif event_direction == -1:
                        ampli[neuron][trial, stim, 1] \
                            = np.argmin(_analog_sm, axis=-1)

                    # correct index back to analog_signal reference
                    ampli[neuron][trial, stim, 1] \
                        += _pse_lower
                    ampli[neuron][trial, stim, 0] \
                        = self.analog_signals[neuron][
                            trial, np.int32(
                                ampli[neuron][trial, stim, 1])]
                    ampli[neuron][trial, stim, 0] \
                        -= baseline[neuron][trial, stim, 0]

                    # store time of max_ampli latency in [stim,2]
                    ampli[neuron][trial, stim, 2] = (
                        self.times[neuron][np.int32(
                            ampli[neuron][trial][stim, 1])]
                        - self.times[neuron][np.int32(
                            self.stim_on[neuron][trial][stim])])

                    # ------------------------------
                    # Calculate event onset latencies
                    max_ampli_smoothed_ind = np.int32(
                        ampli[neuron][trial, stim, 1]
                        - _pse_lower)

                    if max_ampli_smoothed_ind < 2:
                        max_ampli_smoothed_ind = 2

                    _analog_sm_deriv = np.gradient(
                        _analog_sm[0:max_ampli_smoothed_ind])

                    if event_direction == 1:
                        max_deriv_ind = np.argmax(_analog_sm_deriv)
                        ampli[neuron][trial, stim, 3] \
                            = _analog_sm_deriv[max_deriv_ind] \
                            * (sample_rate/1000)
                    elif event_direction == -1:
                        max_deriv_ind = np.argmin(_analog_sm_deriv)
                        ampli[neuron][trial, stim, 3] \
                            = _analog_sm_deriv[max_deriv_ind] \
                            * (sample_rate/1000)

                    # determine latency and store in postsynaptic_event_latency
                    if latency_method == 'max_ampli':
                        event_time_index = np.int32(
                            ampli[neuron][trial, stim, 1])
                        stim_time_index = np.int32(
                            self.stim_on[neuron][trial][stim])

                        latency[neuron][trial, stim, 0] \
                            = self.times[neuron][event_time_index] \
                            - self.times[neuron][stim_time_index]
                        latency[neuron][trial, stim, 1] \
                            = ampli[neuron][trial, stim, 1]

                    elif latency_method == 'max_slope':
                        event_time_index = np.int32(
                            max_deriv_ind + _pse_lower)
                        stim_time_index = np.int32(
                            self.stim_on[neuron][trial][stim])

                        latency[neuron][trial, stim, 0] \
                            = self.times[neuron][event_time_index] \
                            - self.times[neuron][stim_time_index]
                        latency[neuron][trial, stim, 1] \
                            = event_time_index

                    elif latency_method == 'baseline_plus_4sd':
                        _thresh = baseline[neuron][trial, stim, 0] \
                            + 4 * baseline[neuron][trial, stim, 1]

                        # store temp vars for troubleshooting
                        self._analog_sm_crop \
                            = _analog_sm[0:max_ampli_smoothed_ind]
                        self._analog_sm = _analog_sm
                        self._thresh = _thresh

                        _ind_1stcross = np.where(
                            _analog_sm > _thresh)[0][0]

                        latency[neuron][trial, stim, 0] \
                            = self.times[neuron][
                                _ind_1stcross + abs_pse_lower]
                        latency[neuron][trial, stim, 1] \
                            = _ind_1stcross + abs_pse_lower

                    elif latency_method == '80_20_line':
                        value_80pc = 0.8 * (ampli[neuron][trial, stim, 0])\
                            + baseline[neuron][trial, stim, 0]
                        value_20pc = 0.2 * (ampli[neuron][trial, stim, 0])\
                            + baseline[neuron][trial, stim, 0]
                        value_80pc_sizeanalog = value_80pc \
                            * np.ones(len(
                                _analog_sm[0:max_ampli_smoothed_ind]))
                        value_20pc_sizeanalog = value_20pc \
                            * np.ones(len(
                                _analog_sm[0:max_ampli_smoothed_ind]))
                        diff_80pc = (_analog[0:max_ampli_smoothed_ind]
                                     - value_80pc_sizeanalog) > 0
                        diff_20pc = (_analog[0:max_ampli_smoothed_ind]
                                     - value_20pc_sizeanalog) > 0

                        if event_direction == 1:
                            ind_80cross = find_last(diff_80pc, tofind=0)
                            ind_20cross = find_last(diff_20pc, tofind=0)
                        elif event_direction == -1:
                            ind_80cross = find_last(diff_80pc, tofind=1)
                            ind_20cross = find_last(diff_20pc, tofind=1)

                        if ind_20cross > ind_80cross or ind_80cross == 0:
                            ind_80cross = np.int32(1)
                            ind_20cross = np.int32(0)

                        val_80cross = _analog_sm[ind_80cross]
                        val_20cross = _analog_sm[ind_20cross]

                        slope_8020_line = (val_80cross - val_20cross) \
                            / (ind_80cross - ind_20cross)

                        vals_8020_line = np.zeros(
                            len(_analog_sm[0:ind_80cross + 1]))
                        vals_8020_line = [(val_80cross - (ind_80cross - i)
                                           * slope_8020_line)
                                          for i in range(ind_80cross)]

                        vals_baseline = baseline[neuron][trial, stim, 0] \
                            * np.ones(len(_analog_sm[0:ind_80cross]))
                        diff_sq_8020_line = (
                            (vals_baseline - vals_8020_line)**2
                            + (_analog[0:ind_80cross] - vals_8020_line)**2)

                        intercept_8020_ind = np.argmin(diff_sq_8020_line)

                        event_time_index = intercept_8020_ind \
                            + _pse_lower
                        stim_time_index = self.stim_on[neuron][trial][stim]
                        latency[neuron][trial, stim, 0] = \
                            self.times[neuron][event_time_index] \
                            - self.times[neuron][stim_time_index]
                        latency[neuron][trial, stim, 1] = event_time_index

        self.ampli = ampli
        self.latency = latency
        self.baseline = baseline

        # self.height is for legacy support of earlier versions.
        self.height = ampli

        print('\nAdded ampli. \nAdded latency. \nAdded baseline.')

        return

    def add_sucfail_sorting(self, thresh=False, thresh_dir=False):
        """
        Adds trial-by-trial success/failure sorting to the events, where
        failures are stored as masked elements in the following attributes:
        .ampli, .latency, .baseline. A copy of the mask is stored in .mask.

        Failures are determined in one of two ways. First, the method can use
        a dynamic threshold, mean + 3*S.D. (thresh = false).
        Alternately, the method allows a user-specified threshold which can be
        constant for all neurons (type(thresh) = float) or can be specified for
        each neuron (len(thresh) = num_neurons).

        Parameters
        ---------------
        thresh : bool, float or list
            Determines how the threshold for event failures and successes is
            calculated.

            - If thresh = False, the threshold for event failures is
            automatically specified as the mean +- 3*S.D. for each trace.
            - Otherwise, the threshold is determined by the user
            for all neurons.
            (if type(thresh) == float), or for each neuron individually
            (if type(thresh) == list)

        thresh_dir : bool
            Determines whether the threshold is unidirectional (True: in the
            direction previously specified for events) or bidirectional (False)

        Attributes added to class instance
        ------------
        .ampli, .latency, .baseline are modified to be np.masked.arrays, with
        the masked elements corresponding to failures. In addition, the
        following attribute is added:

        self.mask : np.ndarray
            An boolean array denoting which trials/stims are failures (True)
            or successes (False).
                .mask[neuron][trial, stim]

        """

        postsynaptic_event = self.ampli
        postsynaptic_event_latency = self.latency
        baseline = self.baseline

        num_neurons = len(postsynaptic_event)
        thresh = np.array(thresh, dtype=np.float)

        self.mask = np.empty(num_neurons, dtype=np.ndarray)

        postsynaptic_event_successes = np.empty(num_neurons, dtype=np.ndarray)
        postsynaptic_event_latency_successes = np.empty(num_neurons,
                                                        dtype=np.ndarray)

        dynamic_thresholding = False
        if type(thresh) is bool and thresh is False:
            dynamic_thresholding = True
        elif (len(np.atleast_1d(thresh)) is not num_neurons and
              len(np.atleast_1d(thresh)) == 1):
            thresh = np.ones(num_neurons) * thresh

        for neuron in range(num_neurons):
            postsynaptic_event_successes[neuron] = np.ma.array(
                np.copy(postsynaptic_event[neuron]))
            postsynaptic_event_latency_successes[neuron] = np.ma.array(
                np.copy(postsynaptic_event_latency[neuron]))

            ampli_tocompare = np.abs(postsynaptic_event[neuron][:, :, 0])

            if dynamic_thresholding is True:
                thresh_tocompare = 4 * baseline[neuron][:, :, 1]
            else:
                thresh_thisneuron = thresh[neuron]
                thresh_tocompare = np.abs(thresh_thisneuron) * np.ones_like(
                    baseline[neuron][:, :, 1])

            diff_tocompare = ampli_tocompare - thresh_tocompare
            if thresh_dir is False:
                mask_tocompare = diff_tocompare < 0
            elif thresh_dir is True:
                mask_tocompare = diff_tocompare > 0

            mask_tocompare_full_pes = np.ma.empty(
                [mask_tocompare.shape[0], mask_tocompare.shape[1],
                 postsynaptic_event[neuron].shape[2]])
            for shape_3d in range(postsynaptic_event[neuron].shape[2]):
                mask_tocompare_full_pes[:, :, shape_3d] = mask_tocompare

            mask_tocompare_full_lat = np.ma.empty(
                [mask_tocompare.shape[0], mask_tocompare.shape[1],
                 postsynaptic_event_latency[neuron].shape[2]])
            for shape_3d in range(postsynaptic_event_latency[neuron].shape[2]):
                mask_tocompare_full_lat[:, :, shape_3d] = mask_tocompare

            postsynaptic_event_successes[neuron].mask = mask_tocompare_full_pes
            postsynaptic_event_latency_successes[neuron].mask \
                = mask_tocompare_full_lat

            self.ampli[neuron] = np.ma.masked_array(
                self.ampli[neuron],
                mask=mask_tocompare_full_pes)
            self.latency[neuron] = np.ma.masked_array(
                self.latency[neuron],
                mask=mask_tocompare_full_pes[:, :, 0:2])
            self.mask[neuron] = mask_tocompare_full_pes[:, :, 0]
            self.baseline[neuron] = np.ma.masked_array(
                self.baseline[neuron],
                mask=mask_tocompare_full_pes[:, :, 0:2])

        print('\nAdded succ/fail sorting to: '
              '\n\tampli \n\tlatency, \n\tbaseline')

        return

    def add_inverted_sucfail_sort(self):
        """ Adds inverted success/failure sorting, where successes are
        masked instead of failures. See add_sucfail_sorting() for
        more information on the basic operation.

        The inverted sorting is stored in the .ampli_fails attribute.
        """
        postsynaptic_event_successes = self.ampli
        num_neurons = len(postsynaptic_event_successes)

        self.ampli_fails = np.empty(num_neurons, dtype=np.ndarray)

        postsynaptic_event_failures = np.empty(num_neurons, dtype=np.ndarray)

        for neuron in range(num_neurons):
            success_mask = np.ma.getmask(postsynaptic_event_successes[neuron])
            failure_mask = np.logical_not(success_mask)

            postsynaptic_event_failures[neuron] = np.ma.array(
                np.copy(postsynaptic_event_successes[neuron]),
                mask=failure_mask)
            self.ampli_fails[neuron] = np.ma.array(
                np.copy(postsynaptic_event_successes[neuron]),
                mask=failure_mask)

        return

    def add_normalized_ampli(self):
        """Adds normalized amplitude measurement to the class instance as
        .ampli_norm. Amplitudes are normalized to the mean ampli for each
        stimulus delivered to each neuron.

        Parameters
        ------------


        Attributes added to class instance
        ------------
        self.ampli_norm : np.ndarray
            An array of amplitudes and associated metrics.
                .ampli_norm[neuron][trial, stim, ampli_stat], where
                    ampli_stat=0: max amplitude (norm.);
                    ampli_stat=1: index of max amplitude;
                    ampli_stat=2: time from stim onset to max amplitude.
        """

        postsynaptic_event = self.ampli

        num_neurons = len(postsynaptic_event)

        postsynaptic_event_normalized = np.empty(num_neurons, dtype=np.ndarray)
        self.ampli_norm = np.empty(num_neurons, dtype=np.ndarray)
        avg_ampl = np.empty(num_neurons)

        for neuron in range(num_neurons):
            postsynaptic_event_normalized[neuron] = np.ma.copy(
                postsynaptic_event[neuron])
            self.ampli_norm[neuron] = np.ma.copy(
                postsynaptic_event[neuron])

            avg_ampl = np.mean(postsynaptic_event[neuron][:, 0, 0])

            if type(postsynaptic_event[neuron]) is np.ma.core.MaskedArray:
                current_neuron_mask = postsynaptic_event[neuron][:, :, 0].mask

                postsynaptic_event_normalized_temp = np.array(
                    postsynaptic_event[neuron][:, :, 0]) / avg_ampl
                postsynaptic_event_normalized[neuron][:, :, 0] \
                    = np.ma.array(postsynaptic_event_normalized_temp,
                                  mask=current_neuron_mask)
            else:
                postsynaptic_event_normalized[neuron][:, :, 0] /= avg_ampl

            self.ampli_norm[neuron] = postsynaptic_event_normalized[neuron]

        print('\nAdded ampli_norm.')

        return

    def pool_stat_across_neurs(self, name, pool_index=0, mask='suc'):
        """For an event statistic of a given name (eg 'ampli'), pools all
        trial-by-trial values of this statistic across all neurons while
        preserving stimulus order.

        Only considers the minimum number of stimuli that are 'common'
        to all neurons, such that the returned array has shape
        [n_stim, n_neur] and contains the statistics of note.

        Parameters
        ------------------
        name : str
            Name of the event statistic. (eg 'ampli' or 'latency' or
            'baseline'.)

        pool_index : ind
            Index of attribute's array to pool over. Mostly for internal use.

        mask : str
            Can either be 'suc', meaning only successes are considered,
            or 'all', meaning successes/failures are considered while
            pooling.

        Returns
        ---------------
        stimpooled_postsynaptic_events : array
            An array of  [n_stim, n_neur] consisting of individual
            statistics of the desired type (denoted by the kwarg name)
            which are pooled over all trialtypes and neurons.
        """

        if type(name) is str:
            postsynaptic_event = self.__getattribute__(name)
        else:
            postsynaptic_event = name

        num_neurons = len(postsynaptic_event)

        # Calculate min stims:
        common_stims = 10000
        for neuron in range(num_neurons):
            if common_stims > postsynaptic_event[neuron][:, :, :].shape[1]:
                common_stims = postsynaptic_event[neuron][:, :, :].shape[1]

        # Pool data
        stimpooled_postsynaptic_events = np.ma.array(
            np.empty([common_stims, 0]))

        for neuron in range(num_neurons):
            if mask == 'suc':
                stimpooled_postsynaptic_events = np.ma.append(
                    stimpooled_postsynaptic_events,
                    np.ma.transpose(
                        postsynaptic_event[neuron]
                        [:, 0:common_stims, pool_index]),
                    axis=1)

            elif mask == 'all':
                stimpooled_postsynaptic_events = np.ma.append(
                    stimpooled_postsynaptic_events,
                    np.ma.transpose(
                        postsynaptic_event[neuron]
                        [:, 0:common_stims, pool_index].data),
                    axis=1)

        return stimpooled_postsynaptic_events

    def add_sucrate(self, byneuron=False):
        """Computes postsynaptic event success rates (ie successes / all_stims)
        and stores this as an attribute .success_rate.

        Parameters
        ----------------
        byneuron : bool
            If False, does not store success rates by neuron.

        Attributes added to class instance
        ------------
        self.success_rate : np.ndarray
            Stores success rate for each stimulus number,
            pooled across neurons.
                .success_rate[common_stim, sucrate_stat], where
                    sucrate_stat=0 : mean
                    sucrate_stat=1: std. dev.
                    sucrate_stat=2: standard error of mean.

        Example
        -------------
        >>> print(f'mean sucrate for stim 0: {o.success_rate[0, 0]}'
            mean sucrate for stim 0: 0.8
        >>> print(f'sem sucrate for stim 2: {o.success_rate[2, 2]}'
            sem sucrate for stim 2: 0.1
        """
        postsynaptic_event = self.mask

        num_neurons = len(postsynaptic_event)

        # Calculate min stims:
        common_stims = 10000
        for neuron in range(num_neurons):
            if common_stims > len(postsynaptic_event[neuron][0, :, 0]):
                common_stims = len(postsynaptic_event[neuron][0, :, 0])

        success_rate_neur = np.zeros([common_stims, num_neurons])
        success_rate = np.zeros([common_stims, 3])

        for neuron in range(num_neurons):
            count_fails_temp = np.sum(
                postsynaptic_event[neuron][:, 0:common_stims, 0].mask,
                axis=0)
            count_total_temp = postsynaptic_event[neuron].shape[0]
            success_rate_neur[:, neuron] = (
                count_total_temp - count_fails_temp) / count_total_temp

        success_rate[:, 0] = np.mean(success_rate_neur, axis=1)
        success_rate[:, 1] = np.std(success_rate_neur, axis=1)
        success_rate[:, 2] = np.std(success_rate_neur, axis=1) \
            / np.sqrt(np.sum(success_rate_neur, axis=1))

        if byneuron is True:
            success_rate = []
            success_rate = success_rate_neur

        self.success_rate = success_rate

        return self

    def special_sucrate(self, byneuron=False):
        """Computes special success rate. See add_sucrate() for details.
        """
        mask = self.mask

        num_neurons = len(mask)

        # Calculate min stims:
        common_stims = 10000
        for neuron in range(num_neurons):
            if common_stims > len(mask[neuron][0, :]):
                common_stims = len(mask[neuron][0, :])

        success_rate_neur = np.zeros([common_stims, num_neurons])
        success_rate = np.zeros([common_stims, 3])

        for neuron in range(num_neurons):
            count_fails_temp = np.sum(mask[neuron][:, 0:common_stims], axis=0)
            count_total_temp = mask[neuron].shape[0]
            success_rate_neur[:, neuron] = (
                count_total_temp - count_fails_temp) / count_total_temp

        success_rate[:, 0] = np.mean(success_rate_neur, axis=1)
        success_rate[:, 1] = np.std(success_rate_neur, axis=1)
        success_rate[:, 2] = np.std(success_rate_neur, axis=1) \
            / np.sqrt(np.sum(success_rate_neur, axis=1))

        if byneuron is True:
            success_rate = []
            success_rate = success_rate_neur

        self.success_rate = success_rate

        return success_rate

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

        Called by self.remove_unclamped_aps to propagate a changed mask
        across all class instance attributes.
        """
        postsynaptic_events = self.ampli
        mask = self.mask
        num_neurons = len(postsynaptic_events)
        for neuron in range(num_neurons):
            for lastind in range(postsynaptic_events[neuron].shape[2]):
                self.ampli[neuron].mask[:, :, lastind] = mask[neuron]
                self.ampli_norm[neuron].mask[:, :, lastind] = mask[neuron]
                self.ampli_fails[neuron].mask[:, :, lastind] = ~mask[neuron]

                if lastind < self.latency[neuron].shape[2]:
                    self.latency[neuron].mask[:, :, lastind] = mask[neuron]
                if lastind < self.baseline[neuron].shape[2]:
                    self.baseline[neuron].mask[:, :, lastind] = mask[neuron]
                if lastind < self.decay[neuron].shape[2]:
                    self.decay[neuron].mask[:, :, lastind] = mask[neuron]

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
        ampli_norm = self.ampli_norm
        ampli = self.ampli
        latency = self.latency

        num_neurons = len(ampli_norm)

        for neuron in range(num_neurons):
            to_replace = np.argwhere(
                ampli_norm[neuron][:, :, 0] > thresh)

            ampli_norm[neuron][to_replace[:, 0],
                               to_replace[:, 1], :] = np.nan
            ampli_norm[neuron].mask[to_replace[:, 0],
                                    to_replace[:, 1], :] = True

            ampli[neuron][to_replace[:, 0], to_replace[:, 1], :] = np.nan
            ampli[neuron].mask[to_replace[:, 0], to_replace[:, 1], :] = True

            latency[neuron][to_replace[:, 0], to_replace[:, 1], :] = np.nan
            latency[neuron].mask[to_replace[:, 0], to_replace[:, 1], :] = True

            self.mask[neuron] = np.ma.mask_or(
                ampli_norm[neuron][:, :, 0].mask,
                self.ampli[neuron][:, :, 0].mask)

        self._propagate_mask()

        print('\nMasked APs')

    def add_decay(self, t_prestim=0, t_poststim=10, plotting=False,
                  fn='monoexp_normalized_plusb'):
        """Fits each post-synaptic event with an exponential decay fuction
        and stores the fitted parameters in self.decay.

        Decay equation variables correspond to the fitted variables for
        the equation used (see the kwarg fn for more info).
        - monoexponential decay: lambda1, b.
        - biexponential decay: lambda1, lambda2, vstart2, b.

        Parameters
        ------------
        prestim : float
            Time before stimulus, in ms, to include in signal
            used to compute decay.

        poststim : float
            Time after stimulus, in ms, to include in signal
            used to compute decay.

        plotting : bool
            Whether to plot examples of decay fits (True) or not (False).

        fn : str
            Exponential decay function to use.
            - 'monoexp_normalized_plusb': y = e^(-t * lambda1) + b
            - 'biexp_normalized_plusb': y = e^(-t * lambda1)
                + vstart * e^(-t / lambda2) + b

            (In all cases, the more traditional decay tau can be computed
            as tau= 1/lambda).

        Attributes added
        ------------
        self.decay : np.ndarray
            An array of fitted decay parameters.
                .decay[neuron][trial, stim, decay_param].

                - If fn='monoexp_normalized_plusb', then
                    decay_param=0 : lambda1
                    decay_param=1 : b

                - If fn='biexp_normalized_plusb', then
                    decay_param=0 : lambda1
                    decay_param=1 : lambda2
                    decay_param=2 : vstart2
                    decay_param=3 : b
        """

        # Import variables from synappy wrapper
        analog_signals = self.analog_signals
        postsynaptic_events = self.ampli
        baseline = self.baseline
        times = self.times

        num_neurons = len(postsynaptic_events)

        def biexp_normalized_plusb(time_x, lambda1, lambda2, vstart2, b):
            y = np.exp(time_x * (-1) * lambda1) \
                + vstart2 * np.exp(time_x * (-1) * lambda2) + b
            return y

        def monoexp_normalized_plusb(time_x, lambda1, b):
            y = np.exp(time_x * (-1) * lambda1) + b
            return y

        if fn == 'monoexp_normalized_plusb':
            num_vars = 2
            vars_guess = [100, 0]
        elif fn == 'biexp_normalized_plusb':
            num_vars = 4
            vars_guess = [100, 100, 1, 0]

        fitted_vars = np.empty(num_neurons, dtype=np.ndarray)
        fitted_covari = np.empty_like(fitted_vars)

        for neuron in range(num_neurons):
            sample_rate = np.int32(np.round(
                1 / (times[neuron][1] - times[neuron][0])))
            prestim_ind = np.int32(t_prestim * sample_rate / 1000)
            poststim_ind = np.int32(t_poststim * sample_rate / 1000)

            num_trials = postsynaptic_events[neuron].shape[0]
            num_stims = postsynaptic_events[neuron].shape[1]

            if type(postsynaptic_events[neuron]) is np.ma.core.MaskedArray:
                fitted_vars[neuron] = np.ma.array(np.empty(
                    [num_trials, num_stims, num_vars], dtype=np.ndarray))
                fitted_vars[neuron].mask = postsynaptic_events[neuron].mask
                fitted_covari[neuron] = np.ma.array(np.empty(
                    [num_trials, num_stims, num_vars], dtype=np.ndarray))
                fitted_covari[neuron].mask = postsynaptic_events[neuron].mask

            else:
                fitted_vars[neuron] = np.empty(
                    [num_trials, num_stims, num_vars], dtype=np.ndarray)
                fitted_covari[neuron] = np.empty(
                    [num_trials, num_stims, num_vars], dtype=np.ndarray)

            for trial in range(num_trials):
                for stim in range(num_stims):

                    if (type(postsynaptic_events[neuron])
                            == np.ma.core.MaskedArray and
                            postsynaptic_events[neuron][trial, stim, 0] is not
                            np.ma.masked):

                        event_ind_min = postsynaptic_events[neuron][
                            trial, stim, 1] - prestim_ind
                        event_ind_max = event_ind_min + poststim_ind

                        postsynaptic_curve = analog_signals[neuron][
                            trial, event_ind_min:event_ind_max] \
                            - baseline[neuron][trial, stim, 0]
                        postsynaptic_curve /= np.mean(postsynaptic_curve[0:2])

                        times_forfit = times[neuron][0:poststim_ind]

                        try:
                            [popt, pcov] = sp_opt.curve_fit(
                                monoexp_normalized_plusb, times_forfit,
                                postsynaptic_curve, p0=vars_guess)

                        except RuntimeError:
                            popt = np.ones(num_vars) * 10000
                            pcov = 10000

                        except ValueError:
                            print(postsynaptic_curve, 'neuron: ', neuron,
                                  'trial: ', trial, 'stim: ', stim)

                        fitted_vars[neuron][trial, stim, :] = popt[:]

                    elif (type(postsynaptic_events[neuron])
                          == np.ma.core.MaskedArray and
                          postsynaptic_events[neuron][trial, stim, 0] is
                          np.ma.masked):

                        fitted_vars[neuron][trial, stim, :] = np.ones(
                            num_vars) * 10000
                        fitted_vars[neuron][trial, stim, :].mask = np.ones(
                            num_vars, dtype=np.bool)

                    elif (type(postsynaptic_events[neuron]) is not
                          np.ma.core.MaskedArray):
                        event_ind_min = postsynaptic_events[neuron][
                            trial, stim, 1] - prestim_ind
                        event_ind_max = event_ind_min + poststim_ind

                        postsynaptic_curve = analog_signals[neuron][
                            trial, event_ind_min:event_ind_max] \
                            - baseline[neuron][trial, stim, 0]
                        postsynaptic_curve /= postsynaptic_curve[0]

                        times_forfit = times[neuron][0:poststim_ind]

                        try:
                            [popt, pcov] = sp_opt.curve_fit(
                                monoexp_normalized_plusb, times_forfit,
                                postsynaptic_curve, p0=vars_guess)

                        except RuntimeError:
                            popt = np.ones(num_vars) * 10000

                        fitted_vars[neuron][trial, stim, :] = popt[:]

            if plotting is True:
                if type(postsynaptic_events[neuron]) == np.ma.core.MaskedArray:
                    first_nonmasked_trial = (np.argwhere(
                        postsynaptic_events[neuron][:, 0, 1].mask is False)
                                             [0][0])

                    postsynaptic_curve = (analog_signals[neuron][
                        first_nonmasked_trial,
                        postsynaptic_events[neuron]
                        [first_nonmasked_trial, 0, 1] - prestim_ind:
                        postsynaptic_events[neuron]
                        [first_nonmasked_trial, 0, 1] + poststim_ind]
                        - baseline[neuron][first_nonmasked_trial, 0, 0])

                    y_fitted = (postsynaptic_events[neuron]
                                [first_nonmasked_trial, 0, 0]
                                * monoexp_normalized_plusb(
                                    times[neuron][0:poststim_ind + prestim_ind],
                                    fitted_vars[neuron][
                                        first_nonmasked_trial, 0, 0],
                                    fitted_vars[neuron][
                                        first_nonmasked_trial, 0, 1]))

                    plt.figure()
                    plt.plot(times[neuron][0:poststim_ind + prestim_ind],
                             y_fitted, 'r')
                    plt.plot(times[neuron][0:poststim_ind + prestim_ind],
                             postsynaptic_curve, 'b')

                elif type(postsynaptic_events[neuron]) is np.array:
                    postsynaptic_curve = (analog_signals[neuron]
                                          [0, postsynaptic_events[neuron]
                                           [0, 0, 1] - prestim_ind:
                                           postsynaptic_events[neuron][0, 0, 1]
                                           + poststim_ind]
                                          - baseline[neuron][0, 0, 0])
                    y_fitted = (postsynaptic_events[neuron][0, 0, 0]
                                * monoexp_normalized_plusb(
                                    times[neuron][0:poststim_ind + prestim_ind],
                                    fitted_vars[neuron][0, 0, 0],
                                    fitted_vars[neuron][0, 0, 1]))
                    plt.figure()
                    plt.plot(times[neuron][0:poststim_ind + prestim_ind],
                             y_fitted, 'r')
                    plt.plot(times[neuron][0:poststim_ind + prestim_ind],
                             postsynaptic_curve, 'b')

            # convert from lambda to tau
            fitted_ones = np.ones([fitted_vars[neuron][:, :, 0].shape[0],
                                   fitted_vars[neuron][:, :, 0].shape[1]])

            if type(postsynaptic_events[neuron]) is np.ma.core.MaskedArray:
                fitted_vars[neuron][:, :, 0] = fitted_ones / np.array(
                    fitted_vars[neuron][:, :, 0])
                fitted_vars[neuron][:, :, 0] = _get_median_filtered(
                    fitted_vars[neuron][:, :, 0], threshold=10)
                fittedvarmask = np.ma.mask_or(
                    postsynaptic_events[neuron][:, :, 0].mask,
                    fitted_vars[neuron][:, :, 0].mask)
                fitted_vars[neuron][:, :, 0].mask = fittedvarmask

            else:
                fitted_vars[neuron][:, :, 0] = (fitted_ones
                                                / fitted_vars[neuron][:, :, 0])
                fitted_vars[neuron][:, :, 0] = _get_median_filtered(
                    fitted_vars[neuron][:, :, 0], threshold=10)

        self.decay = fitted_vars

        print('\nAdded decay')

        return

    def add_integral(self, t_integral=1000, cdf_bins=100):
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
        self.integral : np.ndarray
            An array of integral values (mV*sec or pA*sec).
                .integral[neuron][trial, stim]
        self.integral_cdf : np.ndarray
            An array containing the cumulative distribution of the integral,
            with bins=cdf_bins, over the entire t_integral time. (units are
            fractional integral from 0->1.)
                .integral_cdf[neuron][trial, stim][cdf_bin]
        """

        analog_signals = self.analog_signals
        postsynaptic_events = self.ampli
        baseline = self.baseline
        times = self.times
        stim_on = self.stim_on

        num_neurons = len(analog_signals)

        integral = np.empty(num_neurons, dtype=np.ndarray)
        cdf_integral = np.empty(num_neurons, dtype=np.ndarray)

        # Iterate through neurons, trials and stims
        for neuron in range(num_neurons):
            sample_rate = np.int32(
                np.round(1 / (times[neuron][1] - times[neuron][0])))

            # Calculate number of trials and fill in integral
            num_trials = postsynaptic_events[neuron].shape[0]
            num_stims = postsynaptic_events[neuron].shape[1]
            integral[neuron] = np.zeros([num_trials, num_stims])
            cdf_integral[neuron] = np.zeros(
                [num_trials, num_stims, int(cdf_bins)])

            # Integral calculation
            for trial in range(num_trials):
                for stim in range(num_stims):
                    int_start = int(stim_on[neuron][stim])
                    int_end = int((t_integral/1000) * sample_rate + int_start)

                    analog_toint = (
                        analog_signals[neuron][trial, int_start:int_end]
                        - baseline[neuron][trial, stim, 0])
                    time_toint = times[neuron][int_start:int_end]
                    integral[neuron][trial] = sp_integrate.trapz(
                        analog_toint, time_toint)

                    # Cumulative distribution
                    for nbin in range(cdf_bins):
                        nbin_plusone = nbin + 1
                        curr_cdf_fraction = nbin_plusone / cdf_bins

                        int_start = int(stim_on[neuron][stim])
                        int_end = int(
                            int_start + ((t_integral/1000) * sample_rate)
                            * curr_cdf_fraction)

                        analog_toint = (analog_signals[neuron][
                            trial, int_start:int_end]
                                        - baseline[neuron][trial, stim, 0])
                        time_toint = times[neuron][int_start:int_end]

                        cdf_integral[neuron][trial, stim, nbin] = (
                            sp_integrate.trapz(analog_toint, time_toint)
                            / integral[neuron][trial, stim])

        self.integral = integral
        self.cdf_integral = cdf_integral
        print('\nAdded integral')

        return

    def add_norm_integral(self, t_integral=1000,
                          cdf_bins=100):
        """
        Compute the normalized integral for each post-synaptic event.
        Integrals are normalized to the mean integral value for that
        stimulus delivered to that particular neuron, and are stored
        in .norm_integral and .norm_cdf_integral.

        Parameters
        -----------------
        t_integral : float
            The total post-stimulus time to integrate, in milliseconds.

        cdf_bins : int
            Number of bins for the cumulative integral

        Attributes added
        ------------
        self.norm_integral : np.ndarray
            An array of normalized integral values ().
                .norm_integral[neuron][trial, stim]
        self.norm_integral_cdf : np.ndarray
            An array containing the cumulative distribution of the integral,
            with bins=cdf_bins, over the entire t_integral time. (units are
            fractional integral from 0->1.)
                .integral_cdf[neuron][trial, stim][cdf_bin]
        """

        # Import needed vars from self
        analog_signals = self.analog_signals
        postsynaptic_events = self.ampli
        baseline = self.baseline
        times = self.times
        stim_on = self.stim_on

        # Define number of neurons
        num_neurons = len(analog_signals)

        # Predefine vars
        integral = np.empty(num_neurons, dtype=np.ndarray)
        cdf_integral = np.empty(num_neurons, dtype=np.ndarray)

        # Iterate through neurons, trials and stims
        for neuron in range(num_neurons):
            sample_rate = np.int32(
                np.round(1 / (times[neuron][1] - times[neuron][0])))

            # Calculate number of trials and fill in integral
            num_trials = postsynaptic_events[neuron].shape[0]
            num_stims = postsynaptic_events[neuron].shape[1]
            integral[neuron] = np.zeros([num_trials, num_stims])
            cdf_integral[neuron] = np.zeros(
                [num_trials, num_stims, int(cdf_bins)])

            for trial in range(num_trials):
                for stim in range(num_stims):
                    int_start = int(stim_on[neuron][stim])
                    int_end = int((t_integral/1000) * sample_rate + int_start)

                    analog_toint = ((analog_signals[neuron]
                                    [trial, int_start:int_end]
                                    - baseline[neuron][trial, stim, 0])
                                    / postsynaptic_events[neuron]
                                    [trial, stim, 0])
                    time_toint = times[neuron][int_start:int_end]
                    integral[neuron][trial] = sp_integrate.trapz(
                        analog_toint, time_toint)

                    # Calculate cumulative distribution of integral
                    for nbin in range(cdf_bins):
                        nbin_plusone = nbin + 1
                        curr_cdf_fraction = nbin_plusone / cdf_bins

                        int_start = int(stim_on[neuron][stim])
                        int_end = int(int_start
                                      + ((t_integral/1000) * sample_rate)
                                      * curr_cdf_fraction)

                        analog_toint = (analog_signals[neuron]
                                        [trial, int_start:int_end]
                                        - baseline[neuron][trial, stim, 0])
                        time_toint = times[neuron][int_start:int_end]

                        cdf_integral[neuron][trial, stim, nbin] \
                            = sp_integrate.trapz(
                                analog_toint, time_toint) / integral[
                                    neuron][trial, stim]

        self.norm_integral = integral
        self.norm_cdf_integral = cdf_integral
        print('\nAdded norm. integral')

        return

    def add_all(self, kwargs_add_ampli={'event_sign': 'pos'},
                kwargs_add_integral={},
                kwargs_add_decay={},
                kwargs_mask_unclamped_aps=False,
                kwargs_add_sucfail_sorting=False):
        """
        Convenience method which takes an initialized EphysObject with
        a .stim_on attribute (stimulus onsets), and quantifies
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
            the .add_sucfail_sorting() method

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

        kwargs_add_sucfail_sorting : bool or dict
            If False, does not add success/failure sorting of events.
            If a dict, adds suc/fail sorting to events, using the
            dict as kwargs to .add_sucfail_sorting().

        Attributes added
        ------------
        See .add_ampli(), .add_integral(), .add_decay(),
        .mask_unclamped_aps() and .add_sucfail_sorting() for more info on the
        particular attributes added.
        """

        self.add_ampli(**kwargs_add_ampli)
        self.add_integral(**kwargs_add_integral)
        self.add_decay(**kwargs_add_decay)

        if type(kwargs_mask_unclamped_aps) is dict:
            self.mask_unclamped_aps(**kwargs_mask_unclamped_aps)

        if type(kwargs_add_sucfail_sorting) is dict:
            self.add_sucfail_sorting(**kwargs_add_sucfail_sorting)

        return

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
