# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 14:51:03 2016

@author: michaellynn
""

****SynapPy is a data analysis tool for patch-clamp synaptic physiologists who
    work with .abf files and want to quickly quantify post-synaptic event
    statistics and visualize the results over multiple trials or neurons.


****Synappy works with either evoked or spontaneous events and includes a rapid
    and Pythonic set of methods to add post-synaptic event statistics, including
    amplitude, baseline, decay kinetics, rise-time, and release probability.
****Synappy works with both current clamp and voltage clamp data, and can also
    be used to analyze spike statistics and timing in current clamp
****SynapPy additionally includes intelligent data visualization tools and
    sophisticated data-quality vetting tools.

#-------------------------------------------------------------------------------
#A. AN INTRODUCTION TO SYNAPPY
#-------------------------------------------------------------------------------

###############
1.
###############
*SynapPy first loads the files into an instance of a specialized class
*containing the specified signal channels and the times as attributes:
    .analog_signals
        [neuron][trial, time_indices]
    .stim_signals
        [neuron][trial, time_indices]
    .times
        [neuron][times]

###############
2.
###############
*Through the .add_stim_on() method, one can then add either evoked
*(event_type = 'stim') or spontaneous (event_type = 'spontaneous') events into
*the .stim_on attribute:
    .stim_on
        [neuron][trial][stim_indices]


###############
3.
###############
*For each event, one can then add a variety of post-synaptic event statistics.
*These are added through the .add_all() method, or through individual methods
*for more granuarity (e.g. .add_ampli(), .add_latency(); a.dd_decays()).
*The postsynaptic event statistics are automatically stored in attributes which
*can be accessed at a later time:

    -----------
    .height
    -----------
    #Stores baseline-subtracted peak amplitude of PSP

    [neuron][trial, stim, [height_params]
        where [height_params] = [ampli, ampli_ind,
                time_of_max_ampli_from_stim, ???]]
    -----------
    .baseline
    -----------
    #Stores values for baseline signal

    [neuron][trial, stim, [baseline_params]]
	    where [baseline_params] = [mean_baseline, stdev_baseline]

    -----------
    .latency
    -----------
    #Stores latency from stimulus onset to foot of PSP

    [neuron][trial, stim, [latency_params]]
        where [latency_params] = [latency_sec, ind_latency_sec]

    -----------
    .height_norm
    -----------
    #Stores baseline-subtracted peak ampli normalized to 1 within a cell

    [neuron][trial, stim, [height_params]
        where [height_params] = [normalized_ampli, norm_ampli_ind,
            time_of_max_ampli_from_stim, ??]]

    -----------
    .decay
    -----------
    #Stores statistics for the decay tau of PSP

    [neuron][trial, stim, [tau_params]]
	    where [tau_params] = [tau, baseline_offset]


###############
4.
###############
*By default, SynapPy intelligently filters out data if events are not above 4*SD(baseline),
*or if their decay constant (tau) is nonsensical. These events are masked but kept
*in the underlying data structure, providing a powerful tool to both analyze
*release probability/failure rate, or alternatively spike probability.


*The main dependencies are: numpy, scipy, matplotlib and neo (ver 0.4+ recommended)



#-------------------------------------------------------------------------------
#B. TYPICAL COMMANDS AND THEIR USAGE, AND AN EXAMPLE PIPELINE
#-------------------------------------------------------------------------------

###############
1.
###############
*Load files, add event statistics, and recover these statistics for further
*analysis

    import synappy as syn

    event1 = syn.load(['15d20004.abf', '15d20007.abf', '15d20020.abf'])

            #Give a list of filenames as first argument
            #can also specify trial ranges [default:all], input channels [default:first]
            #stim channels [default:last] and a downsampling ratio for analog signals [default:2]
            #(this last property to help with rapid analysis)


    event1.add_all(event_direction = 'down', latency_method = 'max_slope')

            #automatically adds all relevant stats. Many options here to change stat properties.
            #Note: includes filtering out of unclamped aps; and filtering out of events with nonsensical decay


    event1.height_norm[neuron]

            #fetch normalized height stats for that neuron. dim0 = trials, dim1 = stims.
            #The raw data behind each attribute can be fetched this way.

###############
2.
###############
*Plot event statistics with useful built-in plotting tools

    event1.plot('height')

            #main data visualization tool. Plots event attribute.
            #Makes a separate figure for each neuron, then plots stim_num on x-axis and attribute on y-axis.
            #plots are color-coded (blue are early trials, red are late trials, grey are fails)


    event1.plot_corr('height, 'decay')

            #plots correlation between two attributes within event1.
            #same format/coloring as event1.plot.


    syn.plot_events(event1.attribute, event2.attribute)

            #compare attributes from two data groups (different conditions, cell types, etc.)


    syn.plot_statwrappers(stat1, stat2, ind = 0, err_ind = 2)

            #compare statfiles on two events, and can also give dim1indices of statfiles to plot.
            #eg to plot means +-sterr, syn.plot_statwrappers(stat1, stat2, ind = 0, err_ind = 2)



###############
3.
###############
*Built-in functions and methods

----Useful functions built into SynapPy package----
    syn.pool(event_1.attribute)

        #Pools this attribute over [stims, :]


    syn.get_stats(event_1.attribute, byneuron = False)

        #Gets statistics for this attribute over stimtrain (out[stim,:]) or neuron if byneuron is True (out[neuron,:])
        #dim2: 1->mean, 2->std, 3->sterr, 4->success_rate_mean, 5->success_rate_stdev
        #eg if byneuron = True, out[neuron, 3] would give sterr for that neuron, calculated by pooling across all trials/stims.


---Useful methods which are part of the synwrapper class---:
    synwrapper.propagate_mask(): propagate synwrapper.mask through to all other attributes.
    synwrapper.add_ampli() adds .height and .latency
    synwrapper.add_sorting() adds .mask and sorts [.height, .latency]
    synwrapper.add_invertedsort() adds .height_fails
    synwrapper.add_normalized() adds .height_norm
    synwrapper.add_decay() adds .decay
    synwrapper.remove_unclamped_aps() identifies events higher than 5x (default) amp.
                                    It updates the .mask with this information, and then .propagate_mask()

    synwrapper.add_all() is a general tool to load all stats.


"""

# imports
import neo
import numpy as np
import scipy as sp
import scipy.signal as sp_signal
import scipy.optimize as sp_opt
import scipy.integrate as sp_integrate
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from types import SimpleNamespace

from bokeh.plotting import figure, output_file, show
import bokeh

# synwrapper class :
# ** TODO height->ampli
# ** TODO printing of what happened and what could happen

# Define synwrapper: A wrapper for synaptic event attributes (eg height, latency) from one dataset.
class synwrapper(object):

# ** init :
    def __init__(self):
        pass

# ** add_stim_on :
    def add_stim_on(self,
                    event_type='stim',
                    stim_thresh=2,
                    spont_filtsize=1001,
                    spont_threshampli=3,
                    spont_threshderiv=-1.2,
                    spont_thresh_pos_deriv=0.1,
                    savgol_polynomial=3):

        analogs = self.analog_signals
        stims = self.stim_signals

        if event_type == 'stim':
            stim_on = find_stims(analogs, stims, thresh=stim_thresh)
        elif event_type == 'spontaneous':
            stim_on = find_spontaneous(analogs,
                                       filt_size=spont_filtsize,
                                       thresh_ampli=spont_threshampli,
                                       thresh_deriv=spont_threshderiv,
                                       thresh_pos_deriv=spont_thresh_pos_deriv,
                                       savgol_polynomial=savgol_polynomial)

        self.stim_on = stim_on
        print('\nAdded stim_on (event_type = ', event_type, ')')

# ** add_ampli :
    def add_ampli(self, event_direction = 'up',
                  baseline_lower = 4, baseline_upper = 0.2,
                  PSE_search_lower = 5, PSE_search_upper = 30,
                  smoothing_width = False, latency_method = 'max_height'):

        # *** setup :
        #Load variables from synappy
        analog_signals = self.analog_signals
        stim_on = self.stim_on
        times = self.times

        num_neurons = len(analog_signals)


        #Define new vars to store event amplitude and latency
        pse_base = np.empty(num_neurons, dtype = np.ndarray)
        pse_ampli = np.empty(num_neurons, dtype = np.ndarray)         #postsynaptic_event stores ampli for each event
        pse_latcy = np.empty(num_neurons, dtype = np.ndarray)  #postsynaptic_event_latency stores latency data for each event


        #Determine direction of postsynaptic events
        if event_direction == 'up':
            event_direction = 1
        if event_direction == 'down':
            event_direction = -1

        # *** extract amplis :
        ##-----Iterate through neurons to extract event statistics-----#
        for neuron in range(num_neurons):
            #Define vars exclusive for this neuron
            num_trials = len(analog_signals[neuron])
            sample_rate = np.int32(np.round(1 / (times[neuron][1] - times[neuron][0])))

            #For this neuron, compute sample-rate-dependent variables: smoothing width,
            #baseline width indexes.
            if smoothing_width == False:
                smoothing_width = 2
                savgol_width = np.int32(smoothing_width * sample_rate / 1000) + 1
            else:
                smoothing_width_ind = np.int32(smoothing_width * (sample_rate / 1000)) + 1

            abs_base_lower = np.int32(baseline_lower * sample_rate / 1000)
            abs_base_upper = np.int32(baseline_upper * sample_rate / 1000)
            abs_pse_lower = np.int32(PSE_search_lower * sample_rate / 1000)
            abs_pse_upper = np.int32(PSE_search_upper * sample_rate / 1000)

            max_stims = 0
            for trial in range(num_trials):
                if len(stim_on[neuron][trial]) > max_stims:
                    max_stims = len(stim_on[neuron][trial])

            pse_base[neuron] = np.empty([num_trials, max_stims, 2], dtype = np.ndarray)
            pse_ampli[neuron] = np.empty([num_trials, max_stims, 4], dtype = np.ndarray)
            pse_latcy[neuron] = np.empty([num_trials, max_stims, 2], dtype = np.ndarray)


            for trial in range(num_trials):
                num_stims = len(stim_on[neuron][trial])

                to_mask = np.arange(max_stims-1, num_stims-1, -1)
                pse_base[neuron][trial,to_mask] = 0
                pse_ampli[neuron][trial, to_mask] = 0
                pse_latcy[neuron][trial, to_mask] = 0

                for stim in range(num_stims):

                    # Calculate inds for this instance
                    _base_lower = np.int32(
                        stim_on[neuron][trial][stim] - abs_base_lower)
                    _base_upper = np.int32(
                        stim_on[neuron][trial][stim] - abs_base_upper)
                    _pse_lower = np.int32(
                        stim_on[neuron][trial][stim] + abs_pse_lower)
                    _pse_upper = np.int32(
                        stim_on[neuron][trial][stim] + abs_pse_upper)

                    # ------------------------------
                    # Calculate event baseline
                    pse_base[neuron][trial, stim, 0] = np.mean(
                        analog_signals[neuron][
                            trial, _base_lower:
                            _base_upper])
                    pse_base[neuron][trial, stim, 1] = np.std(
                        analog_signals[neuron][
                            trial, _base_lower:
                            _base_upper])

                    # -------------------------------
                    # Calculate event amplitude
                    _analog = analog_signals[neuron][
                        trial, _pse_lower:_pse_upper]
                    _analog_sm = sp_signal.savgol_filter(
                        _analog, savgol_width, 3)

                    # calculate max event ampli [stim,0] and its index [stim,1]
                    if event_direction == 1:
                        pse_ampli[neuron][trial, stim, 1] \
                            = np.argmax(_analog_sm, axis=-1)
                    elif event_direction == -1:
                        pse_ampli[neuron][trial, stim, 1] \
                            = np.argmin(_analog_sm, axis=-1)

                    # correct index back to analog_signal reference
                    pse_ampli[neuron][trial, stim, 1] \
                        += _pse_lower
                    pse_ampli[neuron][trial, stim, 0] \
                        = analog_signals[neuron][
                            trial, np.int32(
                                pse_ampli[neuron][trial, stim, 1])]
                    pse_ampli[neuron][trial, stim, 0] \
                        -= pse_base[neuron][trial, stim, 0]
                    # store time of max_height latency in [stim,2]
                    pse_ampli[neuron][trial, stim, 2] \
                        = times[neuron][np.int32(
                            pse_ampli[neuron][trial][stim, 1])] \
                            - times[neuron][
                                np.int32(stim_on[neuron][trial][stim])]

                    # ------------------------------
                    # Calculate event onset latencies
                    max_height_smoothed_ind = np.int32(
                        pse_ampli[neuron][trial, stim, 1]
                        - _pse_lower)

                    if max_height_smoothed_ind < 2:
                        max_height_smoothed_ind = 2

                    _analog_sm_deriv = np.gradient(
                        _analog_sm[0:max_height_smoothed_ind])

                    if event_direction == 1:
                        max_deriv_ind = np.argmax(_analog_sm_deriv)
                        pse_ampli[neuron][trial, stim, 3] \
                            = _analog_sm_deriv[max_deriv_ind] \
                            * (sample_rate/1000)
                    elif event_direction == -1:
                        max_deriv_ind = np.argmin(_analog_sm_deriv)
                        pse_ampli[neuron][trial, stim, 3] \
                            = _analog_sm_deriv[max_deriv_ind] \
                            * (sample_rate/1000)

                    # determine latency and store in postsynaptic_event_latency
                    if latency_method == 'max_height':
                        event_time_index = np.int32(
                            pse_ampli[neuron][trial, stim, 1])
                        stim_time_index = np.int32(
                            stim_on[neuron][trial][stim])

                        pse_latcy[neuron][trial, stim, 0] \
                            = times[neuron][event_time_index] \
                            - times[neuron][stim_time_index]
                        pse_latcy[neuron][trial, stim, 1] \
                            = pse_ampli[neuron][trial, stim, 1]

                    elif latency_method == 'max_slope':
                        event_time_index = np.int32(
                            max_deriv_ind + _pse_lower)
                        stim_time_index = np.int32(
                            stim_on[neuron][trial][stim])

                        pse_latcy[neuron][trial, stim, 0] \
                            = times[neuron][event_time_index] \
                            - times[neuron][stim_time_index]
                        pse_latcy[neuron][trial, stim, 1] \
                            = event_time_index

                    elif latency_method == 'baseline_plus_4sd':
                        _thresh = pse_base[neuron][trial, stim, 0] \
                            + 4 * pse_base[neuron][trial, stim, 1]

                        # store temp vars for troubleshooting
                        self._analog_sm_crop \
                            = _analog_sm[0:max_height_smoothed_ind]
                        self._analog_sm = _analog_sm
                        self._thresh = _thresh

                        _ind_1stcross = np.where(
                            _analog_sm > _thresh)[0][0]

                        pse_latcy[neuron][trial, stim, 0] \
                            = times[neuron][
                                _ind_1stcross + abs_pse_lower]
                        pse_latcy[neuron][trial, stim, 1] \
                            = _ind_1stcross + abs_pse_lower

                    elif latency_method == '80_20_line':
                        value_80pc = 0.8 * (pse_ampli[neuron][trial, stim, 0])\
                            + pse_base[neuron][trial, stim, 0]
                        value_20pc = 0.2 * (pse_ampli[neuron][trial, stim, 0])\
                            + pse_base[neuron][trial, stim, 0]
                        value_80pc_sizeanalog = value_80pc \
                            * np.ones(len(
                                _analog_sm[0:max_height_smoothed_ind]))
                        value_20pc_sizeanalog = value_20pc \
                            * np.ones(len(
                                _analog_sm[0:max_height_smoothed_ind]))
                        diff_80pc = (_analog[0:max_height_smoothed_ind]
                                     - value_80pc_sizeanalog) > 0
                        diff_20pc = (_analog[0:max_height_smoothed_ind]
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

                        vals_baseline = pse_base[neuron][trial, stim, 0] \
                            * np.ones(len(_analog_sm[0:ind_80cross]))
                        diff_sq_8020_line = (vals_baseline - vals_8020_line)**2 \
                            + (_analog[0:ind_80cross] - vals_8020_line)**2

                        intercept_8020_ind = np.argmin(diff_sq_8020_line)

                        event_time_index = intercept_8020_ind \
                            + _pse_lower
                        stim_time_index = stim_on[neuron][trial][stim]
                        pse_latcy[neuron][trial,stim,0] = \
                            times[neuron][event_time_index] \
                            - times[neuron][stim_time_index]
                        pse_latcy[neuron][trial,stim,1] = event_time_index

        self.height = pse_ampli
        self.latency = pse_latcy
        self.baseline = pse_base

        print('\nAdded height. \nAdded latency. \nAdded baseline.')

        return


    #-----------Add success/fail sorting to neuron--------#
    #This allows either a dynamic threshold, 3*S.D. (thresh = false);
    #or it allows a user-specified threshold which can be constant for all
    #neurons (type(thresh) = float) or can be specified for each neuron
    #(len(thresh) = num_neurons)
    def add_sorting(self, thresh = False, thresh_dir = False):
        postsynaptic_event = self.height
        postsynaptic_event_latency = self.latency
        baseline = self.baseline

        num_neurons = len(postsynaptic_event)
        thresh = np.array(thresh, dtype = np.float)

        self.mask = np.empty(num_neurons, dtype = np.ndarray)


        postsynaptic_event_successes = np.empty(num_neurons, dtype = np.ndarray)
        postsynaptic_event_latency_successes = np.empty(num_neurons, dtype = np.ndarray)

        dynamic_thresholding = False
        if type(thresh) is bool and thresh == False:
            dynamic_thresholding = True
        elif len(np.atleast_1d(thresh)) is not num_neurons and len(np.atleast_1d(thresh)) == 1:
            thresh = np.ones(num_neurons) * thresh

#
        for neuron in range(num_neurons):
            postsynaptic_event_successes[neuron] = np.ma.array(np.copy(postsynaptic_event[neuron]))
            postsynaptic_event_latency_successes[neuron] = np.ma.array(np.copy(postsynaptic_event_latency[neuron]))

            height_tocompare = np.abs(postsynaptic_event[neuron][:, :, 0])

            if dynamic_thresholding is True:
                thresh_tocompare = 4 * baseline[neuron][:,:,1]
            else:
                thresh_thisneuron = thresh[neuron]
                thresh_tocompare = np.abs(thresh_thisneuron) * np.ones_like(baseline[neuron][:,:,1])

            diff_tocompare = height_tocompare - thresh_tocompare
            if thresh_dir is False:
                mask_tocompare = diff_tocompare < 0
            elif thresh_dir is True:
                mask_tocompare = diff_tocompare > 0

            mask_tocompare_full_pes = np.ma.empty([mask_tocompare.shape[0], mask_tocompare.shape[1], postsynaptic_event[neuron].shape[2]])
            for shape_3d in range(postsynaptic_event[neuron].shape[2]):
                mask_tocompare_full_pes[:,:,shape_3d] = mask_tocompare

            mask_tocompare_full_lat = np.ma.empty([mask_tocompare.shape[0], mask_tocompare.shape[1], postsynaptic_event_latency[neuron].shape[2]])
            for shape_3d in range(postsynaptic_event_latency[neuron].shape[2]):
                mask_tocompare_full_lat[:,:,shape_3d] = mask_tocompare


            postsynaptic_event_successes[neuron].mask = mask_tocompare_full_pes
            postsynaptic_event_latency_successes[neuron].mask = mask_tocompare_full_lat

            self.height[neuron] = np.ma.masked_array(self.height[neuron], mask = mask_tocompare_full_pes)
            self.latency[neuron] = np.ma.masked_array(self.latency[neuron], mask = mask_tocompare_full_pes[:,:,0:2])
            self.mask[neuron] = mask_tocompare_full_pes[:,:,0]
            self.baseline[neuron] = np.ma.masked_array(self.baseline[neuron], mask = mask_tocompare_full_pes[:,:,0:2])

        print('\nAdded succ/fail sorting to: \n\theight \n\tlatency, \n\tbaseline')

        return


    def add_invertedsort(self):
        postsynaptic_event_successes = self.height
        num_neurons = len(postsynaptic_event_successes)

        self.height_fails = np.empty(num_neurons, dtype = np.ndarray)


        postsynaptic_event_failures = np.empty(num_neurons, dtype = np.ndarray)

        for neuron in range(num_neurons):
            success_mask = np.ma.getmask(postsynaptic_event_successes[neuron])
            failure_mask = np.logical_not(success_mask)

            postsynaptic_event_failures[neuron] = np.ma.array(np.copy(postsynaptic_event_successes[neuron]), mask = failure_mask)
            self.height_fails[neuron] = np.ma.array(np.copy(postsynaptic_event_successes[neuron]), mask = failure_mask)

        #print('\nAdded height_fails.')

        return

    def add_normalized(self):
         postsynaptic_event = self.height

         num_neurons = len(postsynaptic_event)

         postsynaptic_event_normalized = np.empty(num_neurons, dtype = np.ndarray)
         self.height_norm = np.empty(num_neurons, dtype = np.ndarray)
         avg_ampl = np.empty(num_neurons)

         for neuron in range(num_neurons):
             postsynaptic_event_normalized[neuron] = np.ma.copy(postsynaptic_event[neuron])
             self.height_norm[neuron] = np.ma.copy(postsynaptic_event[neuron])

             avg_ampl = np.mean(postsynaptic_event[neuron][:,0,0])

             if type(postsynaptic_event[neuron]) is np.ma.core.MaskedArray:
                 current_neuron_mask = postsynaptic_event[neuron][:,:,0].mask

                 postsynaptic_event_normalized_temp = np.array(postsynaptic_event[neuron][:,:,0]) / avg_ampl
                 postsynaptic_event_normalized[neuron][:,:,0] = np.ma.array(postsynaptic_event_normalized_temp, mask = current_neuron_mask)
             else:
                 postsynaptic_event_normalized[neuron][:,:,0] /= avg_ampl

             self.height_norm[neuron] = postsynaptic_event_normalized[neuron]

         print('\nAdded height_norm.')


         return


    def pool(self, name, pool_index = 0, mask = 'suc'):
        if type(name) is str:
            postsynaptic_event = self.__getattribute__(name)
        else:
            postsynaptic_event = name


        num_neurons = len(postsynaptic_event)

        #Calculate min stims:
        common_stims = 10000
        for neuron in range(num_neurons):
            if common_stims > postsynaptic_event[neuron][:,:,:].shape[1] :
                common_stims = postsynaptic_event[neuron][:,:,:].shape[1]

        #Pool data
        stimpooled_postsynaptic_events = np.ma.array(np.empty([common_stims,0]))

        for neuron in range(num_neurons):
            if mask == 'suc':
                stimpooled_postsynaptic_events = np.ma.append(stimpooled_postsynaptic_events, np.ma.transpose(postsynaptic_event[neuron][:,0:common_stims, pool_index]), axis = 1)
            elif mask == 'all':
                stimpooled_postsynaptic_events = np.ma.append(stimpooled_postsynaptic_events, np.ma.transpose(postsynaptic_event[neuron][:,0:common_stims, pool_index].data), axis = 1)


    def add_sucrate(self, byneuron = False):
        postsynaptic_event = self.mask

        num_neurons = len(postsynaptic_event)

        #Calculate min stims:
        common_stims = 10000
        for neuron in range(num_neurons):
            if common_stims > len(postsynaptic_event[neuron][0,:,0]):
                common_stims = len(postsynaptic_event[neuron][0,:,0])

        success_rate_neur = np.zeros([common_stims, num_neurons])
        success_rate = np.zeros([common_stims, 3])

        for neuron in range(num_neurons):
            count_fails_temp = np.sum(postsynaptic_event[neuron][:,0:common_stims,0].mask, axis = 0)
            count_total_temp = postsynaptic_event[neuron].shape[0]
            success_rate_neur[:, neuron] = (count_total_temp - count_fails_temp) / count_total_temp

        success_rate[:,0] = np.mean(success_rate_neur, axis = 1)
        success_rate[:,1] = np.std(success_rate_neur, axis = 1)
        success_rate[:,2] = np.std(success_rate_neur, axis = 1) / np.sqrt(np.sum(success_rate_neur, axis = 1))


        if byneuron is True:
            success_rate = []
            success_rate = success_rate_neur

        self.success_rate = success_rate
        #print('\nAdded success_rate.')




        return self

    def special_sucrate(self, byneuron = False):
        mask = self.mask

        num_neurons = len(mask)

        #Calculate min stims:
        common_stims = 10000
        for neuron in range(num_neurons):
            if common_stims > len(mask[neuron][0,:]):
                common_stims = len(mask[neuron][0,:])

        success_rate_neur = np.zeros([common_stims, num_neurons])
        success_rate = np.zeros([common_stims, 3])

        for neuron in range(num_neurons):
            count_fails_temp = np.sum(mask[neuron][:,0:common_stims], axis = 0)
            count_total_temp = mask[neuron].shape[0]
            success_rate_neur[:, neuron] = (count_total_temp - count_fails_temp) / count_total_temp

        success_rate[:,0] = np.mean(success_rate_neur, axis = 1)
        success_rate[:,1] = np.std(success_rate_neur, axis = 1)
        success_rate[:,2] = np.std(success_rate_neur, axis = 1) / np.sqrt(np.sum(success_rate_neur, axis = 1))


        if byneuron is True:
            success_rate = []
            success_rate = success_rate_neur

        self.success_rate = success_rate
        #print('\nAdded success_rate.')




        return success_rate

    ###Define function ephys_summarystats which takes a postsynaptic event and outputs (mean, sd, sterr, numevents)
    def get_stats(self, arr_name, pooling_index = 0, mask = 'suc'):
        postsynaptic_event = self.__getattribute__(arr_name)

        stimpooled_postsynaptic_events = self.pool(postsynaptic_event, pooling_index, mask = mask)
        if type(postsynaptic_event[0]) is np.ma.core.MaskedArray:
            success_rate = self.special_sucrate(self)
            num_stims = len(stimpooled_postsynaptic_events)
            stats_postsynaptic_events = np.zeros([num_stims, 5])

            for stim in range(num_stims):
                num_nonmasked_stims = len(stimpooled_postsynaptic_events[stim,:]) - np.ma.count_masked(stimpooled_postsynaptic_events[stim,:])

                stats_postsynaptic_events[stim,0] = np.mean(stimpooled_postsynaptic_events[stim, :])
                stats_postsynaptic_events[stim,1] = np.std(stimpooled_postsynaptic_events[stim, :])
                stats_postsynaptic_events[stim,2] = np.std(stimpooled_postsynaptic_events[stim, :]) / np.sqrt(num_nonmasked_stims)
                stats_postsynaptic_events[stim,3] = success_rate[stim, 0]
                stats_postsynaptic_events[stim,4] = success_rate[stim, 1]
        else:
            num_stims = len(stimpooled_postsynaptic_events)
            stats_postsynaptic_events = np.zeros([num_stims, 3])

            for stim in range(num_stims):
                num_nonmasked_stims = len(stimpooled_postsynaptic_events[stim,:]) - np.ma.count_masked(stimpooled_postsynaptic_events[stim,:])

                stats_postsynaptic_events[stim,0] = np.mean(stimpooled_postsynaptic_events[stim, :])
                stats_postsynaptic_events[stim,1] = np.std(stimpooled_postsynaptic_events[stim, :])
                stats_postsynaptic_events[stim,2] = np.std(stimpooled_postsynaptic_events[stim, :]) / np.sqrt(num_nonmasked_stims)


        return (stats_postsynaptic_events)

    def get_median_filtered(signal, threshold=3):

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

            signal = np.ma.array(signal, mask = combined_mask_2)

        return signal

    def propagate_mask(self):
        postsynaptic_events = self.height
        mask = self.mask
        num_neurons = len(postsynaptic_events)
        for neuron in range(num_neurons):
            for lastind in range(postsynaptic_events[neuron].shape[2]):
                self.height[neuron].mask[:, :, lastind] = mask[neuron]
                self.height_norm[neuron].mask[:, :, lastind] = mask[neuron]
                self.height_fails[neuron].mask[:, :, lastind] = ~mask[neuron]

                if lastind < self.latency[neuron].shape[2]:
                    self.latency[neuron].mask[:, :, lastind] = mask[neuron]
                if lastind < self.baseline[neuron].shape[2]:
                    self.baseline[neuron].mask[:, :, lastind] = mask[neuron]
                if lastind < self.decay[neuron].shape[2]:
                    self.decay[neuron].mask[:, :, lastind] = mask[neuron]

    def remove_unclamped_aps(self, thresh_ratio = 5):
        height_norm = self.height_norm
        height = self.height
        latency = self.latency

        num_neurons = len(height_norm)

        for neuron in range(num_neurons):
            to_replace = np.argwhere(height_norm[neuron][:,:,0] > thresh_ratio)

            height_norm[neuron][to_replace[:, 0], to_replace[:, 1] ,:] = np.nan
            height_norm[neuron].mask[to_replace[:, 0], to_replace[:, 1] ,:] = True

            height[neuron][to_replace[:, 0], to_replace[:, 1] ,:] = np.nan
            height[neuron].mask[to_replace[:, 0], to_replace[:, 1] ,:] = True

            latency[neuron][to_replace[:, 0], to_replace[:, 1] ,:] = np.nan
            latency[neuron].mask[to_replace[:, 0], to_replace[:, 1] ,:] = True

            self.mask[neuron] = np.ma.mask_or(height_norm[neuron][:,:,0].mask, self.height[neuron][:,:,0].mask)

        self.propagate_mask()

        print('\nMasked APs')


    def add_decays(self, prestim = 0, poststim = 10, plotting = False, fn = 'monoexp_normalized_plusb', update_mask = False):
        '''

        '''

        ##------SETUP------#
        #Import variables from synappy wrapper
        analog_signals = self.analog_signals
        postsynaptic_events = self.height
        baseline = self.baseline
        times = self.times

        num_neurons = len(postsynaptic_events)

        def biexp_normalized_plusb(time_x, lambda1, lambda2, vstart2, b):
            y = np.exp(time_x * (-1) * lambda1) + vstart2 * np.exp(time_x * (-1) * lambda2) + b#+  np.exp(time_x * (-1) * lambda2)
            return y

        def monoexp_normalized_plusb(time_x, lambda1, b):
            y = np.exp(time_x * (-1) * lambda1) + b #+  np.exp(time_x * (-1) * lambda2)
            return y

        num_vars = 2

        fitted_vars = np.empty(num_neurons, dtype = np.ndarray)
        fitted_covari = np.empty_like(fitted_vars)

        for neuron in range(num_neurons):
            sample_rate = np.int32(np.round(1 / (times[neuron][1] - times[neuron][0])))
            poststim_ind = np.int32(poststim * sample_rate / 1000)


            num_trials = postsynaptic_events[neuron].shape[0]
            num_stims = postsynaptic_events[neuron].shape[1]

            if type(postsynaptic_events[neuron]) is np.ma.core.MaskedArray:
                fitted_vars[neuron] = np.ma.array(np.empty([num_trials, num_stims, num_vars], dtype = np.ndarray))
                fitted_vars[neuron].mask = postsynaptic_events[neuron].mask
                fitted_covari[neuron] = np.ma.array(np.empty([num_trials, num_stims, num_vars], dtype = np.ndarray))
                fitted_covari[neuron].mask = postsynaptic_events[neuron].mask

            else:
                fitted_vars[neuron] = np.empty([num_trials, num_stims, num_vars], dtype = np.ndarray)
                fitted_covari[neuron] = np.empty([num_trials, num_stims, num_vars], dtype = np.ndarray)


            for trial in range(num_trials):
                for stim in range(num_stims):

                    if type(postsynaptic_events[neuron]) is np.ma.core.MaskedArray and  postsynaptic_events[neuron][trial, stim, 0] is not np.ma.masked:
                        event_ind_min = postsynaptic_events[neuron][trial,stim,1] - prestim
                        event_ind_max = event_ind_min + poststim_ind

                        postsynaptic_curve = analog_signals[neuron][trial, event_ind_min : event_ind_max] - baseline[neuron][trial, stim, 0]
                        postsynaptic_curve /= np.mean(postsynaptic_curve[0:2])

                        vars_guess = [100, 0]
                        times_forfit = times[neuron][0:poststim_ind]
                        try:
                            [popt, pcov] = sp_opt.curve_fit(monoexp_normalized_plusb, times_forfit, postsynaptic_curve, p0 = vars_guess) #Dfun = jacob_exp)#, maxfev = 250) #p0 = [100, postsynaptic_events[neuron][trial,stim,0]]) #0.5*postsynaptic_events[neuron][trial,stim,0]])
                        except RuntimeError:
                            popt = np.ones(num_vars) * 10000
                            pcov = 10000
                        except ValueError:
                            print(postsynaptic_curve, 'neuron: ', neuron, 'trial: ', trial, 'stim: ', stim)
                        fitted_vars[neuron][trial, stim, :] = popt[:]
                    elif type(postsynaptic_events[neuron]) is np.ma.core.MaskedArray and postsynaptic_events[neuron][trial, stim, 0] is np.ma.masked:
                        fitted_vars[neuron][trial, stim, :] = np.ones(num_vars) * 10000
                        fitted_vars[neuron][trial, stim, :].mask = np.ones(num_vars, dtype = np.bool)
                    elif type(postsynaptic_events[neuron]) is not np.ma.core.MaskedArray:
                        event_ind_min = postsynaptic_events[neuron][trial,stim,1] - prestim
                        event_ind_max = event_ind_min + poststim_ind

                        postsynaptic_curve = analog_signals[neuron][trial, event_ind_min : event_ind_max] - baseline[neuron][trial, stim, 0]
                        postsynaptic_curve /= postsynaptic_curve[0]

                        vars_guess = [100, 0]
                        times_forfit = times[neuron][0:poststim_ind]
                        try:
                            [popt, pcov] = sp_opt.curve_fit(monoexp_normalized_plusb, times_forfit, postsynaptic_curve, p0 = vars_guess) #, Dfun = jacob_exp)#, maxfev = 250) #p0 = [100, postsynaptic_events[neuron][trial,stim,0]]) #0.5*postsynaptic_events[neuron][trial,stim,0]])
                        except RuntimeError:
                            popt = np.ones(num_vars) * 10000
                            #pcov = 10000


                        fitted_vars[neuron][trial,stim, :] = popt[:]


            if plotting is True:
                """Specify plotting type based on whether masking has been performed or not.
                """

                if type(postsynaptic_events[neuron]) is np.ma.core.MaskedArray:
                    first_nonmasked_trial = np.argwhere(postsynaptic_events[neuron][:,0,1].mask == False)[0][0]

                    postsynaptic_curve = analog_signals[neuron][first_nonmasked_trial, postsynaptic_events[neuron][first_nonmasked_trial,0,1] - prestim : postsynaptic_events[neuron][first_nonmasked_trial,0,1] + poststim_ind] - baseline[neuron][first_nonmasked_trial, 0, 0]
                    y_fitted = postsynaptic_events[neuron][first_nonmasked_trial,0,0] * monoexp_normalized_plusb(times[neuron][0:poststim_ind + prestim], fitted_vars[neuron][first_nonmasked_trial,0,0], fitted_vars[neuron][first_nonmasked_trial,0,1])
                    plt.figure()
                    plt.plot(times[neuron][0:poststim_ind + prestim], y_fitted, 'r')
                    plt.plot(times[neuron][0:poststim_ind + prestim], postsynaptic_curve, 'b')



                elif type(postsynaptic_events[neuron]) is np.array:
                    postsynaptic_curve = analog_signals[neuron][0, postsynaptic_events[neuron][0,0,1] - prestim : postsynaptic_events[neuron][0,0,1] + poststim_ind] - baseline[neuron][0, 0, 0]
                    y_fitted = postsynaptic_events[neuron][0,0,0] * monoexp_normalized_plusb(times[neuron][0:poststim_ind + prestim], fitted_vars[neuron][0,0,0], fitted_vars[neuron][0,0,1])
                    plt.figure()
                    plt.plot(times[neuron][0:poststim_ind + prestim], y_fitted, 'r')
                    plt.plot(times[neuron][0:poststim_ind + prestim], postsynaptic_curve, 'b')

            #convert from lambda to tau
            fitted_ones = np.ones([fitted_vars[neuron][:,:,0].shape[0], fitted_vars[neuron][:,:,0].shape[1]])

            if type(postsynaptic_events[neuron]) is np.ma.core.MaskedArray:
                fitted_vars[neuron][:,:,0] = fitted_ones / np.array(fitted_vars[neuron][:,:,0])
                fitted_vars[neuron][:,:,0] = get_median_filtered(fitted_vars[neuron][:,:,0], threshold=10)
                fittedvarmask = np.ma.mask_or(postsynaptic_events[neuron][:,:,0].mask, fitted_vars[neuron][:,:,0].mask)
                fitted_vars[neuron][:,:,0].mask = fittedvarmask

            else:
                fitted_vars[neuron][:,:,0] = fitted_ones / fitted_vars[neuron][:,:,0]
                fitted_vars[neuron][:,:,0] = get_median_filtered(fitted_vars[neuron][:,:,0], threshold=10)

            if update_mask is True and type(postsynaptic_events[neuron]) is np.ma.core.MaskedArray:
                self.mask[neuron] = np.ma.mask_or(postsynaptic_events[neuron][:,:,0].mask, fitted_vars[neuron][:,:,0].mask)
        self.decay = fitted_vars

        print('\nAdded decay')

        if update_mask is True and type(postsynaptic_events[neuron]) is np.ma.core.MaskedArray:
            self.propagate_mask()
            print('\n\nAdded round2 of succ/fail sorting to: \n\theight \n\theight_norm, \n\tlatency, \n\tbaseline')

        return

    #Add an integral (area under curve) from each event time. Length is the time post-event to integrate
    #until, in milliseconds.
    #
    #Dynamic length: Scale integration length dynamically based on the fitted EPSP decay (4 * decay)
    def add_integral(self, length = 1000, dynamic_length = False, cdf_bins = 100):
        #Import needed vars from self
        analog_signals = self.analog_signals
        postsynaptic_events = self.height
        baseline = self.baseline
        times = self.times
        stim_on = self.stim_on

        #Define number of neurons
        num_neurons = len(analog_signals)

        #Predefine vars
        integral = np.empty(num_neurons, dtype = np.ndarray)
        cdf_integral = np.empty(num_neurons, dtype = np.ndarray)


        #Condition if dynamic length is not selected.
        if dynamic_length is False:

            #Iterate through neurons, trials and stims
            for neuron in range(num_neurons):
                sample_rate = np.int32(np.round(1 / (times[neuron][1] - times[neuron][0])))

                #Calculate number of trials and fill in integral
                num_trials = postsynaptic_events[neuron].shape[0]
                num_stims = postsynaptic_events[neuron].shape[1]
                integral[neuron] = np.zeros([num_trials, num_stims])
                cdf_integral[neuron] = np.zeros([num_trials, num_stims, int(cdf_bins)])

                for trial in range(num_trials):
                    for stim in range(num_stims):
                        #In each instance: Find the required stretch of analog_signals to integrate
                        #over, and the corresponding times, then integrate over this range and store
                        #it in integral.
                        int_start = int(stim_on[neuron][stim])
                        int_end = int((length/1000) * sample_rate + int_start)

                        analog_toint = analog_signals[neuron][trial, int_start : int_end] - baseline[neuron][trial, stim, 0]
                        time_toint = times[neuron][int_start : int_end]
                        integral[neuron][trial] = sp_integrate.trapz(analog_toint, time_toint)

                        #Calculate cumulative distribution of integral in 100 bins
                        for nbin in range(cdf_bins):
                            nbin_plusone = nbin + 1
                            curr_cdf_fraction = nbin_plusone / cdf_bins

                            int_start = int(stim_on[neuron][stim])
                            int_end = int(int_start + ((length/1000) * sample_rate) * curr_cdf_fraction)

                            analog_toint = analog_signals[neuron][trial, int_start : int_end] - baseline[neuron][trial, stim, 0]
                            time_toint = times[neuron][int_start : int_end]

                            cdf_integral[neuron][trial, stim, nbin] = sp_integrate.trapz(analog_toint, time_toint) / integral[neuron][trial, stim]


        self.integral = integral
        self.cdf_integral = cdf_integral
        print('\nAdded integral')

        return

    
    def add_norm_integral(self, length=1000, dynamic_length = False, cdf_bins=100):
        # Import needed vars from self
        analog_signals = self.analog_signals
        postsynaptic_events = self.height
        baseline = self.baseline
        times = self.times
        stim_on = self.stim_on

        #Define number of neurons
        num_neurons = len(analog_signals)

        #Predefine vars
        integral = np.empty(num_neurons, dtype = np.ndarray)
        cdf_integral = np.empty(num_neurons, dtype = np.ndarray)


        #Condition if dynamic length is not selected.
        if dynamic_length is False:

            #Iterate through neurons, trials and stims
            for neuron in range(num_neurons):
                sample_rate = np.int32(np.round(1 / (times[neuron][1] - times[neuron][0])))

                #Calculate number of trials and fill in integral
                num_trials = postsynaptic_events[neuron].shape[0]
                num_stims = postsynaptic_events[neuron].shape[1]
                integral[neuron] = np.zeros([num_trials, num_stims])
                cdf_integral[neuron] = np.zeros([num_trials, num_stims, int(cdf_bins)])

                for trial in range(num_trials):
                    for stim in range(num_stims):
                        #In each instance: Find the required stretch of analog_signals to integrate
                        #over, and the corresponding times, then integrate over this range and store
                        #it in integral.
                        int_start = int(stim_on[neuron][stim])
                        int_end = int((length/1000) * sample_rate + int_start)

                        analog_toint = (analog_signals[neuron][trial, int_start:int_end]
                                        - baseline[neuron][trial, stim, 0]) \
                                        / postsynaptic_events[neuron][trial, stim, 0]
                        time_toint = times[neuron][int_start : int_end]
                        integral[neuron][trial] = sp_integrate.trapz(analog_toint, time_toint)

                        #Calculate cumulative distribution of integral in 100 bins
                        for nbin in range(cdf_bins):
                            nbin_plusone = nbin + 1
                            curr_cdf_fraction = nbin_plusone / cdf_bins

                            int_start = int(stim_on[neuron][stim])
                            int_end = int(int_start + ((length/1000) * sample_rate) * curr_cdf_fraction)

                            analog_toint = analog_signals[neuron][trial, int_start : int_end] - baseline[neuron][trial, stim, 0]
                            time_toint = times[neuron][int_start : int_end]

                            cdf_integral[neuron][trial, stim, nbin] = sp_integrate.trapz(analog_toint, time_toint) / integral[neuron][trial, stim]


        self.norm_integral = integral
        self.norm_cdf_integral = cdf_integral
        print('\nAdded norm. integral')

        return


    def add_all(self, event_direction = 'down', latency_method = 'max_height', ap_filter = True, sort_thresh = False,
                keep_above = False, delay_mask_update = True, upper_search = 30, decay_fitsize = 1000, plot_decays = False):
        self.add_ampli(event_direction = event_direction, latency_method = latency_method, PSE_search_upper = upper_search)
        self.add_sorting(thresh = sort_thresh, thresh_dir = keep_above)
        self.add_normalized()
        self.add_invertedsort()
        self.add_norm_integral()
        self.add_decays(update_mask = delay_mask_update, poststim = decay_fitsize, plotting = plot_decays)
        if ap_filter is True:
            self.remove_unclamped_aps()

    def plot(self, arr_name, xaxis = 'stim_num', yax = False, ylim = False, by_neuron = False,
             ind = 0, hist = False):

        arr = self.__getattribute__(arr_name)
        num_neurons = len(arr)

        if 'height' == arr_name:
            yax = 'Event amplitude (pA)'
        elif 'height_norm' == arr_name:
            yax = 'Normalized event amplitude'
        elif 'baseline' == arr_name:
            yax = 'Baseline holding current (pA)'
        elif 'decay' == arr_name:
            yax = 'tau (s)'
            hist = True
        elif 'latency' == arr_name:
            yax = 'Latency (s)'


        if yax  is False:
            yax = ' '

        if hist is False:

            for neuron in range(num_neurons):
                print('\nNeuron: ', neuron)
                x_1 = range(1, len(arr[neuron][0,:,0]) + 1)
                num_trials = len(arr[neuron][:,0,0])

                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                for trial in range(num_trials):
                    ratio_thistrial = trial / (num_trials)
                    red_thistrial = 1 / (1 + np.exp( -5 * (ratio_thistrial - 0.5)))
                    color_thistrial = [red_thistrial, 0.2, 1 - red_thistrial]
                    if type(arr[neuron]) is np.ma.core.MaskedArray:
                        ax.plot(x_1, arr[neuron][trial, :, ind].filled(np.nan),'.', color = color_thistrial, alpha = 0.6)
                        ma = arr[neuron][trial, :, ind].mask
                        inv_ma = np.logical_not(ma)
                        new_pse = np.ma.array(np.array(arr[neuron][trial,:,ind]), mask = inv_ma)
                        ax.plot(x_1, new_pse.filled(np.nan),'.', color = '0.7', alpha = 0.6)
                    else:
                        ax.plot(x_1, arr[neuron][trial,:,ind],'.', color = color_thistrial, alpha = 0.6)


                ax.set_xlabel('Stimulation number')
                ax.set_ylabel(yax)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticks_position('bottom')
                xlim_curr = ax.get_xlim()
                ylim_curr = ax.get_ylim()
                ax.set_xlim([xlim_curr[0], xlim_curr[1] + 1])
                if arr_name == 'latency' or arr_name == 'height_norm':
                    ax.set_ylim([0, ylim_curr[1]])




                if ylim is not False:
                    ax.set_ylim(ylim)
                #name = name_5ht + '_' + name_gaba + '_' name_freq + '.jpg'
                plt.show()

        elif hist is True:

            for neuron in range(num_neurons):

                print('\nNeuron: ', neuron)
                num_trials = len(arr[neuron][:,0,0])
                array_to_plot = arr[neuron][:, :, ind]

                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                if type(arr[neuron]) is np.ma.core.MaskedArray:
                    histpool_thisneuron = array_to_plot.compressed()
                    ax.hist(histpool_thisneuron, bins = 30, facecolor = [0.2, 0.4, 0.8], normed = True, alpha = 0.6, linewidth = 0.5)
                else:
                    histpool_thisneuron = array_to_plot.flatten()
                    ax.hist(histpool_thisneuron, n = 30, facecolor = [0.2, 0.4, 0.8], normed = True, alpha = 0.6, linewidth = 0.5)


                ax.set_xlabel('Decay (tau) (s)')
                ax.set_ylabel('Number')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticks_position('bottom')
                xlim_curr = ax.get_xlim()


                plt.show()


    def plot_corr(self, arr_name, arr_name2, ind1 = 0, ind2 = 0, xlabel = False, ylabel = False):
        arr1 = self.__getattribute__(arr_name)
        arr2 = self.__getattribute__(arr_name2)

        if xlabel is False:
            xlabel = arr_name
        if ylabel is False:
            ylabel = arr_name2

        num_neurons = len(arr1)

        for neuron in range(num_neurons):
            print('\nNeuron: ', neuron)

            plt.figure()
            num_trials = arr1[neuron].shape[0]
            for trial in range(num_trials):
                ratio_thistrial = trial / num_trials
                red_thistrial = 1 / (1 + np.exp( -5 * (ratio_thistrial - 0.5)))
                color_thistrial = [red_thistrial, 0.2, 1 - red_thistrial]

                if type(arr1[neuron]) is np.ma.core.MaskedArray or type(arr2[neuron]) is np.ma.core.MaskedArray:
                    plt.plot(arr1[neuron][trial,:,ind1].filled(np.nan), arr2[neuron][trial,:,ind2].filled(np.nan), '.', color = color_thistrial)

                    inv_ma = np.logical_not(arr1[neuron][trial, :, ind1].mask)
                    new_arr1 = np.ma.array(np.array(arr1[neuron][trial, :, ind1]), mask = inv_ma)
                    new_arr2 = np.ma.array(np.array(arr2[neuron][trial, :, ind2]), mask = inv_ma)
                    plt.plot(new_arr1.filled(np.nan), new_arr2.filled(np.nan),'.', color = '0.6')

                else:
                    plt.plot(arr1[neuron][trial,:,ind1], arr2[neuron][trial,:,ind2], '.r')

                plt.xlabel(arr_name)
                plt.ylabel(arr_name2)
            plt.show()




### ---- core accelerated operations on synwrapper classes ----
# - The idea is to take readable pandas DataFrame objects (from synwrapper), and use
#   these core functions to jit-accelerate the DataFrame processing and pass
#   them back to the synwrapper


#
#@jit
#def add_ampli_core(analog_signals, stim_on, times, event_direction = 'up',
#                                   baseline_lower = 4, baseline_upper = 0.2,
#                                   PSE_search_lower = 5, PSE_search_upper = 30,
#                                   smoothing_width = False, latency_method = 'max_height'):
#
#    #Determine direction of postsynaptic events
#    if event_direction is 'up' :
#        event_direction = 1
#    if event_direction is 'down':
#        event_direction = -1
#
#    #Initialize classes for storing new variables associated with postsynaptic events
#    num_neurons = len(analog_signals)
#    baseline = np.empty(num_neurons, dtype = np.ndarray)
#    postsynaptic_event = np.empty(num_neurons, dtype = pd.DataFrame)
#    postsynaptic_event_latency = np.empty(num_neurons, dtype = pd.DataFrame)
#
#
#    #Iterate through neurons
#    for neuron in range(num_neurons):
#        num_trials = len(analog_signals[neuron])    #Define size of trials and stims
#
#        for trial in range(num_trials):
#            num_stims = len(stim_on[neuron][trial])
#
#            #Find sampling rate and size of smoothing kernel
#            sample_rate = np.int32(np.round(1 / (times[neuron][1] - times[neuron][0])))
#            if smoothing_width == False:
#                smoothing_width = 2
#                smoothing_width_ind = np.int32(smoothing_width * sample_rate / 1000) + 1
#            else:
#                smoothing_width_ind = np.int32(smoothing_width * (sample_rate / 1000)) + 1
#
#
#            #convert baseline and PSE search bounds to indices
#            baseline_lower_index = np.int32(baseline_lower * sample_rate / 1000)
#            baseline_upper_index = np.int32(baseline_upper * sample_rate / 1000)
#
#            PSE_search_lower_index = np.int32(PSE_search_lower * sample_rate / 1000)
#            PSE_search_upper_index = np.int32(PSE_search_upper * sample_rate / 1000)
#
#            baseline[neuron] = np.empty([num_trials, num_stims, 2], dtype = np.ndarray)
#            postsynaptic_event[neuron] = np.empty([num_trials, num_stims, 4], dtype = np.ndarray)
#            postsynaptic_event_latency[neuron] = np.empty([num_trials, num_stims, 2], dtype = np.ndarray)
#
#
#            #postsynaptic_event[neuron][trial, stim, b],
#                #b = 0: index of max, b = 1: val of max, b = 2: normalized val of max
#                #b = 3: latency
#
#            for stim in range(num_stims):
#                baseline_lower_thistrial = np.int32(stim_on[neuron][stim] - baseline_lower_index)
#                baseline_upper_thistrial = np.int32(stim_on[neuron][stim] - baseline_upper_index)
#
#                PSE_search_lower_thistrial = np.int32(stim_on[neuron][stim] + PSE_search_lower_index)
#                PSE_search_upper_thistrial = np.int32(stim_on[neuron][stim] + PSE_search_upper_index)
#
#                #calculate mean baseline for this trial, stim. Store in baseline[neuron][trial][stim, 0]
#                #baseline[neuron][trial][stim, 1] stores stdev.
#                for trial in range(num_trials):
#
#                    baseline[neuron][trial, stim, 0] = np.mean(analog_signals[neuron][trial, baseline_lower_thistrial:baseline_upper_thistrial])
#                    baseline[neuron][trial, stim, 1] = np.std(analog_signals[neuron][trial, baseline_lower_thistrial:baseline_upper_thistrial])
#
#                #Use boxcar-moving-avg to smooth analog signal. Calculate this in analog_smoothed
#                #and the derivative in
#
#                analog_presmoothed_input = analog_signals[neuron][:, PSE_search_lower_thistrial:PSE_search_upper_thistrial]
#                analog_smoothed = sp_signal.savgol_filter(analog_presmoothed_input, smoothing_width_ind, 3)
#
#                #calculate max PSE height [stim,0] and its index [stim,1] for this trial, stim
#                if event_direction == 1:
#                    postsynaptic_event[neuron][:, stim, 1] = np.argmax(analog_smoothed, axis = -1)
#                elif event_direction == -1:
#                    postsynaptic_event[neuron][:, stim, 1] = np.argmin(analog_smoothed, axis = -1)
#                #correct index back to analog_signal reference
#                postsynaptic_event[neuron][:, stim, 1] += PSE_search_lower_thistrial
#                postsynaptic_event[neuron][:, stim, 0] = [analog_signals[neuron][i, np.int32(postsynaptic_event[neuron][i,stim,1])] for i in range(num_trials)]
#                postsynaptic_event[neuron][:, stim, 0] -=  baseline[neuron][:, stim, 0]      #correct EPSP val by subtracting baseline measurement
#                #store time of max_height latency in [stim,2]
#                postsynaptic_event[neuron][:, stim, 2] =  [times[neuron][np.int32(postsynaptic_event[neuron][trial][stim,1])] - times[neuron][np.int32(stim_on[neuron][stim])] for trial in range(num_trials)]
#
#                #derivative calcs. Go to trial indexing due to uneven size of arays from stim-on to max-height.
#                for trial in range(num_trials):
#                    max_height_smoothed_ind = np.int32(postsynaptic_event[neuron][trial,stim,1] - PSE_search_lower_thistrial)
#
#                    if max_height_smoothed_ind < 2:
#                        max_height_smoothed_ind = 2
#                    analog_smoothed_deriv = np.gradient(analog_smoothed[trial, 0:max_height_smoothed_ind])
#
#                    if event_direction == 1:
#                        max_deriv_ind = np.argmax(analog_smoothed_deriv)
#                        postsynaptic_event[neuron][:, stim, 3] = analog_smoothed_deriv[max_deriv_ind] * (sample_rate/1000)
#                    elif event_direction == -1:
#                        max_deriv_ind = np.argmin(analog_smoothed_deriv)
#                        postsynaptic_event[neuron][:, stim, 3] = analog_smoothed_deriv[max_deriv_ind] * (sample_rate/1000)
#
#
#                    #Based on latency_method, determine latency and store in postsynaptic_event_latency
#                    if latency_method == 'max_height':
#                        event_time_index = np.int32(postsynaptic_event[neuron][trial, stim, 1])
#                        stim_time_index = np.int32(stim_on[neuron][stim])
#
#                        postsynaptic_event_latency[neuron][trial, stim,0] =  times[neuron][event_time_index] - times[neuron][stim_time_index]
#                        postsynaptic_event_latency[neuron][trial, stim,1] = postsynaptic_event[neuron][trial, stim, 1]
#
#                    elif latency_method == 'max_slope':
#                        event_time_index = np.int32(max_deriv_ind + PSE_search_lower_thistrial)
#                        stim_time_index = np.int32(stim_on[neuron][stim])
#
#                        postsynaptic_event_latency[neuron][trial, stim, 0] = times[neuron][event_time_index] - times[neuron][stim_time_index]
#                        postsynaptic_event_latency[neuron][trial, stim, 1] = event_time_index
#
#                    elif latency_method == 'baseline_plus_4sd':
#                        signal_base_diff = ((analog_smoothed[trial,0:max_height_smoothed_ind] - (baseline[neuron][trial, stim, 0] + 4 * baseline[neuron][trial, stim, 1])) ** 2 ) > 0
#                        signal_base_min_ind = inifind_last(signal_base_diff, tofind = 0)
#                        postsynaptic_event_latency[neuron][trial, stim,0] = times[neuron][signal_base_min_ind + PSE_search_lower_index]
#                        postsynaptic_event_latency[neuron][trial, stim,1] = signal_base_min_ind + PSE_search_lower_index
#
#                    elif latency_method == '80_20_line':
#                        value_80pc = 0.8 * (postsynaptic_event[neuron][trial,stim,0]) + baseline[neuron][trial,stim,0]
#                        value_20pc = 0.2 * (postsynaptic_event[neuron][trial,stim,0]) + baseline[neuron][trial,stim,0]
#                        value_80pc_sizeanalog =  value_80pc * np.ones(len(analog_smoothed[trial, 0:max_height_smoothed_ind]))
#                        value_20pc_sizeanalog =  value_20pc * np.ones(len(analog_smoothed[trial, 0:max_height_smoothed_ind]))
#
#    #                        diff_80pc = (analog_smoothed[trial, 0:max_height_smoothed_ind] - value_80pc_sizeanalog) > 0
#    #                        diff_20pc = (analog_smoothed[trial, 0:max_height_smoothed_ind] - value_20pc_sizeanalog) > 0
#                        diff_80pc = (analog_presmoothed_input[trial, 0:max_height_smoothed_ind] - value_80pc_sizeanalog) > 0
#                        diff_20pc = (analog_presmoothed_input[trial, 0:max_height_smoothed_ind] - value_20pc_sizeanalog) > 0
#
#
#                        if event_direction is 1:
#                            ind_80cross = find_last(diff_80pc, tofind = 0)
#                            ind_20cross = find_last(diff_20pc, tofind = 0)
#                        elif event_direction is -1:
#                            ind_80cross = find_last(diff_80pc, tofind = 1)
#                            ind_20cross = find_last(diff_20pc, tofind = 1)
#
#                        if ind_20cross > ind_80cross or ind_80cross == 0:
#                            ind_80cross = np.int32(1)
#                            ind_20cross = np.int32(0)
#
#                        val_80cross = analog_smoothed[trial, ind_80cross]
#                        val_20cross = analog_smoothed[trial, ind_20cross]
#
#
#                        slope_8020_line = (val_80cross - val_20cross) / (ind_80cross - ind_20cross)
#
#                        vals_8020_line = np.zeros(len(analog_smoothed[trial, 0:ind_80cross + 1]))
#                        vals_8020_line = [(val_80cross - (ind_80cross - i)*slope_8020_line) for i in range(ind_80cross)]
#
#                        vals_baseline = baseline[neuron][trial,stim,0] * np.ones(len(analog_smoothed[trial, 0:ind_80cross]))
#                        #diff_sq_8020_line = (vals_baseline - vals_8020_line) ** 2 + (analog_smoothed[trial, 0:ind_80cross] - vals_8020_line) ** 2
#                        diff_sq_8020_line = (vals_baseline - vals_8020_line) ** 2 + (analog_presmoothed_input[trial, 0:ind_80cross] - vals_8020_line) ** 2
#
#                        intercept_8020_ind = np.argmin(diff_sq_8020_line)
#
#                        event_time_index = intercept_8020_ind + PSE_search_lower_thistrial
#                        stim_time_index = stim_on[neuron][stim]
#                        postsynaptic_event_latency[neuron][trial,stim,0] = times[neuron][event_time_index] - times[neuron][stim_time_index]
#                        postsynaptic_event_latency[neuron][trial,stim,1] = event_time_index
#
#    self.height =  postsynaptic_event
#    self.latency = postsynaptic_event_latency
#    self.baseline = baseline
##        self.upslope =
#
#    print('\nAdded height. \nAdded latency. \nAdded baseline.')
#
#
#    return


# * synappy functions :
### ---- synappy functions ----
###      functionality is syn.find_stims and syn.load.


# ** load
def load(files, trials=None, input_channel=None, stim_channel=None,
         stim_thresh=3, spontaneous=False, filt_size=1001,
         thresh_ampli=3, thresh_deriv=-1.2):

    print('\n\n----New Group---')

    num_neurons = len(files)
    neurons_range = np.int32(range(num_neurons))

    block = np.empty(num_neurons, dtype=np.ndarray)
    analog_signals = np.empty(num_neurons, dtype=np.ndarray)
    stim_signals = np.empty(num_neurons, dtype=np.ndarray)

    times = np.empty(num_neurons, dtype=np.ndarray)

    # block = [neo.AxonIO(filename=files[i]).read()[0] for i in neurons_range]
    block = [neo.AxonIO(filename=files[i]).read(
        signal_group_mode="split-all")[0] for i in neurons_range]

    #Check for presence of optional variables, create them if they don't exist
    if trials is None:
        trials = np.empty(num_neurons, dtype=np.ndarray)
        for neuron in range(num_neurons):
            trials[neuron] = np.array([1, len(block[neuron].segments)],
                                      dtype=np.int)

    if input_channel is None:
        input_channel = np.int8(np.zeros(num_neurons))
    elif type(input_channel) is int:
        input_channel = np.ones(num_neurons) * input_channel

    if stim_channel is None:
        stim_channel = np.int8((-1) * np.ones(num_neurons))
    elif type(stim_channel) is int:
        stim_channel = np.ones(num_neurons) * stim_channel

    #Populate analog_signals and times from raw data in block
    for neuron in range(num_neurons):
        num_trials = len(np.arange(trials[neuron][0], trials[neuron][1] + 1))
        numtimes_full = len(block[neuron].segments[0].analogsignals[0].times)
        numtimes_wanted = numtimes_full

        times[neuron] = np.linspace(
            block[neuron].segments[0].analogsignals[0].times[0].magnitude,
            block[neuron].segments[0].analogsignals[0].times[-1].magnitude,
            num=numtimes_wanted)

        analog_signals[neuron] = np.empty((num_trials, numtimes_wanted))
        stim_signals[neuron] = block[neuron].segments[0].analogsignals[np.int8(
            stim_channel[neuron])][:]

        for trial_index, trial_substance in enumerate(
                block[neuron].segments[trials[neuron][0] -
                                       1:trials[neuron][1]]):
            analog_signals[neuron][
                trial_index, :] = trial_substance.analogsignals[np.int8(
                    input_channel[neuron])][:].squeeze()

#    #Find stim onsets
#    if spontaneous is False:
#        stim_on = find_stims(stim_signals, stim_thresh)
#        for neuron in range(num_neurons):
#            #stim_on[neuron] /= downsampling_ratio
#            stim_on[neuron] = np.int32(stim_on[neuron])
#    elif spontaneous is True:
#        stim_on = find_spontaneous(analog_signals, filt_size = filt_size, thresh_ampli = thresh_ampli, thresh_deriv = thresh_deriv)
#
#    #str_name = 'postsynaptic_events_' + name
#
    synaptic_wrapper = synwrapper()
    synaptic_wrapper.analog_signals = analog_signals
    synaptic_wrapper.stim_signals = stim_signals
    synaptic_wrapper.times = times
    #synaptic_wrapper.name = str_name
    print('\nInitialized. \nAdded analog_signals. \nAdded times.')

    return (synaptic_wrapper)


# ** find_stims :
def find_stims(analogs, stims, thresh):
    num_neurons = len(analogs)

    stim_on = np.empty(num_neurons, dtype=np.ndarray)

    #Find start of stim-on in stim_signals[neuron][:,stim_channel[neuron],:] for each trial.
    #Store in light_on(a): a>trial number, light_on_ind(a)>time index of light-on for that trial
    for neuron in range(num_neurons):
        num_trials = analogs[neuron].shape[0]

        stim_on[neuron] = np.empty(num_trials, dtype=np.ndarray)

        #Fill trial 0 with first stim crossings
        all_crossings = np.where(stims[neuron] > thresh)[0]
        stim_on[neuron][0] = np.array([all_crossings[0]])

        for crossing_ind in np.arange(1, len(all_crossings)):
            if all_crossings[crossing_ind -
                             1] != all_crossings[crossing_ind] - 1:
                stim_on[neuron][0] = np.append(stim_on[neuron][0],
                                               all_crossings[crossing_ind])

        #Now fill all trials with trial 0
        for trial in np.arange(1, num_trials):
            stim_on[neuron][trial] = stim_on[neuron][0]

    print('\nAdded stim_on:')
    for neuron in range(num_neurons):
        print('\tNeuron ', neuron, ': ', len(stim_on[neuron][0]), ' event(s)')

    return stim_on


# ** find_spontaneous :
def find_spontaneous(analog_signals,
                     filt_size=501,
                     thresh_ampli=3,
                     thresh_deriv=-1.2,
                     thresh_pos_deriv=0.1,
                     savgol_polynomial=3):
    num_neurons = len(analog_signals)
    stim_on = np.empty(num_neurons, dtype=np.ndarray)

    for neuron in range(num_neurons):
        num_trials = analog_signals[neuron].shape[0]
        stim_on[neuron] = np.empty(num_trials, dtype=np.ndarray)

        for trial in range(num_trials):
            stim_on[neuron][trial] = np.zeros(1)
            trial_analog = sp.signal.savgol_filter(
                analog_signals[neuron][trial, :], filt_size, savgol_polynomial)
            trial_gradient = np.gradient(trial_analog)

            #Create shifted array
            deriv_lessthan = np.where(trial_gradient < thresh_deriv)[0]
            deriv_lessthan_shifted = np.roll(deriv_lessthan, shift=1)
            deriv_lessthan_shifted[0] = 2
            event_start = deriv_lessthan[deriv_lessthan -
                                         deriv_lessthan_shifted > 1]
            event_start = np.ma.array(event_start.copy(),
                                      mask=np.zeros(len(event_start)))

            deriv_greaterthan = np.where(trial_gradient > thresh_pos_deriv)[0]
            deriv_greaterthan_shifted = np.roll(deriv_greaterthan, shift=1)
            deriv_greaterthan_shifted[0] = 2
            event_end = deriv_greaterthan[deriv_greaterthan -
                                          deriv_greaterthan_shifted > 1]

            #Set the initial current_event_finish: the first value in event_end falling
            #after the first event_start.
            current_event_finish_ind = np.where(
                event_end > event_start[0])[0][0]
            current_event_finish = event_end[current_event_finish_ind]

            for ind, event in enumerate(event_start[:-1]):
                #First, determine whether we are outside of an 'event' and update
                #current_event_finish accordingly
                if event > current_event_finish:
                    try:
                        current_event_finish_ind = np.where(
                            event_end > event)[0][0]
                        current_event_finish = event_end[
                            current_event_finish_ind]
                    except:
                        current_event_finish = event_end[
                            current_event_finish_ind]

                #Determine if the next event_start also falls before the next event_end. If so,
                #more than one event_start is being detected per actual event, so mask the
                #current event.
                if event_start[ind + 1] < current_event_finish:
                    event_start.mask[ind] = True

            #Save the identified event_starts in the appropriate index in stim_on
            stim_on[neuron][trial] = event_start.compressed().astype(np.int32)

            #Delete low and high components of the signal
            to_delete_low = np.where(stim_on[neuron][trial] < 1600)
            stim_on[neuron][trial] = np.delete(stim_on[neuron][trial],
                                               to_delete_low)

            to_delete_high = np.where(
                stim_on[neuron][trial] > len(analog_signals[0][0]) - 251)
            stim_on[neuron][trial] = np.delete(stim_on[neuron][trial],
                                               to_delete_high)

    print('\nAdded stim_on (spontaneous events):')
    for neuron in range(num_neurons):
        num_trials = analog_signals[neuron].shape[0]

        print('\tNeuron ', neuron, ': ')
        for trial in range(num_trials):
            print('\t\tTrial ', int(trial), ': ', len(stim_on[neuron][trial]),
                  ' event(s)')

    return (stim_on)


# ** add_events :
def add_events(wrapper,
               event_type='stim',
               stim_thresh=2,
               spont_filtsize=1001,
               spont_threshampli=3,
               spont_threshderiv=-1.2,
               savgol_polynomial=3):
    analogs = wrapper.analog_signals
    stims = wrapper.stim_signals

    if event_type == 'stim':
        stim_on = find_stims(analogs, stims, thresh=stim_thresh)
    elif event_type == 'spontaneous':
        stim_on = find_spontaneous(analogs,
                                   filt_size=spont_filtsize,
                                   thresh_ampli=spont_threshampli,
                                   thresh_deriv=spont_threshderiv,
                                   savgol_polynomial=savgol_polynomial)

    wrapper.stim_on = stim_on

    return


# ** export :
#export takes an abf file and exports it as a matlab file (other filetypes are not implemented yet)
#Channels is an optional nxm matrix of channel numbers to load, for n neurons with m channels each.
#Filetype can either be 'matlab' or 'csv'.
#Names is an optional list of n names to save the loaded file as.
def export(files, channels='all', filetype='matlab', names=None):
    print('\n\n----New Group---')

    num_neurons = len(files)
    neurons_range = np.int32(range(num_neurons))

    block = np.empty(num_neurons, dtype=np.ndarray)
    analog_signals = np.empty(num_neurons, dtype=np.ndarray)
    times = np.empty(num_neurons, dtype=np.ndarray)

    block = [neo.AxonIO(filename=files[i]).read()[0] for i in neurons_range]

    #Define number of trials for each file
    trials = np.empty(num_neurons, dtype=np.ndarray)
    for neuron in range(num_neurons):
        trials[neuron] = np.array([1, len(block[neuron].segments)],
                                  dtype=np.int)

    #Define channels if channels is 'all', otherwise not
    if channels == 'all':
        channels = np.empty(num_neurons, dtype=np.ndarray)
        for neuron in range(num_neurons):
            channels[neuron] = np.arange(
                0, len(block[neuron].segments[0].analogsignals))

    #Populate analog_signals and times from raw data in block
    for neuron in range(num_neurons):
        num_trials = len(np.arange(trials[neuron][0], trials[neuron][1] + 1))
        numtimes_full = len(block[neuron].segments[0].analogsignals[0].times)
        numtimes_wanted = numtimes_full

        times[neuron] = np.linspace(
            block[neuron].segments[0].analogsignals[0].times[0].magnitude,
            block[neuron].segments[0].analogsignals[0].times[-1].magnitude,
            num=numtimes_wanted)

        analog_signals[neuron] = np.empty(
            (len(channels[neuron]), num_trials, numtimes_wanted))

        for channel_ind, channel in enumerate(channels[neuron]):
            for trial_index, trial_substance in enumerate(
                    block[neuron].segments[trials[neuron][0] -
                                           1:trials[neuron][1]]):
                analog_signals[neuron][int(
                    channel), trial_index, :] = trial_substance.analogsignals[
                        int(channel)][:].squeeze()

    #Save file
    for neuron in range(num_neurons):
        #Define filename
        if names == None:
            current_name = files[neuron]
        else:
            current_name = names[neuron]

        if filetype == 'matlab':
            sp.io.savemat(current_name, {
                'analogSignals': analog_signals[neuron],
                'times': times[neuron]
            },
                          do_compression=True)


### ---- Other useful functions ----


# ** find_last :
#Find last index in an array which has value of tofind.
def find_last(arr, tofind=1):
    for ind, n in enumerate(reversed(arr)):
        if n == tofind or ind == len(arr) - 1:
            return (len(arr) - ind)


#def jacob_exp(pars, x, y, monoexp_normalized_plusb):
#    deriv = np.empty([len(x), 2])
#    deriv[:, 0] = -1 * x * np.exp(-1 * pars[0] * x)
#    deriv[:, 1] = 1 * np.ones(len(x))
#    return deriv


# ** pool :
#syn.pool takes an attribute and pools it across stims: out[stim,:]
def pool(synaptic_wrapper_attribute, pool_index=0):
    #Returns a stim x m matrix where stim is stim-num and [stim,:] is raw data for that n across all neurons, trials

    postsynaptic_event = synaptic_wrapper_attribute
    num_neurons = len(postsynaptic_event)

    #Calculate min stims:
    common_stims = 10000
    for neuron in range(num_neurons):
        if common_stims > postsynaptic_event[neuron][:, :, :].shape[1]:
            common_stims = postsynaptic_event[neuron][:, :, :].shape[1]

    #Pool data
    stimpooled_postsynaptic_events = np.ma.array(np.empty([common_stims, 0]))

    for neuron in range(num_neurons):
        stimpooled_postsynaptic_events = np.ma.append(
            stimpooled_postsynaptic_events,
            np.ma.transpose(
                postsynaptic_event[neuron][:, 0:common_stims, pool_index]),
            axis=1)

    return stimpooled_postsynaptic_events


# ** get_sucrate :
#syn.get_sucrate takes an attribute and returns success rate stats over the stim train
#out[stim,0] = mean. out[stim,1] is stdev. out[stim,2] is sterr.
def get_sucrate(synaptic_wrapper_attribute, byneuron=False):
    num_neurons = len(synaptic_wrapper_attribute)

    #Calculate min stims:
    common_stims = 10000
    for neuron in range(num_neurons):
        if common_stims > len(synaptic_wrapper_attribute[neuron][0, :, 0]):
            common_stims = len(synaptic_wrapper_attribute[neuron][0, :, 0])

    success_rate_neur = np.zeros([common_stims, num_neurons])
    success_rate = np.zeros([common_stims, 3])

    for neuron in range(num_neurons):
        count_fails_temp = np.sum(
            synaptic_wrapper_attribute[neuron].mask[:, 0:common_stims, 0],
            axis=0)
        count_total_temp = synaptic_wrapper_attribute[neuron].mask.shape[0]
        success_rate_neur[:, neuron] = (count_total_temp -
                                        count_fails_temp) / count_total_temp

    success_rate[:, 0] = np.mean(success_rate_neur, axis=1)
    success_rate[:, 1] = np.std(success_rate_neur, axis=1)
    success_rate[:, 2] = np.std(success_rate_neur, axis=1) / np.sqrt(
        np.sum(success_rate_neur, axis=1))

    if byneuron is True:
        success_rate = []
        success_rate = success_rate_neur

    return success_rate


# ** get_stats :
def get_stats(synaptic_wrapper_attribute, pooling_index=0, byneuron=False):
    #If byneuron = False: get_stats returns a (stim x 5) array where stim is the stim number and [stim,0:4] is mean, std, sterr, mean success_rate, std success_rate.
    #If byneuron = True: get_stats returns a (neur x 3) array where neur is the neuron number and [neur,0:2] is mean, std, median.

    postsynaptic_event = synaptic_wrapper_attribute

    if byneuron is False:
        stimpooled_postsynaptic_events = pool(postsynaptic_event,
                                              pooling_index)
        if type(postsynaptic_event[0]) is np.ma.core.MaskedArray:
            success_rate = get_sucrate(postsynaptic_event)
            num_stims = len(stimpooled_postsynaptic_events)
            stats_postsynaptic_events = np.zeros([num_stims, 5])

            for stim in range(num_stims):
                num_nonmasked_stims = len(
                    stimpooled_postsynaptic_events[stim, :]
                ) - np.ma.count_masked(stimpooled_postsynaptic_events[stim, :])

                stats_postsynaptic_events[stim, 0] = np.mean(
                    stimpooled_postsynaptic_events[stim, :].astype(np.float64))
                stats_postsynaptic_events[stim, 1] = np.std(
                    stimpooled_postsynaptic_events[stim, :].astype(np.float64))
                stats_postsynaptic_events[stim, 2] = np.std(
                    stimpooled_postsynaptic_events[stim, :].astype(
                        np.float64)) / np.sqrt(num_nonmasked_stims)
                stats_postsynaptic_events[stim, 3] = success_rate[stim, 0]
                stats_postsynaptic_events[stim, 4] = success_rate[stim, 1]
        else:
            num_stims = len(stimpooled_postsynaptic_events)
            stats_postsynaptic_events = np.zeros([num_stims, 3])

            for stim in range(num_stims):
                num_nonmasked_stims = len(
                    stimpooled_postsynaptic_events[stim, :]
                ) - np.ma.count_masked(stimpooled_postsynaptic_events[stim, :])

                stats_postsynaptic_events[stim, 0] = np.mean(
                    stimpooled_postsynaptic_events[stim, :])
                stats_postsynaptic_events[stim, 1] = np.std(
                    stimpooled_postsynaptic_events[stim, :].astype(np.float64))
                stats_postsynaptic_events[stim, 2] = np.std(
                    stimpooled_postsynaptic_events[stim, :].astype(
                        np.float64)) / np.sqrt(num_nonmasked_stims)

    elif byneuron is True:
        num_neurons = len(postsynaptic_event)
        stats_postsynaptic_events = np.zeros([num_neurons, 3])

        for neuron in range(num_neurons):
            stats_postsynaptic_events[neuron, 0] = np.mean(
                postsynaptic_event[neuron][:, :, pooling_index].flatten())
            stats_postsynaptic_events[neuron, 1] = np.std(
                postsynaptic_event[neuron]
                [:, :, pooling_index].flatten().astype(np.float64))
            stats_postsynaptic_events[neuron, 2] = np.median(
                np.ma.compressed(postsynaptic_event[neuron]
                                 [:, :, pooling_index].flatten().astype(
                                     np.float64)))
    return (stats_postsynaptic_events)


# ** get_median_filtered :
def get_median_filtered(signal, threshold=3):

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
            s = np.array([0])
        else:
            s = difference / float(median_difference)
        mask = np.int32(s > threshold)
        signal[mask] = np.median(signal)

        mask = np.int32(s > threshold)
        mask_2 = np.int32(signal < 0)
        signal[mask] = 0
        signal[mask_2] = 0

        combined_mask_1 = np.ma.mask_or(mask, mask_2)
        combined_mask_2 = np.ma.mask_or(combined_mask_1, original_mask)

        signal = np.ma.array(signal, mask=combined_mask_2)

    return signal

# * plotting tools :
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

    return

###----- Advanced plotting tools to compare groups of recordings ----

    #plot_events compares two synwrapper attributes (eg group1.height, group2.height)

    #plot_statwrapper compares two stat files on attributes. For ex, to plot means +- standard error:
    #stat1 = syn.get_stats(group1.height), stat2 = syn.get_stats(group2.height), syn.plot_statwrapper(stat1, stat2, ind = 0, err_ind = 2)

# ** plot_events :    
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
    mask_for_ana[np.int32(wrap.height[neuron][trial, :, 1].compressed())] = 0

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
