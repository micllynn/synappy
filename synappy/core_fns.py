# imports
import neo
import numpy as np
import scipy as sp


def find_stims(ephysobj, thresh):
    """
    Finds stimulus onsets from an EphysObject class instance.

    Parameters
    ---------------
    ephysobj : EphysObject
        An EphysObject class instance with analog_signals and stim_signals.

    thresh: float
        Stimulus threshold, typically in volts (stimulus channel unit).
        (Stimulus channel must be inspected to determine exactly thresh.)

    Returns
    -------------
    stim_on : np.ndarray
        Array with format stim_on[n_neur][n_trial][stim_ind],
        where stim_ind is the stimulus onset index for a particular
        neuron and trial.
    """

    num_neurons = len(ephysobj.analog_signals)
    stim_on = np.empty(num_neurons, dtype=np.ndarray)

    for neuron in range(num_neurons):
        num_trials = ephysobj.analog_signals[neuron].shape[0]

        stim_on[neuron] = np.empty(num_trials, dtype=np.ndarray)

        # Fill trial 0 with first stim crossings
        all_crossings = np.where(
            ephysobj.stim_signals[neuron] > thresh)[0]
        stim_on[neuron][0] = np.array([all_crossings[0]])

        for crossing_ind in np.arange(1, len(all_crossings)):
            if all_crossings[crossing_ind -
                             1] != all_crossings[crossing_ind] - 1:
                stim_on[neuron][0] = np.append(stim_on[neuron][0],
                                               all_crossings[crossing_ind])

        # Now fill all trials with trial 0
        for trial in np.arange(1, num_trials):
            stim_on[neuron][trial] = stim_on[neuron][0]

    print('\nAdded stim_on:')
    for neuron in range(num_neurons):
        print('\tNeuron ', neuron, ': ', len(stim_on[neuron][0]), ' event(s)')

    return stim_on


def find_spontaneous(ephysobj,
                     filt_size=501,
                     savgol_polynomial=3,
                     thresh_ampli=3,
                     thresh_deriv=-1.2,
                     thresh_pos_deriv=0.1):
    """
    Finds spontaneous synaptic events from an EphysObject class instance.
    Called by ephysobj.find_stims().

    The detection method is a first derivative method related to that
    described by Ankri, Legendre, Faber & Korn (1994), J Neurosci Methods.
    The process is as follows:
        1. Smooth the trace with a Savitzky-Golay filter, with filter
        size filt_size and polynomial order savgol_polynomial.
        2. Take first derivative of trace and detect crossings of
        a first derivative threshold specified by thresh_deriv.
        3. Iterate through each crossing. If the crossing also has
        a minimum amplitude of thresh_ampli, then store the event index
        in stim_on[neur][trial][event_ind].
        4. For the current event, compute when the derivative next
        crosses thresh_pos_deriv, corresponding to entering the decay
        phase of the event. The next detected event can only occur
        beyond this point.

    Parameters
    ---------------
    ephysobj : EphysObject
        An EphysObject class instance with analog_signals and stim_signals.

    filt_size : int
        Filter size of savgol filter, in indices. Must be odd.

    savgol_polynomial : int
        Savgol filter polynomial.

    thresh_ampli : float
        Minimum threshold for event amplitude, in pA or mV depending
        on recording type.
        (Not implemented yet.)

    thresh_deriv : float
        The first derivative threshold for event starts (rising phase).
        Units of pA/index or mV/index.

    thresh_pos_deriv : float
        The first derivative threshold for event ends (falling phase).
        Units of pa/index or mV/index. The next event can only occur
        after this threshold has been reached.

    Returns
    -------------
    stim_on : np.ndarray
        Array with format stim_on[n_neur][n_trial][spont_event_ind],
        where spont_event_ind is the spontaneous event index for a
        particular neuron and trial.
    """

    num_neurons = len(ephysobj.analog_signals)
    stim_on = np.empty(num_neurons, dtype=np.ndarray)

    for neuron in range(num_neurons):
        num_trials = ephysobj.analog_signals[neuron].shape[0]
        stim_on[neuron] = np.empty(num_trials, dtype=np.ndarray)

        for trial in range(num_trials):
            stim_on[neuron][trial] = np.zeros(1)
            trial_analog = sp.signal.savgol_filter(
                ephysobj.analog_signals[neuron][trial, :],
                filt_size, savgol_polynomial)
            trial_gradient = np.gradient(trial_analog)

            # Detect spontaneous event start and end inds.
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

            # Set the initial current_event_finish: the first value in
            # event_end falling after the first event_start.
            current_event_finish_ind = np.where(
                event_end > event_start[0])[0][0]
            current_event_finish = event_end[current_event_finish_ind]

            for ind, event in enumerate(event_start[:-1]):
                # First, determine whether we are outside of an 'event'
                # and update current_event_finish accordingly
                if event > current_event_finish:
                    try:
                        current_event_finish_ind = np.where(
                            event_end > event)[0][0]
                        current_event_finish = event_end[
                            current_event_finish_ind]
                    except:
                        current_event_finish = event_end[
                            current_event_finish_ind]

                # Determine if the next event_start also falls before the
                # next event_end. If so, more than one event_start is being
                # detected per actual event, so mask the current event.
                if event_start[ind + 1] < current_event_finish:
                    event_start.mask[ind] = True

            stim_on[neuron][trial] = event_start.compressed().astype(np.int32)

            # Delete low and high components of the signal
            to_delete_low = np.where(stim_on[neuron][trial] < 1600)
            stim_on[neuron][trial] = np.delete(stim_on[neuron][trial],
                                               to_delete_low)

            to_delete_high = np.where(
                stim_on[neuron][trial]
                > len(ephysobj.analog_signals[0][0]) - 251)
            stim_on[neuron][trial] = np.delete(stim_on[neuron][trial],
                                               to_delete_high)

    print('\nAdded stim_on (spontaneous events):')

    for neuron in range(num_neurons):
        num_trials = ephysobj.analog_signals[neuron].shape[0]
        print('\tNeuron ', neuron, ': ')

        for trial in range(num_trials):
            print('\t\tTrial ', int(trial), ': ', len(stim_on[neuron][trial]),
                  ' event(s)')

    return stim_on


def export(files, channels=None, export_format='matlab', names=None):
    """
    export() takes a set of .abf files and exports them to another format
    (eg matlab).

    Parameters
    ------------
    channels : None or list
        If None, all channels are loaded. Optionally, a list in the format
        channels[neuron][channel_ind] can be provided.

    export_format : str
        The format of file to export to. Currently, only 'matlab' is
        implemented.

    names : None or list
         If None, the exported files are stored with the same names as
        the original files. Optionally, a list of export filenames can be
        provided.

    """
    print('\n\n----New Group---')

    num_neurons = len(files)
    neurons_range = np.int32(range(num_neurons))

    block = np.empty(num_neurons, dtype=np.ndarray)
    analog_signals = np.empty(num_neurons, dtype=np.ndarray)
    times = np.empty(num_neurons, dtype=np.ndarray)

    block = [neo.AxonIO(filename=files[i]).read()[0] for i in neurons_range]

    # Define number of trials for each file
    trials = np.empty(num_neurons, dtype=np.ndarray)
    for neuron in range(num_neurons):
        trials[neuron] = np.array([1, len(block[neuron].segments)],
                                  dtype=np.int)

    # Define channels if channels is 'all', otherwise not
    if channels is None:
        channels = np.empty(num_neurons, dtype=np.ndarray)
        for neuron in range(num_neurons):
            channels[neuron] = np.arange(
                0, len(block[neuron].segments[0].analogsignals))

    # Populate analog_signals and times from raw data in block
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

    # Save file
    for neuron in range(num_neurons):
        # Define filename
        if names is None:
            current_name = files[neuron]
        else:
            current_name = names[neuron]

        if export_format == 'matlab':
            sp.io.savemat(
                current_name,
                {'analogSignals': analog_signals[neuron],
                 'times': times[neuron]},
                do_compression=True)

    return
