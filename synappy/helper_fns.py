# imports
import numpy as np


def find_last(arr, val):
    """Given an array, finds the last instance of a given
    value and returns its ind.

    Parameters
    ----------------
    arr : np.array
        Array to search

    tofind : float
        Value to find the last instance of.

    Returns
    ----------------
    ind : ind
        Index where the last instance of value is.
    """
    for ind, n in enumerate(reversed(arr)):
        if n == val or ind == len(arr) - 1:
            return (len(arr) - ind)


def get_stats(synaptic_wrapper_attribute, pooling_index=0, byneuron=False):
    """Calculates statistics

    Parameters
    ----------------

    Returns
    ----------------

    """

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
def _get_median_filtered(signal, threshold=3):

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

