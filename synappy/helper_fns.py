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

