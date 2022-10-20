# imports
from .classes import EphysObject
import neo
from types import SimpleNamespace
import numpy as np


def load(fnames, trials=None, input_channel=None, stim_channel=None,
         verbose=True):
    """
    Loads a dataset of .abf files into an EphysObject class instance.
    One input channel is loaded, corresponding to the recorded signal,
    and one stim channel is loaded, corresponding to a TTL pulse triggering
    stimulation (electrical, optogenetic, etc.)

    Parameters
    ---------------
    fnames : list
        A list of strings corresponding to .abf filenames to load

    trials : list or None
        A list of arrays corresponding to trials to load from each neuron.
        - 0-indexed
        - example: trials = [[0, 5], [0, 8], [1, 10]] would load
          trials 0 to 5 (inclusive) from first neuron, etc.
        - if None, defaults to loading all trials.

    input_channel : list or None
        A list of input channels for each neuron.
        - example: input_channel=[0, 0, 2] loads channel 0 for
        neurons 0 and 1, and channel 2 for neuron 2.
        - if None, defaults to loading first input channel available.

    stim_channel : list or None
        A list of stim channels for each neuron.
        - example: stim_channel=[0, 0, 2]
        - if None, defaults to loading last channel in the recording.

    verbose : bool
        Whether to print progress statements (True) or not (False).

    Returns
    -------------
    syn_obj : EphysObject
        An instance of the EphysObject class, containing input and stimulus
        signals, recording information, and any quantified event statistics.

        - syn_obj.sig[neuron][trial, time_index]
        - syn_obj.sig_units[neuron]
        - syn_obj.sig_stim[neuron][trial]
    """

    syn_obj = EphysObject()
    syn_obj.fnames = fnames

    n_neurs = len(fnames)

    if verbose is True:
        print(f'Loading {n_neurs} recordings...')

    syn_obj.sig = np.empty(n_neurs, dtype=np.ndarray)
    syn_obj.sig_stim = np.empty(n_neurs, dtype=np.ndarray)
    syn_obj.t = np.empty(n_neurs, dtype=np.ndarray)
    syn_obj._sampling_rate = np.empty(n_neurs)

    # Check for presence of optional variables, create them if they don't exist
    if input_channel is None:
        input_channel = np.zeros(n_neurs, dtype=np.int8)
    elif type(input_channel) is int:
        input_channel = (np.ones(n_neurs) * input_channel).astype(np.int8)

    if stim_channel is None:
        stim_channel = (-1 * np.ones(n_neurs)).astype(np.int8)
    elif type(stim_channel) is int:
        stim_channel = (np.ones(n_neurs) * stim_channel).astype(np.int8)

    # Populate sig and times from raw data in block
    for neuron in range(n_neurs):
        reader = neo.rawio.AxonRawIO(filename=fnames[neuron])
        reader.parse_header()
        n_trials_in_rec = reader.header['nb_segment'][0]
        _in_ch = input_channel[neuron]
        _stim_ch = stim_channel[neuron]

        # store time vector
        n_samples = reader.get_analogsignal_chunk(
            block_index=0,
            seg_index=0,
            channel_indexes=[_in_ch]).shape[0]
        syn_obj._sampling_rate[neuron] = reader.get_signal_sampling_rate()
        _t_interv = 1 / syn_obj._sampling_rate[neuron]
        _t_stop = _t_interv * n_samples
        syn_obj.t[neuron] = np.arange(0, _t_stop, _t_interv)

        # setup trials if not initialized
        if trials is not None:
            _trials = np.arange(trials[neuron][0],
                                trials[neuron][1]+1)
        elif trials is None:
            _trials = np.arange(n_trials_in_rec)
        n_trials = len(_trials)

        # Store stim
        _raw_sig_stim = reader.get_analogsignal_chunk(
            block_index=0,
            seg_index=0,
            channel_indexes=[_stim_ch])
        _float_sig_stim = reader.rescale_signal_raw_to_float(
            _raw_sig_stim,
            dtype='float64',
            channel_indexes=[_stim_ch])[:, 0]
        syn_obj.sig_stim[neuron] = _float_sig_stim

        # store analog signal
        syn_obj.sig[neuron] = np.empty((n_trials, n_samples))
        for rel_trial, abs_trial in enumerate(_trials):
            _raw_sig = reader.get_analogsignal_chunk(
                block_index=0,
                seg_index=abs_trial,
                channel_indexes=[_in_ch])
            _float_sig = reader.rescale_signal_raw_to_float(
                _raw_sig,
                dtype='float64',
                channel_indexes=[_in_ch])[:, 0]
            syn_obj.sig[neuron][rel_trial, :] = _float_sig
            syn_obj.sig_units = reader.header['signal_channels'][
                _in_ch]['units']

        # print summary of this neuron
        if verbose is True:
            print(f'\tfile {fnames[neuron]}:')
            print(f'\t\ttrials: {n_trials}')
            print(f'\t\tunits: {syn_obj.sig_units}')
            print(f'\t\tdur: {_t_stop}s')

    if verbose is True:
        print('Added .sig \nAdded .sig_stim \nAdded .t')

    return syn_obj


def load_all_channels(fname):
    """Internal function called by preview() in the visualize module.
    Loads all channels in an .abf recording including their units.
    """
    rec = SimpleNamespace()

    # Populate sig and times from raw data in block
    reader = neo.rawio.AxonRawIO(filename=fname)
    reader.parse_header()
    n_trials = reader.header['nb_segment'][0]
    n_sigs = reader.header['signal_channels'].shape[0]

    rec.sig = np.empty(n_sigs, dtype=np.ndarray)
    rec.t = np.empty(n_sigs, dtype=np.ndarray)
    rec.units = np.empty(n_sigs, dtype=list)

    for ind_sig in range(n_sigs):
        _raw_sig = reader.get_analogsignal_chunk(
            block_index=0,
            seg_index=0,
            channel_indexes=[ind_sig])

        # time
        n_samples = _raw_sig.shape[0]
        _t_interv = 1 / reader.get_signal_sampling_rate()
        _t_stop = _t_interv * n_samples
        rec.t[ind_sig] = np.arange(0, _t_stop, _t_interv)

        # analog signal
        rec.sig[ind_sig] = np.empty((n_trials, n_samples))
        for trial in range(n_trials):
            _raw_sig = reader.get_analogsignal_chunk(
                block_index=0,
                seg_index=trial,
                channel_indexes=[ind_sig])
            _float_sig = reader.rescale_signal_raw_to_float(
                _raw_sig,
                dtype='float64',
                channel_indexes=[ind_sig])[:, 0]
            rec.sig[ind_sig][trial, :] = _float_sig
            rec.units[ind_sig] = reader.header['signal_channels'][
                ind_sig]['units']

    return rec
