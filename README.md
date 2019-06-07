# SynapPy: Synaptic physiology in Python

SynapPy is a data analysis tool for patch-clamp synaptic physiologists who work with .abf files and want to quickly quantify post-synaptic event statistics and visualize the results over multiple trials or neurons.

Synappy works with either evoked or spontaneous events and includes a rapid and Pythonic set of methods to add post-synaptic event statistics, including amplitude, baseline, decay kinetics, rise-time, and release probability.

Synappy works with both current clamp and voltage clamp data, and both excitatory and inhibitory responses. It can also be used to analyze spike statistics and timing in current clamp. All that is needed is to specify the 'direction' (up or down) of the post-synaptic event, and to tune the time-window in which to look for a responses.

SynapPy additionally includes intelligent data visualization tools and sophisticated data-quality vetting tools.


## Getting Started

### Prerequisites
The main dependencies are: numpy, scipy, matplotlib and neo (ver 0.4+ recommended)

### Loading data
SynapPy first loads the files into an instance of a specialized class containing the specified signal channels and the times as attributes:
    .analog_signals
        [neuron][trial, time_indices]
    .stim_signals
        [neuron][trial, time_indices]
    .times
        [neuron][times]

### Adding stimulus onsets
Through the .add_stim_on() method, one can then add either evoked (event_type = 'stim') or spontaneous (event_type = 'spontaneous') events into the .stim_on attribute:
    .stim_on
        [neuron][trial][stim_indices]


### Adding post-synaptic event statistics
For each event, one can then add a variety of post-synaptic event statistics. These are added through the .add_all() method, or through individual methods for more granuarity (e.g. .add_ampli(), .add_latency(); a.dd_decays()). The postsynaptic event statistics are automatically stored in attributes which can be accessed at a later time:

    #---------------
    .height[neuron][trial, stim, [height_params]]
    #---------------
    #Stores baseline-subtracted peak amplitude of PSP
    #[height_params] = [ampli, ampli_ind,
        time_of_max_ampli_from_stim, first_deriv]]

    #---------------
    .baseline[neuron][trial, stim, [baseline_params]]
    #---------------
    #Stores values for baseline signal
    #[baseline_params] = [mean_baseline, stdev_baseline]

    #---------------
    .latency[neuron][trial, stim, [latency_params]]
    #---------------
    #Stores latency from stimulus onset to foot of PSP
    #[latency_params] = [latency_sec, ind_latency_sec]

    #---------------
    .height_norm[neuron][trial, stim, [height_params]]
    #---------------
    #Stores baseline-subtracted peak ampli normalized to 1 within a cell
    #[height_params] = [normalized_ampli, norm_ampli_ind,
            time_of_max_ampli_from_stim, first_deriv]]

    #---------------
    .decay[neuron][trial, stim, [tau_params]]
    #---------------
    #Stores statistics for the decay tau of PSP
    #[tau_params] = [tau, baseline_offset]


### Data quality and further analysis
By default, SynapPy intelligently filters out data if events are not above 4*SD(baseline), or if their decay constant (tau) is nonsensical. These events are masked but kept in the underlying data structure, providing a powerful tool to both analyze release probability/failure rate, or alternatively spike probability.



## An analysis pipeline including commands and their usage

    ###########
    #Load files, add event statistics, and recover these statistics for further analysis
    ###########

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

    ##########
    #Plot event statistics with useful built-in plotting tools
    ##########

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



## Other built-in functions and methods

### Useful functions built into SynapPy package
    syn.pool(event_1.attribute)

        #Pools this attribute over [stims, :]


    syn.get_stats(event_1.attribute, byneuron = False)

        #Gets statistics for this attribute over stimtrain (out[stim,:]) or neuron if byneuron is True (out[neuron,:])
        #dim2: 1->mean, 2->std, 3->sterr, 4->success_rate_mean, 5->success_rate_stdev
        #eg if byneuron = True, out[neuron, 3] would give sterr for that neuron, calculated by pooling across all trials/stims.


### Useful methods which are part of the synwrapper class
    synwrapper.propagate_mask(): propagate synwrapper.mask through to all other attributes.
    synwrapper.add_ampli() adds .height and .latency
    synwrapper.add_sorting() adds .mask and sorts [.height, .latency]
    synwrapper.add_invertedsort() adds .height_fails
    synwrapper.add_normalized() adds .height_norm
    synwrapper.add_decay() adds .decay
    synwrapper.remove_unclamped_aps() identifies events higher than 5x (default) amp.
                                    It updates the .mask with this information, and then .propagate_mask()

    synwrapper.add_all() is a general tool to load all stats.

