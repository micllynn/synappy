Table of Contents
=================

   * [Introduction](#introduction)
   * [Installation](#installation)
   * [Getting started](#getting-started)
      * [The basics](#the-basics)
      * [Built-in event types](#built-in-event-types)
         * [Rewards](#rewards)
         * [GPIO stimuli](#gpio-stimuli)
         * [Audio](#audio)
         * [Looming visual stimulus](#looming-visual-stimulus)
      * [Built-in measurement types](#built-in-measurement-types)
         * [Continuous polling from GPIO:](#continuous-polling-from-gpio)
         * [Notes on acquiring measurements](#notes-on-acquiring-measurements)
      * [Video streaming](#video-streaming)
   * [Advanced usage](#advanced-usage)
      * [Defining stochastic event start times](#defining-stochastic-event-start-times)
      * [Defining stochastic ITIs](#defining-stochastic-itis)
      * [Constructing more complex experiments](#constructing-more-complex-experiments)
   * [Stored data format: HDF5](#stored-data-format-hdf5)
      * [Experiment attributes](#experiment-attributes)
      * [Trial attributes](#trial-attributes)
      * [Measurements](#measurements)
      * [Events](#events)
   * [Creating custom classes](#creating-custom-classes)
      * [Events](#events-1)
      * [Measurements](#measurements-1)

# Introduction

SynapPy is a data analysis tool for patch-clamp synaptic physiologists who work with .abf files and want to rapidly quantify post-synaptic event statistics and visualize the results.

Synappy detects electrically or optically evoked post-synaptic events in either current clamp or voltage clamp, as long as there is an input channel associated with these stimuli. In addition to detecting events, Synappy includes a Pythonic set of methods to quantify post-synaptic event statistics, including amplitude, baseline, decay kinetics, rise-time, and release probability. Finally, Synappy includes basic support for detecting spontaneous (non-stimulus-triggered) events and quantifying their statistics.

SynapPy additionally includes intelligent tools for data visualization and quality control.

# Installation

The main dependencies are: python3, numpy, scipy, matplotlib and neo (ver 0.4+ recommended). Neo is used to parse raw data from the .abf files.

A basic installation using the included setup.py file:
```python
git clone https://github.com/micllynn/synappy/
cd synappy
python3 setup.py install
```

# Getting started

## The basics
We will start by importing the package and loading the data.

```python
import synappy

files = ['test_file.abf', 'test_file2.abf']

syn.load(files, )
```

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

