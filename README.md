Table of Contents
=================

* [Introduction](#introduction)
* [Installation](#installation)
* [Getting started](#getting-started)
  * [Previewing data](#previewing-data)
  * [Detecting and quantifying events](#detecting-and-quantifying-events)
  * [Visualizing event statistics](#visualizing-event-statistics)
  * [Retrieving values](#retrieving-values)
* [Event quantification](#event-quantification)
  * [`add_ampli()`](#-add-ampli---)
  * [`add_ampli_norm()`](#-add-ampli-norm---)
  * [`add_decay()`](#-add-decay---)
  * [`add_integral()`](#-add-integral---)

# Introduction

SynapPy is a Pythonic data visualization and analysis tool for patch-clamp
synaptic physiologists who work with synaptic events stored in Axon Binary
Format files (.abf).

First, SynapPy provides preview and visualization capabilities for .abf
files, including sweep highlighting and averaging. Second, SynapPy
provides a set of Pythonic tools for automated detection and
quantification of post-synaptic events (either electrically or
optically evoked) based on a stimulus trigger channel. SynapPy can
auto-magically quantify amplitude, baseline, integral, decay kinetics
and rise-time, latency, and release probability. Moreover, these event
attributes can be easily overlaid with the raw signal for visual
inspection afterwards. Finally, Synappy includes basic support for
detecting spontaneous (non-stimulus-triggered) events and, of course,
quantifying their statistics.

# Installation

The main dependencies are: python3, numpy, scipy, matplotlib and neo
(ver 0.4+ recommended). Neo is used to parse raw data from the .abf files.

A basic installation using the included setup.py file:
```python
git clone https://github.com/micllynn/synappy/
cd synappy
python3 setup.py install
```

(Make sure you substitute your binary of python here!)

# Getting started

## Previewing data
Let's start by previewing a file:
```python
import synappy

p = syn.preview('ex_file.abf')
```
This plots all channels and sweeps from the file. Sweeps can be
advanced by the indicated buttons, and an average across all sweeps
can be taken.
![preview-ex](/imgs/preview_ex.png)

## Detecting and quantifying events
To work with synaptic events, we first need to import the associated
files.
```python
import synappy

files = ['ex_file_1.abf', 'ex_file_2.abf']
d = syn.load(files, input_channel=0, stimulus_channel=2)
```
Note that `files` can be thought of as a dataset (for example, a
related group of files with the same parameters for stimulation or
drug infusion.) Also note that if needed, we can specify the signal
channel number and stimulus trigger channel number.
(If no arguments are provided, these default to first and last
channels, respectively.)

`syn.load()` stores the signal, time and stimulus information in the
following class attributes:
* `d.sig[neuron][trial, t_ind]`
  * The raw signal for a given neuron, trial and time index.
* `d.sig_stim[neuron][trial, t_ind]`
  * The stimulus trigger signal for a given neuron, trial and time index.
* `d.t[neuron]`
  * Time, in seconds, for a given neuron

Next, we add stimulus-triggered events.
```python
d.add_events()
```
This searches the stimulus channel provided for any pulses, and adds
these locations as event onsets.

Most event statistics can then be added using specific class methods,
described in detail below. Here, we first add the simplest event
statistic, amplitude:
```python
d.add_ampli(event_sign='pos')
```
This quantifies amplitude from baseline, in the direction specified
by event_sign ('pos' or 'neg' to deal with excitatory or inhibitory
events), after the stimulus trigger.

A convenient library of general event statistics, including event decay
and integral, can be added with the following command:
```python
d.add_all(event_sign='pos')
```

## Visualizing event statistics
Most event statistics can be overlaid with the recorded signal in a
single plot for convenient quality control.
```python
d.preview(neur=0, attrs=['ampli', 'baseline'])
```
Here, we've specified a neuron to preview, as well as a list of attributes
(event statistics) to annotate, typically as colored dots overlaid with the
trace. Note that these can be any attributes shown in the verbose printing
from class methods.



## Retrieving values
All event statistics (eg amplitude, etc.) are stored in a logical format as
attributes within the class instance. Taking `.ampli`as an example:

* `d.ampli.data[neuron][trial, event]`
  * Baseline-subtracted maximum amplitude data (in pA or mV) for a given
  neuron, trial and event index.
* `d.ampli.inds[neuron][trial, event]`
  * Indices in .sig of maximum amplitudes.
* `d.ampli.params`
  * Parameters related to the kwargs specified for the associated class
	method. (For example, `d.ampli.params.t_event_upper` specifies the
	max post-stimulus time to search for the maximum event amplitude.


# Event quantification

Here, we detail all the class methods available for measuring and
quantifying synaptic events.

All class methods are fully documented (`help(d.method)`).

## `.add_ampli()`
Computes pre-event baseline values, event amplitudes,
and event latencies (computed in a number of ways).
	
This requires a self.events attribute, created by calling
the method `self.add_events()`. For each stimulus, a baseline
signal is calculated between `t_baseline_lower` and
`t_baseline_upper` before the stimulus onset. Next, a
maximum event amplitude is calculated. Finally, the event
latency (time to the peak amplitude, or alternately
other latency metrics) is computed.

These values are stored as the following attributes in the
EphysObject instance:
	.ampli
	.baseline
	.latency

### Parameters
* `event_sign` : str
	The sign of the events. Can either be 'pos',
	reflecting EPSPs/IPSCs in IC/VC, or 'neg',
	reflecting IPSPs/EPSCs in IC/VC.

* `t_baseline_lower` : float
	The time before stimuli, in ms, from which to
	start computing a pre-event baseline.

* `t_baseline_upper` : float
	The time before stimuli, in ms, from which to
	stop computing a pre-event baseline.

* `t_event_lower` : float
	The time after stimuli, in ms, from which to
	start searching for events.

* `t_event_upper` : float
	The time after stimuli, in ms, from which to
	stop searching for events.

* `t_savgol_filt` : int
	Width of savgol filter applied to data, in ms,
	before computing maximum amplitude.

* `latency_method` : str
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

### Attributes added to class instance
* `.ampli` : SimpleNamespace
	* `.ampli.data[neuron][trial, event]`
	Baseline-subtracted maximum amplitude data
	(in pA or mV).
	* `.ampli.inds[neuron][trial, event]`
	Indices in .sig of maximum amplitudes.
	* `.ampli.params`
	SimpleNamespace storing key params from `.add_ampli()` method
	related to amplitude.
* `.baseline` : SimpleNamespace
	* `.baseline.mean[neuron][trial, event]`
	Mean baseline values (in pA or mV).
	* `.baseline.std[neuron][trial, event]`
	Standard deviation of baseline values (in pA or mV).
	* `.baseline.inds_start[neuron][trial, event]`
	Indices of the start of baseline period in .sig
    * `.baseline.inds_stop[neuron][trial, event]`
	Indices of the end of baseline period in .sig
	* `.baseline.params`
	SimpleNamespace storing key params from `.add_ampli()` method
	related to baseline.
* `.latency` : SimpleNamespace
	* `.latency.data[neuron][trial, event]`
	Event latency from stimulus onset (sec).
	* `.latency.inds[neuron][trial, event]`
	Indices in .sig of event latency.
	* `.latency.params`
	SimpleNamespace storing key params from .add_ampli() method
	related to latency.


## `.add_ampli_norm()`
Adds normalized amplitude measurement to the class instance as
.ampli_norm. Amplitudes are normalized to the mean ampli for each
stimulus delivered to each neuron.

(.ampli must be an existing attribute, through the .add_ampli() method.)

### Attributes added to class instance
* `self.ampli` : SimpleNamespace
	* `.ampli_norm.data[neuron][trial, event]`
	Baseline-subtracted normalized max amplitude data
	(in pA or mV).
	* `.ampli_norm.inds[neuron][trial, event]`
	Indices in .sig of normalized max amplitudes.
	
	
## `.add_decay()`
Fits each post-synaptic event with an exponential decay fuction
and stores the fitted parameters in self.decay.

Decay equation variables correspond to the fitted variables for
the equation used (see the kwarg fn for more info).
- monoexponential decay: lambda1, b.
- biexponential decay: lambda1, lambda2, vstart2, b.

### Parameters
* `t_prestim` : float
	Time before stimulus, in ms, to include in signal
	used to compute decay.

* `t_poststim` : float
	Time after stimulus, in ms, to include in signal
	used to compute decay.

* `plotting` : bool
	Whether to plot examples of decay fits (True) or not (False).

* `fn` : str
	Exponential decay function to use.
	- 'monoexp': y = e^(-t * lambda1) + b
	- 'biexp_normalized_plusb': y = e^(-t * lambda1)
	 + vstart * e^(-t / lambda2) + b

	(In all cases, decay tau can be computed
	as tau= 1/lambda).

### Attributes added to class instance
* `.decay` : SimpleNamespace
	* `.decay.vars[neuron][trial, stim, decay_var]`
	Fitted variables for monoexponential decay.
		- If `fn='monoexp'`, then
		`decay_var=0` : lambda1
        `decay_var=1` : b
        - If `fn='biexp_normalized_plusb'`, then
		`decay_var=0` : lambda1
		`decay_var=1` : lambda2
		`decay_var=2` : vstart2
		`decay_var=3` : b

	* `.decay.covari[neuron][trial, stim, decay_param]`
	Covariance matrices for fitted variables,
	as documented in .decay.vars

	* `.decay.params`
	Parameters related to the decay fitting.
	
	
## `.add_integral()`
Computes the integral for each post-synaptic event.

### Parameters
* `t_integral` : float
	The total post-stimulus time to integrate, in milliseconds.

* `cdf_bins` : int
	Number of bins for the cumulative integral

### Attributes added
* `.integral` : SimpleNamespace
	* `.integral.data[neuron][trial, event]`
	Integral values for each event (in pA*sec or mV*sec).
	* `.integral.inds_start[neuron][trial, event]`
	Indices of the start of integral period in .sig
	* `.integral.inds_stop[neuron][trial, event]`
	Indices of the end of integral period in .sig
	* `.integral.cdf[neuron][trial, event]`
	Cumulative distribution of the integral over time
	for each event.
	* `.integral.params`
	SimpleNamespace storing key params from .add_integral() method
	related to integral.
