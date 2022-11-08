import nest
import numpy as np


def windowed_events(events, window_times, window_size):
    """
    Generate subsets of events which belong to given time windows.
    Assumptions:
    * events are sorted
    * window_times are sorted
    :param events: one-dimensional, sorted list of event times
    :param window_times: the upper (exclusive) boundaries of time windows
    :param window_size: the size of the windows
    :return: generator yielding (window_time, window_events)
    """
    for window_time in reversed(window_times):
        events = events[events < window_time]
        yield window_time, events[events > window_time - window_size]


def get_spike_times(spike_rec, rec_nodes):
    """
       Takes a spike recorder spike_rec and returns the spikes in a list of numpy arrays.
       Each array has all spike times of one sender (neuron) in units of [ms]
    """
    events = nest.GetStatus(spike_rec)[0]['events']
    spikes = []
    for i in rec_nodes:
        idx = np.where(events['senders'] == i)
        spikes.append(events['times'][idx])
    return spikes
