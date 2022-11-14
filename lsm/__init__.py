from __future__ import annotations

import nest
import numpy as np
import math
import scipy.stats as stats

from utils import windowed_events, get_spike_times


def create_iaf_psc_exp(n_E: int, n_I: int) -> nest.NodeCollection(list):
    membrane_voltage_interval = [13.5, 15.0]
    nodes = nest.Create('iaf_psc_exp', n_E + n_I,
                        {'tau_m': 30.0,
                         't_ref': 2.0,
                         'V_th': 15.0,
                         'E_L': 0.0,
                         'V_m': nest.random.uniform(membrane_voltage_interval[0],membrane_voltage_interval[1])})

    nest.SetStatus(nodes, [{'I_e': 13500.0} for _ in nodes])
    # nest.SetStatus(nodes, [{'I_e': np.minimum(14.9, np.maximum(0, np.random.lognormal(2.65, 0.025)))} for _ in nodes])

    return nodes[:n_E], nodes[n_E:]


def connect_tsodyks(nodes_E: nest.NodeCollection, nodes_I: nest.NodeCollection) -> None:
    tau = 0.1

    A_mu = {
        "EE": 30.0,
        "EI": 60.0,
        "IE": -19.0,
        "II": -19.0
    }

    _lambda = 2

    C = {
        "EE": 0.3,
        "EI": 0.2,
        "IE": 0.4,
        "II": 0.1
    }

    def get_u_0(U, D, F):
        return U / (1 - (1 - U) * np.exp(-1 / (F / tau)))

    def get_x_0(U, D, F):
        return (1 - np.exp(-1 / (D / tau))) / (1 - (1 - get_u_0(U, D, F)) * np.exp(-1 / (D / tau)))


    def connect(src: nest.NodeCollection,
                trg: nest.NodeCollection,
                conn_type: str,
                syn_param: dict[str, float]) -> None:
        D_euclid = 1.4142 if src == trg else 0.
        nest.Connect(src, trg,
                     {'rule': 'pairwise_bernoulli', 'p': C[conn_type] * math.exp(-(D_euclid / _lambda) ** 2)},
                     dict({'synapse_model': 'tsodyks_synapse',
                           'weight': np.random.normal(A_mu[conn_type] * 1e3, 1)},
                          **syn_param))

    def _syn_param(tau_psc: float, UDF: dict[str, float], delay: float) -> dict[str, float]:
        return dict({"tau_psc": tau_psc,
                     "U": UDF["U"], # utilization
                     "u": get_u_0(UDF["U"], UDF["tau_rec"], UDF["tau_fac"]),
                     "x": get_x_0(UDF["U"], UDF["tau_rec"], UDF["tau_fac"]),
                     "tau_rec": UDF["tau_rec"], # recovery time constant in ms
                     "tau_fac": UDF["tau_fac"], # facilitation time constant in ms
                     "delay": delay
                    }, **UDF)

    @staticmethod
    def _gaussian(U_mu: float, D_mu: float, F_mu: float) -> list(float):
        U = np.random.normal(U_mu, 0.5)
        tau_rec = np.random.normal(D_mu, 0.5)
        tau_fac = np.random.normal(F_mu, 0.5)

        return {
            "U": U if U > 0 and U < 1 else np.random.uniform(0, 1),
            "tau_rec": tau_rec if tau_rec > 0 and tau_rec < 1 else np.random.uniform(0, 1),
            "tau_fac": tau_fac if tau_fac > 0 and tau_fac < 1 else np.random.uniform(0, 1)
        }

    connect(nodes_E, nodes_E, 'EE', _syn_param(tau_psc=3.0, UDF=_gaussian(.5, 1.1, .05), delay=1.5))
    connect(nodes_E, nodes_I, 'EI', _syn_param(tau_psc=3.0, UDF=_gaussian(.05, .125, 1.2), delay=0.8))
    connect(nodes_I, nodes_E, 'IE', _syn_param(tau_psc=6.0, UDF=_gaussian(.25, .7, .02), delay=0.8))
    connect(nodes_I, nodes_I, 'II', _syn_param(tau_psc=6.0, UDF=_gaussian(.32, .144, .06), delay=0.8))


def inject_noise(nodes_E, nodes_I):
    p_rate = 25.0  # this is used to simulate input from neurons around the populations
    A_noise = 1.0  # strength of synapses from noise input [pA]
    delay = dict(distribution='normal_clipped', mu=10., sigma=20., low=3., high=200.)

    noise = nest.Create('poisson_generator', 1, {'rate': p_rate})

    @staticmethod
    def truncated_normal(mean, stddev, minval, maxval):
        return np.clip(np.random.normal(mean, stddev), minval, maxval)

    nest.Connect(noise,
                 nodes_E + nodes_I,
                 syn_spec={#'model': 'static_synapse',
                           #'weight': np.random.normal(0, math.sqrt(32)),
                           'weight': np.random.normal(A_noise, 0.7 * A_noise),
                           'delay': truncated_normal(10., 20., 3., 200.)
    })


class LSM(object):
    def __init__(self, n_exc, n_inh, n_rec,
                 create=create_iaf_psc_exp, connect=connect_tsodyks, inject_noise=inject_noise):
        neurons_exc, neurons_inh = create(n_exc, n_inh)
        connect(neurons_exc, neurons_inh)
        inject_noise(neurons_exc, neurons_inh)

        self.exc_nodes = neurons_exc
        self.inh_nodes = neurons_inh
        self.inp_nodes = neurons_exc
        self.rec_nodes = neurons_exc[:n_rec]

        self.n_exc = n_exc
        self.n_inh = n_inh
        self.n_rec = n_rec

        self._rec_detector = nest.Create('spike_recorder')

        nest.Connect(self.rec_nodes, self._rec_detector)

    def get_states(self, times, tau):
        spike_times = get_spike_times(self._rec_detector, self.rec_nodes)
        return LSM._get_liquid_states(spike_times, times, tau)

    @staticmethod
    def compute_readout_weights(states, targets, reg_fact=0):
        """
        Train readout with linear regression
        :param states: numpy array with states[i, j], the state of neuron j in example i
        :param targets: numpy array with targets[i], while target i corresponds to example i
        :param reg_fact: regularization factor; 0 results in no regularization
        :return: numpy array with weights[j]
        """
        if reg_fact == 0:
            w = np.linalg.lstsq(states, targets)[0]
        else:
            w = np.dot(np.dot(np.linalg.inv(reg_fact * np.eye(np.size(states, 1)) + np.dot(states.T, states)),
                              states.T),
                       targets)
        return w

    @staticmethod
    def compute_prediction(states, readout_weights):
        return np.dot(states, readout_weights)

    @staticmethod
    def _get_liquid_states(spike_times, times, tau, t_window=None):
        n_neurons = np.size(spike_times, 0)
        n_times = np.size(times, 0)
        states = np.zeros((n_times, n_neurons))
        if t_window is None:
            t_window = 3 * tau
        for n, spt in enumerate(spike_times):
            for i, (t, window_spikes) in enumerate(windowed_events(np.array(spt), times, t_window)):
                states[n_times - i - 1, n] = sum(np.exp(-(t - window_spikes) / tau))
        return states


