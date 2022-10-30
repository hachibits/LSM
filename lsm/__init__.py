import nest
import numpy as np


def create_iaf_psc_exp(n_E, n_I):
    model_parameters = {
        'C_m': 250.0,
        'E_L': -70.0,
        't_ref': 2.0,
        'tau_syn_ex': 2.0,
        'tau_syn_in': 2.0,
        'V_m': nest.random.uniform(-70.0,-60.0),
        'V_reset': -70.0,
        'V_th': -55.0,
    }
    nodes = nest.Create('iaf_psc_exp', n_E + n_I, model_parameters)
    nest.SetStatus(nodes, [{'I_e': 13.5} for _ in nodes])
    # nest.SetStatus(nodes, [{'I_e': np.minimum(14.9, np.maximum(0, np.random.lognormal(2.65, 0.025)))} for _ in nodes])

    return nodes[:n_E], nodes[n_E:]

def connect_tsodyks(nodes_E, nodes_I):
    n_syn_exc = 2
    n_syn_inh = 1

    def connect(src, trg, model, n_syn=10):
        nest.CopyModel("tsodyks_synapse", model)
        nest.Connect(src, trg,
                     {'rule': 'fixed_indegree', 'indegree': n_syn},
                     {'weight': 10.0 if model[0] == 'E' else -10.0})

    connect(nodes_E, nodes_E, "EE", n_syn_exc)
    connect(nodes_E, nodes_I, "EI", n_syn_exc)
    connect(nodes_I, nodes_E, "IE", n_syn_inh)
    connect(nodes_I, nodes_I, "II", n_syn_inh)

class LSM(object):
    def __init__(self, n_exc, n_inh, n_rec,
                 create=create_iaf_psc_exp, connect=connect_tsodyks):
        neurons_exc, neurons_inh = create(n_exc, n_inh)
        connect(neurons_exc, neurons_inh)

        self.exc_nodes = neurons_exc
        self.inh_nodes = neurons_inh
        self.inp_nodes = neurons_exc
        self.rec_nodes = neurons_exc[:n_rec]

        self.n_exc = n_exc
        self.n_inh = n_inh
        self.n_rec = n_rec

        self._rec_detector = nest.Create('spike_recorder', 1)

        nest.Connect(self.rec_nodes, self._rec_detector)

