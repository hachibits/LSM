import nest
import numpy as np


def create_iaf_psc_exp(n_E, n_I):
    nodes = nest.Create('iaf_psc_exp', n_E + n_I,
                        {'C_m': 250.0,
                         'E_L': -70.0,
                         't_ref': 2.0,
                         'tau_syn_ex': 2.0,
                         'tau_syn_in': 2.0,
                         'V_m': nest.random.uniform(-70.0,-60.0),
                         'V_reset': -70.0,
                         'V_th': -55.0})

    nest.SetStatus(nodes, [{'I_e': 13.5} for _ in nodes])
    # nest.SetStatus(nodes, [{'I_e': np.minimum(14.9, np.maximum(0, np.random.lognormal(2.65, 0.025)))} for _ in nodes])

    return nodes[:n_E], nodes[n_E:]


def connect_tsodyks(nodes_E, nodes_I):
    n_syn_exc = 2
    n_syn_inh = 1

    w_scale = 10.0
    J_EE = w_scale * 5.0
    J_EI = w_scale * 25.0
    J_IE = w_scale * -20.0
    J_II = w_scale * -20.0

    def connect(src, trg, J, n_syn, syn_param):
        nest.Connect(src, trg,
                     {'rule': 'fixed_indegree', 'indegree': n_syn},
                     dict({'model': 'tsodyks_synapse', 'delay': 0.1,
                           'weight': {"distribution": "normal_clipped", "mu": J, "sigma": 0.7 * abs(J),
                                      "low" if J >= 0 else "high": 0.
                           }},
                          **syn_param))

    def _syn_param(tau_psc, tau_rec, tau_fac, U):
        return {"tau_psc": tau_psc,
                "tau_rec": tau_rec, # recovery time constant in ms
                "tau_fac": tau_fac, # facilitation time constant in ms
                "U": U, # utilization
                "u": 0.0,
                "x": 1.0
                }

    connect(nodes_E, nodes_E, J_EE, n_syn_exc, _syn_param(tau_psc=2.0, tau_fac=1.0, tau_rec=813., U=0.59))
    connect(nodes_E, nodes_I, J_EI, n_syn_exc, _syn_param(tau_psc=2.0, tau_fac=1790.0, tau_rec=399., U=0.049))
    connect(nodes_I, nodes_E, J_IE, n_syn_inh, _syn_param(tau_psc=2.0, tau_fac=376.0, tau_rec=45., U=0.016))
    connect(nodes_I, nodes_I, J_II, n_syn_inh, _syn_param(tau_psc=2.0, tau_fac=21.0, tau_rec=706., U=0.25))


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

