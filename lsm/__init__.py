from __future__ import annotations

import nest
import numpy as np


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

    n_syn_exc = 2
    n_syn_inh = 1

    A_mu = {
        "EE": 30.0, 
        "EI": 60.0,
        "IE": -19.0,
        "II": -19.0
    }

    def get_u_0(U, D, F):
        return U / (1 - (1 - U) * np.exp(-1 / (F / tau)))

    def get_x_0(U, D, F):
        return (1 - np.exp(-1 / (D / tau))) / (1 - (1 - get_u_0(U, D, F)) * np.exp(-1 / (D / tau)))


    def connect(src: nest.NodeCollection,
                trg: nest.NodeCollection,
                conn_type: str,
                n_syn: int,
                syn_param: dict[str, float]) -> None:
        nest.Connect(src, trg,
                     {'rule': 'fixed_indegree', 'indegree': n_syn},
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
        return {
            "U": np.random.normal(U_mu, 0.5),
            "tau_rec": np.random.normal(D_mu, 0.5),
            "tau_fac": np.random.normal(F_mu, 0.5)
        }

    connect(nodes_E, nodes_E, 'EE', n_syn_exc, _syn_param(tau_psc=3.0, UDF=_gaussian(.5, 1.1, .05), delay=1.5))
    connect(nodes_E, nodes_I, 'EI', n_syn_exc, _syn_param(tau_psc=3.0, UDF=_gaussian(.05, .125, 1.2), delay=0.8))
    connect(nodes_I, nodes_E, 'IE', n_syn_inh, _syn_param(tau_psc=6.0, UDF=_gaussian(.25, .7, .02), delay=0.8))
    connect(nodes_I, nodes_I, 'II', n_syn_inh, _syn_param(tau_psc=6.0, UDF=_gaussian(.32, .144, .06), delay=0.8))


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


