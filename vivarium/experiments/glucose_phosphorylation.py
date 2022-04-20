"""
======================================
Toy Glucose Phosphorylation Experiment
======================================

This is a toy example referred to by the documentation.
"""

import os

from vivarium.composites.injected_glc_phosphorylation import (
    InjectedGlcPhosphorylation,
)
from vivarium.core.composition import (
    simulate_experiment,
    EXPERIMENT_OUT_DIR,
)
from vivarium.plots.simulation_output import plot_simulation_output
from vivarium.core.emitter import (
    path_timeseries_from_data,
    timeseries_from_data,
)
from vivarium.core.engine import Engine


NAME = 'glucose_phosphorylation'
OUT_DIR = os.path.join(EXPERIMENT_OUT_DIR, NAME)


def glucose_phosphorylation_experiment(config=None):
    if config is None:
        config = {}
    default_config = {
        'injected_glc_phosphorylation': {},
        'emitter': {
            'type': 'timeseries',
        },
        'initial_state': {},
    }
    default_config.update(config)
    config = default_config
    compartment = InjectedGlcPhosphorylation(
        config['injected_glc_phosphorylation'])
    compartment_dict = compartment.generate()
    experiment = Engine({
        'processes': compartment_dict['processes'],
        'topology': compartment_dict['topology'],
        'emitter': config['emitter'],
        'initial_state': config['initial_state'],
    })
    return experiment


def run_experiment():
    experiment = glucose_phosphorylation_experiment()
    settings = {
        'timestep': 1,
        'return_raw_data': True,
    }
    data = simulate_experiment(experiment, settings)
    experiment.end()
    return data


def test_experiment():
    data = run_experiment()
    path_ts = path_timeseries_from_data(data)
    # At every timestep, the changes in ADP and G6P should be equal, and
    # the change in GLC should be the same but with the opposite sign.
    atp = path_ts[('cell', 'ATP')]
    adp = path_ts[('cell', 'ADP')]
    g6p = path_ts[('cell', 'G6P')]
    glc = path_ts[('cell', 'GLC')]

    assert len(atp) == len(adp) == len(g6p) == len(glc)

    for i in range(len(atp) - 1):
        delta_adp = adp[i + 1] - adp[i]
        delta_g6p = g6p[i + 1] - g6p[i]
        delta_glc = glc[i + 1] - glc[i]
        delta_atp = atp[i + 1] - atp[i]

        assert delta_adp == delta_g6p, 'index: {}'.format(i)
        assert delta_atp > 0, 'index: {}'.format(i)
        assert delta_glc < 0, 'index: {}'.format(i)

    print(path_ts)


def main():
    test_experiment()
    data = run_experiment()
    agents_plot_settings = {
        'agents_key': 'agents',
    }
    plot_simulation_output(
        timeseries_from_data(data),
        agents_plot_settings,
        OUT_DIR,
        'simulation',
    )


if __name__ == '__main__':
    main()
