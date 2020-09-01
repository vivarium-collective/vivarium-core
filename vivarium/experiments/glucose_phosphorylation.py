"""
======================================
Toy Glucose Phosphorylation Experiment
======================================

This is a toy example referred to by the documentation.
"""

from vivarium.core.experiment import Experiment
from vivarium.compartments.injected_glc_phosphorylation import (
    InjectedGlcPhosphorylation,
)

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
    experiment = Experiment({
        'processes': compartment_dict['processes'],
        'topology': compartment_dict['topology'],
        'emitter': config['emitter'],
        'initial_state': config['initial_state'],
    })
    return experiment
