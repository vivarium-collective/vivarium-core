"""
================================================
Toy Injected Glucose Phosphorylation Compartment
================================================

This is a toy example referenced in the documentation.
"""

from vivarium.core.process import Generator

from vivarium.processes.glucose_phosphorylation import GlucosePhosphorylation
from vivarium.processes.injector import Injector


class InjectedGlcPhosphorylation(Generator):

    defaults = {
        'glucose_phosphorylation': {
            'k_cat': 1e-2,
        },
        'injector': {
            'substrate_rate_map': {
                'GLC': 1e-4,
                'ATP': 1e-3,
            },
        },
    }

    def __init__(self, config):
        super(InjectedGlcPhosphorylation, self).__init__(config)

    def generate_processes(self, config):
        injector = Injector(self.config['injector'])
        glucose_phosphorylation = GlucosePhosphorylation(
            self.config['glucose_phosphorylation'])

        return {
            'injector': injector,
            'glucose_phosphorylation': glucose_phosphorylation,
        }

    def generate_topology(self, config):
        return {
            'injector': {
                'internal': ('cell', ),
            },
            'glucose_phosphorylation': {
                'cytoplasm': ('cell', ),
                'nucleoside_phosphates': ('cell', ),
                'global': ('global', ),
            },
        }
