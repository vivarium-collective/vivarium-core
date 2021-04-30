from typing import Any, Dict
import logging as log

from vivarium.core.process import Deriver


def daughter_phylogeny_id(mother_id):
    return [
        str(mother_id) + '0',
        str(mother_id) + '1']


def pass_threshold(value, config):
    threshold = config['threshold']
    if value >= threshold:
        return True
    return False


def get_divide_update(daughter_composer, mother_id, daughter_ids):
    daughter_updates = []
    for daughter_id in daughter_ids:
        composer = daughter_composer.generate({
            'agent_id': daughter_id})
        daughter_updates.append({
            'key': daughter_id,
            'processes': composer['processes'],
            'topology': composer['topology'],
            'initial_state': {}})
    return {
        '_divide': {
            'mother': mother_id,
            'daughters': daughter_updates}}


class Division(Deriver):
    """ Division Process """
    defaults: Dict[str, Any] = {
        'daughter_ids_function': daughter_phylogeny_id,
        'condition_function': pass_threshold,
        'condition_config': {
            'threshold': None
        }
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.condition_function = self.parameters['condition_function']
        self.condition_config = self.parameters['condition_config']

        # must provide a composer to generate new daughters
        self.agent_id = self.parameters['agent_id']
        self.composer = self.parameters['composer']
        self.daughter_ids_function = self.parameters['daughter_ids_function']

    def ports_schema(self):
        return {
            'variable': {},
            'agents': {
                '*': {}}}

    def next_update(self, timestep, states):
        variable = states['variable']
        if self.condition_function(variable, self.condition_config):
            daughter_ids = self.daughter_ids_function(self.agent_id)
            divide_update = get_divide_update(
                self.composer, self.agent_id, daughter_ids)
            log.info(
                'DIVIDE! \n--> MOTHER: %s \n--> DAUGHTERS: %s',
                str(self.agent_id), str(daughter_ids))
            return {'agents': divide_update}
        return {}
