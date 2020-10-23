from __future__ import absolute_import, division, print_function

import logging as log

from vivarium.library.dict_utils import deep_merge_combine_lists
from vivarium.core.process import Process



def nested_set(dic, keys, value):
    ''' Make nested dictionary

    Makes an embedded dict from a list of keys, with a value set at the deepest key.
    '''
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value


class TimelineProcess(Process):

    name = 'timeline'

    defaults = {
        'time_step': 1.0,
        'timeline': []}

    def __init__(self, parameters=None):
        super(TimelineProcess, self).__init__(parameters)

        # sort the timeline
        timeline = []
        for new_event in self.parameters['timeline']:
            if not timeline:
                timeline.append(new_event)
                continue

            new_time = new_event[0]
            for event_index, event in enumerate(timeline):
                time = event[0]
                if new_time == time:
                    # merge events
                    timeline[event_index][1].update(new_event[1])
                    break
                elif event_index == len(timeline) - 1:
                    # append as last event
                    timeline.append(new_event)
                    break
                elif new_time < time:
                    # next
                    continue
                elif new_time > time:
                    next_time = timeline[event_index + 1][0]
                    if new_time < next_time:
                        # add event into middle of timeline
                        timeline = timeline[:event_index + 1] + [new_event] + timeline[event_index + 2:]
                        break

        self.timeline = timeline

        # get ports
        self.timeline_ports = {'global': ['time']}
        for event in self.timeline:
            for state in event[1].keys():
                port = {state[0]: [state[1:]]}
                self.timeline_ports = deep_merge_combine_lists(self.timeline_ports, port)

    def ports_schema(self):

        schema = {
            port: {
                '*': {}}
            for port in list(self.timeline_ports.keys())
            if port not in ['global']}

        schema.update({
            'global': {
                'time': {
                    '_default': 0,
                    '_updater': 'accumulate'}}})
        return schema

    def next_update(self, timestep, states):
        time = states['global']['time']
        update = {'global': {'time': timestep}}
        for (t, change_dict) in self.timeline:
            if time >= t:
                for path_to_variable, value in change_dict.items():
                    # make embedded dict with keys listed in path_to_variable
                    update_at_path = {}
                    update_value = {
                        '_value': value,
                        '_updater': 'set'}
                    nested_set(update_at_path, path_to_variable, update_value)
                    update = deep_merge_combine_lists(update, update_at_path)

                self.timeline.pop(0)
                log.info('timeline update: {}'.format(update))
        return update
