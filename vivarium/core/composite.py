from typing import Dict, Any, Optional

from vivarium.core.store import generate_state
from vivarium.core.process import _override_schemas, assoc_in, _get_parameters, Process
from vivarium.core.types import State, Topology, HierarchyPath, Schema, TuplePath
from vivarium.library.datum import Datum
from vivarium.library.dict_utils import deep_merge
from vivarium.library.topology import convert_path_to_tuple, inverse_topology


class Composite(Datum):
    """Composite parent class.

    Contains keys for processes and topology
    """
    processes: Dict[str, Any] = {}
    topology: Dict[str, Any] = {}
    _schema: Dict[str, Any] = {}
    defaults: Dict[str, Any] = {
        'processes': processes,
        'topology': topology,
        '_schema': _schema}

    def __init__(
            self,
            config: Dict[str, Any]
    ) -> None:
        super().__init__(config)
        _override_schemas(self._schema, self.processes)

    def generate_store(self, config):
        initial_state = self.initial_state(config)
        return generate_state(
            self.processes,
            self.topology,
            initial_state
        )

    def initial_state(self, config: Optional[dict] = None) -> Optional[State]:
        """ Merge all processes' initial states
        Arguments:
            config (dict): A dictionary of configuration options. All
            subclass implementation must accept this parameter, but
            some may ignore it.
        Returns:
            (dict): Subclass implementations must return a dictionary
            mapping state paths to initial values.
        """
        return _get_composite_state(
            processes=self.processes,
            topology=self.topology,
            state_type='initial',
            config=config)

    def default_state(self, config: Optional[dict] = None) -> Optional[State]:
        """ Merge all processes' default states
        Arguments:
            config (dict): A dictionary of configuration options. All
            subclass implementation must accept this parameter, but
            some may ignore it.
        Returns:
            (dict): Subclass implementations must return a dictionary
            mapping state paths to default values.
        """
        return _get_composite_state(
            processes=self.processes,
            topology=self.topology,
            state_type='default',
            config=config)

    def merge(
            self,
            composite: Optional['Composite'] = None,
            processes: Optional[Dict[str, 'Process']] = None,
            topology: Optional[Topology] = None,
            path: Optional[HierarchyPath] = None,
            schema_override: Optional[Schema] = None,
    ) -> None:
        composite = composite or Composite({})
        processes = processes or {}
        topology = topology or {}
        path = path or tuple()
        path = convert_path_to_tuple(path)
        schema_override = schema_override or {}

        # get the processes and topology to merge
        merge_processes = {}
        merge_topology = {}
        if composite:
            merge_processes.update(composite['processes'])
            merge_topology.update(composite['topology'])
        deep_merge(merge_processes, processes)
        deep_merge(merge_topology, topology)
        merge_processes = assoc_in({}, path, merge_processes)
        merge_topology = assoc_in({}, path, merge_topology)

        # merge with instance processes and topology
        deep_merge(self.processes, merge_processes)
        deep_merge(self.topology, merge_topology)
        self._schema.update(schema_override)
        _override_schemas(self._schema, self.processes)

    def get_parameters(self) -> Dict:
        """Get the parameters for all :term:`processes`.
        Returns:
            A map from process names to parameters.
        """
        return _get_parameters(self.processes)


def _get_composite_state(
        processes: Dict[str, 'Process'],
        topology: Any,
        state_type: Optional[str] = 'initial',
        path: Optional[TuplePath] = None,
        initial_state: Optional[State] = None,
        config: Optional[dict] = None,
) -> Optional[State]:

    path = path or tuple()
    initial_state = initial_state or {}
    config = config or {}

    for key, node in processes.items():
        subpath = path + (key,)
        subtopology = topology[key]

        if isinstance(node, dict):
            state = _get_composite_state(
                processes=node,
                topology=subtopology,
                state_type=state_type,
                path=subpath,
                initial_state=initial_state,
                config=config.get(key),
            )
        elif isinstance(node, Process):
            if state_type == 'initial':
                # get the initial state
                process_state = node.initial_state(config.get(node.name))
            elif state_type == 'default':
                # get the default state
                process_state = node.default_state()
            state = inverse_topology(path, process_state, subtopology)

        initial_state = deep_merge(initial_state, state)

    return initial_state
