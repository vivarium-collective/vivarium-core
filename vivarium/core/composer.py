"""
==============================================
Composite, Composer, and MetaComposer Classes
==============================================
"""

import abc
import copy
from typing import Dict, Any, Optional, Iterable, List

from vivarium.core.process import (
    _override_schemas, assoc_in, _get_parameters, Process)
from vivarium.core.store import (
    Store, generate_state)
from vivarium.core.types import (
    Processes, Topology, HierarchyPath, State, Schema)
from vivarium.library.datum import Datum
from vivarium.library.dict_utils import deep_merge
from vivarium.library.topology import inverse_topology


def _get_composite_state(
        processes: Dict[str, 'Process'],
        topology: Any,
        state_type: Optional[str] = 'initial',
        path: Optional[HierarchyPath] = None,
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


class Composite(Datum):
    """Composite parent class.

    Contains keys for processes and topology
    """
    processes: Dict[str, Any] = {}
    topology: Dict[str, Any] = {}
    defaults: Dict[str, Any] = {
        'processes': dict,
        'topology': dict}

    def __init__(
            self,
            config: Optional[Dict[str, Any]] = None
    ) -> None:
        config = config or {}
        super().__init__(config)
        self._schema = config.get('_schema', {})
        _override_schemas(self._schema, self.processes)

    def generate_store(self, config: Optional[dict] = None) -> Store:
        config = config or {}
        initial_state = self.initial_state(config)
        return generate_state(
            self.processes,
            self.topology,
            initial_state)

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


class Composer(metaclass=abc.ABCMeta):
    defaults: Dict[str, Any] = {}

    def __init__(self, config: Optional[dict] = None) -> None:
        """Base class for :term:`composer` classes.

        Composers generate :term:`composites`.

        All :term:`composer` classes must inherit from this class.

        Args:
            config: Dictionary of configuration options that can
                override the class defaults.
        """
        config = config or {}
        if 'name' in config:
            self.name = config['name']
        elif not hasattr(self, 'name'):
            self.name = self.__class__.__name__

        self.config = copy.deepcopy(self.defaults)
        self.config = deep_merge(self.config, config)
        self.schema_override = self.config.pop('_schema', {})

    def generate_store(self, config: Optional[dict] = None) -> Store:
        composite = self.generate()
        return composite.generate_store(config)

    @abc.abstractmethod
    def generate_processes(
            self,
            config: Optional[dict]) -> Processes:
        """Generate processes dictionary.

        Every subclass must override this method.

        Args:
            config: A dictionary of configuration options. All
                subclass implementation must accept this parameter, but
                some may ignore it.

        Returns:
            Subclass implementations must return a dictionary
            mapping process names to instantiated and configured process
            objects.
        """
        return {}  # pragma: no cover

    @abc.abstractmethod
    def generate_topology(self, config: Optional[dict]) -> Topology:
        """Generate topology dictionary.

        Every subclass must override this method.

        Args:
            config: A dictionary of configuration options. All
                subclass implementation must accept this parameter, but
                some may ignore it.

        Returns:
            Subclass implementations must return a :term:`topology`
            dictionary.
        """
        return {}  # pragma: no cover

    def generate(
            self,
            config: Optional[dict] = None,
            path: HierarchyPath = ()) -> Composite:
        """Generate processes and topology dictionaries.

        Args:
            config: Updates values in the configuration declared
                in the constructor.
            path: Tuple with ('path', 'to', 'level') associates
                the processes and topology at this level.

        Returns:
            Dictionary with keys ``processes``, which has a value of a
            processes dictionary, and ``topology``, which has a value of
            a topology dictionary. Both are suitable to be passed to the
            constructor for
            :py:class:`vivarium.core.engine.Engine`.
        """
        if config is None:
            config = self.config
        else:
            default = copy.deepcopy(self.config)
            config = deep_merge(default, config)

        processes = self.generate_processes(config)
        topology = self.generate_topology(config)
        _override_schemas(self.schema_override, processes)

        return Composite({
            'processes': assoc_in({}, path, processes),
            'topology': assoc_in({}, path, topology),
        })

    def initial_state(self, config: Optional[dict] = None) -> Optional[State]:
        """ Merge all processes' initial states

        Every subclass may override this method.

        Arguments:
            config (dict): A dictionary of configuration options. All
                subclass implementation must accept this parameter, but
                some may ignore it.

        Returns:
            dict: Subclass implementations must return a dictionary
            mapping state paths to initial values.
        """
        composite = self.generate(config)
        return composite.initial_state(config)

    def get_parameters(self) -> dict:
        """Get the parameters for all :term:`processes`.

        Returns:
            A map from process names to dictionaries of those processes'
            parameters.
        """
        composite = self.generate()
        return composite.get_parameters()


class MetaComposer(Composer):

    def __init__(
            self,
            composers: Iterable[Any] = tuple(),
            config: Optional[dict] = None,
    ) -> None:
        """A collection of :py:class:`Composer` objects.

        The MetaComposer can be used to create composites that combine
        all the composers in the collection.

        Args:
            composers: Initial collection of composers.
            config: Initial configuration.
        """
        super().__init__(config)
        self.composers: List = list(composers)

    def generate_processes(
            self,
            config: Optional[dict] = None
    ) -> Dict[str, Any]:
        # TODO(Eran)-- override composite.config with config
        processes: Dict = {}
        for composer in self.composers:
            new_processes = composer.generate_processes(composer.config)
            if set(processes.keys()) & set(new_processes.keys()):
                raise ValueError(
                    f"{set(processes.keys())} and "
                    f"{set(new_processes.keys())} "
                    f"in contain overlapping keys")
            processes.update(new_processes)
        return processes

    def generate_topology(
            self,
            config: Optional[dict] = None
    ) -> Topology:
        """Do not override this method."""
        topology: Topology = {}
        for composer in self.composers:
            new_topology = composer.generate_topology(composer.config)
            if set(topology.keys()) & set(new_topology.keys()):
                raise ValueError(
                    f"{set(topology.keys())} and {set(new_topology.keys())} "
                    f"contain overlapping keys")
            topology.update(new_topology)
        return topology

    def add_composer(
            self,
            composer: Composer,
            config: Optional[Dict] = None,
    ) -> None:
        """Add a composer to the collection of stored composers.

        Args:
             composer: The composer to add.
             config: The composer's configuration, which will be merged
                 with the stored config.
        """
        if config:
            self.config.update(config)
        self.composers.append(composer)

    def add_composers(
            self,
            composers: List,
            config: Optional[Dict] = None,
    ) -> None:
        """Add multiple composers to the collection of stored composers.

        Args:
            composers: The composers to add.
            config: Configuration for the composers, which will be
                merged with the stored config.
        """
        if config:
            self.config.update(config)
        self.composers.extend(composers)
