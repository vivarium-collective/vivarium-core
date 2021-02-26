"""
============================
Composer and Process Classes
============================
"""

import abc
import copy
from typing import Dict, Any, Optional, Iterable, List

from vivarium.core.process import _override_schemas, _get_composite_state, Process, assoc_in
from vivarium.core.types import State, Topology, Schema, HierarchyPath
from vivarium.library.datum import Datum
from vivarium.library.dict_utils import deep_merge


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
            schema_override: Optional[Schema] = None,
    ) -> None:
        composite = composite or Composite({})
        processes = processes or {}
        topology = topology or {}
        schema_override = schema_override or {}

        if composite:
            processes.update(composite['processes'])
            topology.update(composite['topology'])

        for process in processes.values():
            assert isinstance(process, Process)

        self.processes.update(processes)
        self.topology.update(topology)
        self._schema.update(schema_override)
        _override_schemas(self._schema, self.processes)

    def get_parameters(self) -> Dict:
        """Get the parameters for all :term:`processes`.
        Returns:
            A map from process names to parameters.
        """
        return _get_parameters(self.processes)


def _get_parameters(
        processes: Optional[Dict[str, 'Process']] = None
) -> Dict:
    processes = processes or {}
    parameters: Dict = {}
    for key, value in processes.items():
        if isinstance(value, Process):
            parameters[key] = value.parameters
        elif isinstance(value, dict):
            parameters[key] = _get_parameters(value)
    return parameters


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

    @abc.abstractmethod
    def generate_processes(
            self,
            config: Optional[dict]) -> Dict[str, Any]:
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
            :py:class:`vivarium.core.experiment.Experiment`.
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


class AggregateComposer(Composer):

    def __init__(
            self,
            composers: Iterable[Any] = tuple(),
            config: Optional[dict] = None,
    ) -> None:
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
                    f"{set(processes.keys())} and {set(new_processes.keys())} "
                    f"in contain overlapping keys")
            processes.update(new_processes)
        return processes

    def generate_topology(
            self,
            config: Optional[dict] = None
    ) -> Topology:
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
        if config:
            self.config.update(config)
        self.composers.append(composer)

    def add_composers(
            self,
            composers: List,
            config: Optional[Dict] = None,
    ) -> None:
        if config:
            self.config.update(config)
        self.composers.extend(composers)