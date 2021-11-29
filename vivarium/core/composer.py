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
    Processes, Topology, HierarchyPath, State, Schema, Steps, Flow)
from vivarium.library.datum import Datum
from vivarium.library.dict_utils import deep_merge, deep_merge_check
from vivarium.library.topology import inverse_topology


def _get_composite_state(
        processes: Processes,
        steps: Steps,
        topology: Any,
        state_type: Optional[str] = 'initial',
        path: Optional[HierarchyPath] = None,
        initial_state: Optional[State] = None,
        config: Optional[dict] = None,
) -> Optional[State]:
    path = path or tuple()
    initial_state = initial_state or {}
    config = config or {}

    try:
        processes = copy.deepcopy(processes)
    except TypeError as e:
        print(e)
    deep_merge_check(processes, steps)

    for key, node in processes.items():
        subpath = path + (key,)
        subtopology = topology[key]

        if isinstance(node, dict):
            state = _get_composite_state(
                processes=node,
                steps={},
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
    processes: Processes = {}
    steps: Steps = {}
    flow: Flow = {}
    topology: Topology = {}
    defaults: Dict[str, Any] = {
        'processes': {},
        'steps': {},
        'flow': {},
        'topology': {},
    }

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
            steps=self.steps,
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
            steps=self.steps,
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

        Every subclass must override this method. For backwards
        compatibility, :py:class:`vivarium.core.process.Step` objects
        may be included in the returned dictionary, but this practice is
        discouraged and may be disallowed in a future release.

        Args:
            config: A dictionary of configuration options. All
                subclass implementation must accept this parameter, but
                some may ignore it.

        Returns:
            Subclass implementations must return a dictionary
            mapping process names to instantiated and configured
            :py:class:`vivarium.core.process.Process` objects.
        """
        return {}  # pragma: no cover

    def generate_steps(self, config: Optional[dict]) -> Steps:
        '''Generate the steps dictionary.

        Subclasses that want to include :term:`steps` should override
        this method. This method is the preferred way to specify steps,
        though they may also be returned by
        :py:meth:`generate_processes`.

        Args:
            config: A dictionary of configuration options. All
                subclass implementation must accept this parameter, but
                some may ignore it.

        Returns:
            Subclass implementations should return a dictionary mapping
            step names to instantiated and configured
            :py:class:`vivarium.core.process.Step` objects.
        '''
        _ = config
        return {}  # pragma: no cover

    def generate_flow(self, config: Optional[dict]) -> Flow:
        '''Generate the flow of :term:`step` dependencies.

        Args:
            config: A dictionary of configuration options. All
                subclass implementation must accept this parameter, but
                some may ignore it.

        Returns:
            Subclass implementations should return a dictionary mapping
            step names to sequences (e.g. lists or tuples) of
            :term:`paths`. **Steps with no dependencies must be
            included,** but they should be mapped to an empty sequence.
            Any steps returned by :py:meth:`generate_steps` or
            :py:meth:`generate_processes` that are not included in the
            flow will be treated as if they depend on every step
            previously added to the :term:`engine`.
        '''
        _ = config
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
            Dictionary with the following keys

            * ``processes``: Generated by :py:meth:`generate_processes`.
            * ``steps``: Generated by :py:meth:`generate_steps`.
            * ``flow``: Generated by :py:meth:`generate_flow`.
            * ``topology``: Generated by :py:meth:`generate_topology`.

            The values of these keys are all dictionaries suitable to be
            passed to the constructor for
            :py:class:`vivarium.core.engine.Engine`.
        """
        if config is None:
            config = self.config
        else:
            default = copy.deepcopy(self.config)
            config = deep_merge(default, config)

        processes = self.generate_processes(config)
        steps = self.generate_steps(config)
        flow = self.generate_flow(config)
        topology = self.generate_topology(config)
        _override_schemas(self.schema_override, processes)

        return Composite({
            'processes': assoc_in({}, path, processes),
            'steps': assoc_in({}, path, steps),
            'flow': assoc_in({}, path, flow),
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

    def _generate(
            self, config: Optional[dict], method: str
    ) -> Dict[str, Any]:
        combined: Dict = {}
        for composer in self.composers:
            func = getattr(composer, method)
            composer_config = copy.deepcopy(composer.config)
            deep_merge(composer_config, config)
            new = func(composer_config)
            if set(combined.keys()) & set(new.keys()):
                raise ValueError(
                    f"{set(combined.keys())} and "
                    f"{set(new.keys())} "
                    f"contain overlapping keys. "
                    f"They were produced by Composer.{method}()")
            combined.update(new)
        return combined

    def generate_processes(
            self,
            config: Optional[dict] = None
    ) -> Dict[str, Any]:
        return self._generate(config, 'generate_processes')

    def generate_steps(
            self,
            config: Optional[dict] = None
    ) -> Dict[str, Any]:
        return self._generate(config, 'generate_steps')

    def generate_flow(
            self,
            config: Optional[dict] = None
    ) -> Dict[str, Any]:
        return self._generate(config, 'generate_flow')

    def generate_topology(
            self,
            config: Optional[dict] = None
    ) -> Topology:
        return self._generate(config, 'generate_topology')

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
