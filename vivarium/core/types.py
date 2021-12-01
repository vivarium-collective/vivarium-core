"""
=====
Types
=====
"""

from typing import Tuple, Dict, Any, Union, Sequence


#: Relative path between nodes in the :term:`hierarchy`. Like Unix file
#: paths, ".." refers to the parent directory.
HierarchyPath = Tuple[str, ...]

#: Mapping from :term:`ports` to paths that specify which node in the
#: :term:`hierarchy` should be wired to each port.
Topology = Dict[str, Union[HierarchyPath, dict, object]]

#: Mapping from processes names to Processes, which can be embedded in
#: a hierarchy.
Processes = Dict[str, Any]

#: Mapping from step names to Steps, which can be embedded in a
#: hierarchy.
Steps = Dict[str, Any]

#: Mapping from step names to sequences of HierarchyPaths that specify
#: the step's dependencies.
Flow = Dict[str, Sequence[HierarchyPath]]

#: A dictionary that specifies a :term:`schema`.
Schema = Dict[str, Any]

#: A dictionary that has the form of a :term:`schema`, except instead of
#: specifying the properties of each :term:`variable`, it specifies each
#: variable's value.
State = Dict[str, Any]

#: A dictionary defining an :term:`update`.
Update = Dict[str, Any]

#: A dictionary that specifies the :term:`processes` and
#: :term:`topology` of a :term:`composite`.
CompositeDict = Dict[str, Any]
# TODO(jerry): ^ Dict values should be Union[Process, Topology, CompositeDict]
#  but Process would make recursive imports and CompositeDict would make
#  recursive types. Fix this by switching from a dict to a class or dataclass.

#: A dictionary that contains the retrieved output of an :term:`experiment`
OutputDict = Dict[Union[Tuple, str], Any]
