'''
=====
Types
=====
'''

from typing import Tuple, NewType, Dict, Any


#: Relative path between nodes in the :term:`hierarchy`. Like Unix file
#: paths, ".." refers to the parent directory.
HierarchyPath = Tuple[str, ...]

#: Mapping from :term:`ports` to paths that specify which node in the
#: :term:`hierarchy` should be wired to each port.
Topology = NewType('Topology', dict)

#: A dictionary that specifies a :term:`schema`.
Schema = NewType('Schema', Dict[str, Any])

#: A dictionary that has the form of a :term:`schema`, except instead of
#: specifying the properties of each :term:`variable`, it specifies each
#: variable's value.
State = NewType('State', Dict[str, Any])

#: A dictionary defining an :term:`update`.
Update = NewType('Update', Dict[str, Any])

#: A dictionary that specifies the :term:`processes` and
#: :term:`topology` of a :term:`composite`.
CompositeDict = NewType('CompositeDict', Dict[str, Any])
