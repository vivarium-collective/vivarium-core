from typing import Union, Tuple, NewType, Dict, Any


Path = Union[Tuple[str, ...], Tuple[()]]
Topology = NewType('Topology', dict)
Schema = NewType('Schema', Dict[str, Any])
State = NewType('State', Dict[str, Any])
Update = NewType('Update', Dict[str, Any])
CompositeDict = NewType('CompositeDict', Dict[str, Any])
