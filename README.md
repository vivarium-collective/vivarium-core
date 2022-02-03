# Vivarium Core

[![PyPI](https://img.shields.io/pypi/v/vivarium-core)](https://pypi.org/project/vivarium-core/)
[![documentation](https://github.com/vivarium-collective/vivarium-core/actions/workflows/docs.yml/badge.svg)](https://vivarium-core.readthedocs.io/en/latest/)
[![lint](https://github.com/vivarium-collective/vivarium-core/actions/workflows/pylint.yml/badge.svg)](https://github.com/vivarium-collective/vivarium-core/actions/workflows/pylint.yml?query=branch%3Amaster)
[![pytest](https://github.com/vivarium-collective/vivarium-core/actions/workflows/pytest.yml/badge.svg)](https://github.com/vivarium-collective/vivarium-core/actions/workflows/pytest.yml?query=branch%3Amaster)
[![mypy](https://github.com/vivarium-collective/vivarium-core/actions/workflows/mypy.yml/badge.svg)](https://github.com/vivarium-collective/vivarium-core/actions/workflows/mypy.yml?query=branch%3Amaster)


Vivarium Core provides a process interface and simulation engine for composing 
and executing integrative multi-scale models.

## Concept

Computational systems biology requires software for multi-algorithmic model 
composition, which allows many modeling efforts to be extended, combined, and 
simulated together. We need an "interface protocol" -- analogous to TCP/IP for 
the Internet -- which allows diverse pieces of simulation software to connect, 
communicate, and synchronize seamlessly into large, complex, and open-ended 
networks that anyone can contribute to.

Vivarium addresses the challenges of model reuse and multi-scale integration by 
explicitly separating the interface that connects models from the frameworks that 
implement them. Vivarium's modular interface makes individual simulation tools into 
modules that can be wired together in composite multi-scale models, parallelized 
across multiple CPUs, and run with Vivarium's discrete-event simulation engine.

The figure below illustrates the key terminology of Vivarium's interface.
* (**a**) A *Process*, shown as a rectangular flowchart symbol, is a modular model that contains parameters, 
an update function, and ports.
* (**b**) A *Store*, shown as the flowchart symbol for a database, holds the state variables and schemas that 
determine how to handle updates. 
* (**c**) *Composites* are bundles of Processes and Stores wired together by a bipartite network called a *Topology*, 
with Processes connecting to Stores through their ports. 
* (**d**) *Compartments* are Stores with inner sub-Stores and Processes. Processes can connect across compartments via 
boundary stores.
* (**e**) Compartments are embedded in a *Hierarchy* -- depicted as a place network with discrete layers, 
with outer compartments shown above and inner compartments below.

<p align="center">
    <img src="https://github.com/vivarium-collective/vivarium-core/blob/master/doc/_static/interface.png?raw=true" width="500">
</p>


## Getting Started

Vivarium Core can be installed as a python library like this:

```console
$ pip install vivarium-core
```

To get started using Vivarium Core, see our
[documentation](https://vivarium-core.readthedocs.io/en/latest/getting_started.html)
and [tutorial
notebook](https://vivarium-core.readthedocs.io/en/latest/tutorials.html).

If you want to contribute to Vivarium Core, see our [contribution
guidelines](CONTRIBUTING.md).

## License

Copyright (C) 2019-2022 The Vivarium Authors

Licensed under the Apache License, Version 2.0 (the "License"); you may
not use this project except in compliance with the License. You may
obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

See [`LICENSE.txt`](LICENSE.txt) for a copy of the full license, and see
[`AUTHORS.md`](AUTHORS.md) for a list of the Vivarium Authors.
