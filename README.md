# Vivarium Core

[![Read the Docs](https://img.shields.io/readthedocs/vivarium-core)](https://vivarium-core.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/vivarium-core)](https://pypi.org/project/vivarium-core/)
![GitHub branch checks state](https://img.shields.io/github/checks-status/vivarium-collective/vivarium-core/master)

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

<p align="center">
    <img src="https://github.com/vivarium-collective/vivarium-core/blob/master/doc/_static/interface.png?raw=true" width="500">
</p>

Vivarium's model interface, illustrating the formal structure of the framework.
* (**a**) A *Process*, shown as a rectangular flowchart symbol, is a modular model that contains parameters, 
an update function, and ports.
* (**b**) A *Store*, shown as the flowchart symbol for a database, holds the state variables and schemas that 
determines how to handle updates. 
* (**c**) *Composites* are bundles of Processes and Stores wired together by a bipartite network called a *Topology*, 
with Processes connecting to Stores through their ports. 
* (**d**) *Compartments* are Stores with inner Processes and sub-Stores -- like a folder with internal files.
* (**e**) Compartments are embedded in a *Hierarchy* -- depicted as a place network with discrete layers, 
with outer compartments are shown above and inner compartments below.

## Getting Started

Vivarium Core can be installed as a python library like this:

```console
$ pip install vivarium-core
```

To get started using Vivarium Core, see our
[documentation](https://vivarium-core.readthedocs.io/) and
[tutorial notebook](https://vivarium-core.readthedocs.io/en/latest/tutorials.html).

If you want to contribute to Vivarium Core, see our [contribution
guidelines](CONTRIBUTING.md).

## License

Copyright (C) 2019-2022 The Vivarium Authors

Licensed under the Apache License, Version 2.0 (the "License"); you may
not use this file except in compliance with the License. You may obtain
a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

See [`LICENSE.txt`](LICENSE.txt) for a copy of the full license, and see
[`AUTHORS.md`](AUTHORS.md) for a list of the Vivarium Authors.
