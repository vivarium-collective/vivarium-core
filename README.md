# Vivarium-core

Vivarium-core provides a process interface and engine for composing multi-scale models in computational biology.

![vivarium](doc/_static/multiscale.png)
** Vivarium's multi-scale framework.** 
(**a**) Processes and stores are the basic elements of the framework. Processes declare *parameters*, *ports* that connect to 
stores, and an *update function* that computes how the state variables unfold over time. Stores hold the state variables, 
and determine how the variables are handled with properties such as *units*, *updaters*, *dividers*, *emitters*, and more. 
(**b**) A *compartment* is a composite of processes created with a bipartite graph called a topology that declares how 
processes connect to stores through their ports. Boundary stores reach outside of the compartment, allowing it to connect 
with other compartments above or below. 
(**c**) A *hierarchy* of embedded compartments is shown as a place graph with the higher compartments containing those below. 
(**d**) Two coupled processes operating at different time scales, showing their separated updates of a shared store, and 
an advancing temporal *front*.
(**e**) A topology update shows the addition of a compartment in the time-step after a division update message is 
sent---other topology updates might include merging, engulfing, deleting, or adding.


## Documentation and Tutorials
Visit [Vivarium documentation](https://vivarium-core.readthedocs.io/)

## Concept

A Vivarium is a "place of life" -- an enclosure for raising organisms in controlled environments for observation or
research. Typical vivaria include aquariums or terrariums.  The vivarium provided in this repository is a computational
vivarium for developing multi-scale models of cells in dynamic environments. Its framework is a synthesis of
hybrid modeling, agent-based modeling, multi-scale simulation, and modular programming.

## Installation
vivarium-core can be used as a python library. To install:

```
$ pip install vivarium-core
```
