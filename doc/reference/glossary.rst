========
Glossary
========

.. note:: Some fully-capitalized words and phrases have the meanings
    specified in :rfc:`2119`.

.. glossary::

    ABM
    Agent-Based Model
    Agent-based model
    agent-based model
    Agent-Based Models
    Agent-based models
    agent-based models
        Agent-based modeling is a modeling paradigm in which
        population-level phenomena are emergent from interactions among
        simple agents. An agent-based model is a model constructed using
        this paradigm.

    Boundary Store
    boundary store
    Boundary Stores
    boundary stores
        :term:`Compartments` interact through boundary stores that
        represent how the compartments affect each other. For example,
        between an environment compartment and a cell compartment, there
        might be a boundary store to track the flux of metabolites from
        the cell to the environment and vice versa.

    Compartment
    compartment
    Compartments
    compartments
        Compartments are :term:`composites` at a single level of a :term:`hierarchy`.
        Each compartment is like an agent in an :term:`agent-based model`, which can
        be nested in an environment and interact with neighbors, parent, and child
        compartments. These interactions are possible through :term:`boundary stores`
        that connect internal processes to states outside of the compartment. Thus,
        a model might contain a compartment for an environment which contains
        two child compartments for the two cells in the environment. For
        more details, see our :doc:`guide to compartments</guides/compartments>`.

    composer
    Composer
        An object with a generate method, which returns a :term:`Composite` with
        :term:`Processes` and :term:`Topologies`.

    Composite
    composite
    Composites
    composites
        Composites are dictionaries with special keys for :term:`processes` and
        :term:`topology`. The `processes` key map to a dictionary with initialized
        processes, and the `topology` specifies how they are wire to :term:`stores`.

    Deriver
    deriver
    Derivers
    derivers
        Derivers run after all processes have run for a
        :term:`timepoint` and compute values from the state of the
        model. These computed values are generally stored in the
        ``global`` :term:`store`. For example, one common deriver uses
        the cell's mass and density to compute the volume. Derivers are
        written just like processes, except with a different superclass,
        :py:class:`vivarium.core.process.Deriver`.

    Divider
    divider
    Dividers
    dividers
        When a cell divides, we have to decide how to generate the
        states of its daughter cells. Dividers specify how to generate
        these daughter cells, for example by assigning half of the value
        of the variable in the mother cell to each of the daughter
        cells. We assign a divider to each variable in the
        :term:`schema`. For more details, see the documentation for
        :py:mod:`vivarium.core.registry`.

    Embedded Timeseries
    Embedded timeseries
    embedded timeseries
        An embedded timeseries has nearly the same shape as a simulation
        state dictionary, only each variable's value is a list of values
        over time, and there is an additional ``time`` key. For details,
        see the guide on :doc:`simulation data formats
        </guides/simulation_data_formats>`.

    Emitter
    emitter
    Emitters
    emitters
        While a simulation is running, the current state is stored in
        :term:`stores`, but this information is overwritten at each
        timestep with an updated state. When we want to save off
        variable values for later analysis, we send these data to
        one of our emitters, each of which formats the data for a
        storage medium, for example a database or a Kafka message. We
        then query the emitter to get the formatted data.

    Exchange
    exchange
    Exchanges
    exchanges
        The flux between a cell and its environment. This is stored in a
        :term:`boundary store`.

    Experiment
    experiment
    Experiments
    experiments
        Vivarium defines simulations using
        :py:class:`vivarium.core.experiment.Experiment` objects. These
        simulations can contain arbitrarily nested :term:`compartments`,
        and you can run them to simulate your model over time. See the
        documentation for the ``Experiment`` class and our :doc:`guide
        to experiments </guides/experiments>` for more details.

    Inner
    inner
        A once-removed internal node position relative to a given node in
        the :term:`tree`. Nodes can have multiple inners connected to them.
        The reciprocal relationship is an :term:`outer`, but a node can have
        at most one outer.

    Masking
    masking
        When Vivarium passes stores to processes, it includes only the
        variables the process has requested. We call this filtering
        masking.

    MSM
    Multiscale Model
    Multiscale model
    multiscale model
    Multiscale Models
    Multiscale models
    multiscale models
        Multiscale models use different spatial and temporal scales for
        their component sub-models. For example, Vivarium models a
        cell's internal processes and the interactions between cells and
        their environment at different temporal scales since these
        processes require different degrees of temporal precision.

    Outer
    outer
        A once-removed external node position relative to a given node in
        the :term:`tree`. Each node, except for the top-most node, has one
        outer node that it exists within. The reciprocal relationship is an
        :term:`inner`, but a node can have many inners.

    Path Timeseries
    Path timeseries
    path timeseries
        A path timeseries is a flattened form of an :term:`embedded
        timeseries` where keys are paths in the simulation state
        dictionary and values are lists of the :term:`variable` value
        over time. We describe simulation data formats in more detail in
        our guide to :doc:`simulation data formats
        </guides/simulation_data_formats>`.

    Port
    port
    Ports
    ports
        When a :term:`process` needs access to part of the model state,
        it will be provided a :term:`store`. The ports of a process are
        what the process calls those stores. When running a process, you
        provide a store to each of the process's ports. Think of the
        ports as physical ports into which a cable to a store can be
        plugged.

    Process
    process
    Processes
    processes
        A process in Vivarium models a cellular process by defining how
        the state of the model should change at each timepoint, given
        the current state of the model. During the simulation, each
        process is provided with the current state of the model and
        the timestep, and the process returns an update that changes
        the state of the model. Each process is an instance of a
        :term:`process class`.

        To learn how to write a process, check out
        :doc:`our process-writing tutorial</tutorials/write_process>`.
        For a detailed guide to processes, see :doc:`our guide to
        processes </guides/processes>`.

    Process Class
    Process class
    process class
    Process Classes
    Process classes
    process classes
        A process class is a Python class that defines a process's
        model. These classes can be instantiated, and optionally
        configured, to create :term:`processes`. Each process class must
        subclass either :py:class:`vivarium.core.process.Process`
        or another process class.

    Raw Data
    Raw data
    raw data
        The primary format for simulation data is "raw data." See the
        guide on
        :doc:`simulation data formats
        </guides/simulation_data_formats>`.

    Schema
    schema
    Schemas
    schemas
        A schema defines the properties of a set of :term:`variables` by
        associating with each variable a set of :term:`schema key-value
        pairs`.

    Schema Key
    Schema key
    schema key
    Schema Keys
    Schema keys
    schema keys
    Schema Value
    Schema value
    schema value
    Schema Values
    Schema values
    schema values
    Schema Key-Value Pair
    Schema key-value pair
    schema key-value pair
    Schema Key-Value Pairs
    Schema key-value pairs
    schema key-value pairs
        Each :term:`variable` is defined by a set of schema key-value
        pairs. The available keys are defined in
        :py:attr:`vivarium.core.tree.Store.schema_keys`. These
        keys are described in more detail in the documentation for
        :py:class:`vivarium.core.tree.Store`.

    Serializer
    serializer
    Serializers
    serializers
        A serializer is an object that converts data of a certain type
        into a format that can transmitted and stored.

    Store
    store
    Stores
    stores
        The state of the model is broken down into stores, each of which
        represents the state of some physical or conceptual subset of
        the overall state. For example, a cell model might have a store
        for the proteins in the cytoplasm, another for the transcripts
        in the cytoplasm, and one for the transcripts in the nucleus.
        Each :term:`variable` must belong to exactly one store.

    Template
    template
    Templates
    templates
        A template describes a genetic element, its binding site, and
        the available downstream termination sites on genetic material.
        A chromosome has operons as its templates which include sites
        for RNA binding and release. An mRNA transcript also has
        templates which describe where a ribosome can bind and will
        subsequently release the transcript. Templates are defined in
        :term:`template specifications`.

    Template Specification
    Template specification
    template specification
    Template Specifications
    Template specifications
    template specifications
        Template specifications define :term:`templates` as
        :py:class:`dict` objects with the following keys:

        * **id** (:py:class:`str`): The template name. You SHOULD use
          the name of the associated operon or transcript.
        * **position** (:py:class:`int`): The index in the genetic
          sequence of the start of the genetic element being described.
          In a chromosome, for example, this would denote the start of
          the modeled operon's promoter. On mRNA transcripts (where we
          are describing how ribosomes bind), this SHOULD be set to
          ``0``.

          .. todo:: Is position 0 or 1 indexed?

        * **direction** (:py:class:`int`): ``1`` if the template should
          be read in the forward direction, ``-1`` to proceed in the
          reverse direction.  For mRNA transcripts, this SHOULD be ``1``.
        * **sites** (:py:class:`list`): A list of binding sites. Each
          binding site is specified as a :py:class:`dict` with the
          following keys:

            * **position** (:py:class:`int`): The offset in the sequence
              from the template *position* to the start of the binding
              site.  This value is not currently used and MAY be set to
              0.
            * **length** (:py:class:`int`): The length, in base-pairs,
              of the binding site. This value is not currently used and
              MAY be set to 0.
            * **thresholds** (:py:class:`list`): A list of tuples, each
              of which has a factor name as the first element and a
              concentration threshold as the second. When the
              concentration of the factor exceeds the threshold, the
              site will bind the factor. For example, in an operon the
              factor would be a transcription factor.

        * **terminators** (:py:class:`list`): A list of terminators,
          which halt reading of the template. As such, which genes are
          encoded on a template depends on which terminator halts
          transcription or translation. Each terminator is specified as
          a :py:class:`dict` with the following keys:

            * **position** (:py:class:`int`): The index in the genetic
              sequence of the terminator.
            * **strength** (:py:class:`int`): The relative strength of
              the terminator. For example, if there remain two
              terminators ahead of RNA polymerase, the first of strength
              3 and the second of strength 1, then there is a 75% chance
              that the polymerase will stop at the first terminator. If
              the polymerase does not stop, it is guaranteed to stop at
              the second terminator.
            * **products** (:py:class:`list`): A list of the genes that
              will be transcribed or translated should
              transcription/translation halt at this terminator.

    Timepoint
    timepoint
    Timepoints
    timepoints
        We discretize time into timepoints and update the model state at
        each timepoint. We collect data from the model at each
        timepoint. Note that each compartment may be running with
        different timesteps depending on how finely we need to
        discretize time.

        .. todo:: How does this work with the returned timeseries data?

    Timeseries
    timeseries
        "Timeseries" can refer to the general way in whcih we store
        simulation data or to an :term:`embedded timeseries`. See the
        guide on :doc:`simulation data formats
        </guides/simulation_data_formats>` for details.

    Timestep
    timestep
    Timesteps
    timesteps
        The amount of time elapsed between two timepoints. This is the
        amount of time for which processes compute an update. For
        example, if we discretize time into two-second intervals, then
        each process will be asked to compute an update for how the
        state changes over the next two seconds. The timestep is two
        seconds.

    Topology
    topology
    Topologies
    topologies
        A topology defines how :term:`stores` are associated to
        :term:`ports`. This tells Vivarium which store to pass to each
        port of each process during the simulation. See the constructor
        documentation for
        :py:class:`vivarium.core.experiment.Experiment` for a more
        detailed specification of the form of a topology.

    Hierarchy
    hierarchy
    Hierarchies
    hierarchies
    Compartment Hierarchy
    compartment hierarchy
    Tree
    tree
    Trees
    trees
        We nest the :term:`stores` of a model to form a tree called a
        hierarchy. Each internal node is a store and each leaf node is a
        :term:`variable`. This tree can be traversed like a directory
        tree, and stores are identified by paths. For details see the
        :doc:`hierarchy guide <../guides/hierarchy>`. Note that this
        used to be called a *tree*.

    Update
    update
    Updates
    updates
        An update describes how the model state should change due to the
        influence of a :term:`process` over some period of time (usually
        a :term:`timestep`).

    Updater
    updater
    Updaters
    updaters
        An updater describes how an update should be applied to the
        model state to produce the updated state. For example, the
        update could be added to the old value or replace it. Updaters
        are described in more detail in the documentation for
        :py:mod:`vivarium.core.registry`.

    Variable
    variable
    Variables
    variables
        The state of the model is a collection of variables.  Each
        variable stores a piece of information about the full model
        state. For example, the concentration of glucose in the
        cytoplasm might be a variable, while the concentration of
        glucose-6-phosphate in the cytoplasm is another variable. The
        extracellular concentration of glucose might be a third
        variable. As these examples illustrate, variables are often
        track the amount of a molecule in a physical region. Exceptions
        exist though, for instance whether a cell is dead could also be
        a variable.

        Each variable is defined by a set of
        :term:`schema key-value pairs`.

    WCM
    Whole-Cell Model
    Whole-cell model
    whole-cell model
    Whole-Cell Models
    Whole-cell models
    whole-cell models
        Whole-cell models seek to simulate a cell by modeling the
        molecular mechanisms that occur within it. For example, a cell's
        export of antibiotics might be modeled by the transcription of
        the appropriate genes, translation of the produced transcripts,
        and finally complexation of the translated subunits. Ideally the
        simulated phenotype is emergent from the modeled processes,
        though many such models also include assumptions that simplify
        the model.
