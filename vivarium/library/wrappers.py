from vivarium.core.process import Process


def make_logging_process(process_class, logging_port_name="log_update"):
    """
    Given a subclass of Process, returns a new subclass that behaves exactly
    the same except that it logs all of its updates in a port with name given by
    logging_port_name.

    The returned class has the same name as process_class, but prefixed with 'Logging_'.

    Args:
        process_class: The Process class to be logged
        logging_port_name: Name of the port in which updates will be stored
                           ('log_update' by default.)

    Returns:
        logging_process: the logging version of process_class.
    """

    if not issubclass(process_class, Process):
        raise ValueError(f'process_class must be a subclass of Process.')

    logging_process = type(f"Logging_{process_class.__name__}",
                           (process_class,),
                           {})
    __class__ = logging_process  # set __class__ manually so super() knows what to do

    def ports_schema(self):
        ports = super().ports_schema()  # get the original port structure
        ports[logging_port_name] = {'_default': {}, '_updater': 'set', '_emit': True}  # add a new port
        return ports

    def next_update(self, timestep, states):
        update = super().next_update(timestep, states)  # get the original update
        log_update = {logging_port_name: update}  # log the update
        return {**update, **log_update}

    logging_process.ports_schema = ports_schema
    logging_process.next_update = next_update

    return logging_process
