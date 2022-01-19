from vivarium.core.engine import Engine
from vivarium.core.control import run_library_cli
from vivarium.experiments.engine_tests import get_env_view_composite
from vivarium.core.process import Step


class TopView(Step):
    defaults = {}

    def __init__(self, parameters=None):
        super().__init__(parameters)

    def ports_schema(self):
        return {
            'top': '**',
            'log': {}
        }

    def next_update(self, timestep, states):
        top = states['top']
        import ipdb; ipdb.set_trace()
        return {}


def test_bigraph_view():
    top_view_steps = {'top_view': TopView()}
    top_view_flow = {'top_view': []}
    top_view_topology = {
        'top_view': {
            'top': (),  # connect to the top
            'log': ('log_update',),
        }
    }

    composite = get_env_view_composite()
    composite.merge(
        steps=top_view_steps,
        topology=top_view_topology,
        flow=top_view_flow,
    )

    # run the simulation
    experiment = Engine(composite=composite)
    experiment.update(20)
    data = experiment.emitter.get_data()

    import ipdb;
    ipdb.set_trace()


scans_library = {
    '0': test_bigraph_view,
}

# python vivarium/experiments/bigraph_view.py -n [name]
if __name__ == '__main__':
    run_library_cli(scans_library)