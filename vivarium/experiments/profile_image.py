import json
import os
import multiprocessing

from vivarium.core.composition import EXPERIMENT_OUT_DIR
from vivarium.experiments.profile_runtime import ComplexModelSim, run_scan, plot_scan_results

VM_OUT_DIR = os.path.join(EXPERIMENT_OUT_DIR, 'vm_scan_results')


def scan_parallel_save():
    total_processes = 50
    scan_values = [
        {
            'number_of_processes': total_processes,
            'number_of_parallel_processes': total_processes
        }
    ]

    sim = ComplexModelSim()
    sim.process_sleep = 1e-2
    sim.experiment_time = 50
    saved_stats = run_scan(sim,
                           scan_values=scan_values)

    n_cpus = multiprocessing.cpu_count()

    # save results
    os.makedirs(VM_OUT_DIR, exist_ok=True)
    fig_path = os.path.join(VM_OUT_DIR, f'{n_cpus}cpus.json')
    with open(fig_path, 'w') as outfile:
        json.dump(saved_stats, outfile)


def plot_vm_scan_results():
    path_to_json = VM_OUT_DIR
    # finds json files
    json_files = [
        pos_json for pos_json in os.listdir(path_to_json)
        if pos_json.endswith('.json')]

    # load json
    saved_stats = []
    for index, js in enumerate(json_files):
        with open(os.path.join(path_to_json, js)) as json_file:
            json_text = json.load(json_file)
            saved_stats.extend(json_text)

    # plot
    plot_scan_results(saved_stats,
                      parallel_plot=True,
                      out_dir=path_to_json,
                      filename=f'scan_parallel_processes_vm')


# python vivarium/experiments/profile_image.py
if __name__ == '__main__':
    scan_parallel_save()
