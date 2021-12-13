"""

Steps:
 - make an image with borealis:
        `python borealis/gce.py -o image=vivarium-profiling-image3
        -o custom-cpu=8 -o custom-memory=8GB -l '' -m 'metadata=True'
        vivarium-profile-8cpus`
 - ssh into new image: `gcloud compute ssh vivarium-profile-8cpus-0`
 - run profile_image.py in the VM instance:
        `python3 vivarium/experiments/profile_image.py`
 - exit out of VM (ctrl + D) and retrieve results:
        `gcloud compute scp vivarium-profile-8cpus-0:
        vivarium-core/out/vm/scan_results.json
        out/vm/8cpu_results.json`

"""
import argparse
import json
import os
import multiprocessing

from vivarium.core.composition import BASE_OUT_DIR
from vivarium.experiments.profile_runtime import (
    ComplexModelSim, run_scan, plot_scan_results)

VM_OUT_DIR = os.path.join(BASE_OUT_DIR, 'vm')


def scan_parallel_save() -> None:
    n_cpus = multiprocessing.cpu_count()

    total_processes = 50
    scan_values = [
        {
            'number_of_processes': total_processes,
            'number_of_parallel_processes': total_processes,
            'number_of_cpus': n_cpus,
        }
    ]

    sim = ComplexModelSim()
    sim.process_sleep = 1e-2
    sim.experiment_time = 50
    saved_stats = run_scan(sim,
                           scan_values=scan_values)

    # save results
    os.makedirs(VM_OUT_DIR, exist_ok=True)
    fig_path = os.path.join(VM_OUT_DIR, 'scan_results.json')
    with open(fig_path, 'w') as outfile:
        json.dump(saved_stats, outfile)


def plot_vm_scan_results() -> None:
    path_to_json = VM_OUT_DIR

    # finds json files
    json_files = [
        pos_json for pos_json in os.listdir(path_to_json)
        if pos_json.endswith('.json')]

    # load json
    saved_stats = []
    for js in json_files:
        with open(os.path.join(path_to_json, js)) as json_file:
            json_text = json.load(json_file)
            saved_stats.extend(json_text)

    number_of_processes = None
    for saved_stat in saved_stats:
        if number_of_processes:
            assert number_of_processes == saved_stat['number_of_processes'], \
                "all scans must have the same number of processes"
        else:
            number_of_processes = saved_stat['number_of_processes']

    # plot
    plot_scan_results(saved_stats,
                      cpus_plot=True,
                      row_height=2.5,
                      out_dir=path_to_json,
                      title=f'n vCPUs running {number_of_processes} '
                            f'parallel processes',
                      filename='scan_vCPUs')


# python vivarium/experiments/profile_image.py
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='profile vm instances')
    parser.add_argument(
        '--plot', '-p', action='store_true')
    parser_args = parser.parse_args()

    if parser_args.plot:
        plot_vm_scan_results()
    else:
        scan_parallel_save()
