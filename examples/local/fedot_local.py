"""
Example with XGBoost (local)
============================
This example executes the xgboost benchmark locally with random configurations on the CC18 openml tasks.

To run this example please install the necessary dependencies via:
``pip3 install .[xgboost_example]``
"""

import argparse
from time import time

from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from hpobench.benchmarks.ml.fedot_benchmarks import FedotBenchmark
from hpobench.util.openml_data_manager import get_openmlcc18_taskids


def linear_pipeline():
    pipeline = PipelineBuilder().add_node('knn').add_node('rf').build()
    return pipeline


def run_experiment(on_travis: bool = False):
    task_ids = get_openmlcc18_taskids()
    for task_no, task_id in enumerate(task_ids):

        if on_travis and task_no == 5:
            break

        print(f'# ################### TASK {task_no + 1} of {len(task_ids)}: Task-Id: {task_id} ################### #')
        if task_id == 167204:
            continue  # due to memory limits

        b = FedotBenchmark(task_id=task_id, pipeline=linear_pipeline())
        cs = b.get_configuration_space()
        start = time()
        num_configs = 1
        for i in range(num_configs):
            configuration = cs.sample_configuration()
            print(configuration)
            result_dict = b.objective_function(configuration.get_dictionary())
            valid_loss = result_dict['function_value']
            train_loss = result_dict['info']['train_loss']

            result_dict = b.objective_function_test(configuration)
            test_loss = result_dict['function_value']

            print(f'[{i+1}|{num_configs}] - Test {test_loss:.4f} '
                  f'- Valid {valid_loss:.4f} - Train {train_loss:.4f}')
        print(f'Done, took totally {time()-start:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='HPOBench CC Datasets', description='HPOBench on the CC18 data sets.',
                                     usage='%(prog)s --array_id <task_id>')

    parser.add_argument('--on_travis', action='store_true',
                        help='Flag to speed up the run on the continuous integration tool \'travis\'. This flag can be'
                             'ignored by the user')

    args = parser.parse_args()
    run_experiment(on_travis=args.on_travis)
