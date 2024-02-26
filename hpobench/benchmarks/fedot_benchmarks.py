"""
Changelog:
==========

0.0.1:
* First implementation of the new XGB Benchmarks.
"""
from copy import deepcopy
import time
from typing import Union, Dict

import ConfigSpace as CS
import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.optimisers.objective import MetricsObjective, PipelineObjectiveEvaluate
from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from golem.core.tuning.iopt_tuner import get_node_parameters_for_iopt
from golem.core.tuning.tuner_interface import BaseTuner
from hpobench.abstract_benchmark import AbstractBenchmark

from hpobench.util.rng_helper import get_rng

from hpobench.dependencies.ml.ml_benchmark_template import MLBenchmark
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

__version__ = '0.0.1'

def linear_pipeline():
    pipeline = PipelineBuilder().add_node('knn').add_node('rf').build()
    return pipeline


class FedotBenchmark(MLBenchmark):
    def __init__(self,
                 task_id: int,
                 rng: Union[np.random.RandomState, int, None] = None,
                 valid_size: float = 0.33,
                 data_path: Union[str, None] = None,
                 pipeline: Pipeline = None):
        pipeline = pipeline or linear_pipeline()
        self.init_pipeline = deepcopy(pipeline)
        self.pipeline = pipeline
        self.search_space = PipelineSearchSpace()
        super().__init__(task_id, rng, valid_size, data_path)
        self.input_train = InputData(idx=np.arange(0, len(self.train_X)),
                                     features=self.train_X,
                                     target=self.train_y.to_numpy(),
                                     task=Task(TaskTypesEnum.classification),
                                     data_type=DataTypesEnum.table)

        self.valid_input = InputData(idx=np.arange(0, len(self.valid_X)),
                                     features=self.valid_X,
                                     target=self.valid_y.to_numpy(),
                                     task=Task(TaskTypesEnum.classification),
                                     data_type=DataTypesEnum.table)

        self.test_input = InputData(idx=np.arange(0, len(self.test_X)),
                                    features=self.test_X,
                                    target=self.test_y.to_numpy(),
                                    task=Task(TaskTypesEnum.classification),
                                    data_type=DataTypesEnum.table)

        self.objective = MetricsObjective([ClassificationMetricsEnum.accuracy,
                                           ClassificationMetricsEnum.f1,
                                           ClassificationMetricsEnum.precision])
        data_split = DataSourceSplitter(cv_folds=3).build(self.input_train)
        self.objective_eval = PipelineObjectiveEvaluate(self.objective,
                                                        data_split,
                                                        eval_n_jobs=-1)

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """Parameter space to be optimized --- contains the hyperparameters
        """
        parameters_dict = {}
        for node_id, node in enumerate(self.pipeline.nodes):
            operation_name = node.name
            float_parameters_dict, discrete_parameters_dict = get_node_parameters_for_iopt(self.search_space,
                                                                                           node_id,
                                                                                           operation_name)
            parameters_dict.update({**float_parameters_dict, **discrete_parameters_dict})
        cs = CS.ConfigurationSpace(seed=seed, space=parameters_dict)

        return cs

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameter(CS.Constant('subsample', value=1))
        return fidelity_space

    def init_model(self,
                   config: Union[CS.Configuration, Dict],
                   fidelity: Union[CS.Configuration, Dict, None] = None,
                   rng: Union[int, np.random.RandomState, None] = None):
        """ Function that returns the model initialized based on the configuration and fidelity
        """
        if isinstance(config, CS.Configuration):
            config = dict(config)
        BaseTuner.set_arg_graph(self.pipeline, config)

        return self.pipeline

    def _train_objective(self,
                         config: Dict,
                         fidelity: Dict,
                         shuffle: bool,
                         rng: Union[np.random.RandomState, int, None] = None,
                         evaluation: Union[str, None] = "valid"):

        if rng is not None:
            rng = get_rng(rng, self.rng)

        # initializing model
        model = self.init_model(config, fidelity, rng)

        # preparing data
        if evaluation == "valid":
            train_input = self.valid_input
        else:
            train_input = self.input_train

        # fitting the model with subsampled data
        start = time.time()
        model.fit(train_input)
        model_fit_time = time.time() - start
        # computing statistics on training data
        _start = time.time()
        fitness = self.objective(model, reference_data=self.test_input)
        scores = dict(zip(self.objective.metric_names, fitness.values))
        score_cost = time.time() - _start
        train_loss = 1 - abs(scores["accuracy"])
        return model, model_fit_time, train_loss, scores, score_cost

    @AbstractBenchmark.check_parameters
    def objective_function(self,
                           configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict, None] = None,
                           shuffle: bool = False,
                           rng: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:
        """Function that evaluates a 'config' on a 'fidelity' on the validation set
        """
        model, model_fit_time, train_loss, train_scores, train_score_cost = self._train_objective(
            configuration, fidelity, shuffle, rng, evaluation="val"
        )

        _start = time.time()
        fitness = self.objective(model, reference_data=self.valid_input)
        val_scores = dict(zip(self.objective.metric_names, fitness.values))
        val_score_cost = time.time() - _start
        val_loss = 1 - abs(val_scores["accuracy"])

        _start = time.time()
        fitness = self.objective(model, reference_data=self.test_input)
        test_scores = dict(zip(self.objective.metric_names, fitness.values))
        test_score_cost = time.time() - _start
        test_loss = 1 - abs(test_scores["accuracy"])

        info = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss,
            'model_cost': model_fit_time,
            'train_scores': train_scores,
            'train_costs': train_score_cost,
            'val_scores': val_scores,
            'val_costs': val_score_cost,
            'test_scores': test_scores,
            'test_costs': test_score_cost,
            # storing as dictionary and not ConfigSpace saves tremendous memory
            'fidelity': fidelity,
            'config': configuration,
        }

        return {
            'function_value': info['val_loss'],
            'cost': model_fit_time + info['val_costs'],
            'info': info
        }

    # pylint: disable=arguments-differ
    @AbstractBenchmark.check_parameters
    def objective_function_test(self,
                                configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[CS.Configuration, Dict, None] = None,
                                shuffle: bool = False,
                                rng: Union[np.random.RandomState, int, None] = None,
                                **kwargs) -> Dict:
        """Function that evaluates a 'config' on a 'fidelity' on the test set
        """
        model, model_fit_time, train_loss, train_scores, train_score_cost = self._train_objective(
            configuration, fidelity, shuffle, rng, evaluation="test"
        )
        _start = time.time()
        fitness = self.objective(model, reference_data=self.test_input)
        test_scores = dict(zip(self.objective.metric_names, fitness.values))
        test_score_cost = time.time() - _start
        test_loss = 1 - abs(test_scores["accuracy"])

        info = {
            'train_loss': train_loss,
            'val_loss': None,
            'test_loss': test_loss,
            'model_cost': model_fit_time,
            'train_scores': train_scores,
            'train_costs': train_score_cost,
            'val_scores': dict(),
            'val_costs': dict(),
            'test_scores': test_scores,
            'test_costs': test_score_cost,
            # storing as dictionary and not ConfigSpace saves tremendous memory
            'fidelity': fidelity,
            'config': configuration,
        }

        return {
            'function_value': float(info['test_loss']),
            'cost': float(model_fit_time + info['test_costs']),
            'info': info
        }
