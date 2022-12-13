from avalanche.training.templates.supervised import SupervisedTemplate, SupervisedPlugin
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.supervised import EWC
from torch.nn import Module
from torch.optim import Optimizer
from typing import Optional, List


class EWCPretrained(EWC):
    def __init__(self,
                 model: Module,
                 optimizer: Optimizer,
                 criterion,
                 ewc_lambda: float,
                 mode: str = "separate",
                 decay_factor: Optional[float] = None,
                 keep_importance_data: bool = False,
                 train_mb_size: int = 1,
                 train_epochs: int = 1,
                 eval_mb_size: int = None,
                 device=None,
                 plugins: Optional[List[SupervisedPlugin]] = None,
                 evaluator: EvaluationPlugin = default_evaluator,
                 eval_every=-1,
                 **base_kwargs):
        super(EWCPretrained, self).__init__(model=model,
                                            optimizer=optimizer,
                                            criterion=criterion,
                                            ewc_lambda=ewc_lambda,
                                            mode=mode,
                                            decay_factor=decay_factor,
                                            keep_importance_data=keep_importance_data,
                                            train_mb_size=train_mb_size,
                                            train_epochs=train_epochs,
                                            eval_mb_size=eval_mb_size,
                                            device=device,
                                            plugins=plugins,
                                            evaluator=evaluator,
                                            eval_every=eval_every,
                                            **base_kwargs)

    def manual_importance(self, strategy):
        # compute importance_method
        self.plugins[0].after_training_exp(strategy=strategy)

        # update the counter
        self.clock.train_exp_counter += 1

class EWC(SupervisedTemplate):
    """Elastic Weight Consolidation (EWC) strategy.

    See EWC plugin for details.
    This strategy does not use task identities.
    """

    def __init__(
            self,
            model: Module,
            optimizer: Optimizer,
            criterion,
            ewc_lambda: float,
            mode: str = "separate",
            decay_factor: Optional[float] = None,
            keep_importance_data: bool = False,
            train_mb_size: int = 1,
            train_epochs: int = 1,
            eval_mb_size: int = None,
            device=None,
            plugins: Optional[List[SupervisedPlugin]] = None,
            evaluator: EvaluationPlugin = default_evaluator,
            eval_every=-1,
            **base_kwargs
    ):
        """Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param ewc_lambda: hyperparameter to weigh the penalty inside the total
               loss. The larger the lambda, the larger the regularization.
        :param mode: `separate` to keep a separate penalty for each previous
               experience. `onlinesum` to keep a single penalty summed over all
               previous tasks. `onlineweightedsum` to keep a single penalty
               summed with a decay factor over all previous tasks.
        :param decay_factor: used only if mode is `onlineweightedsum`.
               It specify the decay term of the importance_method matrix.
        :param keep_importance_data: if True, keep in memory both parameter
                values and importances for all previous task, for all modes.
                If False, keep only last parameter values and importances.
                If mode is `separate`, the value of `keep_importance_data` is
                set to be True.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param base_kwargs: any additional
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """
        ewc = EWCPlugin(ewc_lambda, mode, decay_factor, keep_importance_data)
        if plugins is None:
            plugins = [ewc]
        else:
            plugins.append(ewc)

        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )
