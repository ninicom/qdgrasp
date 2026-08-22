# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Base callbacks for Ultralytics training, validation, prediction, and export processes."""

from collections import defaultdict
from copy import deepcopy

# Trainer callbacks ----------------------------------------------------------------------------------------------------


def on_pretrain_routine_start(trainer):
    """Called at the beginning of the pre-training routine, before data loading and model setup."""


def on_pretrain_routine_end(trainer):
    """Called at the end of the pre-training routine, after data loading and model setup are complete."""


def on_train_start(trainer):
    """Called when the training starts, before the first epoch begins."""


def on_train_epoch_start(trainer):
    """Called at the start of each training epoch, before batch iteration begins."""


def on_train_batch_start(trainer):
    """Called at the start of each training batch, before the forward pass."""


def optimizer_step(trainer):
    """Called during the optimizer step. Reserved for custom integrations; not called by default."""


def on_before_zero_grad(trainer):
    """Called before the gradients are set to zero. Reserved for custom integrations; not called by default."""


def on_train_batch_end(trainer):
    """Called at the end of each training batch, after the backward pass. Optimizer step may be deferred by
    accumulation.
    """


def on_train_epoch_end(trainer):
    """Called at the end of each training epoch, after all batches but before validation."""


def on_fit_epoch_end(trainer):
    """Called at the end of each fit epoch (train + val), after validation and any checkpoint save."""


def on_model_save(trainer):
    """Called when the model checkpoint is saved, after validation."""


def on_train_end(trainer):
    """Called when the training ends, after final evaluation of the best model."""


def on_params_update(trainer):
    """Called when the model parameters are updated. Reserved for custom integrations; not called by default."""


def teardown(trainer):
    """Called during the teardown of the training process."""


# Validator callbacks --------------------------------------------------------------------------------------------------


def on_val_start(validator):
    """Called when the validation starts."""


def on_val_batch_start(validator):
    """Called at the start of each validation batch."""


def on_val_batch_end(validator):
    """Called at the end of each validation batch."""


def on_val_end(validator):
    """Called when the validation ends."""


# Predictor callbacks --------------------------------------------------------------------------------------------------


def on_predict_start(predictor):
    """Called when the prediction starts."""


def on_predict_batch_start(predictor):
    """Called at the start of each prediction batch."""


def on_predict_batch_end(predictor):
    """Called at the end of each prediction batch."""


def on_predict_postprocess_end(predictor):
    """Called after the post-processing of the prediction ends."""


def on_predict_end(predictor):
    """Called when the prediction ends."""


# Exporter callbacks ---------------------------------------------------------------------------------------------------


def on_export_start(exporter):
    """Called when the model export starts."""


def on_export_end(exporter):
    """Called when the model export ends."""


default_callbacks = {
    # Run in trainer
    "on_pretrain_routine_start": [on_pretrain_routine_start],
    "on_pretrain_routine_end": [on_pretrain_routine_end],
    "on_train_start": [on_train_start],
    "on_train_epoch_start": [on_train_epoch_start],
    "on_train_batch_start": [on_train_batch_start],
    "optimizer_step": [optimizer_step],
    "on_before_zero_grad": [on_before_zero_grad],
    "on_train_batch_end": [on_train_batch_end],
    "on_train_epoch_end": [on_train_epoch_end],
    "on_fit_epoch_end": [on_fit_epoch_end],  # fit = train + val
    "on_model_save": [on_model_save],
    "on_train_end": [on_train_end],
    "on_params_update": [on_params_update],
    "teardown": [teardown],
    # Run in validator
    "on_val_start": [on_val_start],
    "on_val_batch_start": [on_val_batch_start],
    "on_val_batch_end": [on_val_batch_end],
    "on_val_end": [on_val_end],
    # Run in predictor
    "on_predict_start": [on_predict_start],
    "on_predict_batch_start": [on_predict_batch_start],
    "on_predict_postprocess_end": [on_predict_postprocess_end],
    "on_predict_batch_end": [on_predict_batch_end],
    "on_predict_end": [on_predict_end],
    # Run in exporter
    "on_export_start": [on_export_start],
    "on_export_end": [on_export_end],
}


def get_default_callbacks():
    """Get the default callbacks for Ultralytics training, validation, prediction, and export processes.

    Returns:
        (dict): Dictionary of default callbacks for various training events. Each key represents an event during the
            training process, and the corresponding value is a list of callback functions executed when that
            event occurs.

    Examples:
        >>> callbacks = get_default_callbacks()
        >>> print(list(callbacks.keys()))  # show all available callback events
        ['on_pretrain_routine_start', 'on_pretrain_routine_end', ...]
    """
    return defaultdict(list, deepcopy(default_callbacks))


def add_integration_callbacks(instance):
    """Add integration callbacks to the instance's callbacks dictionary.

    All third-party/SaaS experiment-tracking integrations and Ultralytics HUB/Platform
    telemetry have been removed (PLAN.md M1: "loại ... integrations ngoài scope").
    This is currently a no-op kept for call-site compatibility with
    dexgrasp/engine/trainer.py and friends; a dexgrasp-native logging integration, if
    any, would be wired in here.

    Args:
        instance (Trainer | Predictor | Validator | Exporter): The object instance to which callbacks will be added. The
            type of instance determines which callbacks are loaded.

    Examples:
        >>> from ultralytics.engine.trainer import BaseTrainer
        >>> trainer = BaseTrainer()
        >>> add_integration_callbacks(trainer)
    """
    callbacks_list: list[dict] = []

    # Add the callbacks to the callbacks dictionary
    for callbacks in callbacks_list:
        for k, v in callbacks.items():
            if v not in instance.callbacks[k]:
                instance.callbacks[k].append(v)
