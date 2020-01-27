from collections import defaultdict, namedtuple
from logging import getLogger
from typing import List

import numpy as np

logger = getLogger('logger')


def get_epoch_shifting_wrapper(ctrl, shift):
    return lambda epoch: ctrl(epoch + shift)


def get_decayed_lr(initial_lr, lr_decay):
    # epoch starts from 1
    return lambda epoch: initial_lr / (1 + (epoch - 1) * lr_decay)


def get_const_controller(value):
    return lambda epoch: value


def predict_for_samples(graph, session, outputs, batch_generator):
    labels = [[] for _ in range(len(outputs))]

    for batch in batch_generator():
        feed = {graph["inputs"][key]: value for key, value in batch.items()}

        batch_labels = session.run([graph["outputs"][out] for out in outputs], feed)

        for i, elem in enumerate(batch_labels):
            labels[i] += list(elem)

    return labels


def multitask_scheduler(task_frequencies: List[int]):
    while True:
        for task, freq in enumerate(task_frequencies):
            for _ in range(freq):
                yield task


TaskTrainMeta = namedtuple(
    "TaskTrainMeta",
    ["task_name", "graph", "batcher_factory", "controllers", "classifier", "early_stopping_callback"])


def train_for_samples(session, epoch_num, task_metas: List[TaskTrainMeta], scheduler=None):

    if len(task_metas) > 1 and scheduler is None:
        raise Exception("no scheduler provided in train_for_samples for multitask training")

    running_batchers = list(meta.batcher_factory() for meta in task_metas)

    for epoch in range(1, epoch_num + 1):
        losses = defaultdict(list)
        logger.info("Epoch " + str(epoch))

        while True:
            task_index = next(scheduler) if scheduler is not None else 0
            batch = next(running_batchers[task_index], None)

            if batch is None:
                # main task
                if task_index == 0:
                    break
                # restart batcher
                running_batchers[task_index] = task_metas[task_index].batcher_factory()
                batch = next(running_batchers[task_index])

            task_meta = task_metas[task_index]
            feed = _create_feed(batch, task_meta, epoch)
            batch_losses = _run_train_ops(feed, session, task_meta.graph)
            for loss_op, loss in batch_losses.items():
                losses[loss_op].append(loss)

        for loss_op in sorted(losses.keys(), key=lambda x: str(x)):
            logger.info("Loss {}: ".format(loss_op) + str(np.mean(losses[loss_op])))

        early_stopping = False
        for task_meta in task_metas:
            if task_meta.early_stopping_callback(task_meta.classifier, epoch):
                logger.info(f"Early stopping triggered by {task_meta.task_name}")
                early_stopping = True

        if early_stopping:
            break

        # restart main batcher
        running_batchers[0] = task_metas[0].batcher_factory()


def _create_feed(batch, meta: TaskTrainMeta, epoch):
    graph = meta.graph

    feed = {}
    for name, controller in meta.controllers.items():
        feed[graph["inputs"][name]] = controller(epoch)

    for key, value in batch.items():
        feed[graph["inputs"][key]] = value

    return feed


def _run_train_ops(feed, session, graph):
    losses = {}

    for loss_op, train_op in zip(graph["losses"], graph["train_ops"]):
        loss, _ = session.run([loss_op, train_op], feed)
        losses[loss_op] = loss

    return losses


def get_char_padding_size(props):
    char_kernel_sizes = props.get("char_kernel_sizes", [])
    return max(char_kernel_sizes) if char_kernel_sizes else 0
