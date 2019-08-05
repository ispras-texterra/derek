import logging

import sys


def init_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def update_progress(progress, loss=None):
    barLength = 20  # Modify this to change the length of the progress bar
    block = int(round(barLength*progress))
    loss_msg = "" if loss is None else "loss={:06.4f}".format(loss)

    text = "\rProgress: [{0}] {1:06.2f}% {2}".format("#"*block + "-"*(barLength-block), progress*100, loss_msg)
    if progress >= 1:
        text += '\n'
    sys.stderr.write(text)
    sys.stderr.flush()
