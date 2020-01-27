import os


def get_saving_hook(out_path, name):
    def save(classifier, epoch):
        model = name + '{:03}'.format(epoch)
        path = os.path.join(out_path, model)
        os.makedirs(path, exist_ok=True)
        classifier.save(path)

    return save


def get_specific_epoch_hook(hook, epoch):
    def func(classifier, e):
        if e == epoch:
            hook(classifier, e)

    return func
