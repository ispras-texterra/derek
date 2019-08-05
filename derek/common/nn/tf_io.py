import tensorflow as tf
import os

from derek.common.io import load_with_pickle, save_with_pickle


def save_tf_model_with_saver(path, name, session, saver):
    model_path = os.path.join(path, name)
    os.makedirs(model_path, exist_ok=True)
    saver.save(session, os.path.join(model_path, 'model'))


def load_tf_model_with_saver(path, name, session):
    model_path = os.path.join(path, name)
    saver = tf.train.import_meta_graph(os.path.join(model_path, 'model.meta'))
    saver.restore(session, os.path.join(model_path, "model"))

    return saver


def save_classifier(path, extractor, feature_computer, graph, session, saver):
    extractor_path = os.path.join(path, 'extractor')
    os.makedirs(extractor_path, exist_ok=True)
    extractor.save(extractor_path)

    feature_computer_path = os.path.join(path, 'feature_computer')
    os.makedirs(feature_computer_path, exist_ok=True)
    feature_computer.save(feature_computer_path)

    save_tf_model_with_saver(path, 'model', session, saver)
    save_with_pickle(_convert_save_graph(graph), path, "graph.pkl")


def load_classifier(path, extractor_class, feature_computer_class, session):
    extractor = extractor_class.load(os.path.join(path, 'extractor'))
    feature_computer = feature_computer_class.load(os.path.join(path, "feature_computer"))
    saver = load_tf_model_with_saver(path, "model", session)
    graph = _convert_load_graph(session, load_with_pickle(path, "graph.pkl"))

    return extractor, feature_computer, graph, saver


def _convert_save_graph(graph):
    if type(graph) == dict:
        return {key: _convert_save_graph(value) for key, value in graph.items()}
    elif type(graph) == list:
        return [_convert_save_graph(value) for value in graph]
    else:
        return graph.name


def _convert_load_graph(session, graph):
    if type(graph) == dict:
        return {key: _convert_load_graph(session, value) for key, value in graph.items()}
    elif type(graph) == list:
        return [_convert_load_graph(session, value) for value in graph]
    else:
        try:
            return session.graph.get_tensor_by_name(graph)
        except ValueError:
            return session.graph.get_operation_by_name(graph)
