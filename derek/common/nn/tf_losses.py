import tensorflow as tf
import numpy as np


def get_loss(hidden, gold_idx, out_size, props, labels_mask=None):
    loss_type = props.get('loss', 'cross_entropy')
    if loss_type == 'wang_margin':
        loss, labels, scores = _create_wang_loss(hidden, gold_idx, out_size)
    elif loss_type == 'santos_margin':
        loss, labels, scores = _create_santos_loss(hidden, gold_idx, out_size, props)
    elif loss_type == 'cross_entropy':
        if out_size == 2:
            losses, labels, scores = _get_sigmoid_binary_losses(hidden, gold_idx)
        else:
            losses, labels, scores = _get_softmax_cross_entopy_losses(hidden, gold_idx, out_size, labels_mask)
        loss = tf.reduce_mean(losses, axis=0, name='loss')
    else:
        raise Exception("Unknown loss!")

    l2 = props.get('l2', 0)
    if l2 > 0:
        loss += l2 * _compute_l2_norm()

    return loss, labels, scores


def get_seq_labelling_loss(hidden, seq_len, gold_idx, out_size, props):
    loss_type = props.get("loss", "cross_entropy")
    if loss_type == "cross_entropy":
        if out_size == 2:
            losses, labels, scores = _get_sigmoid_binary_losses(hidden, gold_idx)
        else:
            losses, labels, scores = _get_softmax_cross_entopy_losses(hidden, gold_idx, out_size)

        # we should ignore loss on padding tokens
        mask = tf.sequence_mask(seq_len, dtype=tf.float32)
        loss = tf.reduce_sum(losses * mask) / tf.reduce_sum(tf.to_float(seq_len), name='loss')
    elif loss_type == "crf":
        # scores is a [batch_size] tensor with ovrl sequence score
        loss, labels, scores = _get_crf_loss(hidden, gold_idx, seq_len, out_size)
    else:
        raise Exception("Unknown loss!")

    l2 = props.get('l2', 0)
    if l2 > 0:
        loss += l2 * _compute_l2_norm()

    return loss, labels, scores


def _compute_l2_norm():
    return tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()])


def _get_softmax_cross_entopy_losses(hidden, gold_idx, out_size, labels_mask=None):
    logits = tf.layers.dense(hidden, out_size)
    if labels_mask is not None:
        labels_mask = tf.cast(labels_mask, tf.float32)
        logits = logits * labels_mask + tf.constant(np.finfo(np.float32).min) * (1 - labels_mask)

    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=gold_idx)
    label, score = tf.argmax(logits, -1, name="label"), tf.nn.softmax(logits, name='scores')

    return losses, label, score


def _get_sigmoid_binary_losses(hidden, gold_idx):
    logits = tf.layers.dense(hidden, 1)
    gold_idx = tf.to_float(tf.expand_dims(gold_idx, -1))

    losses = tf.squeeze(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=gold_idx), -1)

    pos_score = tf.nn.sigmoid(logits)
    label = tf.to_int32(tf.round(tf.squeeze(pos_score, -1)), name="label")
    score = tf.concat([1 - pos_score, pos_score], -1, name="scores")

    return losses, label, score


def _get_crf_loss(hidden, gold_idx, seq_len, out_size):
    logits = tf.layers.dense(hidden, out_size)
    losses, tp = tf.contrib.crf.crf_log_likelihood(logits, gold_idx, seq_len)
    loss = tf.reduce_mean(-losses)
    # score is a [batch_size] tensor with ovrl sequence score
    label, score = tf.contrib.crf.crf_decode(logits, tp, seq_len)
    normalized_score = score / tf.to_float(seq_len)

    return loss, label, normalized_score


def _create_wang_loss(hidden, labels, out_size):
    label_embeddings_matrix = tf.get_variable('label_embeddings', [out_size, hidden.shape[1].value])
    label_embeddings = tf.nn.embedding_lookup(label_embeddings_matrix, labels)

    losses = []
    scores = []

    for label in range(out_size):
        cond = tf.equal(labels, tf.ones_like(labels) * label)
        pos = tf.squeeze(tf.where(cond), 1)  # label positions in batch
        predicted_embeddings = tf.gather(hidden, pos)  # scores in chosen positions
        true_embeddings = tf.gather(label_embeddings, pos)

        other_embeddings = []

        for other in range(out_size):
            if other == label:
                continue
            other_idx = tf.ones_like(tf.gather(labels, pos)) * other
            other_embeddings.append(tf.nn.embedding_lookup(label_embeddings_matrix, other_idx))

        other_scores = [_wang_score(predicted_embeddings, x) for x in other_embeddings]
        other_max = tf.reduce_max(other_scores, axis=1)
        losses.append(1 + _wang_score(predicted_embeddings, true_embeddings) - other_max)

        embedding = label_embeddings_matrix[label]
        tiled_embedding = tf.tile([embedding], [tf.shape(hidden)[0], 1])
        scores.append(_wang_score(hidden, tiled_embedding))

    loss = tf.reduce_mean(tf.concat(losses, 0), name='loss')
    scores = tf.stack(scores, axis=1, name="scores")
    label = tf.argmin(scores, 1, name="label")

    return loss, label, scores


def _wang_score(first, second):
    return tf.norm(tf.nn.l2_normalize(first) - tf.nn.l2_normalize(second), axis=1)


def _create_santos_loss(hidden, labels, out_size, props):
    pos_margin = props['pos_margin']
    neg_margin = props['neg_margin']
    scale = props['scale']

    santos_scores = tf.layers.dense(hidden, out_size)

    losses = []
    for label in range(out_size):
        cond = tf.equal(labels, tf.ones_like(labels) * label)
        pos = tf.squeeze(tf.where(cond), 1)  # label positions in batch
        label_scores = tf.gather(santos_scores, pos)  # scores in chosen positions
        other_scores = tf.concat((label_scores[:, :label], label_scores[:, label + 1:]), 1)  # scores of other labels
        other_max = tf.reduce_max(other_scores, 1)
        label_losses = tf.log(1 + tf.exp(scale * (pos_margin - label_scores[:, label]))) + \
            tf.log(1 + tf.exp(scale * (neg_margin + other_max)))
        losses.append(label_losses)

    loss = tf.reduce_mean(tf.concat(losses, 0), name='loss')
    label, scores = tf.argmax(santos_scores, 1, name="label"), tf.nn.softmax(santos_scores, name='scores')

    return loss, label, scores
