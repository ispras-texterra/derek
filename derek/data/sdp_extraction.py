import networkx as nx

from derek.data.model import Document, Sentence


def create_dep_graph(doc: Document, sent: Sentence) -> nx.DiGraph:
    """
    :param doc: Document
    :param sent: Sentence
    :return: nx.Digraph with doc token indices from sentence as nodes and edges from parent to child in dependency tree
        also including "token_str" attribute for nodes and "deprel" attribute for edges
    """
    dt_distances = doc.token_features["dt_head_distances"]
    dt_labels = doc.token_features["dt_labels"]
    g = nx.DiGraph()

    for idx in range(sent.start_token, sent.end_token):
        head_distance = dt_distances[idx]
        label = dt_labels[idx]
        if idx not in g:
            g.add_node(idx, token_str=doc.tokens[idx])

        if head_distance == 0:
            g.nodes[idx]["root"] = True
            continue

        head_idx = idx + head_distance
        if head_idx not in g:
            g.add_node(head_idx, token_str=doc.tokens[head_idx])

        g.add_edge(head_idx, idx, deprel=label)

    return g


def compute_sdp(doc: Document, sent: Sentence, source_doc_idx: int, target_doc_idx: int) -> list:
    """
        returns SDP between source_doc_idx and target_doc_idx from sent as list of doc token indices,
        including source_doc_idx and target_doc_idx
    """
    dep_graph = create_dep_graph(doc, sent)
    undirected_graph = dep_graph.to_undirected()
    return nx.shortest_path(undirected_graph, source_doc_idx, target_doc_idx)


def compute_sdp_subtree(doc: Document, sent: Sentence, sdp: list) -> set:
    """
        returns tree under SDP computed for two tokens in sent as set of doc token indices
    """
    sdp_subtree = set(sdp)
    head_distances = doc.token_features["dt_head_distances"]
    changed = True

    while changed:
        changed = False
        for i in range(sent.start_token, sent.end_token):
            if i in sdp_subtree:
                continue

            parent = i + head_distances[i]
            if parent in sdp_subtree:
                sdp_subtree.add(i)
                changed = True

    return sdp_subtree
