from collections import defaultdict
from typing import List, IO, Dict, Set

from derek.coref.data.model import CoreferenceChain
from derek.coref.data.chains_collection import collect_chains
from derek.data.model import Document


class CoNLLSerializer:

    def serialize_docs(self, docs: List[Document], fp: IO):
        for doc in docs:
            self.serialize_doc(doc, fp)
            fp.write('\n')

    def serialize_doc(self, doc: Document, fp: IO):
        fp.write('#begin document (' + doc.name + '); \n')
        chains = collect_chains(doc.relations, doc.entities)

        group_positions = self._get_group_positions(chains)
        for sentence in doc.sentences:
            for i, token in enumerate(doc.tokens[sentence.start_token: sentence.end_token]):
                idx = i + sentence.start_token
                group_info = self._get_group_info(idx, group_positions)
                fp.write('\t'.join([doc.name, str(idx), str(i), token, group_info]) + '\n')
            fp.write('\n')
        fp.write('#end document')

    def _get_group_positions(self, chains: List[CoreferenceChain]) -> Dict[int, Dict[str, Set[int]]]:
        ret = defaultdict(dict)

        for i, chain in enumerate(chains):
            for entity in chain.entities:
                if entity.start_token == entity.end_token-1:
                    ret[entity.start_token].setdefault('full', []).append(i)
                else:
                    ret[entity.start_token].setdefault('start', []).append(i)
                    ret[entity.end_token - 1].setdefault('end', []).append(i)

        return ret

    def _get_group_info(self, token_idx: int, group_positions: Dict[int, Dict[str, Set[int]]]):

        if token_idx not in group_positions:
            return '_'

        group_info = ''
        token_groups = group_positions[token_idx]

        for group in token_groups.get('end', []):
            group_info += str(group) + ')'

        for group in token_groups.get('full', []):
            group_info += '(' + str(group) + ')'

        for group in token_groups.get('start', []):
            group_info += '(' + str(group)

        return group_info


