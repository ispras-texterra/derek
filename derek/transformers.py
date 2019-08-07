from typing import List, Dict

from derek.data.transformers import DocumentTransformer, SequenceDocumentTransformer
from derek.common.vectorizers import CompositeVectorizer, FastTextVectorizer


def _single_elem_from_props(props, factories, name):
    if not props:
        return None
    if len(props) > 1:
        raise Exception(f'Just one {name} should be specified')

    alg, alg_props = next(iter(props.items()))
    factory = factories.get(alg, None)
    if factory is None:
        raise Exception(f'{alg} {name} is not currently supported')
    print(f'Using {alg} {name}')
    return factory(alg_props)


def _single_elem_from_props_as_lst(props, factories, name):
    elem = _single_elem_from_props(props, factories, name)
    return [elem] if elem is not None else []


def _bllip(props):
    from tools.common.bllip_processor import BLLIPProcessor
    return BLLIPProcessor.from_props(props)


def _texterra_syntax(props):
    from derek.data.texterra_processing import TexterraTextProcessor, TexterraFeatsProcessor
    return SequenceDocumentTransformer([
        TexterraTextProcessor.from_props(props),
        TexterraFeatsProcessor.from_props(props)
    ])


def _udpipe_tagger(props):
    from babylondigger.wrappers.udpipe1_2 import UDPipeTagger
    return UDPipeTagger.load_udpipe(props["model_path"]), {}


def _tdozat_tagger(props):
    from derek.data.babylondigger_processor import digger_extras_extractor
    from babylondigger.wrappers.tdozat import TDozatTagger
    extra_fes = {}

    tagger_recur_layer = props.get("tagger_recur_layer", False)
    if tagger_recur_layer:
        print("tdozat tagger recurrent layer features would be added as 'tagger_recur_layer' token features")
        extra_fes['tagger_recur_layer'] = digger_extras_extractor('tagger_recur_layer')

    upos_hidden = props.get("upos_hidden", False)
    if upos_hidden:
        print("tdozat tagger UPOS hidden layer features would be added as 'upos_hidden' token features")
        extra_fes['upos_hidden'] = digger_extras_extractor('upos_hidden')

    xpos_hidden = props.get("xpos_hidden", False)
    if xpos_hidden:
        print("tdozat tagger XPOS hidden layer features would be added as 'xpos_hidden' token features")
        extra_fes['xpos_hidden'] = digger_extras_extractor('xpos_hidden')

    return TDozatTagger(props["model_path"], tagger_recur_layer=tagger_recur_layer,
                        upos_hidden=upos_hidden, xpos_hidden=xpos_hidden), extra_fes


def _udpipe_parser(props):
    from babylondigger.wrappers.udpipe1_2 import UDPipeParser
    return UDPipeParser.load_udpipe(props["model_path"]), {}


def _tdozat_parser(props):
    from derek.data.babylondigger_processor import digger_extras_extractor
    from babylondigger.wrappers.tdozat import TDozatParser
    extra_fes = {}

    parser_recur_layer = props.get("parser_recur_layer", False)
    if parser_recur_layer:
        print("tdozat parser recurrent layer features would be added as 'parser_recur_layer' token features")
        extra_fes['parser_recur_layer'] = digger_extras_extractor('parser_recur_layer')
    return TDozatParser(props["model_path"], parser_recur_layer=parser_recur_layer), extra_fes


DIGGER_FACTORIES = {
    "udpipe_tagger": _udpipe_tagger,
    "tdozat_tagger": _tdozat_tagger,
    "udpipe_parser": _udpipe_parser,
    "tdozat_parser": _tdozat_parser
}


def _babylondigger(props):
    from derek.data.babylondigger_processor import BabylonDiggerProcessor
    from babylondigger.processor import DocumentComplexProcessor
    processors = []
    extra_fes = {}
    for proc_props in props:
        elem = _single_elem_from_props(proc_props, DIGGER_FACTORIES, 'babylondigger step')
        if elem is None:
            raise Exception('Empty step given for babylondigger')
        proc, extras = elem
        processors.append(proc)
        extra_fes.update(extras)
    return BabylonDiggerProcessor.from_processor(DocumentComplexProcessor(processors), extra_fes)


_SYNTAX_FACTORIES = {
    "bllip": _bllip,
    "texterra": _texterra_syntax,
    "babylondigger": _babylondigger
}


def _tagger_parser_from_props(props: Dict[str, List[dict]]) -> List[DocumentTransformer]:
    return _single_elem_from_props_as_lst(props, _SYNTAX_FACTORIES, 'POS tagger / syntactic parser')


def _texterra_ner(props):
    from derek.data.texterra_processing import TexterraNerExtrasProvider
    return TexterraNerExtrasProvider.from_props(props)


_NER_FACTORIES = {
    "texterra": _texterra_ner
}


def _ner_from_props(props: Dict[str, List[dict]]) -> List[DocumentTransformer]:
    return _single_elem_from_props_as_lst(props, _NER_FACTORIES, 'NER')


def _collapser_from_props(props: dict, ne=False):
    if props:
        from derek.data.entities_collapser import EntitiesCollapser
        collapser = EntitiesCollapser.from_props({**props, "collapse_with_ne": ne})
        print(f"Using {collapser}")
        return [collapser]
    else:
        return []


_VECTORIZERS_FACTORIES = {
    "fasttext": lambda props: FastTextVectorizer.from_props(props)
}


def _vectorizers_from_props(props: Dict[str, List[dict]]) -> List[DocumentTransformer]:
    vectorizers = []

    for vectorizer_name, vectorizers_props in sorted(props.items(), key=lambda x: x[0]):
        factory = _VECTORIZERS_FACTORIES.get(vectorizer_name, None)

        if factory is None:
            raise Exception(f"{vectorizer_name} vectorizer is not currently supported")

        print(f'Using {len(vectorizers_props)} {vectorizer_name} vectorizers')
        vectorizers.append(CompositeVectorizer([factory(pr) for pr in vectorizers_props], vectorizer_name))

    return vectorizers


def transformer_from_props(props: Dict[str, List[dict]]) -> DocumentTransformer:
    ne_collapser = _collapser_from_props(props.get("ne_collapser", {}), True)
    ents_collapser = _collapser_from_props(props.get("entities_collapser", {}))
    tagger_parser = _tagger_parser_from_props(props.get("tagger_parser", {}))
    ner = _ner_from_props(props.get("ner", {}))
    vectorizers = _vectorizers_from_props(props.get("vectorizers", {}))
    return SequenceDocumentTransformer([*tagger_parser, *ner, *vectorizers, *ents_collapser, *ne_collapser])
