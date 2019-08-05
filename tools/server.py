import json
import argparse
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest

from derek import classifier_for, transformer_from_props
from derek.data.helper import align_raw_entities
from derek.data.model import Document, Paragraph
from tools.common.argparsers import init_segmenter_argparser


app = Flask(__name__)
model = None


class Model:
    def __init__(self, segmenter, transformer, ent_clf, rel_clf):
        self.segmenter = segmenter
        self.transformer = transformer
        self.ent_clf = ent_clf
        self.rel_clf = rel_clf

    def predict_docs(self, raw_docs, need_entities, need_relations):
        return [self.predict_doc(text, entities, need_entities, need_relations) for text, entities in raw_docs]

    def predict_doc(self, text, raw_entities, need_entities, need_relations):
        """
        :param raw_entities: list of {"id","start","end","type"} dicts
        :return: (raw_entities, raw_relations) where:
          raw_entities is list of {"id","start","end","type"} dicts or None
          raw_relations is list of {"first","second","type"} dicts or None
        """

        if self.ent_clf is None and raw_entities is None and (need_entities or need_relations):
            raise BadRequest("Server doesn't support entities recognition")

        if self.rel_clf is None and need_relations:
            raise BadRequest("Server doesn't support relation extraction")

        tokens, sentences, raw_tokens = self.segmenter.segment(text)
        doc = Document("_", tokens, sentences, [Paragraph(0, len(sentences))])
        doc = self.transformer.transform(doc)

        entities = None
        if raw_entities is not None:
            if need_relations:
                entities = align_raw_entities(raw_entities, raw_tokens)
            if not need_entities:
                raw_entities = None
        else:
            if need_entities or need_relations:
                entities = self.ent_clf.predict_doc(doc)
            if need_entities:
                raw_entities = self._to_raw_entities(entities, raw_tokens)

        raw_relations = None
        if need_relations:
            doc = doc.with_entities(entities)
            relations = self.rel_clf.predict_doc(doc)
            raw_relations = self._to_raw_relations(relations)

        return raw_entities, raw_relations

    @staticmethod
    def _to_raw_entities(entities, raw_tokens):
        ret = []
        for i, ent in enumerate(entities):
            start = raw_tokens[ent.start_token][0]
            end = raw_tokens[ent.end_token - 1][1]
            # NER always produces id=="generated"
            ret.append({"id": f"T{i+1}", "start": start, "end": end, "type": ent.type})
        return ret

    @staticmethod
    def _to_raw_relations(relations):
        ret = []
        for rel in relations:
            ret.append({"first": rel.first_entity.id, "second": rel.second_entity.id, "type": rel.type})
        return ret


class View:
    @staticmethod
    def raw_docs_from_json(docs_json):
        if not isinstance(docs_json, list):
            raise BadRequest("list of documents is expected")
        return [View.raw_doc_from_json(doc_json) for doc_json in docs_json]

    @staticmethod
    def raw_doc_from_json(doc_json):
        if "text" not in doc_json:
            raise BadRequest("text should be provided for document")

        text = doc_json["text"]
        if "entities" not in doc_json:
            return text, None
        entities = doc_json["entities"]

        for json_ent in entities:
            for field in {"id", "start", "end", "type"}:
                if field not in json_ent:
                    raise BadRequest("id, start, end, type should be provided for entity")
            start, end = json_ent["start"], json_ent["end"]
            if start < 0 or end <= start or end > len(text):
                raise BadRequest("(0 <= start < end <= text_length) should hold for entity")
        # we are OK to consume raw_entities-like JSON format so no conversion for now
        return text, entities

    @staticmethod
    def raw_docs_to_json(raw_docs):
        return [View.raw_doc_to_json(*raw_doc) for raw_doc in raw_docs]

    @staticmethod
    def raw_doc_to_json(entities, relations):
        ret = {}
        if entities is not None:
            # we are OK to produce raw_entities-like JSON format so no conversion for now
            ret["entities"] = entities
        if relations is not None:
            # we are OK to produce raw_relations-like JSON format so no conversion for now
            ret["relations"] = relations
        return ret


@app.route('/', methods=['POST'])
def controller():
    need_entities = (request.args.get('entities', '0') == '1')
    need_relations = (request.args.get('relations', '0') == '1')
    raw_docs = View.raw_docs_from_json(request.json)

    predictions = model.predict_docs(raw_docs, need_entities, need_relations)

    return jsonify(View.raw_docs_to_json(predictions))


def safe_with_clf(task_name, path, func):
    if path is not None:
        with classifier_for(task_name)(path) as clf:
            func(clf)
    else:
        func(None)


def run_app(host, port, segmenter, transformer, ent_clf, rel_clf):
    global app, model
    model = Model(segmenter, transformer, ent_clf, rel_clf)
    app.run(host=host, port=port)


def main():
    argparser = argparse.ArgumentParser(description='HTTP server for DEREK')
    argparser.add_argument('-remote', dest='remote', action='store_true',
                           help='should listen for remote connections')
    argparser.add_argument('-port', type=int, dest='port', metavar='<port number>',
                           required=False, help='port to listen on')
    argparser.add_argument('-ner', type=str, dest='ner_path', metavar='<NER model path>',
                           required=False, help='path to NER model')
    argparser.add_argument('-rel_ext', type=str, dest='rel_ext_path', metavar='<rel_ext model path>',
                           required=False, help='path to rel_ext model')
    argparser.add_argument('-transformer_props', type=str, dest='transformer_props', metavar='<transformers.json>',
                           required=False, help='path to transformer props')
    parsers = {"main": argparser}
    parsers, segmentor_factory = init_segmenter_argparser(parsers, parsers.keys())

    args = argparser.parse_args()
    host = "0.0.0.0" if args.remote else None

    if args.transformer_props is not None:
        with open(args.transformer_props, 'r', encoding='utf-8') as f:
            transformer_props = json.load(f)
    else:
        transformer_props = {}

    segmenter = segmentor_factory(args)

    with transformer_from_props(transformer_props) as transformer:
        safe_with_clf(
            'ner', args.ner_path,
            lambda ent_clf: safe_with_clf(
                'rel_ext', args.rel_ext_path,
                lambda rel_clf: run_app(host, args.port, segmenter, transformer, ent_clf, rel_clf)
            )
        )


if __name__ == "__main__":
    main()
