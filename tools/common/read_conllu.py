from typing import Generator
from babylondigger.evaluation.conllu_io import CoNLLUReader

from tools.doc_converters.digger_converter import convert_from_digger_to_derek
from derek.data.model import Document


def read_conllu_file(path: str) -> Generator[Document, None, None]:
    # read 1 sentence per doc
    reader = CoNLLUReader(1)

    for i, digger_doc in enumerate(reader.read_from_file(path)):
        yield convert_from_digger_to_derek(digger_doc, "{}_{}".format(path, i))
