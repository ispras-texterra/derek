from typing import Dict, Callable, Iterable, Union
from argparse import ArgumentParser


def _apply_for_each(func: Callable[[ArgumentParser], Iterable[ArgumentParser]],
                    parsers: Dict[str, Union[ArgumentParser, Iterable[ArgumentParser]]], keys: Iterable[str]):
    result = dict(parsers)
    for key in keys:
        if isinstance(parsers[key], list):
            result[key] = [p for parser in parsers[key] for p in func(parser)]
        elif isinstance(parsers[key], ArgumentParser):
            result[key] = [p for p in func(parsers[key])]
        else:
            raise ValueError()
    return result


def init_segmenter_argparser(parsers: Dict[str, Union[ArgumentParser, Iterable[ArgumentParser]]], keys: Iterable[str]):
    def _get_segmenter(args):
        segmenter = getattr(args, 'segmenter', None)
        if segmenter is None:
            return None
        elif segmenter == 'nltk_segm':
            from derek.data.nltk_segmenter import NLTKTextSegmenter
            model = args.segm_model if args.segm_model != '' else None
            print("Using NLTK segmenter")
            return NLTKTextSegmenter(nltk_model=model)
        elif segmenter == 'nltkpost_segm':
            from derek.data.nltk_segmenter import NLTKTextSegmenter
            model = args.segm_model if args.segm_model != '' else None
            print("Using NLTK segmenter with postprocessing")
            return NLTKTextSegmenter(nltk_model=model, post_processing=True)
        elif segmenter == 'texterra_segm':
            from derek.data.texterra_processing import TexterraSegmentor
            print("Using Texterra segmenter")
            return TexterraSegmentor(args.segm_key_path, args.segm_lang)
        elif segmenter == 'ud_segm':
            from derek.data.udpipe_processor import load_udpipe_model, UDPipeTextSegmentor
            print("Using UDPipe segmenter")
            return UDPipeTextSegmentor(load_udpipe_model(args.segm_model))
        raise ValueError()

    def add_subparsers(parser: ArgumentParser):
        result = []
        subparsers = parser.add_subparsers(dest='segmenter')
        subparsers.required = True

        subparser = subparsers.add_parser('nltk_segm', help='use nltk for segmentation')
        result.append(subparser)
        subparser.add_argument('-segm_model', type=str, metavar='<nltk model to use>', default='',
                               help='model to use')

        subparser = subparsers.add_parser('nltkpost_segm', help='use nltk with postprocessing for segmentation')
        result.append(subparser)
        subparser.add_argument('-segm_model', type=str, metavar='<nltk model to use>', default='',
                               help='model to use')

        subparser = subparsers.add_parser('texterra_segm', help='use texterra for segmentation')
        result.append(subparser)
        subparser.add_argument('-segm_key_path', type=str, metavar='<texterra_key_path>',
                               help='file with Texterra API key')

        subparser.add_argument('-segm_lang', type=str, metavar='<texterra_lang>',
                               help='language for Texterra API')

        subparser = subparsers.add_parser('ud_segm', help='use udpipe1.2 for segmentation')
        result.append(subparser)
        subparser.add_argument('segm_model', type=str, metavar='<udpipe_model_path>',
                               help='model to use')

        return result

    return _apply_for_each(add_subparsers, parsers, keys), _get_segmenter


def get_dataset_directories_argparser(description: str):
    def process_args(args):
        names, inputs = args.name, args.input

        if names is None:
            if len(inputs) != 1:
                raise ValueError(
                    'If you have more than 1 collection to process, '
                    'please specify collection name for each input directory')
            names = [None]
        else:
            if len(names) != len(inputs):
                raise ValueError('Please specify collection name for each input directory')

        return list(zip(names, inputs))

    parser = ArgumentParser(description=description)

    parser.add_argument('-name', type=str, action='append', metavar='<input collection name>',
                        help='dataset collection name')
    parser.add_argument('-input', type=str, action='append', metavar='<input directory>', required=True,
                        help='directory with dataset collection files')

    parser.add_argument('-o', type=str, dest='output_directory', metavar='<output directory>',
                        required=True, help='output pkl directory')

    return parser, process_args
