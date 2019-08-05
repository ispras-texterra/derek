import unittest
from itertools import product

from tools.generate_dataset import build_argparser


class TestArgparser(unittest.TestCase):

    parser, _, _, _ = build_argparser()

    def test_empty_parseline(self):
        with self.assertRaises(SystemExit):
            TestArgparser.parser.parse_args([])

    def test_correct_lines(self):
        def _correct_args(reader: str, segmenter: str):
            result = ['-name', 'train', '-input', 'train_dir', '-o', 'output_dir', reader]
            correct_args = {
                'name': ['train'],
                'input': ['train_dir'],
                'output_directory': 'output_dir',
                'reader': reader
            }

            result.append(segmenter)
            correct_args['segmenter'] = segmenter
            if segmenter == 'texterra_segm':
                result.extend(['-segm_key_path', 'segm_texterra_key_path'])
                correct_args['segm_key_path'] = 'segm_texterra_key_path'
            elif segmenter == 'ud_segm':
                result.append('udpipe_model_path')
                correct_args['segm_model'] = 'udpipe_model_path'

            return result, correct_args

        for reader, segmenter in product(
                ['BioNLP', 'ChemProt', 'BRAT'],
                ['nltk_segm', 'nltkpost_segm', 'texterra_segm', 'ud_segm']):
            command_line, correct_args = _correct_args(reader, segmenter)
            args = TestArgparser.parser.parse_args(command_line)

            for key, value in correct_args.items():
                self.assertEqual(value, getattr(args, key, None))
