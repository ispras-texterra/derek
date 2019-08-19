import re
from collections import defaultdict

from derek.common.helper import compose_functions
from derek.data.model import Sentence

DELIMITERS = {"-", "–", "―", "‐", "−", "—", "‒", "‑", "─",
              '•', "·", "∙", "⋅", "·",
              '≥', '≤', '~', "∼", "≈",
              '|', "/", "\\", "∶", "†",
              '™', '®', '©', '∷',
              '\'', '´', '׳'}

PROHIBITED_END_TOKENS = {"sp", "spp", "subsp", "approx", "i.e", "e.g", "b.i.d"}


def fix_joined_tokens(tokens, sentences, raw_tokens, delimeters=None):
    if not delimeters:
        delimeters = DELIMITERS

    subtokens_ranges = []

    for token in tokens:
        new_tokens_ranges = []

        span_start = 0
        for char_idx, char in enumerate(token):
            if char not in delimeters:
                continue

            # previous char delimeter check
            if char_idx != span_start:
                new_tokens_ranges.append((span_start, char_idx))

            new_tokens_ranges.append((char_idx, char_idx + 1))
            span_start = char_idx + 1

        if span_start < len(token):
            new_tokens_ranges.append((span_start, len(token)))

        subtokens_ranges.append(new_tokens_ranges)

    return _fix_sentences_and_tokens(subtokens_ranges, tokens, sentences, raw_tokens)


# TODO implement tests
def regex_fix_joined_tokens(tokens, sentences, raw_tokens, regexs):
    subtokens_ranges = []

    for token in tokens:
        new_tokens_ranges = []
        start = 0
        for regex in regexs:
            match = regex.match(token[start:])
            if match:
                end = match.span()[1]
                new_tokens_ranges.append((start, start + end))
                start += end
            else:
                new_tokens_ranges = []
                break
        subtokens_ranges.append(new_tokens_ranges
                                if new_tokens_ranges and start == len(token)
                                else [(0, len(token))])

    return _fix_sentences_and_tokens(subtokens_ranges, tokens, sentences, raw_tokens)


_CUR_RE = re.compile(r"[A-Z]\.")
_NEXT_RE = re.compile(r"[a-z]{4,}")


def fix_joined_names(tokens, sentences, raw_tokens):
    return regex_fix_joined_tokens(tokens, sentences, raw_tokens, [_CUR_RE, _NEXT_RE])


# TODO implement tests
def expand_names(tokens):
    name2indices = defaultdict(set)
    for idx in range(len(tokens) - 1):
        cur, nxt = tokens[idx], tokens[idx + 1]
        if _CUR_RE.fullmatch(cur) and _NEXT_RE.fullmatch(nxt):
            name2indices[(cur[:-1], nxt)].add(idx)

    tokens = list(tokens)
    for name, indices in name2indices.items():
        min_idx = min(indices)
        replacement = None
        for idx in range(min_idx - 2, -1, -1):
            if tokens[idx].startswith(name[0]) and tokens[idx + 1] == name[1]:
                replacement = tokens[idx]
                break
        if replacement is not None:
            for idx in indices:
                tokens[idx] = replacement

    return tokens


_DIGIT_RE = re.compile(r'[0-9]+')


def _replace_digits_with_zero(token):
    # lots of useless numbers: reduce dimensionality
    token = _DIGIT_RE.sub('0', token)
    return token


_REFERENCE_RE = re.compile(r"(\([^\(\)]*et(\s|\.|\.\s)al[^\(\)]*\))", re.IGNORECASE)
_FIGURE_TABLE_REFS_RE = re.compile(r"(\([^\(\)]*(Fig\.|Figure\s|Figs\.|Table\s|Tables\s)[^\(\)]*\))")
_DIGIT_REFS_RE = re.compile(r"(\((\d+|\d+[-–―‐−—‒‑─\-]\d+)(,\s*(\d+|\d+[-–―‐−—‒‑─\-]\d+))*\))")
_PROCENTS_RE = re.compile(r"(\(\d+%?\))")
_REFERENCE_AND_RE = re.compile(r"(\([A-Z][a-z]+ and [A-Z][a-z]+, \d{4}\))")


def eliminate_references_and_figures(raw_txt):
    matches = {}
    for regex in (_REFERENCE_RE, _FIGURE_TABLE_REFS_RE, _DIGIT_REFS_RE, _REFERENCE_AND_RE, _PROCENTS_RE):
        for match in regex.finditer(raw_txt):
            start, end = match.span()
            if end > matches.get(start, -1):
                matches[start] = end

    matches = sorted(matches.items())
    if len(matches) > 1:
        non_overlapping_matches = [matches[0]]

        for i, (start, end) in enumerate(matches[1:]):
            # current i-th match is i-1 in non_overlapping_matches
            prev_start, prev_end = non_overlapping_matches[i]
            if start >= prev_end:
                non_overlapping_matches.append((start, end))
            elif end > prev_end:
                non_overlapping_matches[i] = (non_overlapping_matches[i][0], end)

        matches = non_overlapping_matches

    new_txt = ""
    prev_match_end = 0
    for match_start, match_end in matches:
        new_txt += raw_txt[prev_match_end:match_start]
        prev_match_end = match_end

    new_txt += raw_txt[prev_match_end:]

    return new_txt, matches


def fix_raw_tokens_after_elimination(raw_tokens, matches):
    """
    :param raw_tokens: list of tuples (start, end), must be sorted and non-overlapping
    :param matches: list of tuples (start, end), must be sorted and non-overlapping
    :return: list of tuples (start, end) fixed due matches elimination from raw text
    """
    if not raw_tokens or not matches:
        return raw_tokens

    fixed = []
    match_idx = 0
    match_start, match_end = matches[0]
    shift = 0

    for tok_start, tok_end in raw_tokens:
        cur_tok_start = tok_start + shift
        cur_tok_end = tok_end + shift

        while match_idx is not None and cur_tok_end > match_start:
            cur_tok_end += match_end - match_start
            if match_start <= cur_tok_start:
                cur_tok_start += match_end - match_start

            match_idx += 1

            if match_idx >= len(matches):
                match_idx = None
            else:
                match_start, match_end = matches[match_idx]

        fixed.append((cur_tok_start, cur_tok_end))
        shift = cur_tok_end - tok_end

    return fixed


def _fix_sentences_and_tokens(subtokens_ranges, tokens, sentences, raw_tokens):
    fixed_tokens = []
    fixed_raw_tokens = []
    fixed_sentences = []

    new_sentence_start = 0
    for sent in sentences:
        new_sentence_length = 0

        for i in range(sent.start_token, sent.end_token):
            new_token_ranges = subtokens_ranges[i]
            token = tokens[i]
            raw_token = raw_tokens[i]

            for rng in new_token_ranges:
                fixed_tokens.append(token[rng[0]:rng[1]])
                fixed_raw_tokens.append((raw_token[0] + rng[0], raw_token[0] + rng[1]))
                new_sentence_length += 1

        fixed_sentences.append(Sentence(new_sentence_start, new_sentence_start + new_sentence_length))
        new_sentence_start += new_sentence_length

    return fixed_tokens, fixed_sentences, fixed_raw_tokens


def fix_sentences_ends(tokens, sentences, prohibited_end_tokens: set = None):
    if prohibited_end_tokens is None:
        prohibited_end_tokens = PROHIBITED_END_TOKENS

    fixed_sentences = []

    if not sentences:
        return fixed_sentences

    sent_start = sentences[0].start_token
    for sent in sentences:
        if len(sent) < 2 or not(
                tokens[sent.end_token - 2] in prohibited_end_tokens and tokens[sent.end_token - 1] == "."):

            fixed_sentences.append(Sentence(sent_start, sent.end_token))
            sent_start = sent.end_token

    if sent_start != sentences[-1].end_token:
        fixed_sentences.append(Sentence(sent_start, sentences[-1].end_token))

    return fixed_sentences


QUOTES = frozenset({
    '\u0022',  # quotation mark (")
    '\u0027',  # apostrophe (')
    '\u00ab',  # left-pointing double-angle quotation mark
    '\u00bb',  # right-pointing double-angle quotation mark
    '\u2018',  # left single quotation mark
    '\u2019',  # right single quotation mark
    '\u201a',  # single low-9 quotation mark
    '\u201b',  # single high-reversed-9 quotation mark
    '\u201c',  # left double quotation mark
    '\u201d',  # right double quotation mark
    '\u201e',  # double low-9 quotation mark
    '\u201f',  # double high-reversed-9 quotation mark
    '\u2039',  # single left-pointing angle quotation mark
    '\u203a',  # single right-pointing angle quotation mark
    '\u300c',  # left corner bracket
    '\u300d',  # right corner bracket
    '\u300e',  # left white corner bracket
    '\u300f',  # right white corner bracket
    '\u301d',  # reversed double prime quotation mark
    '\u301e',  # double prime quotation mark
    '\u301f',  # low double prime quotation mark
    '\ufe41',  # presentation form for vertical left corner bracket
    '\ufe42',  # presentation form for vertical right corner bracket
    '\ufe43',  # presentation form for vertical left corner white bracket
    '\ufe44',  # presentation form for vertical right corner white bracket
    '\uff02',  # fullwidth quotation mark
    '\uff07',  # fullwidth apostrophe
    '\uff62',  # halfwidth left corner bracket
    '\uff63',  # halfwidth right corner bracket
})

_QUOTES_RE = re.compile(r'[{}]'.format(''.join(sorted(QUOTES))))


def _replace_quotes_with_std(token, *, sub_for='"'):
    return _QUOTES_RE.sub(sub_for, token)


def _identity(token: str) -> str:
    return token

class StandardTokenProcessor:
    def __init__(self, lowercase: bool, replace_digits: bool, replace_quotes: bool):
        flags_with_processors = [
            (lowercase, str.lower),
            (replace_digits, _replace_digits_with_zero),
            (replace_quotes, _replace_quotes_with_std)
        ]
        processors = [proc for flag, proc in flags_with_processors if flag]
        if processors:
            self.__processor = compose_functions(processors)
        else:
            self.__processor = _identity


    def __call__(self, token: str) -> str:
        return self.__processor(token)

    @classmethod
    def from_props(cls, props: dict):
        return cls(props.get("lower", False), props.get("replace_digits", False), props.get("replace_quotes", False))
