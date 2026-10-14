import json
import re
from pathlib import Path

import numpy as np
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

M = Path('D:/CCAD_Storage/newtopics/p3')
HF_HUB = Path('D:/CCAD_Storage/hf_home/hub')
MODEL = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
MODEL_COMMIT = 'ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562'
READER = 'Qwen/Qwen2.5-0.5B'
READER_COMMIT = '060db6499f32faf8b98477b0a26969ef7d8b9987'
MODEL_PATH = str(HF_HUB / 'models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots' / MODEL_COMMIT)
READER_PATH = str(HF_HUB / 'models--Qwen--Qwen2.5-0.5B/snapshots' / READER_COMMIT)
DATASET = 'edinburgh-dawg/mmlu-redux-2.0'
DATASET_COMMIT = '372ea425445d51e1ba1188c56e5e893f8138621f'
DATASET_PATH = HF_HUB.parent / 'local' / f'mmlu-redux-2.0@{DATASET_COMMIT}'
LETTERS = 'ABCD'
THINK_END = '</think>'
MAX_BOUNDARIES = 25
TEMPERATURE = 0.6
TOP_P = 0.95
MAX_NEW_TOKENS = 4096
INSTRUCTION = 'Please reason step by step, and put your final answer (a single letter A, B, C, or D) within \\boxed{}.'
BOXED = re.compile(r'\\boxed\{\s*(?:\\text(?:bf)?\{\s*)?\(?\s*([A-D])(?![A-Za-z])')
STATED = re.compile(r'(?:[Aa]nswer|ANSWER|[Oo]ption|[Cc]hoice)\**\s*(?:is\s*:?|:)\s*\**\s*\(?\s*([A-D])(?![A-Za-z])')
LAST_LINE = re.compile(r'^\(?([A-D])(?:\)|[.:]|$)(?![A-Za-z])')
BYTE_DECODER = {c: b for b, c in bytes_to_unicode().items()}


def eligible(row):
    # Screening rule: 2 to 6 correct of 8; training questions also need 4 parsable rollouts for their labels.
    return 2 <= row['n_correct'] <= 6 and (row['role'] == 'test' or row['n_valid'] >= 4)


def user_text(q):
    opts = '\n'.join(f'{LETTERS[i]}. {c}' for i, c in enumerate(q['choices']))
    return f"{q['question']}\n\n{opts}\n\n{INSTRUCTION}"


def prompt_ids(tok, q):
    text = tok.apply_chat_template([{'role': 'user', 'content': user_text(q)}], tokenize=False,
                                   add_generation_prompt=True)
    assert text.endswith('<think>\n'), text[-40:]
    return tok(text, add_special_tokens=False)['input_ids']


def parse_answer(response_text):
    # Final answer = text after </think>: the last \boxed{letter}; else the last stated answer ("answer is C",
    # "**Answer: C**", "correct option is C"); else a last line that is an option letter or starts with one ("C. ...").
    if THINK_END not in response_text:
        return None
    tail = response_text.split(THINK_END)[-1].replace('<｜end▁of▁sentence｜>', '')
    for pattern in (BOXED, STATED):
        hits = pattern.findall(tail)
        if hits:
            return hits[-1]
    lines = [l.strip().strip('*').strip() for l in tail.strip().splitlines() if l.strip()]
    m = LAST_LINE.match(lines[-1]) if lines else None
    return m.group(1) if m else None


def token_bytes(tok, ids):
    added = tok.added_tokens_decoder
    pieces = tok.convert_ids_to_tokens(ids)
    out = []
    for i, p in zip(ids, pieces):
        out.append(added[i].content.encode('utf-8') if i in added else bytes(BYTE_DECODER[c] for c in p))
    return out


def sentence_cuts(tok, reasoning_ids):
    # Candidate boundaries, as numbers of reasoning tokens kept: 0, the token ending each sentence (the Thought Anchors
    # splitter: '.', '?' or '!' followed by a space or newline, or a paragraph break), and the end of the reasoning.
    pieces = token_bytes(tok, reasoning_ids)
    ends = np.cumsum([len(p) for p in pieces])
    text = b''.join(pieces)
    cuts, chunk_has_text = [0], False
    for i in range(len(text)):
        if text[i] not in b' \t\n\r\x0b\x0c':
            chunk_has_text = True
        para = text[i:i + 2] == b'\n\n' or text[i:i + 4] == b'\r\n\r\n'
        sent = i < len(text) - 1 and text[i:i + 1] in (b'.', b'?', b'!') and text[i + 1:i + 2] in (b' ', b'\n')
        if (para or sent) and chunk_has_text:
            cut = int(np.searchsorted(ends, i + 1)) + 1
            if cut > cuts[-1]:
                cuts.append(cut)
            chunk_has_text = False
    if len(reasoning_ids) > cuts[-1] and text[ends[cuts[-1] - 1] if cuts[-1] else 0:].strip():
        cuts.append(len(reasoning_ids))
    return cuts


def chosen_cuts(cuts):
    if len(cuts) <= MAX_BOUNDARIES:
        return list(cuts)
    idx = np.unique(np.round(np.linspace(0, len(cuts) - 1, MAX_BOUNDARIES)).astype(int))
    return [cuts[i] for i in idx]


def reasoning_part(tok, gen_ids):
    end_id = tok.convert_tokens_to_ids(THINK_END)
    assert end_id in gen_ids
    return gen_ids[:gen_ids.index(end_id)]


def tv(p, q):
    return 0.5 * float(np.abs(np.asarray(p, float) - np.asarray(q, float)).sum())


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, rows):
    with open(path, 'a') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
