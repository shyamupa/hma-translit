from __future__ import division
from __future__ import print_function

import logging
import random

from baseline import align_utils
from seq2seq.constants import ALIGN_SYMBOL
from seq2seq.constants import STEP


# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def safe_replace_spaces(s):
    s = s.replace("  ", "#")
    s = s.replace(" ", "")
    s = s.replace("#", " ")
    return s


def subsample_examples(examples, frac, single_token):
    new_examples = []
    for ex in examples:
        fr, en, weight, is_eng = ex
        frtokens, entokens = fr.split(" "), en.split(" ")
        if len(frtokens) != len(entokens): continue
        if single_token:
            if len(frtokens) > 1 or len(entokens) > 1: continue
        for frtok, entok in zip(frtokens, entokens):
            new_examples.append((frtok, entok, weight, is_eng))
    examples = new_examples
    logging.info("new examples %d", len(examples))
    # subsample if needed
    random.shuffle(examples)
    if frac < 1.0:
        tmp = examples[0:int(frac * len(examples))]
        examples = tmp
    elif frac > 1.0:
        tmp = examples[0:int(frac)]
        examples = tmp
    return examples


def read_examples(fpath, native_or_eng="both", remove_spaces=False, weight=1.0):
    examples = []
    bad = 0
    for idx, l in enumerate(open(fpath)):
        parts = l.strip().split('\t')
        if len(parts) == 3:
            fr_sent, en_sent = parts[:2]
            is_eng = True
        elif len(parts) == 2:
            # print(parts)
            fr_sent, en_sent = parts
            is_eng = False
        elif len(parts) == 4:
            fr_sent, en_sent, is_eng = parts[:3]
            is_eng = True if is_eng=="True" else False
        else:
            logging.info("#%d bad line %d %s", bad, idx, parts)
            bad += 1
            continue
        if remove_spaces:
            # fr_sent = fr_sent.replace(" ", "")
            # en_sent = en_sent.replace(" ", "")
            fr_sent = safe_replace_spaces(fr_sent)
            en_sent = safe_replace_spaces(en_sent)
        if native_or_eng == "nat" and not is_eng:
            examples.append((fr_sent, en_sent, weight, is_eng))
        elif native_or_eng == "eng" and is_eng:
            examples.append((fr_sent, en_sent, weight, is_eng))
        elif native_or_eng == "both":
            examples.append((fr_sent, en_sent, weight, is_eng))
        else:
            pass
        if "!!!" in l and not is_eng:
            logging.info("wierd line %s", l)
    num_engs = sum([1 if ex[-1] == True else 0 for ex in examples])
    num_nats = sum([1 if ex[-1] == False else 0 for ex in examples])
    logging.info("read %d examples in \"%s\" mode", len(examples), native_or_eng)
    logging.info("# engs %d", num_engs)
    logging.info("# nats %d", num_nats)
    return examples


def align_examples(examples, seed, algo="mcmc"):
    logging.info("aligning using %d examples", len(examples))

    pairs = [(x, y) for x, y, weight, is_eng in examples]
    is_eng_list = [(weight,is_eng) for x, y, weight, is_eng in examples]
    if algo == "dumb":
        raise NotImplementedError
    else:
        aligned_pairs = align_utils.mcmc_align(pairs, ALIGN_SYMBOL, seed=seed)
    ans = [(ax, ay, weight, is_eng) for (ax, ay), (weight, is_eng) in zip(aligned_pairs, is_eng_list)]
    return ans


def load_aligned_data(examples, seed, mode=None):
    ans = []

    if mode == "mcmc":
        aligned_data = align_examples(examples=examples, seed=seed)
    else:
        # No alignments --> seq2seq
        aligned_data = examples
    for x, y, weight, is_eng in aligned_data:
        if mode == "mcmc":
            raw_x, raw_y = x.replace(ALIGN_SYMBOL, ""), y.replace(ALIGN_SYMBOL, "")
            raw_x, raw_y = ' '.join(list(raw_x)), ' '.join(list(raw_y))
        elif mode == "m2m":
            raise NotImplementedError
        else:
            raw_x, raw_y = x, y
        xs, ys = ' '.join(list(x)), ' '.join(list(y))
        ans.append((raw_x, raw_y, xs, ys, weight, is_eng))
    return ans


def oracle_action(example):
    raw_x, raw_y, x, y, weight, is_eng = example
    x = x.split(' ')
    y = y.split(' ')
    actions = []
    inputs = []
    alignments = list(zip(x, y))
    for idx, a in enumerate(alignments):
        # if 1-to-0 alignment, then step
        if a[1] == ALIGN_SYMBOL:
            actions.append(STEP)
            inputs.append(a[0])
        else:
            actions.append(a[1])
            inputs.append(a[0])
            if idx + 1 < len(alignments) and alignments[idx + 1][0] != ALIGN_SYMBOL:
                actions.append(STEP)
    return inputs,actions

