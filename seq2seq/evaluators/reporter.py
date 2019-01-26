from __future__ import division
import shutil

from collections import Counter, defaultdict
import logging

import numpy as np

from seq2seq.constants import EOS_token, SOS_token
from utils.news_evaluation_script import news_evaluation
from utils.news_evaluation_script.news_evaluation import compute_edit_dist as ED
from seq2seq.constants import STEP

__author__ = 'Shyam'


def get_decoded_words(decoded_outputs):
    ans = []
    # print(decoded_outputs)
    for score, output in decoded_outputs:
        if output[-1] == EOS_token:
            output = output[:-1]
        if output[0] == SOS_token:
            output = output[1:]
        output = [p for p in output if p != STEP]
        output = " ".join(output)
        ans.append((score, output))
    # print(ans)
    return ans


def compute_acc_at_position(pred_dict, gold_dict, pos):
    correct = 0
    for src_word in gold_dict:
        gold = gold_dict[src_word]
        preds = pred_dict[src_word]
        # print(preds,gold)
        if preds[pos] == gold[0]:
            correct += 1
    # print(correct)
    return correct


def print_evauation_details(pred_dict, gold_dict, header="all", vocab=None, beam_width=None):
    acc_map, f, f_best_match, mrr, map_ref, acc_10, edit_dist, nrm_edit_dist = news_evaluation.evaluate(pred_dict=pred_dict,
                                                                                         gold_dict=gold_dict)
    N = len(acc_map)
    if N == 0:
        logging.info("N is 0, returning ...")
        return 0.0
    edit_dist_freqs = Counter(list(edit_dist.values()))
    # for k in edit_dist:
    #     print(k,edit_dist[k])
    mean_ed_at_1 = np.mean(list(edit_dist.values()))
    std_ed_at_1 = np.std(list(edit_dist.values()))
    mean_ned_at_1 = np.mean(list(nrm_edit_dist.values()))
    median_ed_at_1 = np.median(list(edit_dist.values()))
    acc_num = float(sum([acc_map[src_word] for src_word in acc_map.keys()]))
    acc10_num = float(sum([acc_10[src_word] for src_word in acc_10.keys()]))
    accuracy = acc_num / N
    accuracy10 = acc10_num / N
    macro_f1 = float(sum([f[src_word] for src_word in f.keys()])) / N
    logging.info(20 * "*" + header + 20 * "*")
    logging.info('ACC:          %f (%d/%d)', accuracy, acc_num, N)
    logging.info('Mean F-score: %f', macro_f1)
    logging.info('Mean ED@1: %f+-%.3f', mean_ed_at_1,std_ed_at_1)
    logging.info('Mean NED@1: %f', mean_ned_at_1)
    logging.info('Median ED@1: %f', median_ed_at_1)
    for d in range(3):
        logging.info('edit dist of %d: %f (%d/%d)', d, edit_dist_freqs[d] / N, edit_dist_freqs[d], N)

    if beam_width is not None:
        for d in range(beam_width):
            acc_at_d = compute_acc_at_position(pred_dict=pred_dict, gold_dict=gold_dict, pos=d)
            logging.info("acc at %d: %.3f (%d/%d)", d, acc_at_d / N, acc_at_d, N)
    # logging.info('MRR:          %f', float(sum([mrr[src_word] for src_word in mrr.keys()])) / N)
    # logging.info('MAP_ref:      %f', float(sum([map_ref[src_word] for src_word in map_ref.keys()])) / N)
    logging.info('ACC@10:       %f (%d/%d)', accuracy10, acc10_num, N)
    return accuracy, accuracy10


class AccReporter:
    def __init__(self, args, dump_file=None):
        self.best_acc = 0
        self.args = args
        self.best_acc10 = 0
        self.best_eng_acc = 0
        self.best_nat_acc = 0
        self.best_seen = 0
        self.best_epoch = 0
        self.dump_file = dump_file

    def print_details(self, epoch, gold_dict, pred_dict, header="all"):
        beam_width = self.args["beam_width"]
        # epoch, gold_dict, pred_dict, header = "all"
        accuracy, accuracy10 = print_evauation_details(gold_dict=gold_dict, pred_dict=pred_dict,
                                                       header=header, beam_width=beam_width)
        return accuracy, accuracy10

    def report_eval(self, epoch, seen, examples, evaler):
        pred_dict, gold_dict = {}, {}
        eng_pred_dict, eng_gold_dict = {}, {}
        nat_pred_dict, nat_gold_dict = {}, {}
        correct = 0
        correct_nat = 0
        correct_eng = 0
        if self.dump_file is not None:
            out = open(self.dump_file, "w")
        else:
            out = None
        eng_nwords = sum([1 for (_, _, weight, is_eng) in examples if is_eng])
        nat_nwords = sum([1 for (_, _, weight, is_eng) in examples if not is_eng])
        for idx, example in enumerate(examples):
            x, y, weight, is_eng = example
            # print(weight,is_eng)
            if idx > 0 and idx % 200 == 0:
                logging.info("running infer on example %d", idx)

            decoded_outputs = evaler.infer_on_example(sentence=x)
            scores_and_words = get_decoded_words(decoded_outputs)
            decoded_words = [w for s, w in scores_and_words]

            key = x.replace(" ", "")

            pred_dict[key] = decoded_words
            gold_dict[key] = [y]

            if is_eng:
                eng_pred_dict[key] = decoded_words
                eng_gold_dict[key] = [y]
            else:
                nat_pred_dict[key] = decoded_words
                nat_gold_dict[key] = [y]

            if decoded_words[0] == y:
                correct += 1
                if is_eng:
                    correct_eng += 1
                else:
                    correct_nat += 1

            if out is not None:
                edit_dists = ";".join([str(ED(ref=y, candidate=word)) for score, word in scores_and_words])
                beam_outputs = ";".join([word for score, word in scores_and_words])
                beam_scores = ";".join([str(score) for score, word in scores_and_words])
                buf = "%s\t%s\t%s\t%s\t%s\t%s\n" % (x, y, is_eng, beam_outputs, beam_scores, edit_dists)
                out.write(buf)

        logging.info("accuracy %d/%d=%.2f", correct, len(examples), correct / len(examples))
        NAT_ACC = 0.0 if nat_nwords == 0 else correct_nat / nat_nwords
        ENG_ACC = 0.0 if eng_nwords == 0 else correct_eng / eng_nwords
        logging.info("accuracy (nat) %d/%d=%.2f", correct_nat, nat_nwords, NAT_ACC)
        logging.info("accuracy (eng) %d/%d=%.2f", correct_eng, eng_nwords, ENG_ACC)
        if out is not None:
            out.close()
        all_acc, all_acc10 = self.print_details(header="total", epoch=epoch, gold_dict=gold_dict, pred_dict=pred_dict)
        if eng_nwords == 0:
            eng_acc, eng_acc10 = 0, 0
        else:
            eng_acc, eng_acc10 = self.print_details(header="eng", epoch=epoch, gold_dict=eng_gold_dict,
                                                    pred_dict=eng_pred_dict)
        nat_acc, nat_acc10 = self.print_details(header="nat", epoch=epoch, gold_dict=nat_gold_dict,
                                                pred_dict=nat_pred_dict)
        ret_val = False
        if eng_acc > self.best_eng_acc:
            self.best_eng_acc = eng_acc
        if nat_acc > self.best_nat_acc:
            self.best_nat_acc = nat_acc
        if all_acc10 > self.best_acc10:
            self.best_acc10 = all_acc10
        if all_acc > self.best_acc:
            self.best_acc = all_acc
            self.best_seen = seen
            self.best_epoch = epoch
            ret_val = True
        if ret_val is True and self.dump_file is not None:
            bestpred = self.dump_file + '_best.txt'
            logging.info("saving best predictions to file %s",bestpred)
            shutil.copyfile(self.dump_file, bestpred)
        logging.info("best accuracy: %.3f", self.best_acc)
        logging.info("best accuracy@10: %.3f", self.best_acc10)
        logging.info("best after %d mini-batches (%d epoch)", self.best_seen, self.best_epoch)
        logging.info("best eng accuracy: %.3f", self.best_eng_acc)
        logging.info("best nat accuracy: %.3f", self.best_nat_acc)
        return ret_val, self.best_acc
