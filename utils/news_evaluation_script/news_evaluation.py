#!/usr/bin/python

import codecs
import logging
import sys
import getopt
from os.path import basename
import xml.dom.minidom
from xml.dom.minidom import Node
# what we expect to find inside <TransliterationResults> tag...
import editdistance
import numpy as np
RESULT_HEADER_ATTR = ('SourceLang', 'TargetLang', 'GroupID', 'RunID', 'RunType', 'Comments')
# ... and inside <TransliterationCorpus> tag
CORPUS_HEADER_ATTR = ('SourceLang', 'TargetLang', 'CorpusID', 'CorpusType', 'CorpusSize', 'CorpusFormat')

MAX_CANDIDATES = 10


def usage():
    '''
    User's manual
    '''
    print('''
Transliteration results evaluation script for NEWS:
Named Entities Workshop - Shared Task on Transliteration

Usage:
    [python] %s [-h|--help] [-i|--input-file=<filename>]
                [-o|--output-file=<filename>]
                -t|--test-file=<filename>
                --max-candidates=<int>
                [--map-n=<int>]

Options:
    -h, --help         : Print this help and quit

    --check-only       : Only checks that the file is in correct format.
                         When this option is given, only one file is
                         accepted, either stdin or given with -i option.

    -i, --input-file   : Input file with transliteration results in NEWS
                         XML format. If not given, standard input is used.

    -t, --test-file    : Test file with transliteration references in NEWS
                         XML format.

    -o, --output-file  : Output file with contribution of each source word
                         to each metric. If not given, no details are written.
                         The output file contains comma-separated values
                         and can be opened by a spreadsheet application
                         such as Microsoft Excel or OpenOffice Calc.
                         The values in the file are not divided by the
                         number of source names.

    --max-candidates   : Maximum number of transliteration candidates
                         to consider. By default, maximum 10 candidates are
                         considered for evaluation according to the
                         NEWS 2009 whitepaper.


The input files must be in UTF-8.

Example:
    %s -i translit_results.xml -t test.xml -o evaluation_details.csv

The detailed description of the metrics is in the NEWS 2010 whitepaper.

For comments, suggestions and bug reports email to Vladimir Pervouchine
vpervouchine@i2r.a-star.edu.sg.
    ''' % (basename(sys.argv[0]), basename(sys.argv[0])))


def get_options():
    '''
    Extracts command line arguments
    '''
    input_fname = None
    output_fname = None
    test_fname = None
    max_candidates = MAX_CANDIDATES
    check_only = False
    silent = False

    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'hi:o:t:',
                                       ['help', 'input-file=', 'output-file=', 'test-file=',
                                        'check-only', 'silent'])
    except getopt.GetoptError as err:
        sys.stderr.write('Error: %s\n' % err)
        usage()
        sys.exit(1)

    for o, a in opts:
        if o in ('-i', '--input-file'):
            input_fname = a
        elif o in ('-o', '--output-file'):
            output_fname = a
        elif o in ('-t', '--test-file'):
            test_fname = a
        elif o in ('-h', '--help'):
            usage()
            sys.exit()
        elif o in ('--check-only',):
            check_only = True
        elif o in ('--silent',):
            silent = True
        elif o in ('--max-candidates',):
            try:
                max_candidates = int(a)
            except ValueError:
                sys.stderr.write('Error: --max-candidates takes integer argument (you provided %s).\n' % a)
                sys.exit(1)
            if max_candidates < 1:
                sys.stderr.write('Error: --max-candidates must be above 0.\n')
                sys.exit(1)

        else:
            sys.stderr.write('Error: unknown option %s. Type --help to see the options.\n' % o)
            sys.exit(1)

    if check_only:
        if test_fname or output_fname:
            sys.stderr.write('No test file or output file is required to check the input format.\n')
            sys.exit(1)
    else:
        if not test_fname:
            sys.stderr.write('Error: no test file provided.\n')
            sys.exit(1)

    return input_fname, output_fname, test_fname, max_candidates, check_only, silent


def parse_xml(f_in, max_targets=None):
    '''
    Parses XML input and test files with paranoid error checking.
    Returns a tuple of header and content
    Content is a dictionary with source names as keys and contains lists of target names.
    If max_targets is given, the number of target names in the list is cut up to max_targets names.
    Header is a dictionary of header data
    '''

    stderr = codecs.getwriter('utf-8')(sys.stderr)

    doc = xml.dom.minidom.parse(f_in)
    if doc.encoding.lower() != 'utf-8':
        raise IOError('Invalid encoding. UTF-8 is required but %s found' % doc.encoding)

    # try results
    header = doc.getElementsByTagName('TransliterationTaskResults')
    is_results = True
    if not header:
        # try corpus
        is_results = False
        header = doc.getElementsByTagName('TransliterationCorpus')
    if not header:
        raise IOError('Unknown file. TransliterationTaskResults and TransliterationCorpus tags are missing')
    if len(header) > 1:
        raise IOError('Invalid file. Several headers were found')
    header = header[0]

    # parse the comments
    header_data = {}
    if is_results:
        attr_list = RESULT_HEADER_ATTR
    else:
        attr_list = CORPUS_HEADER_ATTR

    for attr in attr_list:
        header_data[attr] = header.getAttribute(attr)

    # parse the data
    data = {}
    for node in doc.getElementsByTagName('Name'):
        # we ignore the name ID unless encounter error
        # get the source name
        s = node.getElementsByTagName('SourceName')
        # import ipdb
        # ipdb.set_trace()
        if not s:
            raise IOError('Invalid file format: one of <Name> nodes does not have <SourceName>')
        if s[0].childNodes[0].nodeType == Node.TEXT_NODE:
            src_name = s[0].childNodes[0].data.strip('" ')  # strip quotes and spaces in case someone adds them
            src_name = src_name.upper()  # convert to uppercase in case it's a language where case matters
        else:
            raise IOError('For Name ID %s no SourceName was found or its format is invalid' % node.getAttribute('ID'))

        # get the targets
        t = node.getElementsByTagName('TargetName')
        if not t:
            raise IOError('Invalid file format: one of <Name> nodes does not have <TargetName>')
        # we'll read target names as tuples: (target_name, ID) so that the list can later be sorted
        # according to the ID, which is going to be removed after that.
        tgt_list = []
        for t_node in t:
            # get the ID, which is the rank for transliteration candidates
            try:
                tgt_id = int(t_node.getAttribute('ID'))
            except ValueError:
                raise IOError(
                    'For name ID %s (%s) one of target names have invalid ID' % (node.getAttribute('ID'), src_name))
            # get the word
            if not t_node.childNodes:
                raise IOError('For name ID %s (%s) one of the target names ID %s is empty' % (
                    node.getAttribute('ID'), src_name, tgt_id))
            if t_node.childNodes[0].nodeType == Node.TEXT_NODE:
                tgt_name = t_node.childNodes[0].data.strip('" ')
                if tgt_name:
                    tgt_name = tgt_name.upper()  # convert to uppercase in case it matters
                    tgt_list.append((tgt_name, tgt_id))
                else:
                    stderr.write(
                        'Warning: Name ID %s (%s) contains empty target words\n' % (node.getAttribute('ID'), src_name))
            else:
                raise IOError('For name ID %s (%s) one of target names ID %s have invalid format' % (
                    node.getAttribute('ID'), src_name, tgt_id))

        # sort by ID
        if not tgt_list:
            stderr.write('Warning: no non-empty target words found for name ID %s (%s). This name is ignored.\n' % (
                node.getAttribute('ID'), src_name))

        else:

            tgt_list.sort(key=lambda x: x[1])
            # check for duplicate IDs: if there are any, they must be adjacent elements after sorting
            # we only care for IDs to be unique in the results file because IDs are ranks there.
            if is_results:
                for i in range(len(tgt_list) - 1):
                    if tgt_list[i][1] == tgt_list[i + 1][1]:
                        raise IOError(
                            'XML results file contains duplicate IDs for transliterations of word %s' % src_name)

            # cut up to max_targets
            if max_targets:
                tgt_list = tgt_list[0:max_targets]

            data[src_name] = [tgt[0] for tgt in tgt_list]  # remove IDs, we don't need them anymore

            # test (codecs.getwriter('utf-8')(sys.stdout)).write('Name: %s\n' % (data[src_name][0]))
            # test raise IOError('%s' % data[src_name][0])

    return header_data, data, is_results


def LCS_length(s1, s2):
    '''
    Calculates the length of the longest common subsequence of s1 and s2
    s1 and s2 must be anything iterable
    The implementation is almost copy-pasted from Wikibooks.org
    '''
    m = len(s1)
    n = len(s2)
    # An (m+1) times (n+1) matrix
    C = [[0] * (n + 1) for i in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                C[i][j] = C[i - 1][j - 1] + 1
            else:
                C[i][j] = max(C[i][j - 1], C[i - 1][j])
    return C[m][n]


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_edit_dist(ref, candidate):
    ref = ref.replace(" ", "")
    candidate = candidate.replace(" ", "")
    return editdistance.eval(ref, candidate)


def f_score(candidate, references):
    '''
    Calculates F-score for the candidate and its best matching reference
    Returns F-score and best matching reference
    '''
    # determine the best matching reference (the one with the shortest ED)
    best_ref = references[0]
    if len(candidate) == 0:
        return 0.0, best_ref, 100, 0
    best_ref_lcs = LCS_length(candidate, references[0])
    for ref in references[1:]:
        lcs = LCS_length(candidate, ref)
        if (len(ref) - 2 * lcs) < (len(best_ref) - 2 * best_ref_lcs):
            best_ref = ref
            best_ref_lcs = lcs

    # try:
    precision = float(best_ref_lcs) / float(len(candidate))
    recall = float(best_ref_lcs) / float(len(best_ref))
    # except:
    #    import ipdb
    #    ipdb.set_trace()
    # edit_dist = levenshtein(best_ref,candidate)
    # edit_dist = Levenshtein.distance(best_ref,candidate)
    edit_dist = compute_edit_dist(ref=best_ref, candidate=candidate)
    nrm_edit_dist = edit_dist / len(best_ref)
    # print("best_ref:", best_ref, "candidate:", candidate, "edit_dist:", edit_dist)
    if best_ref_lcs:
        return 2 * precision * recall / (precision + recall), best_ref, edit_dist, nrm_edit_dist
    else:
        return 0.0, best_ref, edit_dist, nrm_edit_dist


def mean_average_precision(candidates, references, n):
    '''
    Calculates mean average precision up to n candidates.
    '''

    total = 0.0
    num_correct = 0
    for k in range(n):
        if k < len(candidates) and (candidates[k] in references):
            num_correct += 1
        total += float(num_correct) / float(k + 1)

    return total / float(n)


def inverse_rank(candidates, reference):
    '''
    Returns inverse rank of the matching candidate given the reference
    Returns 0 if no match was found.
    '''
    rank = 0
    while (rank < len(candidates)) and (candidates[rank] != reference):
        rank += 1
    if rank == len(candidates):
        return 0.0
    else:
        return 1.0 / (rank + 1)


def evaluate(pred_dict, gold_dict):
    '''
    REMEMBER! -- pred_dict and gold_dict should be word to lists dictionaries.
    The list will be ordered in descending order in the pred_dict.
    If you only have a single reference make sure its in a list.

    Evaluates all metrics to save looping over input_data several times
    n is the map-n parameter
    Returns acc, f_score, mrr, map_ref, map_n
    '''
    mrr = {}
    acc = {}
    f = {}
    f_best_match = {}
    # map_n = {}
    map_ref = {}
    # map_sys = {}
    acc_10 = {}
    edit_dist = {}
    nrm_edit_dist = {}
    for src_word in gold_dict.keys():
        if src_word in pred_dict:
            candidates = pred_dict[src_word]
            references = gold_dict[src_word]

            acc[src_word] = max([int(candidates[0] == ref) for ref in references])  # either 1 or 0

            f[src_word], f_best_match[src_word], edit_dist[src_word], nrm_edit_dist[src_word] = f_score(candidates[0], references)

            mrr[src_word] = max([inverse_rank(candidates, ref) for ref in references])

            # map_n[src_word] = mean_average_precision(candidates, references, n)
            map_ref[src_word] = mean_average_precision(candidates, references, len(references))
            # map_sys[src_word] = mean_average_precision(candidates, references, len(candidates))

            ## compute accuracy at 10- Anoop
            acc_10[src_word] = max([int(ref in candidates) for ref in references])  # either 1 or 0

        else:
            logging.error('Warning: No transliterations found for word %s\n' % src_word)
            mrr[src_word] = 0.0
            acc[src_word] = 0.0
            f[src_word] = 0.0
            edit_dist[src_word] = np.infty
            nrm_edit_dist[src_word] = 1.0
            f_best_match[src_word] = ''
            # map_n[src_word] = 0.0
            map_ref[src_word] = 0.0
            # map_sys[src_word] = 0.0
            # Anoop
            acc_10[src_word] = 0.0

    return acc, f, f_best_match, mrr, map_ref, acc_10, edit_dist, nrm_edit_dist  # added by Anoop


def write_details(output_fname, input_data, test_data, acc, f, f_best_match, mrr, map_ref, acc_10):
    '''
    Writes detailed results to CSV file
    '''
    if output_fname == '-':
        f_out = codecs.getwriter('utf-8')(sys.stdout)
    else:
        f_out = codecs.open(output_fname, 'w', 'utf-8')

    f_out.write('%s\n' % (
        ','.join(['"Source word"', '"First candidate"', '"ACC"', '"ACC-10"', '"F-score"', '"Best matching reference"',
                  '"MRR"', '"MAP_ref"', '"References"'])))

    for src_word in test_data.keys():
        if src_word in input_data:
            first_candidate = input_data[src_word][0]
        else:
            first_candidate = ''

        f_out.write('%s,%s,%f,%f,%f,%s,%f,%f,%s\n' % (
            src_word, first_candidate, acc[src_word], acc_10[src_word], f[src_word], f_best_match[src_word],
            mrr[src_word],
            map_ref[src_word], '"' + ' | '.join(test_data[src_word]) + '"'))

    if output_fname != '-':
        f_out.close()


def main():
    input_fname, output_fname, test_fname, max_candidates, check_only, silent = get_options()
    stderr = codecs.getwriter('utf-8')(sys.stderr)

    if not input_fname:
        f = sys.stdin
    else:
        f = input_fname
    try:
        input_header, input_data, is_results = parse_xml(f, max_targets=max_candidates)
    except IOError as e:
        error_message = e.strerror
        if not error_message:
            error_message = e.message
        stderr.write(u'Error encountered while parsing input: %s.\n' % error_message)
        sys.exit(1)

    if check_only:
        stdout = codecs.getwriter('utf-8')(sys.stdout)

        if not silent:
            if is_results:
                corpus_type = 'testing or reference'
            else:
                corpus_type = 'training or development'
            stdout.write('This is %s corpus\n' % corpus_type)
            for elem in input_header.keys():
                stdout.write('%30s : %-30s\n' % (elem, input_header[elem]))
            stdout.write('Number of words: %d\n' % len(input_data))
        else:
            stdout.write("OK\n")

        sys.exit()

    try:
        test_header, test_data, is_results = parse_xml(test_fname)
    except IOError as e:
        error_message = e.strerror
        if not error_message:
            error_message = e.message
        stderr.write(u'Error encountered while parsing test file. Here is what the parser said:\n%s.\n' % error_message)
        sys.exit(1)

    acc, f, f_best_match, mrr, map_ref, acc_10 = evaluate(input_data, test_data)

    if output_fname:
        write_details(output_fname, input_data, test_data, acc, f, f_best_match, mrr, map_ref, acc_10)

    N = len(acc)
    acc_num = float(sum([acc[src_word] for src_word in acc.keys()]))
    acc10_num = float(sum([acc_10[src_word] for src_word in acc_10.keys()]))
    sys.stdout.write('ACC:          %f (%d/%d)\n' % (acc_num / N, acc_num, N))
    sys.stdout.write('Mean F-score: %f\n' % (float(sum([f[src_word] for src_word in f.keys()])) / N))
    sys.stdout.write('MRR:          %f\n' % (float(sum([mrr[src_word] for src_word in mrr.keys()])) / N))
    sys.stdout.write('MAP_ref:      %f\n' % (float(sum([map_ref[src_word] for src_word in map_ref.keys()])) / N))
    sys.stdout.write('ACC@10:       %f (%d/%d)\n' % (acc10_num / N, acc10_num, N))
    # sys.stdout.write('MAP_%d:       %f\n' % (n, float(sum([map_n[src_word] for src_word in map_n.keys()]))/N))
    # sys.stdout.write('MAP_sys:      %f\n' % (float(sum([map_sys[src_word] for src_word in map_sys.keys()]))/N))


def test():
    stdout = codecs.getwriter('utf-8')(sys.stdout)
    input_header, input_data, is_result = parse_xml('news_results.xml', max_targets=10)
    test_header, test_data, is_result = parse_xml('news_test.xml')
    acc, f, f_best_match, mrr, map_ref = evaluate(input_data, test_data)
    for src_word in test_data.keys():
        stdout.write('%10s ACC=%f\tF-score=%f (%s)\tMRR=%f\tMAP_ref=%f\n' % (
            src_word, acc[src_word], f[src_word], f_best_match[src_word], mrr[src_word], map_ref[src_word]))


if __name__ == '__main__':
    main()
    # test()
