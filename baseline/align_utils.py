import baseline.align as align

def mcmc_align(wordpairs, align_symbol,seed):
    a = align.Aligner(wordpairs, align_symbol=align_symbol,random_seed=seed)
    return a.alignedpairs


def med_align(wordpairs, align_symbol):
    a = align.Aligner(wordpairs, align_symbol=align_symbol, mode='med')
    return a.alignedpairs
