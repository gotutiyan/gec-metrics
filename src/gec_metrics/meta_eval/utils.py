
def read_lines(path):
    '''This function loads a file and removes trailing spaces from each sentence.
    '''
    sents = open(path).read().rstrip().split('\n')
    return [s.rstrip() for s in sents]

