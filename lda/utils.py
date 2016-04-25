from __future__ import absolute_import, unicode_literals  # noqa

import logging
import numbers
import sys
from os import listdir, makedirs
from os.path import isfile, isdir, join
import mimetypes
import re,string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import numpy as np

PY2 = sys.version_info[0] == 2
if PY2:
    import itertools
    zip = itertools.izip


logger = logging.getLogger('lda')


def check_random_state(seed):
    if seed is None:
        # i.e., use existing RandomState
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("{} cannot be used as a random seed.".format(seed))


def matrix_to_lists(doc_word):
    """Convert a (sparse) matrix of counts into arrays of word and doc indices

    Parameters
    ----------
    doc_word : array or sparse matrix (D, V)
        document-term matrix of counts

    Returns
    -------
    (WS, DS) : tuple of two arrays
        WS[k] contains the kth word in the corpus
        DS[k] contains the document index for the kth word

    """
    if np.count_nonzero(doc_word.sum(axis=1)) != doc_word.shape[0]:
        logger.warning("all zero row in document-term matrix found")
    if np.count_nonzero(doc_word.sum(axis=0)) != doc_word.shape[1]:
        logger.warning("all zero column in document-term matrix found")
    sparse = True
    try:
        # if doc_word is a scipy sparse matrix
        doc_word = doc_word.copy().tolil()
    except AttributeError:
        sparse = False

    if sparse and not np.issubdtype(doc_word.dtype, int):
        raise ValueError("expected sparse matrix with integer values, found float values")

    ii, jj = np.nonzero(doc_word)
    if sparse:
        ss = tuple(doc_word[i, j] for i, j in zip(ii, jj))
    else:
        ss = doc_word[ii, jj]

    n_tokens = int(doc_word.sum())
    DS = np.repeat(ii, ss).astype(np.intc)
    WS = np.empty(n_tokens, dtype=np.intc)
    startidx = 0
    for i, cnt in enumerate(ss):
        cnt = int(cnt)
        WS[startidx:startidx + cnt] = jj[i]
        startidx += cnt
    return WS, DS

def files_preprocessing(in_dir,out_dir):
    """
    S
    """
    global i
    i=0
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    if not isdir(out_dir):
        makedirs(out_dir)

    def directory_helper(in_dir):
        global i
        files = listdir(in_dir)
        files = sorted(files)
        for f in files:
            new_path = join(in_dir, f)
            if isfile(new_path) and mimetypes.guess_type(new_path)[0]=='text/plain':
                of = open(new_path).read()
                of = stemmer.stem(of)
                nf = open(join(out_dir,in_dir.replace('/','_')+'_'+str(i)+'.txt'),'wb')
                nf.write(of)
                i+=1
            elif isdir(new_path):
                directory_helper(new_path)

    directory_helper(in_dir)

def files_to_lists(directory):
    """Convert directory of text files into arrays of word and doc indices
    Parameters
    ----------
    directory : path of directory contains text files (documents)
    Returns
    -------
    (V, L, WS, DS) : tuple of two arrays
        V[i] contains ith word in the corpus vocabulary
        L[j] contains the label of jth document 
        (HARD CODED-- LABEL IS EITHER 'Yes' OR 'No' ACCORDING TO FILENAME : tes_Yes_3.txt or tes_No_4.txt)
        WS[k] contains index of the kth word in the corpus
        DS[k] contains the document index for the kth word
    """
    textfiles = [f for f in listdir(directory) if (isfile(join(directory, f)) and mimetypes.guess_type(join(directory, f))[0]=='text/plain')]
    textfiles = sorted(textfiles)
    V = [] #vocabulary list
    print textfiles
    WS=[]
    DS=[]
    L = []
    for f in range(len(textfiles)):
        r = open(join(directory,textfiles[f])).read().lower()
        table=string.maketrans("","")
        t=r.translate(table,string.punctuation)
        for w in re.split(' |\n|\r|\t',t): #need to remove numbers later
            if len(w)>0:
                iw=len(V)
                try:
                    iw = V.index(w)
                except:
                    V+=[w]
                WS+=[iw]
                DS+=[f]
        if 'yes' in textfiles[f].lower().split('_'):
            L+=[1]
        elif 'no' in textfiles[f].lower().split('_'):
            L+=[0]
        else:
            raise ValueError("{} do not contain yes or no label.".format(textfiles[f]))
    return np.asarray(V),np.asarray(L,dtype=np.intc),np.asarray(WS,dtype=np.intc),np.asarray(DS,dtype=np.intc)


def lists_to_matrix(WS, DS):
    """Convert array of word (or topic) and document indices to doc-term array

    Parameters
    -----------
    (WS, DS) : tuple of two arrays
        WS[k] contains the kth word in the corpus
        DS[k] contains the document index for the kth word

    Returns
    -------
    doc_word : array (D, V)
        document-term array of counts

    """
    D = max(DS) + 1
    V = max(WS) + 1
    doc_word = np.empty((D, V), dtype=np.intc)
    for d in range(D):
        for v in range(V):
            doc_word[d, v] = np.count_nonzero(WS[DS == d] == v)
    return doc_word


def dtm2ldac(dtm, offset=0):
    """Convert a document-term matrix into an LDA-C formatted file

    Parameters
    ----------
    dtm : array of shape N,V

    Returns
    -------
    doclines : iterable of LDA-C lines suitable for writing to file

    Notes
    -----
    If a format similar to SVMLight is desired, `offset` of 1 may be used.
    """
    try:
        dtm = dtm.tocsr()
    except AttributeError:
        pass
    assert np.issubdtype(dtm.dtype, int)
    n_rows = dtm.shape[0]
    for i, row in enumerate(dtm):
        try:
            row = row.toarray().squeeze()
        except AttributeError:
            pass
        unique_terms = np.count_nonzero(row)
        if unique_terms == 0:
            raise ValueError("dtm row {} has all zero entries.".format(i))
        term_cnt_pairs = [(i + offset, cnt) for i, cnt in enumerate(row) if cnt > 0]
        docline = str(unique_terms) + ' '
        docline += ' '.join(["{}:{}".format(i, cnt) for i, cnt in term_cnt_pairs])
        if (i + 1) % 1000 == 0:
            logger.info("dtm2ldac: on row {} of {}".format(i + 1, n_rows))
        yield docline


def ldac2dtm(stream, offset=0):
    """Convert an LDA-C formatted file to a document-term array

    Parameters
    ----------
    stream: file object
        File yielding unicode strings in LDA-C format.

    Returns
    -------
    dtm : array of shape N,V

    Notes
    -----
    If a format similar to SVMLight is the source, an `offset` of 1 may be used.
    """
    doclines = stream

    # We need to figure out the dimensions of the dtm.
    N = 0
    V = -1
    data = []
    for l in doclines:
        l = l.strip()
        # skip empty lines
        if not l:
            continue
        unique_terms = int(l.split(' ')[0])
        term_cnt_pairs = [s.split(':') for s in l.split(' ')[1:]]
        for v, _ in term_cnt_pairs:
            # check that format is indeed LDA-C with the appropriate offset
            if int(v) == 0 and offset == 1:
                raise ValueError("Indexes in LDA-C are offset 1")
        term_cnt_pairs = tuple((int(v) - offset, int(cnt)) for v, cnt in term_cnt_pairs)
        np.testing.assert_equal(unique_terms, len(term_cnt_pairs))
        V = max(V, *[v for v, cnt in term_cnt_pairs])
        data.append(term_cnt_pairs)
        N += 1
    V = V + 1
    dtm = np.zeros((N, V), dtype=np.intc)
    for i, doc in enumerate(data):
        for v, cnt in doc:
            np.testing.assert_equal(dtm[i, v], 0)
            dtm[i, v] = cnt
    return dtm
