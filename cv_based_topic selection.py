import numpy as np
import pandas as pd
import feather
import multiprocessing
from gensim import corpora, models
from collections import defaultdict
from itertools import compress
from time import time

INFILE = r'.\output\notes_clean_compact.feather'
INFILE_STOPWORDS = r'.\data\stopwords.csv'
MINIMUM_TERM_N = 25  # if the word frequency is < N it is not used
MAXIMUM_DOC_PROPORTION = 0.3  # if the word appears in more than x% of documents it is not used
NUM_WORKERS = multiprocessing.cpu_count() - 1
OUTFILE = r'.\output\perplexity_report.txt'

# settings to grid search over
N_MIN_TOPIC = 1
N_MAX_TOPIC = 100
STEP_SIZE = 5
CV_FOLDS = 5

# the data
df_compact = feather.read_dataframe(INFILE)
documents = df_compact.notes_standard_transfrom.tolist()

# remove common words and tokenize
stoplist = set(pd.read_csv(r'.\data\stopwords.csv').STOPWORDS_CUSTOM.tolist())
texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]

# document pruning
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > MINIMUM_TERM_N]
         for text in texts]

# bag of words representation for topics
dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_above=MAXIMUM_DOC_PROPORTION)
dictionary.compactify()
bmc_corpus = [dictionary.doc2bow(text) for text in texts]

# set cross-validation strategy
n = df_compact.shape[0]
idx_folds = np.random.randint(0, high=CV_FOLDS, size=n)
print(pd.Series(idx_folds).groupby(idx_folds).size())  # size of each CV fold

# initializing empty dicts to hold results and looping over the grid
grid_logLik = defaultdict(list)
grid_perplx = defaultdict(list)
text_file = open(OUTFILE, 'w')
t0 = time()
for i in range(N_MIN_TOPIC, N_MAX_TOPIC+1, STEP_SIZE):
    print("processing n_topic = %.3f" % i)
    for j in range(CV_FOLDS):
        print("processing cross-validation fold = %.3f" % j)

        # training-test split based on fold
        test_corpus = list(compress(bmc_corpus, [True if idx == j else False for idx in idx_folds]))
        train_corpus = list(compress(bmc_corpus, [False if idx == j else True for idx in idx_folds]))

        # train the model
        model = models.LdaMulticore(corpus=train_corpus
                                    , workers=NUM_WORKERS
                                    , id2word=dictionary
                                    , num_topics=i)

        # compute perplexity on the hold-out fold
        perplex = model.bound(test_corpus)  # this is the likelihood value
        print("Perplexity: %s" % perplex)
        grid_logLik[i].append(perplex)

        per_word_perplex = np.exp2(-perplex / sum(cnt for doc in test_corpus for _, cnt in doc))  # scaling
        # by number of tokens
        print("Per-word Perplexity: %s" % per_word_perplex)
        grid_perplx[i].append(per_word_perplex)
        text_file.write('Finished running n_topic={0:2d}; Perplexity:{1:.3f}; Per-token perplexity:{2:.3f}\n'
                        .format(i, perplex, per_word_perplex))

text_file.close()
print('Grid search completed in {0:.3f}s.'.format(time()-t0))
