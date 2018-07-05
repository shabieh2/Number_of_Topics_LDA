import numpy as np
import pandas as pd
import feather
import multiprocessing
from gensim import corpora, models, matutils
from collections import defaultdict
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from time import time

INFILE = r'.\output\notes_clean_compact.feather'
INFILE_STOPWORDS = r'.\data\stopwords.csv'
MINIMUM_TERM_N = 25  # if the word frequency is < N it is not used
MAXIMUM_DOC_PROPORTION = 0.3  # if the word appears in more than x% of documents it is not used
NUM_WORKERS = multiprocessing.cpu_count() - 1
OUTFILE = r'.\output\perplexity_report.txt'

# settings to grid search over
MODELS_DIR = r'.\kmeans'
MAX_K = 50

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

tfidf = models.TfidfModel(bmc_corpus, normalize=True)
corpus_tfidf = tfidf[bmc_corpus]

# project to 2 dimensions
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)

# write out coordinates to file
fcoords = open(os.path.join(MODELS_DIR, "coords.csv"), 'wb')
for vector in lsi[bmc_corpus]:
    if len(vector) != 2:
        continue
    fcoords.write(b'%6.4f\t%6.4f\n' % (vector[0][1], vector[1][1]))
fcoords.close()

# X = matutils.corpus2dense(lsi[bmc_corpus], len(lsi.projection.s)).T
X = np.loadtxt(os.path.join(MODELS_DIR, "coords.csv"), delimiter="\t")
ks = range(1, MAX_K + 1)

t0 = time()
inertias = np.zeros(MAX_K)
diff = np.zeros(MAX_K)
diff2 = np.zeros(MAX_K)
diff3 = np.zeros(MAX_K)
for k in range(1, MAX_K + 1):
    print('k=%.3f' % k)
    kmeans = KMeans(k).fit(X)
    inertias[k - 1] = kmeans.inertia_
    # first difference
    if k > 1:
        diff[k - 1] = inertias[k - 1] - inertias[k - 2]
    # second difference
    if k > 2:
        diff2[k - 1] = diff[k - 1] - diff[k - 2]
    # third difference
    if k > 3:
        diff3[k - 1] = diff2[k - 1] - diff2[k - 2]

elbow = np.argmin(diff3[3:]) + 3
print('Grid search completed in {0:.3f}s.'.format(time()-t0))

plt.plot(ks, inertias, "b*-")
plt.plot(ks[elbow], inertias[elbow], marker='o', markersize=12,
         markeredgewidth=2, markeredgecolor='r', markerfacecolor=None)
plt.ylabel("Inertia")
plt.xlabel("K")
plt.show()

