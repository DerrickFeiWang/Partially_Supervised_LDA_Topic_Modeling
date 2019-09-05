import pyodbc
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import nltk

# Load Data
server='yourServer'
db= 'yourDatabase'
conn = pyodbc.connect('DRIVER={SQL Server};SERVER=' + server + ';DATABASE=' + db + ';Trusted_Connection=yes')


sqlStr = '''
      SELECT * from yourTable
''' 

df1 = pd.io.sql.read_sql(sqlStr, conn)
df1.head(1)




# Data Pre-processing
stemmer = nltk.stem.SnowballStemmer('english')

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text, n):    # n: the minimum length of a word
    result = []
    for token in gensim.utils.simple_preprocess(text):   # convert document into tokens
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > n:
            result.append(lemmatize_stemming(token))
    return ' '.join(result)



G1_processed_docs = G1['Notes'].apply(lambda x: preprocess(x, 2))
G1_processed_docs = list(G1_processed_docs)

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

Count_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=4500,
                                stop_words='english')

G1_cv = Count_vectorizer.fit_transform(G1_processed_docs)


# Preliminary Topic Test
n_testTpoics = 5

G1_lda = LatentDirichletAllocation(n_components  = 5,max_iter=5,
                                learning_method='online',
                                learning_offset=50.
                               )
G1_lda.fit(G1_cv)
G1_LdaRstMatrix = G1_lda.transform(G1_cv)
G1_LdaRst = G1_LdaRstMatrix.argmax(axis=1)

import pyLDAvis
import pyLDAvis.sklearn

pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(G1_lda, G1_cv, Count_vectorizer, mds='tsne')
panel

# Labeling
G1_Topic = pd.DataFrame(G1_processed_docs, columns = ['ProcessedDoc'])
G1_Topic['drug'] = G1_Topic['ProcessedDoc'].str.contains('drugName').astype('int')
G1_Topic['drug'].value_counts()


# Find the appropriate number of topics
def optimizeNumTopics(NumTopic):
    G1_lda = LatentDirichletAllocation(n_components = NumTopic,
                                       max_iter=3,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state = 4
                                      )
    
    G1_lda.fit(G1_cv)
    G1_LdaRst = G1_lda.transform(G1_cv)
    G1_LdaRst = G1_LdaRst.argmax(axis=1)

    G1_Topic['Topic'] = G1_LdaRst
    ss = G1_Topic.groupby(['drug','Topic']).count().reset_index()
    ss = ss.rename(columns = {'ProcessedDoc':'DocCounts'})

    TargetTopic = ss.loc[(ss['drug'] == 1) & 
                    (ss['DocCounts'] == max(ss['DocCounts'].loc[ss['drug']==1]))
                    ,"Topic"].values[0]

    fp = ss['DocCounts'][ (ss['drug']==0) & (ss['Topic']==TargetTopic) ].values[0]
    tp = ss['DocCounts'][ (ss['drug']==1) & (ss['Topic']==TargetTopic) ].values[0]
    precision = tp/(fp + tp)
    recall = tp/(sum(ss['DocCounts'][ss['drug']==1]))
    f1 = 2 * precision * recall / (precision + recall)
    
    score = G1_lda.score(G1_cv)
    perplexity = G1_lda.perplexity(G1_cv)
    
    
    return [NumTopic, precision,recall, f1, score, perplexity]

RstList = []
for NumTopic in range(2, 20):    
    print('Now testing NumTopic ', NumTopic)
    lst =  optimizeNumTopics(NumTopic)
    RstList.append(lst)

RstDf = pd.DataFrame(RstList, columns = ['NumTopic', 'precision','recall', 'f1', 'score','perplexity'])	

plt.plot(RstDf.NumTopic  , RstDf.precision)
plt.plot(RstDf.NumTopic  , RstDf.recall)
plt.plot(RstDf.NumTopic  , RstDf.f1)
plt.legend()

plt.plot(RstDf.NumTopic  , RstDf.score)
plt.legend()

plt.plot(RstDf.NumTopic  , RstDf.perplexity)
plt.legend()


# Hyper Parameter Tuning
def optimizeAlphaEta(alpha, eta):
    G1_lda = LatentDirichletAllocation(n_components = 9,
                                       max_iter=3,
                                       doc_topic_prior = alpha,
                                       topic_word_prior  = eta,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state = 4
                                      )
    
    G1_lda.fit(G1_cv)
    G1_LdaRst = G1_lda.transform(G1_cv)
    G1_LdaRst = G1_LdaRst.argmax(axis=1)

    G1_Topic['Topic'] = G1_LdaRst
    ss = G1_Topic.groupby(['drug','Topic']).count().reset_index()
    ss = ss.rename(columns = {'ProcessedDoc':'DocCounts'})


    TargetTopic = ss.loc[(ss['drug'] == 1) & 
                    (ss['DocCounts'] == max(ss['DocCounts'].loc[ss['drug']==1]))
                    ,"Topic"].values[0]

    fp = ss['DocCounts'][ (ss['drug']==0) & (ss['Topic']==TargetTopic) ].values[0]
    tp = ss['DocCounts'][ (ss['drug']==1) & (ss['Topic']==TargetTopic) ].values[0]
    precision = tp/(fp + tp)
    recall = tp/(sum(ss['DocCounts'][ss['drug']==1]))
    f1 = 2 * precision * recall / (precision + recall)
    
    score = G1_lda.score(G1_cv)
    perplexity = G1_lda.perplexity(G1_cv)
    
    print('F1 score is ', f1)
    
    return [alpha, eta, precision,recall, f1, score, perplexity]
	
RstList = []
for alpha in np.linspace(0,1,11):    
    for eta in np.linspace(0,1,11):
        print('\n Now testing alpha eta ', alpha, eta)
        lst =  optimizeAlphaEta(alpha, eta)
        RstList.append(lst)
		
RstDf = pd.DataFrame(RstList, columns = ['alpha', 'eta', 'precision','recall', 'f1', 'score','perplexity'])

% matplotlib inline
plt.figure(figsize = (8,8))
plt.imshow(pd.pivot_table(RstDf[['alpha', 'eta', 'f1']], index = ['alpha'], values = 'f1', columns = ['eta']).values
          ,interpolation='nearest', cmap='Reds')
		  
		  