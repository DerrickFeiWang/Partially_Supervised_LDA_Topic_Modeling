# Partially Supervised LDA Topic Modeling for Text Document Classification

Topic modeling is a useful statistical modeling technique for extracting the topics in a collection of text documents. Latent Dirichlet Allocation (LDA) is the most common form of topic modeling. Other modeling methods include Non-negative Matrix factorization (NMF), Probabilistic Latent Semantic Analysis (PLSA) and Singular Value Decomposition (SVD), etc.

If you are not familiar with LDA model, here is a youtube video by Andrius Knispelis who brilliantly explained it within 20 minutes.
[https://www.youtube.com/watch?v=3mHy4OSyRf0&t=690s]

I have tested the performance of LDA and NMF on a collection of medical notes dataset. The LDA method performs much better than NMF on this sepcific dataset. However, I also observed that the result from LDA method is unstable, and it's hard to determine the appropriate number of topics, which were complaint frequently by users.

In order to solve the above mentioned problems, I developed a partially supervised LDA method for medical notes classification.

The big idea is, although we don't know how many topics in the corpus, we can find at least one most important topic that we care the most by running a quick "blind test" LDA with an estimated number of topics and reviewing the sample notes. Therefore, we can seperate the medical notes into two super-classes:

**_T1 vs T0_** 

         T1: Topic of interest, topic 1
         T0: All other topics, topic 2,3,4.....N
         
We can find the T1 medical notes by looking for specific word(s) in them and label them as 1, all other notes will be labeled as 0. These labels can be used in hyper-parameter tuning to determine the appropriate number of topics, as well as keep topic consistency.

## Backgrounds

We have hundreds to thousands of medical notes logged into the database every day in addition to well defined structure data. These notes document moderate to severe symptoms, as well as routine operations that are not suitable to be recorded as structured data, i.e. tables with predefined columns. These medical notes are valuable compliment information to the structure data, but they were rarely analyzed systematically. 

In order to extract information from the medical notes, we tried LDA and NMF topic models to classify the notes into topics. Pilot studies found that the results from NMF models are stable, but less accurate and less human interpretable. LDA models give much better accuracy and human interpretability, however the topic instability can be a big problem when deploying to production.

Here, I developed a partially-supervised LDA method for hyper parameter tuning to improve topic stability and determine the appropriate number of topics.

## Method,  Results and Discussion

### 1. Data Preprocessing

First, we will cpnvert the text documents ("Notes") into word lists by lematization, stemming and removing stop words.
```python
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

```

### 2. Convert word lists to word vector
We will use count vectorizer from the sklearn library for this purpose.

```python
Count_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=4500,
                                stop_words='english')

G1_cv = Count_vectorizer.fit_transform(G1_processed_docs)
```

### 3. Find out an important topic for labeling
Since the notes in our dataset are from one medical treatment for a specific group of patients, we presume the number of topics in these medical notes is small. We run a preliminary test with 5 topics in order to get a flavor of the major topic that we care the most.

```python
n_testTpoics = 5

G1_lda = LatentDirichletAllocation(n_components  = n_testTpoics,
                                learning_method='online',
                                learning_offset=50.
                               )
G1_lda.fit(G1_cv)
G1_LdaRstMatrix = G1_lda.transform(G1_cv)
G1_LdaRst = G1_LdaRstMatrix.argmax(axis=1)
```
We then use the pyLDAvis to visualize the topics and their key words.

```python
import pyLDAvis
import pyLDAvis.sklearn

pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(G1_lda, G1_cv, Count_vectorizer, mds='tsne')
panel
```
![Figure 3](https://user-images.githubusercontent.com/44976640/64359457-f9efd300-cfcd-11e9-826e-4a3491ef7f5b.JPG)
Note: Key words in the above picture were intentionally masked.

From the result of LDAvis, we identified one topic is talking about the usage of an expensive drug, we labeled all notes that mentioned this drug as 1, then use this label to calculate the precision, recall and f1 scores.

```python
G1_Topic = pd.DataFrame(G1_processed_docs, columns = ['ProcessedDoc'])
G1_Topic['drug'] = G1_Topic['ProcessedDoc'].str.contains('drugName').astype('int')
G1_Topic['drug'].value_counts()
```
### 4. Optimize the hyper parameters using f1 score, perplexity and coherence score.

We have labeled the notes with either it's about the drug usage (actual positive, 1) or not (actual negative, 0). After labeling the notes with the topics learnt from the LDA model, we will take the topic ID that consists the most of documents relating the drug as the predicted positive, all other topics are considered as predicted negative. Thus, we can calculate the precision, recall and f1 score with the following code.

#### 4.1 Determine the appropriate number of topics

```python
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
    recall = tp/(sum(ss['DocCounts'][ss['epo']==1]))
    f1 = 2 * precision * recall / (precision + recall)
    
    score = G1_lda.score(G1_cv)
    perplexity = G1_lda.perplexity(G1_cv)
    
    
    return [NumTopic, precision,recall, f1, score, perplexity]
    
RstList = []
for NumTopic in range(2, 20):    
    print('Now testing NumTopic ', NumTopic)
    lst =  optimizeNumTopics(NumTopic)
    RstList.append(lst)
    
```
The above function will return precision,recall, f1, as well as coherence score and perplexity which were provided by default from the sklearn LDA algorithm.

![Figure 4](https://user-images.githubusercontent.com/44976640/64371412-1ac42280-cfe6-11e9-8ad6-885b8ff1ec53.JPG)

With considering f1, perplexity and coherence score in this example, we can decide that 9 topics is a propriate number of topics.

#### 4.2 Hyper parameter tuning and model stability.

Although LDA algorithm suffering from topic un-stable issue, we found that the models with highest f1 scores are relatively consistent with each other. The reason is that LDA was known as a non-convex algorithm. Each run will reach a local optimal, but you can't expect that any given run would be the global optimal. But the highest local optimal are tend to render results close to global optimal, therefore, results from "best" local optimal models are relatively consistent to each other to a certain extent.
    
  
    





