# Partially Supervised LDA Topic Modeling for Text Document Classification

Topic modeling is a useful statistical modeling technique for extracting the topics in a collection of text documents. Latent Dirichlet Allocation (LDA) is the most common form of topic modeling. Other modeling methods include Non-negative Matrix factorization (NMF), Probabilistic Latent Semantic Analysis (PLSA) and Singular Value Decomposition (SVD), etc.

If you are not familiar with LDA model, here is a youtube video by Andrius Knispelis who brilliantly explained it within 20 minutes.
[https://www.youtube.com/watch?v=3mHy4OSyRf0&t=690s]

I have tested the performance of LDA and NMF on a collection of medical notes dataset. The LDA method performs much better than NMF on this sepcific dataset. However, I also observed that the result from LDA method is unstable, and it's hard to choose the appropriate number of topics, which were complaint frequently by others.

In order to solve the above mentioned problems, I developed a partially supervised LDA method for medical notes classification.

The big idea is, although we don't know how many topics in the corpus, we can find at least one most important topic that we care the most by reviewing the sample notes. Therefore, we can seperate the medical notes into two super-classes:

**_T1 vs T0_** 

         T1: Topic of interest, topic 1
         T0: All other topics, topic 2,3,4.....N
         
We can find the T1 medical notes by looking for specific word(s) in them and label them as 1, all other notes will be labeled as 0. These labels can be used in hyper-parameter tuning to determine the appropriate number of topics, as well as keep topic consistency.


 
 
 
 
