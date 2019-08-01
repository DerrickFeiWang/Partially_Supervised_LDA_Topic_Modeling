# Partially Supervised LDA Topic Modeling for Text Document Classification

Topic modeling is a useful statistical modeling technique for extracting the topics in a collection of text documents. Latent Dirichlet Allocation (LDA) is the most common form of topic modeling. Other modeling methods include Non-negative Matrix factorization (NMF), Probabilistic Latent Semantic Analysis (PLSA) and Singular Value Decomposition (SVD), etc.

I have tested the performance of LDA and NMF on a collection of medical notes dataset. The LDA method performs much better than NMF on this sepcific dataset. However, I also observed that the result from LDA method is unstable, and it's hard to choose the appropriate number of topics, which were complaint frequently by others.

In order to solve the above mentioned problems, I developed a partially supervised LDA method for medical notes classification.

The big idea is, although we don't know how many topics in the corpus, we can find at least there is one most important topic that we care the most by reviewing the samples of the medical notes. Therefore, we can seperate the medical notes into two super-classes:

         ### Ti vs To 
         
            Ti: Topic of interest
            To: Other topics
         
 
