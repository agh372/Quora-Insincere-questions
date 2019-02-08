# Quora-Insincere-questions

**Project Name:** Quora Insincere Questions Classification
**Project Team Members:**
Neha Aggarwal
Xiuli Gu
Thingom Bishal Singha 
Arjun Gurudatta Hegde 

## 1. Introduction and Problem Description

The main aim of this project is to predict whether a question asked on Quora is sincere or not. An
insincere question is defined as a question intended to make a statement rather than look for
helpful answers. Some characteristics that can signify that a question is insincere:

- Has a non-neutral tone
- Is disparaging or inflammatory
- Isn't grounded in reality
- Uses sexual content (incest, bestiality, pedophilia) for shock value, and not to seek
    genuine answers

## 2. Dataset description

The dataset has been provided by Kaggle competitions. Along with the training and the test
dataset, word embeddings have been provided.
Data fields:

- qid - unique question identifier
- question_text - Quora question text
- target - a question labeled "insincere" has a value of 1, otherwise 0

Data Size:

- Training dataset – 1306122 rows, 3 columns
- Test Dataset – 56370 rows, 2 columns

Word Embeddings:

- GoogleNews-vectorsnegative300 - htps://code.google.com/archive/p/word2vec/
- glove.840B.300d - https://nlp.stanford.edu/projects/glove/
- paragram_300_sl999 - https://cogcomp.org/page/resource_view/
- wiki-news-300d-1M - https://fasttext.cc/docs/en/english-vectors.html

Target Values (Train dataset):

- Sincere: Total count – 1225312
- Insincere: Total Count – 80810


A brief look of the training data:

## 3. Pre-processing techniques

**1 ) Data Preprocessing with embeddings**

**Text Preprocessing:**

The dataset does not have NaN value.
We use Keras Tokenizer API to preprocess text.

- num_words: the maximum number of words to keep, based on word frequency, set to
    40000.
- Filter out all punctuation, plus tabs and line breaks, except the ' character in a text
    question.
- Convert the texts to lowercase.
- Split string, separator for word splitting is ‘ ’.
- Every word will be treated as a token.
- Ignore words that are out-of-vocabulary, i.e. oov_token=None

**Text embedding:**

The text data is preprocessed in order to bring it with accordance to the given embeddings. It is
then compared with all the embeddings to choose the one which is the best for our dataset. The
following preprocessing has been done on the data:

- Convert them to lowercase.
- Clean all contractions by creating a map.
- Clean all special characters.

We use Keras Embedding layer:

- Size of the vocabulary: 40000, that equals to the maximum number of words in Text
    processing.


- Dimension of the dense embedding is output vector length, set to 300
- Initialize weights to the embeddings matrix that we build using glove.840B.300d.
- We don’t use embeddings_regularizer and embeddings_constraint.
- Length of input sequences: According the distribution of word length of questions,
    average questions length in train and test datasets are similar, we choose the max length
    of questions is 70, although there are quite long questions in train dataset.

**2 ) Data Preprocessing without Embeddings**
After the initial cleaning, as described earlier, the following things were done:

- Texts to lowercase
- Removal of special characters
- Correction/regularization of spellings
- Removal of stop words

Here is the wordcloud of the cleaned corpus for Insincere questions.


Here is the wordcloud for the corpus for sincere questions.

For vectorization of each question text, **tf-idf** vectorization was used. tf-idf stands for Term
Frequency - Inverse Document Frequency. Each word in a document is given a weight which is
proportional to its frequency in the question text but inversely proportional to the frequency in
the corpus.

The length of the vector is 300, which means the 300 most frequent terms in the corpus are used
for this purpose. The terms are defined as n-grams where n can be 1, 2 or 3.

**Balancing the Data**
The data provided is unbalanced with number of sincere questions making up 93.8% of the
dataset. To overcome this issue, data is created for insincere questions by random sampling.

## 4. Related Work

Here are results of some of the top voted kernels for this ongoing competition

(https://www.kaggle.com/c/quora-insincere-questions-classification ):

- 2 LSTM w/ attention Glove : f1-score: 0.6851 https://www.kaggle.com/shujian/mix-of-
    nn-models-based-on-meta-embedding
- Single RNN : f1-score: 0.
    https://www.kaggle.com/shujian/single-rnn-with- 4 - folds-clr
- Bidirectional GRI Model: f1-score: 0.
    https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings


## 5. Proposed solution and methods

The following are the algorithms that we are using for this project.

**1) Deep Learning - RNN**

**Introduction** : Recurrent Neural Network (RNN) can use their internal stored state (also are
referred to as gated state or gated memory) to process sequences of inputs, which allows it to
exhibit temporal dynamic behavior for a time sequence. In order to solve the vanishing gradient
problem which comes with a standard recurrent neural network, two well-defined RNN
networks come up – long short-term memory (LSTM) and gated recurrent units (GRU).
The controlled states unit show in the following:

Note: The pictures are from Wikipedia.

**Implementation:** In this project, we use Keras library to build a RNN network. Train our model
on Google cloud Compute Engine, System Performance: 8 core CPU, 32 GB RAM memory,
Nvidia Tesla P100, 80GB SSD storage, Operating system: ubuntu 16.
Our network Architecture:

In the bidirectional layer, we will choose one of four layers – GRU, LSTM, CuDNNGRU,
CuDNNLSTM, based on the best performance.


Dense_1 and Dense_2 layer are fully-connected layers , we will choose one of activation
function - relu and sigmoid.
Dropout_1 layer is used to reduce the weights.

**2) Deep Learning - CNN**

Convolutional Neural networks are a part of deep learning with complex architecture involving
multiple parameters. CNN have proven to be highly successful in learning and classifying
sentences. The picture below explains the working of CNN on a word embedding.

Picture has been picked from “Zhang Y, Wallace B. A Sensitivity Analysis of (and Practitioners’ Guide to)
Convolutional Neural Networks for Sentence Classification. arXiv preprint arXiv:151003820. 2015; PMID:
463165”
Three filter regions have been depicted here with region sizes: 2,3,4, each of which has 2 filters.
(In are model the region sizes are 1,2,3,5 with total 36 filters.) Filters perform convolutions on
the sentence matrix with has been generated using the best embedding obtained after the
preprocessing of data. Then these filters generate (variable-length) feature maps after which 1-
max pooling is performed over each map, thus picking up the largest number. A univariate
feature vector is generated from all six maps, and these 6 features are concatenated to form a
feature vector for the penultimate layer. The final softmax later then receives this feature vector
as input and uses it to classify the question as 0 or 1 for it being insincere or sincere.


**Implementation:** The model has been developed in Keras library of python. It involves 2 D
convolution network with 1 Dense layer. The complete description of the model has been given
below.

**3) Support Vector Classification**

**Introduction:** Support vector machines are supervised learning models which works on the
Structural Risk Minimization principle. This principle allows us to find a hypothesis h for which


we can guarantee the lowest true error. The lowest true error is guaranteed by the hypothesis
given by this principle. SVMs produce hyperplane

SVMs are known to work well for text categorization because of the following few reasons:

1. SVMs tend to overlook overfitting, so it is capable of handling text data with large feature
spaces.
2. Text data are generally linearly separable
3. Text categorization problems generate document vectors which are sparse which is suitable for
SVMs.

Picture has been picked from
“https://image.slidesharecdn.com/treparel-introductiontextclassification2012- 120712064839 - phpapp02/95/machine-
learning-based-text-classification-introduction- 19 - 728.jpg?cb=1411005878”

**Implementation:** The model has been developed in Scikit learn library of python. SVM with
linear kernel is used to separate the document vectors which belong to sincere questions and
document vectors which belong to insincere questions. 300 weights are assigned to the features
to fit the data. Shrinking heuristic was used to speed up the optimization.

**4) XGBoost**

XGBoost is an ensemble classifier which is an implementation of gradient boosted classifiers.
The algorithm used by XGBoost is a variant of gradient boosting. In Gradient Boosting, weak
learners (like Shallow Decision Trees) iteratively run on the dataset. With each iteration, the next
learner learns from its predecessors to predict the errors of the prior models. These models are
then added together.


The model uses the gradient descent algorithm to minimize its error.

XGBoost, of late, has almost become a silver bullet and is extensively used in competitions. By
using weak learners as base estimators, it overcomes overfitting and aggregation reduces the bias
of the weak learners. Thus it is able to overcome the bias-variance tradeoff.

Src: https://www.kdnuggets.com/2017/10/understanding-machine-learning-algorithms.html

**Implementation** : XGBoost algorithm is implemented by the XGBoost library. XGBoost is a fast
algorithm and provides parallel tree boosting. We have used the XGBClassifier model provided
by the XGBoost library. It provides 3 base learners: gbtree, gblinear, and dart.

## 6. Experimental Results and Analysis

**1) Results from RNN model
Best hyperparameters select:**
After training and testing, we tone the parameters to best value

```
Parameter Value
```
```
Preprocessing and
embedding
```
```
Train and valid data split 10% valid data
```
```
Embedding method glove.840B.300d
```
```
Embedding size 300
```
```
Max features 50000
```
```
Max length of words 70
```
```
Network model RNN layer CuDNNGRU(GPU only)
```
```
Full-connect layer 1 Relu
```

```
Drop out 0.
```
```
Full-connect layer 2 sigmoid
```
```
Loss function Binary cross entropy
```
```
Optimizer adam
```
```
Training and
Prediction
```
```
Epochs 3
```
```
Batch size 1024
```
```
Early stopping None
```
```
Reduce learning rate None
```
```
Performance metrics Accuracy, ROC curve
```
```
Best threshold 0.
```
**Output file:**
Best model file
Best weight file
Submission.scv: Prediction results for Kaggle test data using best threshold

**Performance:**
Prediction accuracy for training data : 0.
Prediction accuracy for validation data : 0.
F1 Score for validation data (maximum): 0.
The validation data ROC curve:


Average raining time: 48s/Epoch, 36us/step

**2) Results from CNN model
Best hyperparameters select:**

```
Parameter Value
```
```
Preprocessing and
embedding
```
```
Train and valid data split 10% valid data
```
```
Embedding method glove.840B.300d
```
```
Embedding size 300
```
```
Max features 50000
```
```
Max length of words 70
```
```
Network model CNN layer Conv2D , MaxPool2D
```
```
Filter Sizes 1,2,3,
```
```
Number of filters 36
```
```
Activation function elu
```
```
Kernel initializer He_normal
```
```
Drop out 0.
```
```
Dense layer activation sigmoid
```
```
Training and Prediction Epochs 2
```
```
Batch size 1024
```
```
Early stopping None
```
```
Reduce learning rate None
```
```
Best threshold 0.
```
**Output file:**
Best model file
Best weight file
Submission.scv: the prediction result for Kaggle test data based on the best threshold


**Performance:**
The training data accuracy (max value): 95.23%
The validation data accuracy (max value): 95.99%
The validation data ROC curve area: 0.

**3) Results from Support Vector machine model
Best hyperparameters select:**

After training and testing, we tone the parameters to best value

```
Parameter Value
```
```
Preprocessing and
embedding
```
```
Train and valid data split 10% valid data
```
```
Embedding method glove.840B.300d
```
```
Max features 300
```
```
Max length of words 70
```
```
SVM model kernel linear
```
```
Decision Function
shape
```
```
One vs Rest
```
```
Size of the kernel 200 MB
```
```
Training and
Prediction
```
```
Coefficients 300
```
```
Max iteration No limit
```
```
Performance metrics Accuracy, Precision,
```

```
Recall, ROC curve
```
**Performance:**
Prediction accuracy for training data : 0.
Prediction accuracy for validation data : 0.
The validation data ROC curve: 0.

**4) Results from XGBoost Classifier
Best hyperparameters select for XGBoost:**
After training and testing, we tone the parameters to best value

```
Parameter Values used
```
```
Preprocessing and
embedding
```
```
Train and valid data split 10% Validation Data
```
```
Embedding method No pre-trained embeddings used.
```
```
Vectorization tf-idf (sklearn)
```
```
No. of features 300
```
```
XGBoost Base Estimator Linear Regresssion, Decision Trees,
Dropout meet Additive Regression
Trees
```
```
Boosting Algorithm gblinear, gbtree, dart
```
```
Max Depth 3, 10, 30
```
```
No. of Estimators 10, 50, 100
```
```
Prediction Iterator ParameterGrid
```

```
Performance metrics Accuracy, f1 Score, ROC curve
```
**Performance:**
Prediction accuracy for training data : 0.
Prediction accuracy for validation data : 0.
F1-score for validation data is 0.
The validation data ROC curve: 0.

## 7. Conclusion

In our project, the best performance is 96% accuracy for validation dataset, and the best AUC is
0.964. Deep learning algorithms (RNN and CNN) have almost the same performance in terms of
accuracy, which is better than SVM and XGboost. Since we use Keras CuDNNRGU that is
GPU-only for RNN, the running time of RNN is obviously less than CNN.


## 8. Contribution of team members

```
Works Contributor
```
```
Data preprocessing and data analysis Neha Aggarwal
Xiuli Gu
Arjun Gurudatta Hegde
Thingom Bishal Singha
```
```
CNN model coding and testing Neha Aggarwal
```
```
RNN model coding and testing Xiuli Gu
```
```
Support Vector Machine coding and testing Arjun Gurudatta Hegde
```
```
XGBoost Classifier Coding and Testing Thingom Bishal Singha
```
```
Project report and Demo Arjun Gurudatta Hegde
Thingom Bishal Singha
Neha Aggarwal
Xiuli Gu
```

## 9. References

References for RNN, Keras and embedding method
https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be
https://keras.io/layers/recurrent/
https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings
https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings

References for CNN, Keras and embedding
[http://www.joshuakim.io/understanding-how-convolutional-neural-network-cnn-perform-text-](http://www.joshuakim.io/understanding-how-convolutional-neural-network-cnn-perform-text-)
classification-with-word-embeddings/
https://www.kaggle.com/shujian/blend-of-lstm-and-cnn-with- 4 - embeddings-1200d

References for SVM and embedding method
[http://www.cs.cornell.edu/~tj/publications/joachims_98a.pdf](http://www.cs.cornell.edu/~tj/publications/joachims_98a.pdf)
https://www.kaggle.com/nhrade/text-classification-using-word-embeddings

References for XGBoost and tf-idf
https://xgboost.readthedocs.io/en/latest/
https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/
https://www.analyticsvidhya.com/blog/2018/09/an-end-to-end-guide-to-understand-the-math-
behind-xgboost/
[http://www.tfidf.com/](http://www.tfidf.com/)
https://scikit-
learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html


