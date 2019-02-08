
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.base import TransformerMixin
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.base import TransformerMixin
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


import operator 
import re
import sys


# In[3]:


#Load embeddings
def get_coefs(word,*arr): 
    return word, np.asarray(arr, dtype='float32')

def load_embedding(file):    
    if file == '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec':
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o)>100)
    else:
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))
        
    return embeddings_index


# In[4]:


def build_vocab(texts):
    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab
def check_coverage(vocab, embeddings_index):
    known_words = {}
    unknown_words = {}
    nb_known_words = 0
    nb_unknown_words = 0
    for word in vocab.keys():
        try:
            known_words[word] = embeddings_index[word]
            nb_known_words += vocab[word]
        except:
            unknown_words[word] = vocab[word]
            nb_unknown_words += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(known_words) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))
    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]

    return unknown_words

def add_lower(embedding, vocab):
    count = 0
    for word in vocab:
        if word in embedding and word.lower() not in embedding:  
            embedding[word.lower()] = embedding[word]
            count += 1


# In[5]:


mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}
def correct_spelling(x, dic):
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x


# In[6]:


contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }
def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text


# In[7]:


punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }
def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text


# In[8]:


def get_processed_data():
    #Loading data
    train = pd.read_csv("../input/train.csv").drop('target', axis=1)
    test = pd.read_csv("../input/test.csv")
    df = pd.concat([train ,test])
    vocab = build_vocab(df['question_text'])
    print("Number of texts: ", df.shape[0])
    
    train = pd.read_csv("../input/train.csv")

    
    #Loading of embeddings
    glove = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    paragram =  '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    wiki_news = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    
    #Google embedding not able to access
    
#     embed_glove = load_embedding(glove)
    embed_paragram = load_embedding(paragram)
#     embed_fasttext = load_embedding(wiki_news)
    print('Successful loading of embeddings.')
    
    #Lowercase (Not required)
    train['lowered_question'] = train['question_text'].apply(lambda x: x.lower())
    df['lowered_question'] = df['question_text'].apply(lambda x: x.lower())
    vocab_low = build_vocab(df['lowered_question'])
    
    #Add words to embeddings
#     add_lower(embed_glove, vocab)
    add_lower(embed_paragram, vocab)
#     add_lower(embed_fasttext, vocab)
    
    #Cleaning Contractions, Removing special characters and Correcting Spellings
    df['treated_question'] = df['lowered_question'].apply(lambda x: clean_contractions(x, contraction_mapping))
    df['treated_question'] = df['treated_question'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
    df['treated_question'] = df['treated_question'].apply(lambda x: correct_spelling(x, mispell_dict))
    train['treated_question'] = train['lowered_question'].apply(lambda x: clean_contractions(x, contraction_mapping))
    train['treated_question'] = train['treated_question'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
    train['treated_question'] = train['treated_question'].apply(lambda x: correct_spelling(x, mispell_dict))
    
    train_df = train['treated_question']
    y = train['target'].values
    
    vocab = build_vocab(df['treated_question'])

#     oov_glove = check_coverage(vocab, embed_glove)
    oov_paragram = check_coverage(vocab, embed_paragram)
#     oov_fasttext = check_coverage(vocab, embed_fasttext)
    
    return train_df, y, embed_paragram    
    
input_df, y, embedding = get_processed_data()


# In[9]:


from sklearn.svm import SVC
from sklearn.svm import LinearSVC


# In[10]:


data = pd.read_csv("../input/train.csv")
y_train = data['target']
y_train.head()


# In[11]:


class EmbeddingHelper(TransformerMixin):
    
    def __init__(self):
        self.voc, self.words = self.insert_text()
        
    #We are inserting the 
    def insert_text(self):
        words = {}
        voc = []

        with open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt', 'r', encoding="utf8") as f:
            for i, line in enumerate(f):
                split = line.split(' ')
                if split[0].isalpha():
                    w = [float(i) for i in split[1:]]
                    words[split[0]] = np.array(w)
                    voc.append(split[0])
        return np.array(voc), words            
    
    def mean_vec(self, doc):
        arr = np.array([self.words[w.lower().strip()] for w in doc if w.lower().strip() in self.words])
        return np.mean(arr , axis=0)
    
    def get_obj(self, X, y=None):
        return self
    
    def to_mean_array(self, X):
        return np.array([self.mean_vec(vec) for vec in X])
    
    def transform(self, X, y=None):
        return self.get_obj(X).to_mean_array(X)


# In[12]:


df = pd.DataFrame({'questions':input_df.values})


# In[13]:


def tokenize(X,size):
    questions = X[:,0]
    tokens = [word_tokenize(vec) for vec in questions[:size]]
    met = EmbeddingHelper()
    X_trans = met.transform(tokens)
    return X_trans


# In[14]:


df.columns


# In[15]:


import nltk
nltk.download('punkt')


# In[16]:


X = df[['questions']].as_matrix()


# In[17]:


from nltk.tokenize import word_tokenize
X_transform = tokenize(X, 99000)


# In[18]:


X_transform


# In[19]:


from sklearn.preprocessing import scale
X_scaled = scale(X_transform)


# In[20]:


X_scaled


# In[21]:


len(y)


# In[22]:


X = df['questions'].as_matrix()


# In[23]:


y = y[0:99000]


# In[24]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled,
                                                    y,test_size=0.10, random_state=2018)


# In[ ]:


svc = SVC(kernel='linear').fit(X_train, y_train)


# In[ ]:


svc


# In[ ]:


len(y_train)


# In[ ]:


y_pred = svc.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(y_test, y_pred)


# In[ ]:


fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')


# In[ ]:


y_pred_train = svc.predict(X_train)


# In[ ]:


accuracy_score(y_train, y_pred_train)

