import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib as mpl
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

df = pd.read_csv('clean_data.csv')

stopwords = set(STOPWORDS)

df_i = df[df['target'] == 1]
df_s = df[df['target'] == 0]

wordcloud_i = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40,
                          random_state=42
                         ).generate(str(df_i['treated_question']))

wordcloud_s = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40,
                          random_state=42
                         ).generate(str(df_s['treated_question']))

fig = plt.figure(1)
plt.imshow(wordcloud_i)
plt.imshow(wordcloud_s)
plt.axis('off')
plt.show()