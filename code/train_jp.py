import argparse
# from sentence_transformers.cross_encoder import CrossEncoder
# from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
# from sentence_transformers import SentenceTransformer, InputExample, losses
# from sentence_transformers import evaluation
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# from hanlp.utils.lang.en.english_tokenizer import tokenize_english
# from janome.tokenizer import Tokenizer
import nltk
from nltk import word_tokenize

import pyprind
import os
import nltk
from nltk import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

from sklearn.datasets import load_svmlight_file
from sklearn import svm, tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
## 保存模型
import joblib

# 词形归并
def lemmatizer(words):
    lemmatizaer = WordNetLemmatizer()
    for word in words:
        word = lemmatizaer.lemmatize(word)
    return words


# 词干还原
def stemmer(words):
    porter_stemmer = PorterStemmer()
    result = []
    for word in words:
        result.append(porter_stemmer.stem(word))
    return result


# 载入停用词
# en_stopwords = set()
# jp_stopwords = set()
# es_stopwords = nltk.corpus.stopwords.words('spanish')
# with open('stopwords/en_stopwords.txt', 'r', encoding='utf8') as infile:
#     for line in infile:
#         line = line.rstrip('\n')
#         if line:
#             en_stopwords.add(line.lower())
#
# with open('stopwords/jp_stopwords.txt', 'r', encoding='utf8') as infile:
#     for line in infile:
#         line = line.rstrip('\n')
#         if line:
#             jp_stopwords.add(line)
#
#
# def tokenize_us(text):
#     text = str(text)
#     text = text.lower()
#     tokens = tokenize_english(text)
#     newlist = [word for word in tokens if word not in en_stopwords]
#     return " ".join(stemmer(newlist))
#     # return nltk.word_tokenize(text)
#
#
# t = Tokenizer()
#
#
# def tokenize_jp(text):
#     text = str(text)
#     text = t.tokenize(text, wakati=True)
#     text = list(text)
#     text = [word for word in text if word not in jp_stopwords]
#     return " ".join(text)
#
#
# def tokenize_es(text):
#     text = str(text)
#     text = text.lower()
#     text = word_tokenize(text, "spanish")
#     text = [word for word in text if word not in es_stopwords]
#     return " ".join(text)
# ###produce Japanese(jp)

def get_common2(token):
  if isinstance(token[0],float) or isinstance(token[1],float):return 0.0
  if len(set(token[0].split()))==0:return 0.0
  return len(set(token[0].split())&set(token[1].split()))*1.0 / len(set(token[0].split()))


def get_cosine_sim(*strs):
    try:
        vectors = [t for t in get_vectors(*strs)]
        return cosine_similarity(vectors)[0][1]
    except:
        return 0


def get_vectors(*strs):
    text = [t for t in strs]
    vectorizer = CountVectorizer(text)
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()


def get_tfidf(text, q):
    cv = TfidfVectorizer()
    cv.fit(text)
    return cv.transform(text), cv.transform(q)

def cos(*x):
    return cosine_similarity(x[0].reshape(1,-1), x[1].reshape(1,-1))[0][0]


# def main():
DATA_TASK1_PATH = "../data/task1/"
PRODUCT_CATALOGUE_PATH_FILE = os.path.join(DATA_TASK1_PATH, "new_product_catalogue_jp.csv")
TRAIN_PATH_FILE = os.path.join(DATA_TASK1_PATH, "train-v0.2.csv.zip")
col_query_id = "query_id"
col_query = "query"
col_query_locale = "query_locale"
col_esci_label = "esci_label"
col_product_id = "product_id"
col_product_title = "product_title"
col_product_locale = "product_locale"
col_product_description = "product_description"
col_product_bullet = "product_bullet_point"
col_product_brand = "product_brand"
col_product_color = "product_color_name"
col_gain = 'gain'

col_query_token = "query_token"
col_title_token = "title_token"
col_desc_token = "describe_token"
col_bullet_token = "bullet_token"
col_brand_token = "brand_token"
col_color_token = "color_token"

col_title_ratio = "title_ratio"
col_desc_ratio = "describe_ratio"
col_bullet_ratio = "bullet_ratio"
col_brand_ratio = "brand_ratio"
col_color_ratio = "color_ratio"

esci_label2gain = {
    'exact': 1.0,
    'substitute': 0.1,
    'complement': 0.01,
    'irrelevant': 0.0,
}

# """ 1. Load data """
# locale = "jp"
# df = pd.read_csv(TRAIN_PATH_FILE)
# print("train loaded...")
# df_product_catalogue = pd.read_csv(PRODUCT_CATALOGUE_PATH_FILE)
# print("product loaded...")
# df = df[df[col_query_locale] == locale]  ##筛选jp
# print(len(df))
# df_product_catalogue = df_product_catalogue[df_product_catalogue[col_product_locale] == locale]
# df = pd.merge(
#     df,
#     df_product_catalogue,
#     how='left',
#     left_on=[col_product_id, col_query_locale],
#     right_on=[col_product_id, col_product_locale],
# )
# print("merged...")
#
# df = df[df[col_product_title].notna()]
# list_query_id = df[col_query_id].unique()  ## 查询的unique id
# # dev_size = args.n_dev_queries / len(list_query_id)  ## 划分训练集测试集，吗？
# # list_query_id_train, list_query_id_dev = train_test_split(list_query_id, test_size=dev_size,
# #                                                           random_state=args.random_state)
# print("start token...")
# df[col_gain] = df[col_esci_label].apply(lambda label: esci_label2gain[label])  ##label转化为gain数值
# print("stat query token...")
# df[col_query_token] = df[col_query].apply(lambda x: tokenize_jp(x))
# print("stat title token...")
# df[col_title_token] = df[col_product_title].apply(lambda x: tokenize_jp(x))
# print("stat desc token...")
# df[col_desc_token] = df[col_product_description].apply(lambda x: tokenize_jp(x))
# print("stat bullet token...")
# df[col_bullet_token] = df[col_product_bullet].apply(lambda x: tokenize_jp(x))
# print("stat brand token...")
# df[col_brand_token] = df[col_product_brand].apply(lambda x: tokenize_jp(x))
# print("stat color token...")
# df[col_color_token] = df[col_product_color].apply(lambda x: tokenize_jp(x))
#
# print("over token...")
#
#
# df = df[[col_query_id, col_query, col_query_token,col_product_id,col_title_token, col_desc_token,
#          col_bullet_token, col_brand_token, col_color_token,
#          col_gain]]  ##取查询id，查询内容，结果名称，结果的gain
# # df_train = df[df[col_query_id].isin(list_query_id_train)]
# # df_dev = df[df[col_query_id].isin(list_query_id_dev)]
# print("dump...")
# saved_path = os.path.join(DATA_TASK1_PATH, "train_token_jp.csv")
# df.to_csv(saved_path, index=False)
# print("over dump...")


col_title_term = "title_term"
col_desc_term = "describe_term"
col_bullet_term = "bullet_term"
col_query_title = "query_title"
col_query_desc = "query_describe"
col_query_bullet = "query_bullet"

col_title_tfidf = "title_tfidf"
col_desc_tfidf = "describe_tfidf"
col_bullet_tfidf = "bullet_tfidf"
saved_path = os.path.join(DATA_TASK1_PATH, "train_token_jp.csv")
print("data loading...")
df=pd.read_csv(saved_path)
print("start ratio...")
df[col_title_ratio] = df[[col_title_token,col_query_token]].apply(lambda x:get_cosine_sim(*x),axis=1) #(len(set(df[col_title_token]) & set(df[col_query_tokan])) / len(set(df[col_query_tokan])))
print("start desc ratio...")
df[col_desc_ratio] = df[[col_desc_token,col_query_token]].apply(lambda x:get_cosine_sim(*x),axis=1)#len(set(df[col_desc_token]) & set(df[col_query_tokan])) / len(set(df[col_query_tokan]))
print("start bullet ratio...")
df[col_bullet_ratio] = df[[col_bullet_token,col_query_token]].apply(lambda x:get_cosine_sim(*x),axis=1) #len(set(df[col_bullet_token]) & set(df[col_query_tokan])) / len(set(df[col_query_tokan]))
print("start brand ratio...")
df[col_brand_ratio] = df[[col_brand_token,col_query_token]].apply(lambda x:get_common2(x),axis=1)#len(set(df[col_brand_token]) & set(df[col_query_tokan]))
print("start color ratio...")
df[col_color_ratio] = df[[col_color_token,col_query_token]].apply(lambda x:get_common2(x),axis=1)#len(set(df[col_color_token]) & set(df[col_query_tokan]))
print("over ratio... ")

df.fillna("",inplace=True) ##really important!
total_query = df[col_query_token].values
total_title = df[col_title_token].values
total_desc = df[col_desc_token].values
total_bullet = df[col_bullet_token].values

print("title tfidf...")
term_title_matrix, title_query_matrix = get_tfidf(total_title,total_query)
print("desc tfidf...")
term_desc_matrix, desc_query_matrix = get_tfidf(total_desc,total_query)
print("bulle tfidf...")
term_bullet_matrix, bullet_query_matrix = get_tfidf(total_bullet,total_query)




print("add term...")
df[col_title_term]=[i for i in term_title_matrix]
df[col_desc_term]=[i for i in term_desc_matrix]
df[col_bullet_term]=[i for i in term_bullet_matrix]
df[col_query_title]=[i for i in title_query_matrix]
df[col_query_desc]=[i for i in desc_query_matrix]
df[col_query_bullet]=[i for i in bullet_query_matrix]

print("title tfidf...")
df[col_title_tfidf]=df[[col_query_title,col_title_term]].apply(lambda x:cos(*x),axis=1)
print("desc tfidf...")
df[col_desc_tfidf]=df[[col_query_desc,col_desc_term]].apply(lambda x:cos(*x),axis=1)
print("bullet tfidf...")
df[col_bullet_tfidf]=df[[col_query_bullet,col_bullet_term]].apply(lambda x:cos(*x),axis=1)


df = df[[col_query_id, col_query, col_query_token, col_title_token,col_title_ratio,col_title_tfidf,col_desc_token,col_desc_ratio,col_desc_tfidf,
         col_bullet_token, col_bullet_ratio,col_bullet_tfidf, col_brand_token, col_brand_ratio, col_color_token, col_color_ratio,
         col_gain]]  ##取查询id，查询内容，结果名称，结果的gain
# df_train = df[df[col_query_id].isin(list_query_id_train)]
# df_dev = df[df[col_query_id].isin(list_query_id_dev)]
print("dump...")
saved_path = os.path.join(DATA_TASK1_PATH, "train_features_jp.csv")
df.to_csv(saved_path, index=False)
print("over dump...")

# df=pd.read_csv(saved_path)