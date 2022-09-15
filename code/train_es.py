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

def get_common2(token):
  if isinstance(token[0],float) or isinstance(token[1],float):return 0.0
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

## calculate similarity on Spain(es)
DATA_TASK1_PATH = "../data/task1/"
PRODUCT_CATALOGUE_PATH_FILE = os.path.join(DATA_TASK1_PATH, "new_product_catalogue_es.csv")
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

col_title_term = "title_term"
col_desc_term = "describe_term"
col_bullet_term = "bullet_term"
col_query_title = "query_title"
col_query_desc = "query_describe"
col_query_bullet = "query_bullet"

col_title_tfidf = "title_tfidf"
col_desc_tfidf = "describe_tfidf"
col_bullet_tfidf = "bullet_tfidf"
# saved_path = os.path.join(DATA_TASK1_PATH, "train_token_es.csv")
# print("data loading...")
# df=pd.read_csv(saved_path)
# print("start ratio...")
# df[col_title_ratio] = df[[col_title_token,col_query_token]].apply(lambda x:get_cosine_sim(*x),axis=1) #(len(set(df[col_title_token]) & set(df[col_query_tokan])) / len(set(df[col_query_tokan])))
# print("start desc ratio...")
# df[col_desc_ratio] = df[[col_desc_token,col_query_token]].apply(lambda x:get_cosine_sim(*x),axis=1)#len(set(df[col_desc_token]) & set(df[col_query_tokan])) / len(set(df[col_query_tokan]))
# print("start bullet ratio...")
# df[col_bullet_ratio] = df[[col_bullet_token,col_query_token]].apply(lambda x:get_cosine_sim(*x),axis=1) #len(set(df[col_bullet_token]) & set(df[col_query_tokan])) / len(set(df[col_query_tokan]))
# print("start brand ratio...")
# df[col_brand_ratio] = df[[col_brand_token,col_query_token]].apply(lambda x:get_common2(x),axis=1)#len(set(df[col_brand_token]) & set(df[col_query_tokan]))
# print("start color ratio...")
# df[col_color_ratio] = df[[col_color_token,col_query_token]].apply(lambda x:get_common2(x),axis=1)#len(set(df[col_color_token]) & set(df[col_query_tokan]))
# print("over ratio... ")
#
# df.fillna("",inplace=True) ##really important!
# total_query = df[col_query_token].values
# total_title = df[col_title_token].values
# total_desc = df[col_desc_token].values
# total_bullet = df[col_bullet_token].values
#
# print("title tfidf...")
# term_title_matrix, title_query_matrix = get_tfidf(total_title,total_query)
# print("desc tfidf...")
# term_desc_matrix, desc_query_matrix = get_tfidf(total_desc,total_query)
# print("bulle tfidf...")
# term_bullet_matrix, bullet_query_matrix = get_tfidf(total_bullet,total_query)
#


#
# print("add term...")
# df[col_title_term]=[i for i in term_title_matrix]
# df[col_desc_term]=[i for i in term_desc_matrix]
# df[col_bullet_term]=[i for i in term_bullet_matrix]
# df[col_query_title]=[i for i in title_query_matrix]
# df[col_query_desc]=[i for i in desc_query_matrix]
# df[col_query_bullet]=[i for i in bullet_query_matrix]
#
# print("title tfidf...")
# df[col_title_tfidf]=df[[col_query_title,col_title_term]].apply(lambda x:cos(*x),axis=1)
# print("desc tfidf...")
# df[col_desc_tfidf]=df[[col_query_desc,col_desc_term]].apply(lambda x:cos(*x),axis=1)
# print("bullet tfidf...")
# df[col_bullet_tfidf]=df[[col_query_bullet,col_bullet_term]].apply(lambda x:cos(*x),axis=1)
#
#
# df = df[[col_query_id, col_query, col_query_token, col_title_token,col_title_ratio,col_title_tfidf,col_desc_token,col_desc_ratio,col_desc_tfidf,
#          col_bullet_token, col_bullet_ratio,col_bullet_tfidf, col_brand_token, col_brand_ratio, col_color_token, col_color_ratio,
#          col_gain]]  ##取查询id，查询内容，结果名称，结果的gain
# # df_train = df[df[col_query_id].isin(list_query_id_train)]
# # df_dev = df[df[col_query_id].isin(list_query_id_dev)]
# print("dump...")
saved_path = os.path.join(DATA_TASK1_PATH, "train_features_es.csv")
# df.to_csv(saved_path, index=False)
# print("over dump...")
df=pd.read_csv(saved_path)
# bar = pyprind.ProgPercent(len(df))

def RMSE(gold,pre):
  RMSE = 0.0
    # 2.compute RMSE
  data_len = len(pre)
  RMSE = sum([((pre[i] - gold[i]) ** 2) / data_len for i in range(data_len)]) ** 0.5
  return RMSE

""" 2. Prepare data loaders """
x_train = df[[col_title_ratio,col_title_tfidf, col_desc_ratio,col_desc_tfidf, col_bullet_ratio,col_bullet_tfidf,
              col_brand_ratio, col_color_ratio]].values
y_train = df[col_gain].values
# x_train=np.array(x_train)
# y_train=np.array(y_train)
## 做shuffle+层次抽样
index = np.arange(len(x_train))
np.random.shuffle(index)
x_data = x_train[index]
y_data = y_train[index]
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, random_state=1, stratify=y_data)

params = {'learning_rate': 0.02, 'max_depth': 7, 'min_samples_leaf': 9, 'n_estimators': 300}
# model = GradientBoostingRegressor(max_features=0.8, min_samples_split=5, subsample=0.9, **params)
model = GradientBoostingRegressor(random_state=10)
model.fit(x_train, y_train)


model_saved=DATA_TASK1_PATH+'saved_model/GBDT_es.pkl'
joblib.dump(model,model_saved)

y_pred = model.predict(x_test)
features = [col_title_ratio,col_title_tfidf, col_desc_ratio,col_desc_tfidf, col_bullet_ratio,col_bullet_tfidf,
              col_brand_ratio, col_color_ratio]
importances = model.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(10, 10))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='#1e90ff', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

print(importances)
print("RMSE : " ,RMSE(y_test, y_pred))