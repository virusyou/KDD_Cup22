import argparse
# from sentence_transformers.cross_encoder import CrossEncoder
# from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
# from sentence_transformers import SentenceTransformer, InputExample, losses
# from sentence_transformers import evaluation
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

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


def tokenize(text):
    text = str(text)
    text = text.lower()
    tokens = word_tokenize(text)
    newlist = [word for word in tokens if word not in stopwrod]
    return " ".join(stemmer(newlist))
    # return nltk.word_tokenize(text)


stopwrod = []
from nltk.corpus import stopwords

stopwrord = stopwords.words('english')


def main():
    DATA_TASK1_PATH = "../data/task1/"
    PRODUCT_CATALOGUE_PATH_FILE = os.path.join(DATA_TASK1_PATH, "product_catalogue-v0.2.csv.zip")
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

    col_query_tokan = "query_token"
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

    """ 1. Load data """
    locale = "us"
    df = pd.read_csv(TRAIN_PATH_FILE)
    print("train loaded...")
    df_product_catalogue = pd.read_csv(PRODUCT_CATALOGUE_PATH_FILE)
    print("product loaded...")
    df = df[df[col_query_locale] == locale]  ##筛选Eng
    df_product_catalogue = df_product_catalogue[df_product_catalogue[col_product_locale] == locale]
    df = pd.merge(
        df,
        df_product_catalogue,
        how='left',
        left_on=[col_product_id, col_query_locale],
        right_on=[col_product_id, col_product_locale],
    )
    print("merged...")

    df = df[df[col_product_title].notna()]
    list_query_id = df[col_query_id].unique()  ## 查询的unique id
    # dev_size = args.n_dev_queries / len(list_query_id)  ## 划分训练集测试集，吗？
    # list_query_id_train, list_query_id_dev = train_test_split(list_query_id, test_size=dev_size,
    #                                                           random_state=args.random_state)
    print("start token...")
    df[col_gain] = df[col_esci_label].apply(lambda label: esci_label2gain[label])  ##label转化为gain数值
    df[col_query_tokan] = df[col_query].apply(lambda x: tokenize(x))
    df[col_title_token] = df[col_product_title].apply(lambda x: tokenize(x))
    df[col_desc_token] = df[col_product_description].apply(lambda x: tokenize(x))
    df[col_bullet_token] = df[col_product_bullet].apply(lambda x: tokenize(x))
    df[col_brand_token] = df[col_product_brand].apply(lambda x: tokenize(x))
    df[col_color_token] = df[col_product_color].apply(lambda x: tokenize(x))

    print("over token...")
    print("start ratio...")
    df[col_title_ratio] = len(set(df[col_title_token]) & set(df[col_query_tokan])) / len(set(df[col_query_tokan]))
    df[col_desc_ratio] = len(set(df[col_desc_token]) & set(df[col_query_tokan])) / len(set(df[col_query_tokan]))
    df[col_bullet_ratio] = len(set(df[col_bullet_token]) & set(df[col_query_tokan])) / len(set(df[col_query_tokan]))
    df[col_brand_ratio] = len(set(df[col_brand_token]) & set(df[col_query_tokan]))
    df[col_color_ratio] = len(set(df[col_color_token]) & set(df[col_query_tokan]))
    print("over ratio... ")

    df = df[[col_query_id, col_query, col_query_tokan, col_title_token, col_title_ratio, col_desc_token, col_desc_ratio,
             col_bullet_token, col_bullet_ratio, col_brand_token, col_brand_ratio, col_color_token, col_color_ratio,
             col_gain]]  ##取查询id，查询内容，结果名称，结果的gain
    # df_train = df[df[col_query_id].isin(list_query_id_train)]
    # df_dev = df[df[col_query_id].isin(list_query_id_dev)]
    print("dump...")
    saved_path = os.path.join(DATA_TASK1_PATH, "train_features_us.csv")
    df.to_csv(saved_path, index=False)
    print("over dump...")

    df=pd.read_csv(saved_path)
    bar = pyprind.ProgPercent(len(df))
    """ 2. Prepare data loaders """
    x_train = []
    y_train = []
    for (_, row) in df.iterrows():  ## iterrows 可以返回所有的行索引，以及该行的所有内容
        x_train.append([row[col_title_ratio], row[col_desc_ratio], row[col_bullet_ratio], row[col_brand_ratio],
                       row[col_color_ratio]])
        y_train.append(float(row[col_gain]))
        bar.updata()
    ## 做shuffle+层次抽样
    index = np.arange(len(x_train))
    np.random.shuffle(index)
    x_data = x_train[index, :]
    y_data = y_train[index]
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, random_state=1, stratify=y_data)

    # clf = LogisticRegression()
    # clf.fit(x_train,y_train)
    # y_pred=clf.predict(x_test)
    #
    params = {'learning_rate': 0.02, 'max_depth': 7, 'min_samples_leaf': 9, 'n_estimators': 300}
    # model = GradientBoostingRegressor(max_features=0.8, min_samples_split=5, subsample=0.9, **params)
    model = GradientBoostingRegressor(random_state=10)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    features = x_train.columns
    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=(10, 10))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

    print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
    # print "AUC Score (Train): %f" %metrics.roc_auc_score(y_train,y_pred)

    # ts = pd.read_csv("../input/csc-575-final-project-kaggle-winter-2019/sample_submission_new.csv")
    #
    # ts.relevance = y_pred
    #
    # ts.to_csv("sub_reduced.csv", index=None)

    #
    # train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.train_batch_size, drop_last=True)
    # if args.locale == "us":
    #     dev_samples = {}
    #     query2id = {}
    #     for (_, row) in df_dev.iterrows():
    #         try:
    #             qid = query2id[row[col_query]]
    #         except KeyError:
    #             qid = len(query2id)
    #             query2id[row[col_query]] = qid
    #         if qid not in dev_samples:
    #             dev_samples[qid] = {'query': row[col_query], 'positive': set(), 'negative': set()}
    #         if row[col_gain] > 0:
    #             dev_samples[qid]['positive'].add(row[col_product_title])
    #         else:
    #             dev_samples[qid]['negative'].add(row[col_product_title])  ##负样本为irrelevant
    #     evaluator = CERerankingEvaluator(dev_samples, name='train-eval')
    #
    #     """ 3. Prepare Cross-enconder model:  ## corss-encoder 语义相似度模型
    #         https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_cross-encoder_kd.py
    #     """
    #     model_name = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
    #     num_epochs = 1
    #     num_labels = 1
    #     max_length = 512
    #     default_activation_function = torch.nn.Identity()  # 恒等函数
    #     model = CrossEncoder(
    #         model_name,
    #         num_labels=num_labels,
    #         max_length=max_length,
    #         default_activation_function=default_activation_function,
    #         device=device
    #     )
    #     loss_fct = torch.nn.MSELoss()
    #     evaluation_steps = 5000
    #     warmup_steps = 5000
    #     lr = 7e-6
    #     """ 4. Train Cross-encoder model """
    #     model.fit(
    #         train_dataloader=train_dataloader,
    #         loss_fct=loss_fct,
    #         evaluator=evaluator,
    #         epochs=num_epochs,
    #         evaluation_steps=evaluation_steps,
    #         warmup_steps=warmup_steps,
    #         output_path=f"{args.model_save_path}_tmp",
    #         optimizer_params={'lr': lr},
    #     )
    #     model.save(args.model_save_path)

    # else:
    #     dev_queries = df_dev[col_query].to_list()
    #     dev_titles = df_dev[col_product_title].to_list()
    #     dev_scores = df_dev[col_gain].to_list()
    #     evaluator = evaluation.EmbeddingSimilarityEvaluator(dev_queries, dev_titles, dev_scores)
    #
    #     """ 3. Prepare sentence transformers model:
    #         https://www.sbert.net/docs/training/overview.html
    #     """
    #     model_name = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
    #     model = SentenceTransformer(model_name)
    #     train_loss = losses.CosineSimilarityLoss(model=model)
    #     num_epochs = 1
    #     evaluation_steps = 1000
    #     """ 4. Train Sentence transformer model """
    #     model.fit(
    #         train_objectives=[(train_dataloader, train_loss)],
    #         evaluator=evaluator,
    #         epochs=num_epochs,
    #         evaluation_steps=evaluation_steps,
    #         output_path=args.model_save_path,
    #     )

    ## us 用一种模型，cross encoder； jp和es用另一种模型，SentenceTransformer 吗


if __name__ == "__main__":
    main()