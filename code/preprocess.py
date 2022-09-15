import collections
import re
import os
import pandas as pd
import pyprind
import json

def read_time_machine():
    DATA_TASK1_PATH = "../data/task1/"
    PRODUCT_CATALOGUE_PATH_FILE = os.path.join(DATA_TASK1_PATH, "product_catalogue-v0.2.csv.zip")
    # PRODUCT_CATALOGUE_PATH_FILE = "${DATA_TASK1_PATH}/product_catalogue-v0.2.csv.zip"
    PRODUCT_CATALOGUE_PATH_FILE = os.path.join(DATA_TASK1_PATH, "product_catalogue-v0.2.csv.zip")

    df = pd.read_csv(PRODUCT_CATALOGUE_PATH_FILE)
    lines=[]
    bar=pyprind.ProgPercent(len(df))
    # dict_product_color={}
    All_color=set()
    All_brand=set()
    dict_product_brand={}

    for (_, row) in df.iterrows():  ## iterrows 可以返回所有的行索引，以及该行的所有内容
        # print(_)
        pid, title, desc, bullet, brand, color, locale = row["product_id"], row["product_title"], row[
            "product_description"], row["product_bullet_point"], row["product_brand"], row["product_color_name"], row[
                                                             "product_locale"]
        # dict_product_color[pid]=color
        # dict_product_brand[pid]=brand
        All_color.add(color)
        All_brand.add(brand)

        bar.update()
        # line=str(color)#str(title)+str(desc)+str(bullet)+str(brand)+str(color)
        # lines.append(line)

    # 将每一行中的标点符号全部以空格代替，并把大写字母换成小写字母（正则化）
    # return lines #[re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]
    to_file=os.path.join(DATA_TASK1_PATH,"dict_product_brand.json")
    # print("dump...")
    # json.dump(dict_product_brand,open(to_file,"w"))
    print("dump...")
    All_color_file=os.path.join(DATA_TASK1_PATH,"All_color.txt")
    All_brand_file=os.path.join(DATA_TASK1_PATH,"All_brand.txt")
    All_color=list(All_color)
    All_brand=list(All_brand)
    file = open(All_color_file, mode='a', encoding='utf-8')  ##保存测试集预测数据
    file.truncate(0)  ##清空
    for i in range(len(All_color)):
        #     print(all_news[i])
        file.write(str(All_color[i])+"\n")
    file.close()

    file = open(All_brand_file, mode='a', encoding='utf-8')  ##保存测试集预测数据
    file.truncate(0)  ##清空
    for i in range(len(All_brand)):
        #     print(all_news[i])
        file.write(str(All_brand[i])+"\n")
    file.close()

def get_dict_product_color_freq(text):
    dict_product_color_freq={}
    color_text=text
    bar = len()







read_time_machine()






#
# def tokenize(lines, token='word'):
#     if token == 'word':
#         return [line.split() for line in lines]
#     elif token == 'char':
#         return [list(line) for line in lines]
#     else:
#         print('错误：未知词元类型：' + token)
#
# class Vocab:
#     def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
#         if tokens is None:
#             tokens = []
#         if reserved_tokens is None:
#             reserved_tokens = []
#         # 按出现频率排序
#         counter = count_corpus(tokens)
#         self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
#                                    reverse=True)
#         # 未知词元的索引为0
#         self.idx_to_token = ['<unk>'] + reserved_tokens
#         self.token_to_idx = {token: idx
#                              for idx, token in enumerate(self.idx_to_token)}
#         for token, freq in self._token_freqs:
#             if freq < min_freq:
#                 break
#             if token not in self.token_to_idx:
#                 self.idx_to_token.append(token)
#                 self.token_to_idx[token] = len(self.idx_to_token) - 1
#
#     def __len__(self):
#         return len(self.idx_to_token)
#
#     def __getitem__(self, tokens):
#         if not isinstance(tokens, (list, tuple)):
#             return self.token_to_idx.get(tokens, self.unk)
#         return [self.__getitem__(token) for token in tokens]
#
#     def to_tokens(self, indices):
#         if not isinstance(indices, (list, tuple)):
#             return self.idx_to_token[indices]
#         return [self.idx_to_token[index] for index in indices]
#
# def count_corpus(tokens):
#     # 这里的tokens是1D列表或2D列表
#     if len(tokens) == 0 or isinstance(tokens[0], list):
#         # 将词元列表展平成一个列表
#         tokens = [line for line in tokens ]
#         # tokens = [token for line in tokens for token in line]
#
#     # 统计词频，返回类型是{token:出现次数}
#     print(collections.Counter(tokens))
#     return collections.Counter(tokens)
#
#
# lines = read_time_machine()
# tokens = tokenize(lines)
# vocab = Vocab(tokens)
# print(list(vocab.token_to_idx.items())[:10])
