
import re
import sys

import hanlp
import pandas as pd
import os
import polyglot
import numpy as np
from hanlp.utils.lang.en.english_tokenizer import tokenize_english
from janome.tokenizer import Tokenizer
import nltk
from nltk import word_tokenize

# 载入停用词
en_stopwords = set()
jp_stopwords = set()
es_stopwords = nltk.corpus.stopwords.words('spanish')
with open('stopwords/en_stopwords.txt', 'r',encoding='utf8') as infile:
	for line in infile:
		line = line.rstrip('\n')
		if line:
			en_stopwords.add(line.lower())

with open('stopwords/jp_stopwords.txt', 'r',encoding='utf8') as infile:
	for line in infile:
		line = line.rstrip('\n')
		if line:
			jp_stopwords.add(line)


def latin_filter(name):
#	This function convert latin character into ascii character

#	to N
	name = re.sub(r'\u00d1', r'N', name)

#	to a
	name = re.sub(r'\u00e1', r'a', name)

#	to e
	name = re.sub(r'\u00e9', r'e', name)

#	to i
	name = re.sub(r'\u00ed', r'i', name)

#	to n
	name =re.sub(r'\u00f1','n',name)

#	to o
	name = re.sub(r'\u00f3', r'o', name)

#	to u
	name = re.sub(r'\u00fa', r'u', name)

## 顺便去除html标签
	name = re.sub(r"<([^>]*)>","",name)


	return name

#大写字母转为小写字母
def upper2lower(text:str):
    return text.lower()

##文本清洗
def clear_character(text, locale):
	text=latin_filter(str(text))
	#只取合法字符,数字、英文、西班牙语、日语
	pattern = [  ## /[\u0800-\u4e00]/  ff10 - ffdf 日语， \u4e00-\u9fa5 中文，
		"[^\u4e00-\u9fa5^\u0800-\u4e00^\uff10-\uffdf^a-z^A-Z^0-9^\u002e^\u002f^\u002d^\u007e^\u0025^\u0020^\u002c]",  # save_standing_character
		# "\.$"  # remove_full_stop
		# "./-~%"  ##保留数字信息 . / - ~ % 及空格 ,
	]
	tmp='|'.join(pattern)
	##去除emoji
	# p = '['u'\U0001F300-\U0001F64F' u'\U0001F680-\U0001F6FF' u'\u2600-\u2B55 \U00010000-\U0010ffff]+'
	# text = re.sub(p, '', text)  # 正则匹配，将表情符合替换为空''
    # return re.sub('|'.join(pattern),'', text)
	text= re.sub(tmp,'',text)
	# text = re.sub("[\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", text)
	han_en_tokenizer = tokenize_english
	t = Tokenizer()
	if(locale == 'us'):
		text = upper2lower(text)
		text = han_en_tokenizer(text)
		text = [word for word in text if word not in en_stopwords ]
		# print(text)
	elif(locale == 'jp'):
		text = t.tokenize(text, wakati=True)
		text = list(text)
		text = [word for word in text if word not in jp_stopwords]
	elif(locale == 'es'):
		text = word_tokenize(text, "spanish")
		text = [word for word in text if word not in es_stopwords ]

	text=''.join(text)
	return text


def clean_train(train_path,saved_path):
	colu=["query_id","query","query_locale","product_id","esci_label"]
	df = pd.read_csv(train_path)
	new_df=pd.DataFrame(columns=colu)
	for (_, row) in df.iterrows():  ## iterrows 可以返回所有的行索引，以及该行的所有内容
		qid,query,locale,pid,label=row["query_id"],row["query"],row["query_locale"],row["product_id"],row["esci_label"]
		# print(query)
		query=clear_character(query)
		# print(query)
		element = pd.Series({"query_id":qid,"query":query,"query_locale":locale,"product_id":pid,"esci_label":label})
		new_df = new_df.append(element, ignore_index=True)
		# df["query"][_]=query
		# print(df["query"][_])

	new_df.to_csv(saved_path, index=False)



def clean_catalogue(cata_path,saved_path):
	colu=["product_id","product_title","product_description","product_bullet_point","product_brand","product_color_name","product_locale"]
	df = pd.read_csv(cata_path)
	print(len(df["product_id"]))
	new_df=pd.DataFrame(columns=colu)
	for (index, row) in df.iterrows():  ## iterrows 可以返回所有的行索引，以及该行的所有内容
		if(index == 10):
			sys.exit(123)
		pid,title,desc,bullet,brand,color,locale=row["product_id"],row["product_title"],row["product_description"],row["product_bullet_point"],row["product_brand"],row["product_color_name"],row["product_locale"]
		# print(query)
		desc=clear_character(desc, locale)
		bullet=clear_character(bullet, locale)
		# print(query)
		element = pd.Series({"product_id":pid,"product_title":title,"product_description":desc,"product_bullet_point":bullet,"product_brand":brand,"product_color_name":color,"product_locale":locale})
		new_df = new_df.append(element, ignore_index=True)
		# df["query"][_]=query
		# print(df["query"][_])

	new_df.to_csv(saved_path, index=False)

if __name__ == "__main__":
	text="Amazon Basics Woodcased"
	print(text)
	print(clear_character(text, 'us'))
	DATA_TASK1_PATH = "../data/task1/"
	PRODUCT_CATALOGUE_PATH_FILE = os.path.join(DATA_TASK1_PATH,"product_catalogue-v0.2.csv.zip")
	# PRODUCT_CATALOGUE_PATH_FILE = "${DATA_TASK1_PATH}/product_catalogue-v0.2.csv.zip"
	TRAIN_PATH_FILE = os.path.join(DATA_TASK1_PATH,"train-v0.2.csv.zip")
	SAVED_PATH= os.path.join(DATA_TASK1_PATH,"new_train.csv")
	# clean_train(TRAIN_PATH_FILE,SAVED_PATH)

	SAVED_PATH=os.path.join(DATA_TASK1_PATH,"new_product_catalogue.csv")
	# clean_catalogue(PRODUCT_CATALOGUE_PATH_FILE,SAVED_PATH)
