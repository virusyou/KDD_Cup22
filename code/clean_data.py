
import re
# from pyhanlp import *
import pandas as pd
import os
import pyprind
import numpy as np

#
# def latin_filter(name):
# #	This function convert latin character into ascii character

# #	to A
# 	name = re.sub(r'\xc3[\x80-\x85]', r'A', name)
# 	name = re.sub(r'\xc4[\x80\x82\x84]', r'A', name)
# 	name = re.sub(r'\xc8\xa6', r'A', name)
# #	to AE
# 	name = re.sub(r'\xc3\x86', r'AE', name)
# #	to C
# 	name = re.sub(r'\xc3\x87', r'C', name)
# 	name = re.sub(r'\xc4[\x86\x88\x8a\x8c]', r'C', name)
# #	to D
# 	name = re.sub(r'\xc3\x90', r'D', name)
# 	name = re.sub(r'\xc4[\x8e\x90]', r'D', name)
# #	to E
# 	name = re.sub(r'\xc3[\x88-\x8b]', r'E', name)
# 	name = re.sub(r'\xc4[\x92\x94\x96\x98\x9a]', r'E', name)
# #	to G
# 	name = re.sub(r'\xc4[\x9c\x9e\xa0\xa2]', r'G', name)
# #	to I
# 	name = re.sub(r'\xc3[\x8c-\x8f]', r'I', name)
# 	name = re.sub(r'\xc4[\xa8\xaa\xac\xae\axb0]', r'I', name)
# #	to J
# 	name = re.sub(r'\xc4\xb4', r'J', name)
# #	to K
# 	name = re.sub(r'\xc4\xb6', r'K', name)
# #	to L
# 	name = re.sub(r'\xc4\xbd', r'L', name)
# #	to N
# 	name = re.sub(r'\xc3\x91|\xc5[\x83\x87]', r'N', name)
# #	to O
# 	name = re.sub(r'\xc3[\x92-\x96\x98]', r'O', name)
# 	name = re.sub(r'\xc5[\x8c\x8e\x90]', r'O', name)
# 	name = re.sub(r'\xc6\x9f', r'O', name)
# #	to OE
# 	name = re.sub(r'\xc5\x92', r'OE', name)
# #	to R
# 	name = re.sub(r'\xc5[\x94\x98]', r'R', name)
# #	to S
# 	name = re.sub(r'\xc5[\x9a\x9c\x9e\xa0]', r'S', name)
# #	to T
# 	name = re.sub(r'\xc5[\xa2\xa4]', r'T', name)
# #	to U
# 	name = re.sub(r'\xc3[\x99-\x9c]', r'U', name)
# 	name = re.sub(r'\xc5[\xa8\xaa\xac\xae\xb0\xb2]', r'U', name)
# 	name = re.sub(r'\xc6\xaf', r'U', name)
# #	to Y
# 	name = re.sub(r'\xc3\x9d', r'Y', name)
# #	to Z
# 	name = re.sub(r'\xc5[\xb9\xbb\xbd]', r'Z', name)
# #	to Th
# 	name = re.sub(r'\xc3\x9e', r'Th', name)
# #	to s
# 	name = re.sub(r'\xc3\x9f', r's', name)
# #	to a
# 	name = re.sub(r'\xc3[\xa0-\xa5]', r'a', name)
# 	name = re.sub(r'\xc4[\x81\x83\x85]', r'a', name)
# 	name = re.sub(r'\xc7\x8e', r'a', name)
# 	name = re.sub(r'\xc8[\x83\xa7]', r'n', name)
# #	to ae
# 	name = re.sub(r'\xc3\xa6', r'ae', name)
# #	to c
# 	name = re.sub(r'\xc3\xa7', r'c', name)
# 	name = re.sub(r'\xc4[\x87\x89\x8b\x8d]', r'c', name)
# #	to d
# 	name = re.sub(r'\xc3\xb0', r'd', name)
# 	name = re.sub(r'\xc4[\x8f\x91]', r'd', name)
# #	to e
# 	name = re.sub(r'\xc3[\xa8-\xab]', r'e', name)
# 	name = re.sub(r'\xc4[\x93\x95\x97\x99\x9b]', r'e', name)
# 	name = re.sub(r'\xc8\xa9', r'e', name)
# 	name = re.sub(r'\xc9[\x99\x9b]', r'e', name)
# #	to g
# 	name = re.sub(r'\xc4[\x9d\x9f\xa1\xa3]', r'g', name)
# 	name = re.sub(r'\xc7\xa7', r'g', name)
# #	to h
# 	name = re.sub(r'\xc4[\xa5\xa7]', r'h', name)
# #	to i
# 	name = re.sub(r'\xc3[\xac-\xaf]', r'i', name)
# 	name = re.sub(r'\xc4[\xa9\xab\xad\xaf\xb0\xb1]', r'i', name)
# 	name = re.sub(r'\xc7\x90', r'i', name)
# #	to ij
# 	name = re.sub(r'\xc4\xb3', r'ij', name)
# #	to j
# 	name = re.sub(r'\xc4\xb5', r'j', name)
# #	to k
# 	name = re.sub(r'\xc4\xb7', r'k', name)
# #	to l
# 	name = re.sub(r'\xc4[\xba\xbe]|\xc5\x80', r'l', name)
# #	to n
# 	# name =re.sub(r'\u00f1','n',name)
#
# 	name = re.sub(r'\xc3\xb1', 'n', name)
# 	name = re.sub(r'\xc5[\x84\x86\x88]','n',name)
# 	name = re.sub(r'\xc7\xb9', 'n', name)
# #	to eng
# 	name = re.sub(r'\xc5\x8b', r'eng', name)
# #	to oe
# 	name = re.sub(r'\xc5\x93', r'oe', name)
# #	to o
# 	name = re.sub(r'\xc3[\xb2-\xb6\xb8]', r'o', name)
# 	name = re.sub(r'\xc5[\x8d\x8f\x91]', r'o', name)
# 	name = re.sub(r'\xc7\xab', r'o', name)
# #	to r
# 	name = re.sub(r'\xc5[\x95\x99]', r'r', name)
# 	name = re.sub(r'\xc9\xb9', r'r', name)
# #	to t
# 	name = re.sub(r'\xc5[\xa3\xa5]', r't', name)
# #	to s
# 	name = re.sub(r'\xc5[\x9b\x9d\x9f\xa1]', r's', name)
# #	to u
# 	name = re.sub(r'\xc3[\xb9-\xbc]', r'u', name)
# 	name = re.sub(r'\xc5[\xa9\xab\xad\xaf\xb1\xb3]', r'u', name)
# #	to y
# 	name = re.sub(r'\xc3[\xbd\xbf]', r'y', name)
# #	to z
# 	name = re.sub(r'\xc5[\xba\xbc\xbe]', r'z', name)
# #	to th
# 	name = re.sub(r'\xc3\xbe', r'th', name)
#
# #	to L
# 	name = re.sub(r'\xc5\x81', r'L', name)
# #	to l
# 	name = re.sub(r'\xc5\x82', r'l', name)
# #	haven't done '\xce\xbe', '\xce\xbf', '\xcf\x80', '\xcf\x9a'
# 	name = re.sub(r'[\xc0-\xdf].', '', name)
# 	name = re.sub(r'[\xe0-\xef]..', '', name)
# 	name = re.sub(r'[\xf0-\xf7]...', '', name)
#
# 	return name


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
def clear_character(text):
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
	return text


	##分词
	# text=HanLP.segment(text)
	# print(text)

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
	df = df[df["product_locale"]=="jp"]
	print(len(df["product_id"]))
	new_df=pd.DataFrame(columns=colu)
	bar = pyprind.ProgPercent(len(df))
	for (_, row) in df.iterrows():  ## iterrows 可以返回所有的行索引，以及该行的所有内容
		# print(_)
		pid,title,desc,bullet,brand,color,locale=row["product_id"],row["product_title"],row["product_description"],row["product_bullet_point"],row["product_brand"],row["product_color_name"],row["product_locale"]
		# print(query)
		desc=clear_character(desc)
		bullet=clear_character(bullet)
		# print(query)
		element = pd.Series({"product_id":pid,"product_title":title,"product_description":desc,"product_bullet_point":bullet,"product_brand":brand,"product_color_name":color,"product_locale":locale})
		new_df = new_df.append(element, ignore_index=True)
		# df["query"][_]=query
		# print(df["query"][_])
		bar.update()

	new_df.to_csv(saved_path, index=False)

if __name__ == "__main__":
	text="Amazon Basics Woodcased"
	print(clear_character(text))
	DATA_TASK1_PATH = "../data/task1/"
	PRODUCT_CATALOGUE_PATH_FILE = os.path.join(DATA_TASK1_PATH,"product_catalogue-v0.2.csv.zip")
	# PRODUCT_CATALOGUE_PATH_FILE = "${DATA_TASK1_PATH}/product_catalogue-v0.2.csv.zip"
	TRAIN_PATH_FILE = os.path.join(DATA_TASK1_PATH,"train-v0.2.csv.zip")
	SAVED_PATH= os.path.join(DATA_TASK1_PATH,"new_train.csv")
	# clean_train(TRAIN_PATH_FILE,SAVED_PATH)

	SAVED_PATH=os.path.join(DATA_TASK1_PATH,"new_product_catalogue_jp.csv")
	# clean_catalogue(PRODUCT_CATALOGUE_PATH_FILE,SAVED_PATH)
	tmp=pd.read_csv(TRAIN_PATH_FILE)
	tmp=tmp["esci_label"]
	# print(tmp)