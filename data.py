import numpy as np
import pandas as pd
import sys,os,re
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
from keras.preprocessing import sequence
from model import *
from utils import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

class Prepare_Data(Conf):
    def __init__(self,taskType,confidenceLimit,predictionPath,maxSeqLen,bertPath):
        super(Prepare_Data,self).__init__()
        self.task_type = taskType
        self.confidence_limit = confidenceLimit
        self.prediction_path = predictionPath
        self.max_seq_len = maxSeqLen
        self.bert_path = bertPath

    def Prepare_train_test_data(self,tokenizer,trainBS,evalBS):
        # if not exist original data, download them
        if not os.path.exists('./data/part-i-%s-path-theme-distributions.txt'%self.task_type):
            print('You can download data manually from https://zenodo.org/record/1035500#.Xe3uR5MzZTZ')
            os.system('wget -P ./data %s'%self.download_url[self.task_type][0])
        if not os.path.exists('./data/part-ii-dependency-paths-%s-sorted-with-themes.txt'%self.task_type):
            print('You can download data manually from https://zenodo.org/record/1035500#.Xe3uR5MzZTZ')
            os.system('wget -P ./data %s'%self.download_url[self.task_type][1])

        # build data loder
        path_theme_df = pd.read_csv('./data/part-i-%s-path-theme-distributions.txt'%self.task_type,sep='\t',
                                    usecols=self.usecols[self.task_type])
        path_theme_df['max_value_prob'] = path_theme_df[self.usecols[self.task_type][1:]].max(axis=1) / path_theme_df[self.usecols[self.task_type][1:]].sum(axis=1)
        path_theme_df = path_theme_df.loc[path_theme_df['max_value_prob']>self.confidence_limit].reset_index(drop=True)
        path_theme_df['label_raw'] = path_theme_df[self.usecols[self.task_type][1:]].apply(lambda x:x.argmax(),axis=1)
        path_theme_df.drop(self.usecols[self.task_type][1:],axis=1,inplace=True)
        sentence_df = pd.read_csv('./data/part-ii-dependency-paths-%s-sorted-with-themes.txt'%self.task_type,sep='\t',header=None,
                                  names=['PubMed_ID','sentence_number','start_entity','start_entity_location','end_entity','end_entity_location','start_entity_raw','end_entity_raw','start_entity_database_id','end_entity_database_id','start_entity_type','end_entity_type','path','sentence_tokenized'],
                                  usecols=['start_entity','end_entity','path','sentence_tokenized'],nrows=None)
        sentence_df = sentence_df.loc[~((sentence_df['start_entity'].isna())|(sentence_df['end_entity'].isna()))].reset_index(drop=True)
        sentence_df['path'] = sentence_df['path'].apply(lambda x:x.lower())
        sentence_df['sentence_tokenized'] = sentence_df.apply(lambda x:x['sentence_tokenized'].replace(x['start_entity'],'start_entity').replace(x['end_entity'],'end_entity'),axis=1)
        sentence_df = sentence_df.merge(path_theme_df[['path','label_raw']],how='inner',on='path')

        # encoder labels
        le = preprocessing.LabelEncoder()
        le.fit(sentence_df['label_raw'].values.tolist())
        label_df = pd.DataFrame({'label_raw':le.classes_,'label':list(range(len(le.classes_)))})
        # map label to theme
        theme_dic = self.relation_theme[self.task_type]
        if self.task_type.split('-')[0] != self.task_type.split('-')[1]:
            theme_dic.update(self.relation_theme[self.task_type.split('-')[1]+'-'+self.task_type.split('-')[0]])
        label_df['theme'] = label_df['label_raw'].map(theme_dic)
        label_df.to_csv('./data/%s_label.csv'%self.task_type,index=False)
        sentence_df['label'] = le.transform(sentence_df['label_raw'].values)

        # covert sentences to ids in bert
        ids = sentence_df['sentence_tokenized'].apply(lambda x:tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x))).tolist()
        ids = sequence.pad_sequences(ids,self.max_seq_len, truncating='post', padding='post')

        # split data to train and test
        X_train, X_test, y_train, y_test = train_test_split(ids,sentence_df['label'].values, test_size=0.2, stratify=sentence_df['label'].values, random_state=2020)
        train_data_loader = Data_loader(torch.LongTensor(X_train),torch.LongTensor(y_train),bs=trainBS)
        test_data_loader = Data_loader(torch.LongTensor(X_test),torch.LongTensor(y_test),bs=evalBS)
        return train_data_loader,test_data_loader

    def Prepare_predict_data(self,tokenizer,bs):
        marked_sentence_df = pd.read_csv('./data/%s/marked_sentence.csv'%self.prediction_path)
        marked_sentences = marked_sentence_df.loc[(marked_sentence_df['start_entity_type'].apply(lambda x:x.lower())==self.task_type.split('-')[0])&\
                                                  (marked_sentence_df['end_entity_type'].apply(lambda x:x.lower())==self.task_type.split('-')[1]),'marked_sentence']

        ids = marked_sentences.apply(lambda x:tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x))).tolist()
        ids = sequence.pad_sequences(ids,self.max_seq_len, truncating='post', padding='post')
        # we cannot confirm order of entities, so predict two possibilities
        reverse_marked_sentences = marked_sentence_df.loc[(marked_sentence_df['start_entity_type'].apply(lambda x:x.lower())==self.task_type.split('-')[0])&\
                                                  (marked_sentence_df['end_entity_type'].apply(lambda x:x.lower())==self.task_type.split('-')[1]),'marked_sentence']\
                                                  .apply(lambda x:x.replace('start_entity','init_start_entity').replace('end_entity','start_entity').replace('init_start_entity','end_entity'))
        reverse_ids = reverse_marked_sentences.apply(lambda x:tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x))).tolist()
        reverse_ids = sequence.pad_sequences(reverse_ids,self.max_seq_len, truncating='post', padding='post')
        predict_data_loader = Data_loader(torch.LongTensor(ids),torch.LongTensor(reverse_ids),bs=bs)
        return marked_sentences.values,predict_data_loader
