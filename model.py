import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from utils import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

def Bert_model(taskType,bertPath):
    label_df = pd.read_csv('./data/%s_label.csv'%taskType)
    tokenizer = BertTokenizer.from_pretrained(bertPath,do_lower_case=False)
    model = BertForSequenceClassification.from_pretrained(bertPath, num_labels=label_df['label'].nunique())
    Bert_conf(tokenizer,model)
    return tokenizer,model
