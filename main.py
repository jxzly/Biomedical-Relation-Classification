import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
import argparse
from data import *
from model import *
from utils import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

conf = Conf()

def Parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--task_type',
                      default='chemical-disease', help='task type:chemical-disease,chemical-gene,gene-disease')
    args.add_argument('--confidence_limit', type=float,
                      default=-1.0, help='dependency path lower confidence limit, use suggestion value if it equal -1.0. \
                      suggestion value:0.9 for chemical-disease; 0.5 for chemical-gene; 0.6 for gene-disease; 0.9 for gene-gene')
    args.add_argument('--prediction_path',
                      default='COVID-19', help='prediction data path')
    args.add_argument('--max_seq_len', type=int,
                      default=64, help='padding length of sequence')
    args.add_argument('--bert_path',
                      default='./pretrained/bert-base-cased', help='bert model path')
    args.add_argument('--lr', type=float, default=1e-5)
    args.add_argument('--train_bs', type=int, default=128, help='train batch size')
    args.add_argument('--eval_bs', type=int, default=128, help='evaluate batch size')
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--cuda', type=int, default=0, help='which gpu be used')
    args = args.parse_args()
    return args

args = Parse_args()
# dependency path lower confidence limit, use suggestion value if it equal -1.0
confidence_limit = args.confidence_limit if args.confidence_limit != -1.0 else conf.confidence_limit[args.task_type]
tokenizer = BertTokenizer.from_pretrained(args.bert_path,do_lower_case=False)
Bert_conf(tokenizer)
prepare = Prepare_Data(args.task_type,confidence_limit,args.prediction_path,args.max_seq_len,args.bert_path)

def Load_train_test_data():
    train_data_loader,test_data_loader = prepare.Prepare_train_test_data(tokenizer,args.train_bs,args.eval_bs)
    return train_data_loader,test_data_loader

def Load_predict_data():
    marked_sentences,predict_data_loader = prepare.Prepare_predict_data(tokenizer,args.eval_bs)
    return marked_sentences,predict_data_loader

train_data_loader,test_data_loader = Load_train_test_data()
marked_sentences,predict_data_loader = Load_predict_data()
device = torch.device('cuda:%s'%args.cuda if torch.cuda.is_available() else 'cpu')
print('device:', device)

def Train(evalEpochs=None):
    tokenizer,model = Bert_model(args.task_type,args.bert_path)
    tokenizer.save_pretrained('./model/%s'%args.task_type)
    model = model.to(device)
    model_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(model_params, lr=args.lr)
    for epoch in range(args.epochs):
        running_loss = 0.0
        for data in tqdm(train_data_loader):
            ids, labels = [t.to(device) for t in data]
            optimizer.zero_grad()
            # forward pass
            outputs = model(input_ids=ids,labels=labels)
            loss = outputs[0]
            # backward
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'[epoch {epoch+1}] loss: {running_loss:3f}')
        if evalEpochs != None:
            if (epoch+1)%evalEpochs == 0:
                torch.cuda.empty_cache()
                Evaluate(model)
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained('./model/%s'%args.task_type)
    torch.cuda.empty_cache()
    return

def Evaluate(model=None):
    if model == None:
        tokenizer,model = Bert_model(args.task_type,'./model/%s'%args.task_type)
        model = model.to(device)
    test_preds,test_labels = [],[]
    for data in tqdm(test_data_loader):
        ids, labels = [t.to(device) for t in data]
        outputs = model(input_ids=ids)
        logits = outputs[0]
        _, pred = torch.max(logits.data, 1)
        test_preds.extend(list(pred.cpu().detach().numpy()))
        test_labels.extend(list(labels.cpu().detach().numpy()))
    macro_f1 = f1_score(test_labels,test_preds,average='macro')
    print('test macro f1 score:%.4f'%macro_f1)
    torch.cuda.empty_cache()
    return

def Predict():
    reverse_task_type = args.task_type.split('-')[1] + '-' + args.task_type.split('-')[0]
    def Filter(x):
        if x['init_pred'] in conf.relation_type[args.task_type]:
            if x['reverse_pred'] not in conf.relation_type[reverse_task_type]:
                # init_pred is a correct relation but reverse_pred not
                return 'init_pred'
            else:
                # init_pred and reverse_pred both are correct relations
                if x['init_pred_prob'] >= x['reverse_pred_prob']:
                    # init_pred_prob greater than or equal to reverse_pred_prob
                    return 'init_pred'
                else:
                    return 'reverse_pred'
        else:
            if x['reverse_pred'] not in conf.relation_type[reverse_task_type]:
                # init_pred and reverse_pred both are uncorrect relations
                return 'uncorrect'
            else:
                # reverse_pred is a correct relation but init_pred not
                return 'reverse_pred'
    marked_sentence_df = pd.read_csv('./data/%s/marked_sentence.csv'%args.prediction_path)
    label_df = pd.read_csv('./data/%s_label.csv'%args.task_type)
    tokenizer,model = Bert_model(args.task_type,'./model/%s'%args.task_type)
    model = model.to(device)
    preds = []
    preds_prob = []
    reverse_preds = []
    reverse_preds_prob = []
    for data in tqdm(predict_data_loader):
        ids,reverse_ids = [t.to(device) for t in data]
        outputs = model(input_ids=ids)
        logits = outputs[0]
        pred_prob, pred = torch.max(F.softmax(logits.data,1), 1)
        preds.extend(list(pred.cpu().detach().numpy()))
        preds_prob.extend(list(pred_prob.cpu().detach().numpy()))
        reverse_outputs = model(input_ids=reverse_ids)
        reverse_logits = reverse_outputs[0]
        reverse_pred_prob, reverse_pred = torch.max(F.softmax(reverse_logits.data,1), 1)
        reverse_preds.extend(list(reverse_pred.cpu().detach().numpy()))
        reverse_preds_prob.extend(list(reverse_pred_prob.cpu().detach().numpy()))

    pred_df = pd.DataFrame({'marked_sentence':marked_sentences,'init_pred':preds,'init_pred_prob':preds_prob,'reverse_pred':reverse_preds,'reverse_pred_prob':reverse_preds_prob})
    # map label(0, 1, 2...) to raw label(T, C, Sa...)
    pred_df['init_pred'] = pred_df['init_pred'].replace(dict(label_df.set_index(['label'])['label_raw']))
    pred_df['reverse_pred'] = pred_df['reverse_pred'].replace(dict(label_df.set_index(['label'])['label_raw']))
    # judge the order of a pair of entities
    pred_df['filter'] = pred_df.apply(lambda x:Filter(x), axis=1)
    pred_df['pred'] = pred_df['init_pred']
    pred_df['pred_prob'] = pred_df['init_pred_prob']
    pred_df.loc[pred_df['filter']=='reverse_pred','pred'] = pred_df.loc[pred_df['filter']=='reverse_pred','reverse_pred']
    pred_df.loc[pred_df['filter']=='reverse_pred','pred_prob'] = pred_df.loc[pred_df['filter']=='reverse_pred','reverse_pred_prob']
    pred_df = pred_df.loc[pred_df['filter']!='uncorrect']
    pred_df = marked_sentence_df.merge(pred_df,how='inner',on='marked_sentence')
    pred_df['init_start_entity'] = pred_df['start_entity']
    pred_df['init_start_entity_type'] = pred_df['start_entity_type']
    pred_df.loc[pred_df['filter']=='reverse_pred','start_entity'] = pred_df.loc[pred_df['filter']=='reverse_pred','end_entity']
    pred_df.loc[pred_df['filter']=='reverse_pred','start_entity_type'] = pred_df.loc[pred_df['filter']=='reverse_pred','end_entity_type']
    pred_df.loc[pred_df['filter']=='reverse_pred','end_entity'] = pred_df.loc[pred_df['filter']=='reverse_pred','init_start_entity']
    pred_df.loc[pred_df['filter']=='reverse_pred','end_entity_type'] = pred_df.loc[pred_df['filter']=='reverse_pred','init_start_entity_type']
    pred_df.drop(['init_start_entity','init_start_entity_type'],axis=1,inplace=True)
    pred_df.to_csv('./data/%s/%s_pred.csv'%(args.prediction_path,args.task_type),index=False)
    torch.cuda.empty_cache()
    return

Train(evalEpochs=5)
Evaluate()
Predict()
