# Biomedical-Relation-Classification

[![Project](https://img.shields.io/badge/project-Biomedical--Relation--Classification-orange)](https://github.com/jxzly/Biomedical-Relation-Classification)
[![Application](https://img.shields.io/badge/application-Kaggle--COVID--19--knowledge--graph-brightgreen)](https://www.kaggle.com/daishu/covid-19-knowledge-graph)
[![Issues](https://img.shields.io/badge/github-issues-blue)](https://github.com/jxzly/Biomedical-Relation-Classification/issues)
[![Python](https://img.shields.io/badge/python-%3E%3D3.5-yellow)](https://www.python.org/downloads/)

To understand what relationship types are possible and map unstructured natural language descriptions onto these structured classes, we used labeled sentences in [Percha B 2018](https://academic.oup.com/bioinformatics/article/34/15/2614/4911883) to classify relations bewteen chemicals, genes and diseases.

### Requirements

torch='1.3.1'
transformers='2.7.0'
keras

### Dataset

All train dataset can be found [here](https://zenodo.org/record/1035500#.Xe3uR5MzZTZ). You can download them manually and put them in the `./data`.

We used [Kaggle-COVID-19-dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks) to extract sentences what contains more than two biomedical entities and put them in `./data/COVID-19`. If you have a new dataset, please save it in `./data/YourDatasetName`, the format as follows.

start_entity | end_entity | start_entity_type | end_entity_type | marked_sentence
- | - | - | - | -
COVID-19 | Pharyngitis | Disease | Disease | end_entity, bronchitis, and start_entity represent the most common respiratory tract infections
remdesivir | MERS | Chemical | Disease | The data presented here support testing of the efficacy of start_entity treatment in the context of a end_entity clinical trial
remdesivir | COVID-19 | Chemical | Disease | Drugs are possibly effective for end_entity include: start_entity, lopinavir ritonavir, lopinavir ritonavir combined with interferon-, convalescent plasma, and monoclonal antibodies

### Pretrained model

We used `./pretrained/bert-base-cased` as pretrained model. You can use other pretrained bert model by specifying --bert_path.

### Training

**Parameters:**

`--task_type`: task type:chemical-disease,chemical-gene,gene-disease.

`--confidence_limit`: dependency path lower confidence limit, use suggestion value if it equal -1.0. suggestion value:0.9 for chemical-disease; 0.5 for chemical-gene; 0.6 for gene-disease; 0.9 for gene-gene.

`--prediction_path`: prediction data path.

`--max_seq_len`: padding length of sequence.

`--bert_path`: pretrained bert model path.

`--lr`: learning rate.

`--train_bs`: train batch size.

`--eval_bs`: evaluate batch size.  

`--epochs`: training epochs.

`--cuda`: which gpu be used.

**Fine-tuning models:**

`./model/chemical-disease`

`./model/chemical-gene`

`./model/gene-disease`

`./model/gene-gene`

### Reproducing results

python3 main.py --task_type chemical-disease --prediction_path COVID-19 --max_seq_len 64 --bert_path ./pretrained/bert-base-cased --lr 1e-5 --train_bs 128 --eval_bs 64 --epochs 10 --cuda 0

### COVID-19 knowledge graph

![chemical-COVID-19](https://github.com/jxzly/Biomedical-Relation-Classification/blob/master/figure/chemical-COVID-19.png?raw=true)

![subsample-of-COVID-19](https://github.com/jxzly/Biomedical-Relation-Classification/blob/master/figure/subsample-of-COVID-19.png?raw=true)

### Acknowledgement

This work is supported by [Aladdin Healthcare Technologies](https://aladdinid.com/)
<span style="display: inline-block;"> <img src="https://github.com/jxzly/Biomedical-Relation-Classification/blob/master/figure/Aladdin.png?raw=true" width=40/></span>
