from torch.utils.data import Dataset,TensorDataset,DataLoader

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

class Conf:
    # some information can be found in:
    # Percha B, Altman R B. A global network of biomedical relationships derived from text[J]. Bioinformatics, 2018, 34(15): 2614-2624.
    download_url = {'chemical-disease':['https://zenodo.org/record/1035500/files/part-i-chemical-disease-path-theme-distributions.txt?download=1',
                                             'https://zenodo.org/record/1035500/files/part-ii-dependency-paths-chemical-disease-sorted-with-themes.txt?download=1'],
                    'chemical-gene':['https://zenodo.org/record/1035500/files/part-i-chemical-gene-path-theme-distributions.txt?download=1',
                                                               'https://zenodo.org/record/1035500/files/part-ii-dependency-paths-chemical-gene-sorted-with-themes.txt?download=1'],
                    'gene-disease':['https://zenodo.org/record/1035500/files/part-i-gene-disease-path-theme-distributions.txt?download=1',
                                             'https://zenodo.org/record/1035500/files/part-ii-dependency-paths-gene-disease-sorted-with-themes.txt?download=1'],
                    }
    usecols = {'chemical-disease':['path', 'T', 'C', 'Sa', 'Pr', 'Pa', 'J', 'Mp'],
               'chemical-gene':['path', 'A+', 'A-', 'B', 'E+', 'E-', 'E', 'N', 'O', 'K', 'Z'],
               'gene-disease':['path', 'U', 'Ud', 'D', 'J', 'Te', 'Y', 'G', 'Md', 'X', 'L'],
               'gene-gene':['path', 'B', 'W', 'V+', 'E+', 'E', 'I', 'H', 'Rg', 'Q'],
               }
    relation_type = {'chemical-disease':['T', 'C', 'Sa', 'Pr', 'Pa', 'J'],
                     'disease-chemical':['Mp'],
                     'chemical-gene':['A+', 'A-', 'B', 'E+', 'E-', 'E', 'N'],
                     'gene-chemical':['O', 'K', 'Z'],
                     'gene-disease':['U', 'Ud', 'D', 'J', 'Te', 'Y', 'G'],
                     'disease-gene':['Md', 'X', 'L'],
                     'gene-gene':['B', 'W', 'V+', 'E+', 'E', 'I', 'H', 'Rg', 'Q']
                     }
    relation_theme = {'chemical-disease':{'T':'Treatment/therapy (incl. investigatory)',
                                          'C':'Inhibits cell growth (esp. cancers)',
                                          'Sa':'Side effect/adverse event',
                                          'Pr':'Prevents, suppresses',
                                          'Pa':'Alleviates, reduces',
                                          'J':'Role in pathogenesis',
                                          },
                      'disease-chemical':{'Mp':'Biomarkers (progression)',
                                          },
                      'chemical-gene':{'A+':'Agonism, activation',
                                       'A-':'Antagonism, blocking',
                                       'B':'Binding, ligand (esp. receptors)',
                                       'E+':'Increases expression/production',
                                       'E-':'Decreases expression/production',
                                       'E':'Affects expression/production (neutral)',
                                       'N':'Inhibits',
                                       },
                      'gene-chemical':{'O':'Transport, channels',
                                       'K':'Metabolism, pharmacokinetics',
                                       'Z':'Enzyme activity',
                                       },
                      'gene-disease':{'U':'Causal mutations',
                                      'Ud':'Mutations affect disease course',
                                      'D':'Drug targets',
                                      'J':'Role in pathogenesis',
                                      'Te':'Possible therapeutic effect',
                                      'Y':'Polymorphisms alter risk',
                                      'G':'Promotes progression',
                                      },
                      'disease-gene':{'Md':'Biomarkers (diagnostic)',
                                      'X':'Overexpression in disease',
                                      'L':'Improper regulation linked to disease',
                                      },
                      'gene-gene':{'B':'Binding, ligand (esp. receptors)',
                                   'W':'Enhances response',
                                   'V+':'Activates, stimulates',
                                   'E+':'Increases expression/production',
                                   'E':'Affects expression/production (neutral)',
                                   'I':'Signaling pathway',
                                   'H':'Same protein or complex',
                                   'Rg':'Regulation',
                                   'Q':'Production by cell population',
                                   }
                       }
    confidence_limit = {'chemical-disease':0.9,
                        'chemical-gene':0.5,
                        'gene-disease':0.6,
                        'gene-gene':0.9,
                        }

def Data_loader(x,y=None,bs=128,shuffle=False,numWorkers=0):
    if y is not None:
        data = TensorDataset(x,y)
    else:
        data = TensorDataset(x)
    data_loader = DataLoader(dataset=data,batch_size=bs,shuffle=shuffle,num_workers=numWorkers)
    return data_loader

def Bert_conf(tokenizer=None,model=None):
    # add some tokens to vocab
    if tokenizer != None:
        num_added_toks = tokenizer.add_tokens(['start_entity', 'end_entity'])
        print('We have added', num_added_toks, 'tokens')
    if model != None:
        model.resize_token_embeddings(len(tokenizer))
    return None
