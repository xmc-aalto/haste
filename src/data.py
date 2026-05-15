import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import json
from typing import List
from cluster import cluster_labels
import itertools
from label_grouping import group_labels
from label_grouping_faiss import group_labels_faiss
import math
from preprocess import tokenize_file

#change path of the pre computed clusters file in your running environment or include new pair. (Only use when meta_branch=True )
env2clusterpath = {'guest':'./data/'}

name_map = {'eurlex4k': 'Eurlex-4K','wiki31k': 'Wiki10-31K', 'amazon670k': 'Amazon-670K', 'wiki500k': 'Wiki-500K',
                'amazon3m':'Amazon-3M', 'lfamazontitles131k':'LF-AmazonTitles-131K',
                'lfwikiseealso320k':'LF-WikiSeeAlso-320k', 'amazontitles670k':'AmazonTitles-670K','lfpaper2keywords':'LF-Paper2Keywords-8.6M'}
dtype_map = {'float16':torch.float16, 'bfloat16':torch.bfloat16}



def collate(batch):
    '''
    collate function to be used when sparse label format is needed.
    
    '''
    tokens = []
    attention_mask = []
    labels = []
    group_label_ids = []
    for i, (t, m, l, g) in enumerate(batch):
        tokens.append(t)
        attention_mask.append(m)
        l_coo = [(i, lbl) for lbl in l]
        labels += l_coo
        group_label_ids.append(g)
    return (
        torch.utils.data.default_collate(tokens),
        torch.utils.data.default_collate(attention_mask),
        torch.Tensor(labels).to(torch.int32).contiguous(),
        torch.utils.data.default_collate(group_label_ids),
    )


class DataHandler:
    '''
    Handle all the data reading, preprocessing ,dataset, dataloader and other stuff.
    
    '''
    def __init__(self,cfg,path):
        
        self.cfg = cfg
        self.path = path
        self.device = torch.device(cfg.environment.device)
        self.low_precision_dtype = dtype_map[cfg.training.amp.dtype]
        self.label_map = {}
        self.use_ht_split = cfg.model.ffi.use_ht_split
        self.group_size = cfg.model.ffi.group_size
       # self.read_files()

        if cfg.data.tokenized_loading:
            self.read_tokenized_files()
        else:
            self.read_files()

    
        if cfg.model.auxiliary.use_meta_branch:
            self.group_y = self.load_group(cfg.data.dataset)

        
    def load_group(self,dataset):
        '''
        loading of precomputed cluster file
        '''
        print('Loading cluster groups')
        cluster_path = env2clusterpath[self.cfg.environment.running_env] + name_map[dataset] + f'/label_group{self.cfg.model.auxiliary.group_y_group}.npy'
        print('cluster path:',cluster_path)
        return np.load(cluster_path, allow_pickle=True)




    def read_tokenized_files(self):
        
        if not self.cfg.data.is_lf_data:
            train_raw_texts = self._read_text_files(self.path.train_raw_texts) 
            test_raw_texts = self._read_text_files(self.path.test_raw_texts) 

            self.train_labels = self._read_label_files(self.path.train_labels)
            self.test_labels = self._read_label_files(self.path.test_labels)
            train_filename = str(self.cfg.data.dataset)+'_'+'train'
            test_filename = str(self.cfg.data.dataset)+'_'+'test'

        else:
            train_raw_texts, train_labels = self._read_lf_files(self.path.train_json)
            self.train_labels = train_labels
            test_raw_texts, self.test_labels = self._read_lf_files(self.path.test_json)
            train_filename = str(self.cfg.data.dataset)+'_'+'train'
            test_filename = str(self.cfg.data.dataset)+'_'+'test'
            if self.cfg.data.augment_label_data:
                label_raw_texts,label_labels = self._read_lf_files(self.path.label_json,label_json=True)
                train_raw_texts += label_raw_texts
                self.train_labels += label_labels
                train_filename = str(self.cfg.data.dataset)+'_'+'train_'+'aug'
                test_filename = str(self.cfg.data.dataset)+'_'+'test'

        if not self.use_ht_split:
            if self.cfg.model.ffi.use_shared_group and self.cfg.model.ffi.cluster_based_grouping:
                #groups = group_labels(self.cfg,emb_path=self.cfg.model.ffi.cluster_file,group_size=self.cfg.model.ffi.group_size)
                groups = group_labels_faiss(self.cfg,label_list=None, emb_path=self.cfg.model.ffi.cluster_file,
                                    group_size=self.cfg.model.ffi.group_size,probe_k=512,seed=123)
                #groups = cluster_labels(self.cfg.data.dataset,self.path.bow_train_path,self.path.train_labels,64)
                #all_labels = [lbl for lbl in self.label_map.keys()]
                labels = list(itertools.chain(*groups))
                # remaining_labels = set(all_labels) - set(labels)
                # remaining_labels = list(remaining_labels)
                # labels += remaining_labels
                self.label_map = {}
                for i, k in enumerate(labels):
                    self.label_map[k] = i
            else:
                for i, k in enumerate(sorted(self.label_map.keys())):
                    self.label_map[k] = i
            print(f"label mapping finished for non ht_split")
            
        else:
            sorted_labels = sorted(self.label_map.items(), key=lambda x: x[1], reverse=True)
            if self.cfg.model.ffi.use_shared_group and self.cfg.model.ffi.cluster_based_grouping:
                self.num_head_labels = max(1, int(self.cfg.data.num_labels * self.cfg.model.ffi.head_ratio))
                #modify it to the ceiling of group size.
                if self.num_head_labels > self.group_size:
                    self.num_head_labels = int(math.ceil(self.num_head_labels / self.group_size)) * self.group_size
                all_labels = [lbl for lbl, _ in sorted_labels]
                head_labels = all_labels[:self.num_head_labels]
                tail_labels = all_labels[self.num_head_labels:]

                # exclude heads when clustering tail (cluster_labels expects strings)
               # exclude_for_cluster = [str(l) for l in head_labels]
                #groups = cluster_labels(self.cfg.data.dataset,self.path.bow_train_path,self.path.train_labels,64,exclude_labels=exclude_for_cluster)
                #print(f"min cluster size = {min(len(groups[0]),len(groups[1]),len(groups[2]))}")
                #groups = group_labels(self.cfg,label_list=tail_labels,emb_path=self.cfg.model.ffi.cluster_file,group_size=self.cfg.model.ffi.group_size)
                groups = group_labels_faiss(self.cfg,label_list=tail_labels if self.use_ht_split else None, emb_path=self.cfg.model.ffi.cluster_file,
                                    group_size=self.cfg.model.ffi.group_size,probe_k=512,seed=123)
                tail_labels = list(itertools.chain(*groups))
                labels = head_labels + tail_labels
                # also needs to add the labels thats present in test but not in train
                #so it would be remaining = [all_labels - labels]

                # remaining_labels = set(all_labels) - set(labels)
                # remaining_labels = list(remaining_labels)
                # labels += remaining_labels

                print(f"{len(labels)=}")
                # Reset label_map and assign new indices
                self.label_map = {}
                # for i, k in enumerate(labels):
                #     self.label_map[k] = i
                for k in labels:
                    if k not in self.label_map:
                        self.label_map[k] = len(self.label_map)
            else:
                # Reset label_map and assign new indices
            
                print(f"{len(sorted_labels)=}")
                self.label_map = {}
                for idx, (label, freq) in enumerate(sorted_labels):
                    self.label_map[label] = idx

        
        #print(f"{self.label_map=}")
        print(f"{len(self.label_map)=}")
        #print(f"{self.train_labels=}")
        self.train_nsample = len(train_raw_texts)
        self.test_nsample = len(test_raw_texts)
                
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.encoder.encoder_tokenizer,do_lower_case=True)
        self.pad_token_id = tokenizer.pad_token_id
        self.train_tokenized_filename = train_filename+'_'+str(self.cfg.data.max_len)+'.dat'
        if not os.path.exists(self.train_tokenized_filename):
            print(f"tokenized train file is not available. creating the file")
            tokenize_file(train_raw_texts,self.train_tokenized_filename,tokenizer,self.train_nsample,self.cfg.data.max_len)
        else:
            print(f"tokenized train file exists. File {self.train_tokenized_filename } would be used.")
        
        self.test_tokenized_filename = test_filename+'_'+str(self.cfg.data.max_len)+'.dat'
        if not os.path.exists(self.test_tokenized_filename):
            print(f"tokenized test file is not available. creating the file")
            tokenize_file(test_raw_texts,self.test_tokenized_filename,tokenizer,self.test_nsample,self.cfg.data.max_len)
            
        else:
            print(f"tokenized test file exists. File {self.test_tokenized_filename } would be used")

        
    def read_files(self):
        
        if not self.cfg.data.is_lf_data:
            self.train_raw_texts = self._read_text_files(self.path.train_raw_texts) 
            self.test_raw_texts = self._read_text_files(self.path.test_raw_texts) 

            self.train_labels = self._read_label_files(self.path.train_labels)
            self.test_labels = self._read_label_files(self.path.test_labels)
        else:
            self.train_raw_texts, train_labels = self._read_lf_files(self.path.train_json)
            self.train_labels = train_labels
            self.test_raw_texts, self.test_labels = self._read_lf_files(self.path.test_json)
            if self.cfg.data.augment_label_data:
                label_raw_texts,label_labels = self._read_lf_files(self.path.label_json,label_json=True)
                self.train_raw_texts += label_raw_texts
                self.train_labels += label_labels

        if not self.use_ht_split:
            if self.cfg.model.ffi.use_shared_group and self.cfg.model.ffi.cluster_based_grouping:
                #groups = group_labels(self.cfg,emb_path=self.cfg.model.ffi.cluster_file,group_size=self.cfg.model.ffi.group_size)
                groups = group_labels_faiss(self.cfg,label_list= None, emb_path=self.cfg.model.ffi.cluster_file,
                                    group_size=self.cfg.model.ffi.group_size,probe_k=512,seed=123)
                #groups = cluster_labels(self.cfg.data.dataset,self.path.bow_train_path,self.path.train_labels,64)
                all_labels = [lbl for lbl in self.label_map.keys()]
                label_type = type(next(iter(self.label_map.keys()))) 
                labels = [label_type(x) for x in itertools.chain(*groups)]
                remaining_labels = set(all_labels) - set(labels)
                remaining_labels = list(remaining_labels)
                labels += remaining_labels
                self.label_map = {}
                for i, k in enumerate(labels):
                    self.label_map[k] = i
            else:
                for i, k in enumerate(sorted(self.label_map.keys())):
                    self.label_map[k] = i
            print(f"label mapping finished for non ht_split")
            
        else:
            sorted_labels = sorted(self.label_map.items(), key=lambda x: x[1], reverse=True)
            if self.cfg.model.ffi.use_shared_group and self.cfg.model.ffi.cluster_based_grouping:
                self.num_head_labels = max(1, int(self.cfg.data.num_labels * self.cfg.model.ffi.head_ratio))
                #modify it to the ceiling of group size.
                if self.num_head_labels > self.group_size:
                    self.num_head_labels = int(math.ceil(self.num_head_labels / self.group_size)) * self.group_size
                all_labels = [lbl for lbl, _ in sorted_labels]
                head_labels = all_labels[:self.num_head_labels]
                tail_labels = all_labels[self.num_head_labels:]

                # exclude heads when clustering tail (cluster_labels expects strings)
               # exclude_for_cluster = [str(l) for l in head_labels]
                #groups = cluster_labels(self.cfg.data.dataset,self.path.bow_train_path,self.path.train_labels,64,exclude_labels=exclude_for_cluster)
                #print(f"min cluster size = {min(len(groups[0]),len(groups[1]),len(groups[2]))}")
                #groups = group_labels(self.cfg,label_list=tail_labels,emb_path=self.cfg.model.ffi.cluster_file,group_size=self.cfg.model.ffi.group_size)
                groups = group_labels_faiss(self.cfg,label_list=tail_labels if self.use_ht_split else None, emb_path=self.cfg.model.ffi.cluster_file,
                                    group_size=self.cfg.model.ffi.group_size,probe_k=512,seed=123)
                #tail_labels = list(itertools.chain(*groups))
                #tail_labels = [int(x) for x in itertools.chain(*groups)]

                label_type = type(next(iter(self.label_map.keys())))  # canonical type: int or str
                tail_labels = [label_type(x) for x in itertools.chain(*groups)]
                labels = head_labels + tail_labels
                # also needs to add the labels thats present in test but not in train
                #so it would be remaining = [all_labels - labels]
                remaining_labels = set(all_labels) - set(labels)
                remaining_labels = list(remaining_labels)
                labels += remaining_labels
                print(f"{len(labels)=}")
                # Reset label_map and assign new indices
                self.label_map = {}
                # for i, k in enumerate(labels):
                #     self.label_map[k] = i
                for k in labels:
                    if k not in self.label_map:
                        self.label_map[k] = len(self.label_map)
            else:
                # Reset label_map and assign new indices
            
                print(f"{len(sorted_labels)=}")
                self.label_map = {}
                for idx, (label, freq) in enumerate(sorted_labels):
                    self.label_map[label] = idx
        
        #print(f"{self.label_map=}")
        print(f"{len(self.label_map)=}")
        #print(f"{self.train_labels=}")
        
    def _read_text_files(self,filename):
        container = []
        f = open(filename,encoding="utf8")
        for line in f:
            container.append(line.strip())
    
        return container

    def _read_label_files(self,filename):
        container = []
        f = open(filename,encoding="utf8")
        for line in f:
            for l in line.strip().split():   
                self.label_map[l] = self.label_map.get(l,0)+1 #update the freq count
            container.append(line.strip().split())
            
        return container
    
    # def _read_lf_files(self,file,label_json=False):
    #     text_data = []
    #     labels = []
    #     key = 'title' if 'titles' in  self.cfg.data.dataset else 'content'
    #     if label_json:
    #         key='title'
    #     with open(file) as f:
    #         for i,line in enumerate(f):
    #             data = json.loads(line)
    #             text_data.append(data[key])
    #             if label_json:
    #                 labels.append([i])
    #                 self.label_map[i] = self.label_map.get(i, 0) + 1 
    #             else:
    #                 lbls = data['target_ind']
    #                 for l in lbls:
    #                     self.label_map[l]= self.label_map.get(l,0)+1 #update the freq count 
    #                 labels.append(lbls)

    #     return text_data,labels

    def _read_lf_files(self,file,label_json=False):
        text_data = []
        labels = []
        key = 'title' if 'titles' in self.cfg.data.dataset else 'content'
        if label_json:
            key='title'
        with open(file) as f:
            for i,line in enumerate(f):
                data = json.loads(line)
                if 'titles' in self.cfg.data.dataset or label_json:
                    text_data.append(data["title"])
                else:
                    text_data.append(data["title"] + " " + data["content"])
                if label_json:
                    labels.append([i]) # no need to count this frquency. add 1 to all labels.
                    self.label_map[i] = self.label_map.get(i, 0) + 1 
                else:
                    lbls = data['target_ind']
                    for l in lbls:
                        self.label_map[l]= self.label_map.get(l,0)+1 # count frquency
                    labels.append(lbls)

        return text_data,labels

        
    
    def getDatasets(self):
        
        group_y = None
        if self.cfg.model.auxiliary.use_meta_branch:
            group_y = self.group_y
            

        if self.cfg.data.tokenized_loading:
            train_dset = SimpleTokenizedDataset(self.cfg,self.train_tokenized_filename,self.train_nsample,
                                                self.train_labels,self.label_map,self.pad_token_id,mode='train',task='train')
            test_dset = SimpleTokenizedDataset(self.cfg,self.test_tokenized_filename,self.test_nsample,
                                               self.test_labels,self.label_map,self.pad_token_id,mode='test')
            train_dset_eval = SimpleTokenizedDataset(self.cfg,self.train_tokenized_filename,self.train_nsample,
                                                     self.train_labels,self.label_map,self.pad_token_id,mode='train')
        else:
            train_dset = SimpleDataset(self.cfg,self.train_raw_texts,self.train_labels,self.label_map,group_y,mode='train',task='train')
            test_dset = SimpleDataset(self.cfg,self.test_raw_texts,self.test_labels,self.label_map,None,mode='test')
            train_dset_eval = SimpleDataset(self.cfg,self.train_raw_texts,self.train_labels,self.label_map,None,mode='train')


        
        return train_dset,test_dset,train_dset_eval
    
    
        
    def getDataLoader(self,dset,mode='train'):
        '''
        #currently  separate dataloader for train (sparse labels) and evaluate (dense labels) in order to fasten both process.
        
        '''
        assert mode in ['train','test'], " mode must be either train or test."
        shuffle=False
        batch_size = self.cfg.data.test_batch_size
        p_memory = False
        if mode == 'train':
            shuffle=True
            p_memory = True
            batch_size = self.cfg.data.batch_size

        return DataLoader(dset, batch_size=batch_size, num_workers=self.cfg.data.num_workers, pin_memory=p_memory,
        persistent_workers=True, shuffle=shuffle, collate_fn=collate,prefetch_factor=2)


    
    

class SimpleDataset(Dataset):
    
    def __init__(self,cfg,raw_texts,labels,label_map,group_y,mode='train',task='evaluate'):
        super(SimpleDataset).__init__()
        
        self.cfg = cfg
        self.task = task
        self.group_y = group_y
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.encoder.encoder_tokenizer,do_lower_case=True,use_fast=True)
        self.cls_token_id = [101]  # [self.tokenizer.cls_token_id]
        self.sep_token_id = [102]  # [self.tokenizer.sep_token_id]
        
        self.raw_text = raw_texts
        self.labels = labels
        self.label_map = label_map
        self.mode = mode
        
        #Only use when meta_branch=True
        if group_y is not None: 
            # group y mode
            self.group_y, self.n_group_y_labels = [], group_y.shape[0]
            self.map_group_y = np.empty(len(label_map), dtype=np.longlong)
            for idx, labels in enumerate(group_y):
                self.group_y.append([])
                for label in labels:
                    if self.cfg.data.is_lf_data:
                        self.group_y[-1].append(label_map[int(label)]) #changed original: int(label)
                    else:
                        self.group_y[-1].append(label_map[label]) #changed original: int(label)
                self.map_group_y[self.group_y[-1]] = idx #check this line
                self.group_y[-1]  = np.array(self.group_y[-1])
            self.group_y = np.array(self.group_y,dtype=object)
        
            for i in range(len(self.map_group_y )):
                val = self.map_group_y[i]
                if val<0 or val>self.n_group_y_labels:
                    self.map_group_y[i] = random.choice(range(self.n_group_y_labels))
                    
                    
    def __len__(self):
        return len(self.raw_text)
    
    def __getitem__(self,idx):
        
        padding_length = 0
        #raw_text = clean_str(self.raw_text[idx])
        raw_text = self.raw_text[idx]
        tokens = self.tokenizer.encode(raw_text, add_special_tokens=False,truncation=True, max_length=self.cfg.data.max_len)
        tokens = tokens[:self.cfg.data.max_len-2]
        tokens = self.cls_token_id +tokens + self.sep_token_id
        
        if len(tokens)<self.cfg.data.max_len:
            padding_length = self.cfg.data.max_len - len(tokens)
        attention_mask = torch.tensor([1] * len(tokens) + [0] * padding_length)
        tokens = torch.tensor(tokens+([0]*padding_length))
        
        labels = [self.label_map[i] for i in self.labels[idx] if i in self.label_map]

        if self.group_y is not None:
            group_labels = self.map_group_y[labels] # list of group labels 
            group_label_ids = torch.zeros(self.n_group_y_labels)
            group_label_ids = group_label_ids.scatter(0, torch.tensor(group_labels),torch.tensor([1.0 for i in group_labels])) #group labels in one-hot format
        else:
            group_label_ids = torch.zeros(10)
        
        return tokens, attention_mask, labels, group_label_ids

class SimpleTokenizedDataset(Dataset):
    '''
    Currently Doesn't support meta classifiers.
    
    '''
    
    def __init__(self,cfg,tokenized_filename,nsample,labels,label_map,pad_token_id,mode='train',task='evaluate'):
        super(SimpleTokenizedDataset,self).__init__()
        
        self.cfg = cfg
        self.task = task
        self.mode = mode
        self.labels = labels
        self.label_map = label_map
        self.pad_token_id = pad_token_id 
        self.num_samples = nsample
        self.tokenized_filename = tokenized_filename
        
        self.data_shape = (self.num_samples, self.cfg.data.max_len)
        self._memmap_initialized = False
        self.group_label_ids = torch.zeros(10)

    def _initialize_memmap(self):
        """Lazy initialization of the tokenized memmap array.
        Also helps to avoid multiple copies over num_workers which is the case during loading in the constructor.
        
        """
        if not self._memmap_initialized:
            self.memmap_array = np.memmap(self.tokenized_filename, dtype='int32', mode='r', shape=self.data_shape)
            self._memmap_initialized = True
                               
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self,idx):
        if not self._memmap_initialized:
            self._initialize_memmap()    
        
        tokens = torch.tensor(self.memmap_array[idx])
        attention_mask = (tokens != self.pad_token_id).long()
        labels = [self.label_map[i] for i in self.labels[idx] if i in self.label_map]
        return tokens, attention_mask, labels, self.group_label_ids 
