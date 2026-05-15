import re
import torch
import numpy as np
from functools import wraps
import scipy.sparse as sp
import time
from tqdm import tqdm
import torch.optim as optim


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params, all_param = 0, 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad: trainable_params += param.numel()
    print(f" trainable params: {trainable_params} || all params: {all_param} || trainable %: {100 * trainable_params / all_param}")
    
    return f" trainable params: {trainable_params} || all params: {all_param} || trainable %: {100 * trainable_params / all_param}"


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

def clean_str(string):
    string = string.replace('\n', ' ')
    string = string.replace('_', ' ')
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub("/SEP/", "[SEP]", string)
    return string

def dense_to_sparse(cfg,dense_tensor):
    indices = (dense_tensor==1).nonzero().numpy()
    val = np.ones(len(indices))
    sparse_tensor = sp.csr_matrix((val, (indices[:,0],indices[:,1])), shape=(len(dense_tensor),cfg.data.num_labels))
    return sparse_tensor

@timeit
def Calculate_P_k(cfg,data_loader,model):
    model.eval()
    with torch.no_grad():
        #with torch.autocast(device_type='cuda', dtype=cfg.low_precision_dtype, enabled=cfg.use_low_precision_inference):
        P1, P3, P5 = [], [], []
        for data in tqdm(data_loader):
            tmp1, tmp3, tmp5 = 0, 0, 0
            tokens,mask,labels,_ = data
            tokens, mask, labels = tokens.to(cfg.device),mask.to(cfg.device),labels.to(cfg.device)
            logits,_ = model(tokens,mask)
            _,a5 = logits.topk(5)
            tmp = labels.gather(1,a5)
            tmp1 = tmp[:,:1].sum(dim=1)
            tmp3 = tmp[:,:3].sum(dim=1)*1/3
            tmp5 = tmp[:,:5].sum(dim=1)*1/5
            P1.append(tmp1.mean().item())
            P3.append(tmp3.mean().item())
            P5.append(tmp5.mean().item())

    return np.array(P1).mean(), np.array(P3).mean(), np.array(P5).mean()


    
    
            
class EarlyStopping:
    def __init__(self, patience=4, delta=0, mode='min'):

        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        
        if self.mode=='min':
            if score > self.best_score + self.delta:
                self.counter +=1
                if self.counter >= self.patience:  
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        if self.mode=='max':
            if score < self.best_score + self.delta:
                self.counter +=1
                if self.counter >= self.patience:  
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0

class CosineDecay(object):
    def __init__(self, death_rate, T_max, eta_min=0.001, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self):
        return self.sgd.param_groups[0]['lr']

class LinearDecay(object):
    def __init__(self, death_rate, factor=0.99, frequency=600):
        self.factor = factor
        self.steps = 0
        self.frequency = frequency

    def step(self):
        self.steps += 1

    def get_dr(self, death_rate):
        if self.steps > 0 and self.steps % self.frequency == 0:
            return death_rate*self.factor
        else:
            return death_rate