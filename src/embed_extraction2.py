import random
import numpy as np
import os
import torch
import hydra
from omegaconf import OmegaConf
from omegaconf import open_dict
from hydra.core.config_store import ConfigStore
import itertools
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from config import ( 
        PathWiki31K, PathEurlex4K, PathAmazon670K, PathWiki500K, PathAmazon3M, PathAmazonTitles670K,
        PathLFAmazonTitles131K, PathLFWikiSeeAlso320K,SimpleConfig, validate_config, FFIConfig, PathLFAmazonTitles1P3M
)
from data import DataHandler
from runner import Runner

# Register resolvers for configuration variables based on conditional logic
OmegaConf.register_new_resolver(
    'encoder_feature_size',
    lambda enc_name: 1024 if 'large' in enc_name else 768
)

OmegaConf.register_new_resolver(
    'input_size_select',
    lambda use_penultimate, penultimate_size, feature_layers, feature_dim: (
        penultimate_size if use_penultimate else feature_layers * feature_dim
    )
)



# Initialize configuration store
cs = ConfigStore.instance()
cs.store(name="simple_config", node=SimpleConfig)

# Map dataset names to user-friendly names
DATASET_NAME_MAP = {'eurlex4k': 'Eurlex-4K','wiki31k': 'Wiki10-31K', 'amazon670k': 'Amazon-670K', 'wiki500k': 'Wiki-500K',
                'amazon3m':'Amazon-3M','lfamazontitles131k':'LF-AmazonTitles-131K',
                'lfwikiseealso320k':'LF-WikiSeeAlso-320k','amazontitles670k':'AmazonTitles-670K','lfamazontitles1.3m':'LF-AmazonTitles-1.3M'}

# Path configuration based on the environment
#ENVIRONMENT_TO_PATH = {'guest':'/l/WorkSpace/Datasets/XMC'}  #
ENVIRONMENT_TO_PATH = {'guest':'/scratch/work/nasibun1/projects/Datasets'} 

# Dataset path mapping
DATASET_TO_PATH_OBJECT = {'eurlex4k': PathEurlex4K, 'wiki31k': PathWiki31K, 'amazon670k': PathAmazon670K,'wiki500k' : PathWiki500K,  
            'amazon3m': PathAmazon3M, 'lfamazontitles131k':PathLFAmazonTitles131K,
            'lfwikiseealso320k':PathLFWikiSeeAlso320K,'amazontitles670k':PathAmazonTitles670K, 'lfamazontitles1.3m':PathLFAmazonTitles1P3M}

@hydra.main(version_base="1.2", config_path="../config/",config_name="config") 
def main(cfg: SimpleConfig) -> None:
    '''
    Main function to initialize training process based on configuration.
    '''
    # Unfreeze cfg.dataset to allow modifications
    #OmegaConf.set_struct(cfg.dataset, False)
    
    # Set random seeds for reproducibility
    seed = cfg.dataset.training.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    #print(OmegaConf.to_yaml(cfg))

    
 
    
     # Determine dataset path
    dataset_path = cfg.dataset_path if os.path.exists(cfg.dataset_path) else ENVIRONMENT_TO_PATH[cfg.environment.running_env]
    path = DATASET_TO_PATH_OBJECT[cfg.dataset.data.dataset](dataset_path)

    
    #modify configuration
    if cfg.use_wandb:
        cfg.dataset.training.verbose.use_wandb = True
    if cfg.wandb_runname != "":
        cfg.dataset.training.verbose.wandb_runname = cfg.wandb_runname
    if cfg.log_fname != "":
        cfg.dataset.training.verbose.log_fname = cfg.log_fname
    if cfg.job_num is not None:
        cfg.dataset.training.verbose.job_num = cfg.job_num
 
        
        
        
        

    # cfg.dataset.environment = cfg.environment
    
    # # Re-freeze cfg.dataset
    # OmegaConf.set_struct(cfg.dataset, True)
    
    
    #modify the config file
    # Modify the config file
    cfg2 = cfg.dataset
    with open_dict(cfg2):
        cfg2['environment'] = cfg['environment']
        #if cfg.job_num is not None:
        #cfg2['jobnum'] = cfg.job_num
    cfg = cfg2

    data_handler = DataHandler(cfg,path)
    train_dset, test_dset, train_dset_eval = data_handler.getDatasets()
    train_loader_eval = data_handler.getDataLoader(train_dset_eval,mode='test')
    test_loader = data_handler.getDataLoader(test_dset,mode='test')
    train_loader = data_handler.getDataLoader(train_dset,mode='train')


    data = train_dset.raw_text
    labels = train_dset.labels
    test_labels = test_dset.labels

    train_all_labels  = list(itertools.chain.from_iterable(labels))
    test_all_labels = list(itertools.chain.from_iterable(test_labels))

    all_labels = set(train_all_labels + test_all_labels)
    only_test_labels = all_labels - set(train_all_labels)

    label_to_data = {}
    for i,d in tqdm(enumerate(data)):
        labels_list = labels[i]
        for lbl in labels_list:
            if lbl in label_to_data:
                label_to_data[lbl] +=[d]
            else:
                label_to_data[lbl] = [d]

    #if only_test_labels > 0 -> get some texts for those labels from test_raw_texts and test_labels
    if len(only_test_labels) > 0:
        print(f"There are {len(only_test_labels)} labels that are present in test set but missing in train set.")

        test_data = test_dset.raw_text

        for i,d in tqdm(enumerate(test_data)):
            labels_list = test_labels[i]
            for lbl in labels_list:
                if lbl in only_test_labels:
                    if lbl in label_to_data:
                        label_to_data[lbl] +=[d]
                    else:
                        label_to_data[lbl] = [d]

    for k,v in label_to_data.items():
        label_to_data[k] = ''.join(x for x in v)


    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda").half()

    label_keys, label_texts = [], []
    for k,v in label_to_data.items():
        label_keys.append(k)
        label_texts.append(v)
    
    MAX_CHARS = 8000   # tune
    label_texts = [t[:MAX_CHARS] for t in label_texts]


    print(f"starting to extract embeddings")
    with torch.no_grad():
        query_embs = model.encode(
            label_texts,
            prompt_name="document",
            batch_size=32,                 
            normalize_embeddings=True,     
            convert_to_numpy=True,
            show_progress_bar=True,).astype(np.float32)


    label_to_emb = {k: emb for k, emb in zip(label_keys, query_embs)}
    print("num labels:", len(label_to_emb), "emb dim:", next(iter(label_to_emb.values())).shape)

    out_dir = os.getcwd()  # hydra run dir (per-run)
    save_dir = os.path.join(out_dir, "embeddings")
    os.makedirs(save_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(save_dir, f"{cfg.data.dataset}_label_embs_06B{MAX_CHARS}.npz"),
        embs=query_embs,                     # (N, D)
        labels=np.array(label_keys, dtype=object)  # (N,)
    )

    
if __name__ == '__main__':
    main()


# z = np.load("label_embs.npz", allow_pickle=True)
# labels = z["labels"].tolist()
# embs = z["embs"]
# label_to_emb = {k: embs[i] for i, k in enumerate(labels)}