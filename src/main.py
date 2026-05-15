import random
import numpy as np
import os
import torch
import hydra
from omegaconf import OmegaConf
from omegaconf import open_dict
from hydra.core.config_store import ConfigStore

from config import ( 
        PathWiki31K, PathEurlex4K, PathAmazon670K, PathWiki500K, PathAmazon3M, PathAmazonTitles670K,
        PathLFAmazonTitles131K, PathLFWikiSeeAlso320K, PathLFAmazonTitles1P3M, SimpleConfig, validate_config, FFIConfig, PathLFPaper2keywords
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
                'lfwikiseealso320k':'LF-WikiSeeAlso-320k','amazontitles670k':'AmazonTitles-670K','lfamazontitles1.3m':'LF-AmazonTitles-1.3M',
                'lfpaper2keywords':'LF-Paper2Keywords-8.6M'}

# Path configuration based on the environment
#ENVIRONMENT_TO_PATH = {'guest':'/l/WorkSpace/Datasets/XMC'}  #
ENVIRONMENT_TO_PATH = {'guest':'/scratch/work/nasibun1/projects/Datasets'} 

# Dataset path mapping
DATASET_TO_PATH_OBJECT = {'eurlex4k': PathEurlex4K, 'wiki31k': PathWiki31K, 'amazon670k': PathAmazon670K,'wiki500k' : PathWiki500K,  
            'amazon3m': PathAmazon3M, 'lfamazontitles131k':PathLFAmazonTitles131K,
            'lfwikiseealso320k':PathLFWikiSeeAlso320K,'amazontitles670k':PathAmazonTitles670K, 'lfamazontitles1.3m':PathLFAmazonTitles1P3M,
            'lfpaper2keywords': PathLFPaper2keywords}

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

    cfg.training.optimization.training_steps = len(train_loader)
    validate_config(cfg)
    
    runner = Runner(cfg,path,data_handler)  

    print(f"wandb run name: {cfg.training.verbose.wandb_runname }")
    
    print(f"Training stage: {cfg.training.stage}")
    runner.run_train(train_loader,test_loader,train_loader_eval)

if __name__ == '__main__':
    main()
