import os
from dataclasses import dataclass, field
from omegaconf import DictConfig


@dataclass
class EnvironmentConfig:
    '''
    Configuration related to the running environment of the system.
    '''
    running_env: str = "guest"
    cuda_device_id: int = 0
    device: str = "cuda"

@dataclass
class DataConfig:
    '''
    Configuration for dataset specifics and data handling procedures.
    '''
    dataset: str = "lfamazon131k"
    is_lf_data: bool =  True
    augment_label_data: bool =  True
    use_filter_eval: bool = False
    num_labels: int = 131073
    max_len: int = 128
    num_workers: int = 2
    batch_size: int = 256
    test_batch_size: int = 256
    input_features: int = 768
    
@dataclass
class EncoderConfig:
    '''
    Configuration for the encoder model specifics.
    '''
    encoder_model: str =  "sentence-transformers/msmarco-distilbert-base-v4" #['sentence-transformers/all-roberta-large-v1','bert-base-uncased']
    encoder_tokenizer: str =  "sentence-transformers/msmarco-distilbert-base-v4"
    encoder_ftr_dim: int =  768
    pool_mode: str =  "last_hidden_avg" #[last_nhidden_conlast,last_hidden_avg]
    feature_layers: int =  1
    embed_dropout: float = 0.7
    use_torch_compile: bool = False
    use_ngame_encoder_weights: bool = False
    ngame_checkpoint: str = "./NGAME_ENCODERS/lfamazon131k/state_dict.pt"
   
@dataclass 
class PenultimateConfig:
    '''
    Configuration settings for the penultimate layer, specifying
    whether to use the penultimate layer, its size, and activation function.
    '''
    use_penultimate_layer: bool =  True
    penultimate_size: int = 4096
    penultimate_activation: str = "relu"
  
@dataclass  
class FFIConfig:
    '''
    Configuration for the sparse layer, managing
    aspects of sparsity, pruning, and growth of network connections.
    
    Args:
        use_sparse_layer: Flag to activate sparse layer. If False dense layer would be used automatically.
    
    '''
    use_ht_split: bool = False
    head_ratio: float = 0.05
    use_sparse_layer: bool = True
    use_shared_group: bool = False
    group_size: int = 16
    fan_in: int = 128
    prune_mode: str = "threshold"
    rewire_threshold: float = 0.01
    rewire_fraction: float = 0.25
    growth_mode: str = "random"
    growth_init_mode: str = "zero"
    input_features: int = 768
    output_features: int = 131073 #depends on num_labels in data
    rewire_interval: int = 300
    use_rewire_scheduling: bool = True
    rewire_end_epoch: int = 66   #depends on epoch

@dataclass
class AuxiliaryConfig:
    '''
    Configuration for the auxiliary branches of the model that may influence or enhance the learning
    process with additional objective.
    
    Args:
        use_meta_branch (bool): Flag to activate the auxiliary branch.
        group_y_group (int): Index of auxiliary file varient.
        meta_cutoff_epoch (int): After this many epochs auxiliary branch would be disabled.
        auxloss_scaling (float): Contribution of auxiliary loss to the total loss.
    '''
    use_meta_branch: bool =  False
    group_y_group: int = 0
    meta_cutoff_epoch: int = 5   # varies based on fan_in values
    auxloss_scaling: float = 0.5
    
@dataclass
class PrivateFeatureConfig:
    '''
    For private feature experiments
    '''
    use_private_feature: bool = True
    head_start: int = 256
    head_end: int = 512
    tail_start: int = 0
    tail_end: int = 256
    head_dropout: float = 0.6
    tail_dropout:  float = 0.2

@dataclass
class ModelConfig:
    '''
    Configuration grouping various model component settings.
    '''
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    penultimate: PenultimateConfig = field(default_factory=PenultimateConfig)
    ffi: FFIConfig = field(default_factory=FFIConfig)
    auxiliary: AuxiliaryConfig = field(default_factory=AuxiliaryConfig)
    private_feature: PrivateFeatureConfig = field(default_factory=PrivateFeatureConfig)
    
@dataclass
class AmpConfig:
    '''
    Configuration for Automatic Mixed Precision.
    '''
    enabled: bool = True
    dtype: str = "float16"

@dataclass
class OptimizationConfig:
    loss_fn: str = "sparse_hinge"   # ['bce','squared_hinge']
    encoder_optimizer: str = "adamw"
    xmc_optimizer: str = "sgd"
    epochs: int = 201  
    grad_accum_step: int = 4
    encoder_lr: float = 1.0e-5
    bottleneck_lr: float = 2.0e-4
    meta_lr: float = 5.0e-4
    lr: float = 0.05  # learning rate of final layer
    wd_encoder: float = 0.01   # weight decay on encoder
    wd: float = 1e-4  # weight decay of final layer
    lr_scheduler: str = "CosineScheduleWithWarmup"   #[MultiStepLR,CosineScheduleWithWarmup,ReduceLROnPlateau]
    lr_scheduler_xmc: str = "CosineScheduleWithWarmup"
    warmup_steps: int =  5000
    training_steps: int = 0  #selected at runtime based on batch size and dataloader

    
@dataclass
class EvaluationConfig:
    train_evaluate: bool = True
    train_evaluate_every: int = 10
    test_evaluate_every: int = 1
    A: float = 0.6  # for propensity calculation, epends on dataset
    B: float = 2.6  # for propensity calculation, epends on dataset
    eval_psp: bool = True

    
@dataclass
class VerboseConfig:
    show_iter: bool = False  # print loss during training
    print_iter: int = 2000  # how often (iteration) to print
    use_wandb: bool = False
    wandb_runname: str = "none"
    logging: bool = True
    log_fname: str = "log_amazontitles131k"
    use_checkpoint: bool = False  #whether to use automatic checkpoint
    best_p1: float = 0.462  # to store the model above this performance in case of automatic checkpoint
    job_num: int = 0
    
@dataclass
class CheckPointConfig:
    use_checkpoint: bool = False  #whether to use automatic checkpoint
    checkpoint_file: str = "PBCE_NoLF_NM1"
    best_p1: float = 0.2  # to store the model above this performance in case of automatic checkpoint
    
@dataclass
class TrainingConfig:
    amp: AmpConfig = field(default_factory=AmpConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    verbose: VerboseConfig = field(default_factory=VerboseConfig)
    checkpoint: CheckPointConfig = field(default_factory=CheckPointConfig)


@dataclass
class SimpleConfig:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    input_features: int = 768
    

def validate_config(cfg: DictConfig):
    '''
    Validate the provided configuration to ensure it meets all requirements.
    '''
    # Example validation: check if a specific value meets a condition
    assert cfg.environment.device in ['cpu','cuda','cuda:0'], " Unknown device Selected"
    if 'lf' not in cfg.data.dataset and cfg.data.augment_label_data:
        raise ValueError("Can't Augment Label data for non label feature dataset. make augment_label_data=False or change the dataset")
    if 'lf' not in cfg.data.dataset and cfg.data.use_filter_eval:
        raise ValueError(" Can't use Filter evaluation for Non LF datasets. No reciprocal pairs.")
    assert cfg.training.amp.dtype in ['float16','float32','bfloat16'],"Wrong Datatype is selected."
    if cfg.model.encoder.pool_mode=='last_hidden_avg' and not cfg.model.encoder.feature_layers==1:
            raise ValueError('The selected Pooling mode should have feature_layers=1')
        
    print('All Validation passed..')
    
        
##-------------------------------------------------------------------------------------------------------------------##
##---------------------------------------Dataset Paths Configuration-------------------------------------------------##
##-------------------------------------------------------------------------------------------------------------------##
    
class PathWiki31K:
    '''
    Wiki10-31K Dataset.
    '''
    def __init__(self,root_path):
        
        self.root_folder = 'Wiki10-31K'
        self.dataset_path = os.path.join(root_path,self.root_folder)

        #Raw Data and Labels
        self.train_raw_texts = os.path.join(self.dataset_path,'train_raw_texts.txt')
        self.test_raw_texts = os.path.join(self.dataset_path,'test_raw_texts.txt')
        self.train_labels = os.path.join(self.dataset_path,'train_labels.txt')
        self.test_labels = os.path.join(self.dataset_path,'test_labels.txt')
        #BOW features
        self.bow_path = os.path.join(self.dataset_path,'BOW')
        self.bow_train_path = os.path.join(self.bow_path,'train.txt')
        self.bow_test_path = os.path.join(self.bow_path,'test.txt')
        self.bowXf_path = os.path.join(self.bow_path,'Xf.txt')
        self.bowY_path = os.path.join(self.bow_path,'Y.txt')


class PathEurlex4K:
    '''
    EurLex-4K Dataset.
    
    '''
    def __init__(self,root_path):

        self.root_folder = 'Eurlex-4K'
        self.dataset_path = os.path.join(root_path,self.root_folder)

        #Raw Data and Labels
        self.train_raw_texts = os.path.join(self.dataset_path,'train_texts.txt')
        self.test_raw_texts = os.path.join(self.dataset_path,'test_texts.txt')
        self.train_labels = os.path.join(self.dataset_path,'train_labels.txt')
        self.test_labels = os.path.join(self.dataset_path,'test_labels.txt')
        #BOW features
        self.bow_train_path = os.path.join(self.dataset_path,'train.txt')
        self.bow_test_path = os.path.join(self.dataset_path,'test.txt')
      


class PathAmazon670K:
    '''
    Amazon-670K Dataset
    '''
    def __init__(self,root_path):

        self.root_folder = 'Amazon-670K'
        self.dataset_path = os.path.join(root_path,self.root_folder)

        #Raw Data and Labels
        self.train_raw_texts = os.path.join(self.dataset_path,'train_raw_texts.txt')
        self.test_raw_texts = os.path.join(self.dataset_path,'test_raw_texts.txt')
        
        self.train_labels = os.path.join(self.dataset_path,'train_labels.txt')
        self.test_labels = os.path.join(self.dataset_path,'test_labels.txt')
        
        #BOW features
        self.bow_train_path = os.path.join(self.dataset_path,'train_v1.txt')
        self.bow_test_path = os.path.join(self.dataset_path,'BOW/test.txt')

class PathWiki500K:
    '''
    Wiki-500K Dataset.
    '''
    def __init__(self,root_path):

        self.root_folder = 'Wiki-500K'
        self.dataset_path = os.path.join(root_path,self.root_folder)

        #Raw Data and Labels
        self.train_raw_texts = os.path.join(self.dataset_path,'train_raw_texts.txt')
        self.test_raw_texts = os.path.join(self.dataset_path,'test_raw_texts.txt')
        self.train_labels = os.path.join(self.dataset_path,'train_labels.txt')
        self.test_labels = os.path.join(self.dataset_path,'test_labels.txt')
        #BOW features
        self.bow_train_path = os.path.join(self.dataset_path,'train.txt')
        self.bow_test_path = os.path.join(self.dataset_path,'BOW/test.txt')

class PathAmazon3M:
    '''
    Amazon-3M Dataset.
    
    '''
    def __init__(self,root_path):

        self.root_folder = 'Amazon-3M'
        self.dataset_path = os.path.join(root_path,self.root_folder)

        #Raw Data and Labels
        self.train_raw_texts = os.path.join(self.dataset_path,'train_raw_texts.txt')
        self.test_raw_texts = os.path.join(self.dataset_path,'test_raw_texts.txt')
        
        self.train_labels = os.path.join(self.dataset_path,'train_labels.txt')
        self.test_labels = os.path.join(self.dataset_path,'test_labels.txt')
        
        #BOW features
        self.bow_train_path = os.path.join(self.dataset_path,'train_v1.txt')
        self.bow_test_path = os.path.join(self.dataset_path,'BOW/test.txt')
        
##----------------------------------------------- Label Features Datasets-----------------------###
##-------------------------------------------------------------------------------------------------##

class PathLFAmazonTitles131K:
    '''
    LF-AmazonTitles-131K Dataset.
    '''
    def __init__(self,root_path):

        self.root_folder = 'LF-AmazonTitles-131K'
        self.dataset_path = os.path.join(root_path,self.root_folder)

        #Raw text
        self.raw_text_path = os.path.join(self.dataset_path,'raw_text')
        self.train_json = os.path.join(self.raw_text_path,'trn.json')
        self.test_json = os.path.join(self.raw_text_path,'tst.json')
        self.label_json = os.path.join(self.raw_text_path,'lbl.json')
        self.filter_labels_train = os.path.join(self.raw_text_path,'filter_labels_train.txt')
        self.filter_labels_test = os.path.join(self.raw_text_path,'filter_labels_test.txt')
        #BOW features
        self.bow_train_path = os.path.join(self.dataset_path,'train.txt')
        self.bow_test_path = os.path.join(self.dataset_path,'test.txt')
        
             

class PathLFWikiSeeAlso320K:
    '''
    LF-WikiSeeAlso-320K Dataset.
    '''
    def __init__(self,root_path):

        self.root_folder = 'LF-WikiSeeAlso-320K'
        self.dataset_path = os.path.join(root_path,self.root_folder)
        #Raw text
        self.raw_text_path = os.path.join(self.dataset_path,'raw_text')
        self.train_json = os.path.join(self.raw_text_path,'trn.json')
        self.test_json = os.path.join(self.raw_text_path,'tst.json')
        self.label_json = os.path.join(self.raw_text_path,'lbl.json')
        self.filter_labels_train = os.path.join(self.raw_text_path,'filter_labels_train.txt')
        self.filter_labels_test = os.path.join(self.raw_text_path,'filter_labels_test.txt')
        #BOW features
        self.bow_train_path = os.path.join(self.dataset_path,'train.txt')
        self.bow_test_path = os.path.join(self.dataset_path,'test.txt')


class PathAmazonTitles670K:
    '''
    AmazonTitles-670K Dataset.
    '''
    def __init__(self,root_path):

        self.root_folder = 'AmazonTitles-670K'
        self.dataset_path = os.path.join(root_path,self.root_folder)

        #Raw text
        self.train_json = os.path.join(self.dataset_path,'trn.json')
        self.test_json = os.path.join(self.dataset_path,'tst.json')
        #self.filter_labels_train = os.path.join(self.raw_text_path,'filter_labels_train.txt')
        #self.filter_labels_test = os.path.join(self.raw_text_path,'filter_labels_test.txt')
        #BOW features
        self.bow_train_path = os.path.join(self.dataset_path,'trn_X_Xf.txt')
        self.bow_test_path = os.path.join(self.dataset_path,'tst_X_Xf.txt')



class PathLFAmazonTitles1P3M:
    def __init__(self,root_path):

        self.root_folder = 'LF-AmazonTitles-1.3M'
        self.dataset_path = os.path.join(root_path,self.root_folder)

        #Raw text
        self.raw_text_path = os.path.join(self.dataset_path,'raw_text')
        self.train_json = os.path.join(self.raw_text_path,'trn.json')
        self.test_json = os.path.join(self.raw_text_path,'tst.json')
        self.label_json = os.path.join(self.raw_text_path,'lbl.json')
        self.filter_labels_train = os.path.join(self.raw_text_path,'filter_labels_train.txt')
        self.filter_labels_test = os.path.join(self.raw_text_path,'filter_labels_test.txt')
        #BOW features
        self.bow_train_path = os.path.join(self.dataset_path,'train.txt')
        self.bow_test_path = os.path.join(self.dataset_path,'test.txt')


class PathLFPaper2keywords:
    '''
    LF-Paper2keywords Dataset.
    '''
    def __init__(self, root_path):

        self.root_folder = 'LF-Paper2Keywords-8.6M'
        self.dataset_path = os.path.join(root_path, self.root_folder)
        #Raw text
        self.train_json = os.path.join(self.dataset_path, 'trn.json')
        self.test_json = os.path.join(self.dataset_path, 'tst.json')
        self.label_json = os.path.join(self.dataset_path, 'lbl.json')
