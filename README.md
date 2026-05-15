# HASTE: Hardware Aware Dynamic Sparse Training for Large Output Spaces

Official PyTorch implementation for the paper: "Hardware Aware Dynamic Sparse Training for Large Output Spaces".



## Installation

```bash
conda create -y -n haste python=3.10
conda activate haste
bash setup.sh
pip install torch_sparse-0.0.6.dev295+cu124.pt24-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install .
```

## Label Embeddings for Label Grouping
Label embeddings can be extracted using src/embed_extraction.py script
or download pre-extracted embedings to /embeddings/ using below link 
- [label embeddings](https://drive.google.com/drive/folders/1gm1S19Ko8ouo6a8n8aqiueBnk1qX2OnV?usp=sharing)

## Datasets
Download datasets from the [extreme classification repo.](http://manikvarma.org/downloads/XC/XMLRepository.html)  
or follow the links below
-   [Wiki10-31K](https://www.dropbox.com/scl/fo/pueq58m7lz7r6yt9jtl90/h?rlkey=b7w2cr3eq4y1rc33tgu96huc2&dl=0)
-   [Amazon-670K](https://www.dropbox.com/scl/fo/effa1w2c2l68ql0mcvgmx/h?rlkey=65nsv9otk0w6olyzbrj6b5msd&dl=0)
-   [Wiki-500K](https://www.dropbox.com/scl/fo/3v5wgn396hyzukmhc31fn/h?rlkey=e4ga4g3l8pc7xv6dkc3j0d1pv&dl=0)
-   [LF-AmazonTitles-131K](https://www.dropbox.com/scl/fo/qbt00gbyt35p2h1yz05on/h?rlkey=3bf8dbq3bgns9dvfau4d9d7sx&dl=0)


## Running
1. Setup environment based on the installation instructions above. 
2. Settings and Hyperparameters are managed by [hydra](https://hydra.cc/). See the complete configuration layout for `LF-AmazonTitles-131K` below. For more details check config folder.
3. Add key:value (running-env:root path of datafolders) entry in  env2path dictionary in main.py and add the running-env to environment in yaml file (Optional to avoid typing path during every run).
4. Run `python src/main.py dataset=dataset log_fname=log_dataset ` (step 3 followed).
            <br> <center>OR</center> <br>
5. Run `python src/main.py dataset=dataset  dataset_path=./data log_fname=log_dataset` (step 3 not followed).
>[where `dataset_path` is root path and dataset argument names are  `wiki31k`,  `amazon670k`, `wiki500k`, `amazon3m`, `lfamazontitles131k`, `lfwikiseealso320k`]

Pre-trained Initialization for LF-AmazonTitles-131K can be found [here](https://drive.google.com/drive/folders/16NchlBKPf_nnAP3cKW2NLlVu1rY1z7Vf?usp=sharing)


## Sample Run Scripts
**Wiki31K**
```bash
python src/main.py dataset=lfamazontitles131k log_fname="log_AT131K"

```
**Amazon670K**
```bash
python src/main.py dataset=amazon670k log_fname="log_A670K"

```

## References
[1] [The Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html)

[2] [Towards Memory-Efficient Training for Extremely Large Output Spaces – Learning with 500k Labels on a Single Commodity GPU](https://github.com/xmc-aalto/ecml23-sparse)

[3] [Dynamic Sparse Training with Structured Sparsity](https://openreview.net/pdf?id=kOBkxFRKTA)

[3] [pyxclib](https://github.com/kunaldahiya/pyxclib)

[4] [LightXML: Transformer with dynamic negative sampling for High-Performance Extreme Multi-label Text Classiﬁcation](https://github.com/kongds/LightXML)

[5] [Navigating Extremes: Dynamic Sparsity in Large Output Spaces](https://github.com/xmc-aalto/NeurIPS24-dst)

## You May Also Like

- [Towards Memory-Efficient Training for Extremely Large Output Spaces – Learning with 500k Labels on a Single Commodity GPU](https://github.com/xmc-aalto/ecml23-sparse)

## Config Layout Example for LF-AmazonTitles-131K
```yaml
environment:
  running_env: guest
  cuda_device_id: 0
  device: cuda
data:
  dataset: lfamazontitles131k
  is_lf_data: True
  augment_label_data: True
  use_filter_eval: True
  num_labels: 131073
  max_len: 32
  num_workers: 8
  batch_size: 512
  test_batch_size: 512

model:
  encoder:
    encoder_model: "sentence-transformers/msmarco-distilbert-base-v4" 
    encoder_tokenizer: ${dataset.model.encoder.encoder_model}
    encoder_ftr_dim: 768
    pool_mode: 'last_hidden_avg'
    feature_layers: 1
    embed_dropout: 0.85 
    use_torch_compile: False
    use_ngame_encoder_weights: False
    ngame_checkpoint: ./NGAME_ENCODERS/${dataset.data.dataset}/state_dict.pt

  penultimate:
    use_penultimate_layer: False
    penultimate_size: 4096
    penultimate_activation: relu

  ffi:
    use_sparse_layer: False
    fan_in: 128
    prune_mode: fraction
    rewire_threshold: 0.01
    rewire_fraction: 0.15
    growth_mode: random
    growth_init_mode: zero
    input_features: 768
    output_features: 131073  
    rewire_interval: 1000
    use_rewire_scheduling: True
    rewire_end_epoch: 0.66   #depends on epoch

  auxiliary:
    use_meta_branch: False
    group_y_group: 0
    meta_cutoff_epoch: 20   # varies based on fan_in values
    auxloss_scaling: 0.4
training:
  seed: 42
  amp:
    enabled: False
    dtype: bfloat16

  optimization:
    loss_fn: bce   # ['bce','squared_hinge']
    encoder_optimizer: adamw
    xmc_optimizer: sgd_sr
    epochs: 100   # depends on dataset
    dense_epochs: 100
    grad_accum_step: 1
    encoder_lr: 1.0e-5
    penultimate_lr: 2.0e-4
    meta_lr: 5.0e-4
    lr: 0.05  # learning rate of final layer
    wd_encoder: 0.01   # weight decay on encoder
    wd: 1e-4  # weight decay of final layer
    lr_scheduler: CosineScheduleWithWarmup
    lr_scheduler_xmc: CosineScheduleWithWarmup
    warmup_steps: 5000
    training_steps: 1  #selected at runtime based on batch size and dataloader

  evaluation:
    train_evaluate: True
    train_evaluate_every: 10
    test_evaluate_every: 1
    A: 0.6  # for propensity calculation
    B: 2.6  # for propensity calculation
    eval_psp: True

  verbose:
    show_iter: False  # print loss during training
    print_iter: 2000  # how often (iteration) to print
    use_wandb: False
    wandb_runname: none
    logging: True
    log_fname: log_amazontitles131k
    
  use_checkpoint: False  #whether to use automatic checkpoint
  checkpoint_file: PBCE_NoLF_NM1
  best_p1: 0.2  # In case of automatic checkpoint
```


