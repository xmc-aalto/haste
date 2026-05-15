import torch
from torch import nn
import math
from torch_sparse.rewire import FixedFanIn
from transformers import AutoModel, AutoConfig
from torch.utils.data import Dataset, DataLoader
from shared_ffi.rewire import GroupFixedFanIn
from group_fanin_real import GroupFixedFanInReal

dtype_map = {'float16':torch.float16, 'bfloat16':torch.bfloat16,'float32':torch.float32}

class TransformerEncoder(nn.Module):
    '''
     Custom Transformer Encoder with configurable model components. 
    '''
    def __init__(self, cfg):
        super(TransformerEncoder, self).__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.environment.device)
        self.transformer = self.load_transformer_model(cfg)
        self.pooler = self.create_pooler()
        
    def load_transformer_model(self, cfg):
        """ Load transformer model based on the provided configuration. """
        model_config = AutoConfig.from_pretrained(cfg.model.encoder.encoder_model)
        model_config.gradient_checkpointing = True # For Gradient checkpointing of Encoder
        model_config.output_hidden_states = True
        try:
            model = AutoModel.from_pretrained(
            cfg.model.encoder.encoder_model,
            add_pooling_layer=False,
            config=model_config).to(self.device)


            # THIS is what actually enables it in most HF models
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()  # optionally pass kwargs (see below)

            # Optional: sanity print
            if hasattr(model, "is_gradient_checkpointing"):
                print("HF checkpointing enabled:", model.is_gradient_checkpointing)



            return model
        except Exception as e:
            print(f"Failed to load model with pooling layer removed: {e}")
            return AutoModel.from_pretrained(
                cfg.model.encoder.encoder_model, 
                config=model_config
            ).to(self.device)
      
    def forward(self, tokens,masks):
        '''
        Forward pass through transformer and pooling layers. 
        '''
        return self.pooler(self.transformer(tokens,masks),masks).contiguous()
    
    def create_pooler(self):
        '''
         Create a pooling layer based on the configuration.
        '''
        def pool_last_hidden_avg(tf_output, masks):
            last_hidden_state = tf_output['last_hidden_state']
            input_mask_expanded = masks.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_hidden_state = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_hidden_state / sum_mask
        
        def pool_last_nhidden_conlast(tf_output,masks):
            bert_out = tf_output[-1]
            bert_data = [bert_out[-i][:, 0] for i in range(1, self.cfg.model.encoder.feature_layers+1)]
            return torch.cat(bert_data, dim=-1)
            
        if self.cfg.model.encoder.pool_mode == 'last_hidden_avg':
            return pool_last_hidden_avg
        elif self.cfg.model.encoder.pool_mode == 'last_nhidden_conlast':
            return pool_last_nhidden_conlast
        else:
            raise ValueError('Invalid pooling mode specified in the configuration.')
            


class SimpleTModel(nn.Module):
    '''
     A simple transformer model supporting various configurations and enhancements like LoRA and sparse layers.
    '''

    def __init__(self,cfg,path,group_y_labels):
        super(SimpleTModel,self).__init__()
     
        self.cfg = cfg
        self.path = path
        self.group_y_labels = group_y_labels
        self.device = torch.device(cfg.environment.device)
        if self.cfg.training.amp.enabled:
            self.dtype = torch.float32
        else:
            self.dtype = dtype_map[cfg.training.amp.dtype]
        self.encoder = TransformerEncoder(cfg).to(self.dtype)


        
        self.use_sparse_layer = cfg.model.ffi.use_sparse_layer
        self.use_shared_group = cfg.model.ffi.use_shared_group
        self.group_size = cfg.model.ffi.group_size
        
        num_labels = cfg.data.num_labels
        self.num_head_labels = max(1, int(num_labels * cfg.model.ffi.head_ratio))  # Top p% as head labels

            #modify it to the ceiling of group size.
        if self.num_head_labels > self.group_size:
            self.num_head_labels = int(math.ceil(self.num_head_labels / self.group_size)) * self.group_size

        self.num_tail_labels = num_labels - self.num_head_labels
        self.use_ht_split = cfg.model.ffi.use_ht_split
        
        if cfg.model.private_feature.use_private_feature:
            self.head_input_size = self.cfg.model.private_feature.head_end - self.cfg.model.private_feature.head_start
            self.tail_input_size = self.cfg.model.private_feature.tail_end - self.cfg.model.private_feature.tail_start
        else:
            self.head_input_size = cfg.model.ffi.input_features
            self.tail_input_size = cfg.model.ffi.input_features

        
        
        self.configure_components(cfg)
        
        self.auxloss_scaling = 0
      
        if cfg.model.encoder.use_torch_compile:
            self.encoder = torch.compile(self.encoder)
        
            
    def configure_components(self, cfg):
        """ Configure additional components like dropout, linear layers, etc. based on the model configuration. """
        
        if cfg.model.private_feature.use_private_feature:
            self.head_dropout = nn.Dropout(self.cfg.model.private_feature.head_dropout).to(self.device).to(self.dtype)
            self.tail_dropout = nn.Dropout(self.cfg.model.private_feature.tail_dropout).to(self.device).to(self.dtype)
        else:
            self.dropout = nn.Dropout(cfg.model.encoder.embed_dropout).to(self.device).to(self.dtype)

        if cfg.model.auxiliary.use_meta_branch:
            self.auxloss_scaling = cfg.model.auxiliary.auxloss_scaling
            # self.group_branch = nn.Linear(
            #     cfg.model.encoder.feature_layers * self.encoder.transformer.config.hidden_size,
            #     self.group_y_labels
            # ).to(self.device)
            
            self.group_branch = nn.Linear(
                cfg.model.encoder.feature_layers * 256,
                self.group_y_labels
            ).to(self.device).to(self.dtype)

        if cfg.model.penultimate.use_penultimate_layer:
            self.penultimate = nn.Linear(
                cfg.model.encoder.feature_layers * self.encoder.transformer.config.hidden_size,
                cfg.model.penultimate.penultimate_size
            ).to(self.device).to(self.dtype)

        if cfg.model.ffi.use_sparse_layer:
            self.configure_sparse_layer(cfg)
        else:
            if self.use_ht_split:
                self.head_layer = nn.Linear(self.cfg.model.ffi.input_features,self.num_head_labels).to(self.device).to(self.dtype)  # Dense layer for head labels
                self.tail_layer = nn.Linear(self.cfg.model.ffi.input_features,self.num_tail_labels).to(self.device).to(self.dtype) # Dense layer for tail labels
                nn.init.normal_(self.head_layer.weight, mean=0.0, std=0.02)  # Gaussian initialization
                nn.init.normal_(self.tail_layer.weight, mean=0.0, std=0.02)  # Gaussian initialization
            else:
                self.linear = nn.Linear(cfg.model.ffi.input_features, cfg.data.num_labels).to(self.device).to(self.dtype)
                nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)  # Gaussian initialization
                
            
        print(f"cfg.model.ffi.input_features={cfg.model.ffi.input_features}")
  
            
    def configure_sparse_layer(self, cfg):
        """ Configure sparse layer for the model. """
        
        if self.use_ht_split:
        
            self.head_layer = nn.Linear(self.head_input_size,self.num_head_labels).to(self.device).to(self.dtype)  # Dense layer for head labels
            nn.init.normal_(self.head_layer.weight, mean=0.0, std=0.02)  # Gaussian initialization
            
            if self.use_shared_group:
                self.tail_layer = GroupFixedFanInReal(self.tail_input_size, self.num_tail_labels, fan_in=cfg.model.ffi.fan_in,
                              prune_mode=cfg.model.ffi.prune_mode, init_mode=cfg.model.ffi.growth_init_mode,
                              rewire_threshold=cfg.model.ffi.rewire_threshold, rewire_fraction=cfg.model.ffi.rewire_fraction,
                              group_size=self.group_size,bias=True,score_mode="mean", dtype=self.dtype).to(self.device)
            else:
                self.tail_layer =  FixedFanIn(self.tail_input_size, self.num_tail_labels, fan_in=cfg.model.ffi.fan_in,
                              prune_mode=cfg.model.ffi.prune_mode, init_mode=cfg.model.ffi.growth_init_mode,
                              rewire_threshold=cfg.model.ffi.rewire_threshold, 
                              rewire_fraction=cfg.model.ffi.rewire_fraction).to(self.device).to(self.dtype) #Sparse tail layer
        
           
        else:
            if self.use_shared_group:
                self.linear =  GroupFixedFanInReal(self.tail_input_size, cfg.model.ffi.output_features, fan_in=cfg.model.ffi.fan_in,
                              prune_mode=cfg.model.ffi.prune_mode, init_mode=cfg.model.ffi.growth_init_mode,
                              rewire_threshold=cfg.model.ffi.rewire_threshold, 
                              rewire_fraction=cfg.model.ffi.rewire_fraction,group_size=self.group_size,dtype=self.dtype).to(self.device)
            else:
                self.linear =  FixedFanIn(self.tail_input_size, cfg.model.ffi.output_features, fan_in=cfg.model.ffi.fan_in,
                              prune_mode=cfg.model.ffi.prune_mode, init_mode=cfg.model.ffi.growth_init_mode,
                              rewire_threshold=cfg.model.ffi.rewire_threshold, rewire_fraction=cfg.model.ffi.rewire_fraction,dtype=self.dtype).to(self.device)
    
    def rewire(self):
        if self.use_ht_split:
            self.tail_layer.rewire()   
        else:
            self.linear.rewire()     
        
    def forward(self,tokens,masks):
        ''' Forward pass through the model. '''
            
        out = self.encoder(tokens,masks)
        out = out.to(self.dtype)
        dtype = out.dtype
        
        if self.cfg.model.private_feature.use_private_feature:
            branch_out = self.group_branch(out[:,512:]) if self.cfg.model.auxiliary.use_meta_branch else None
            if self.cfg.model.penultimate.use_penultimate_layer:
                out = self.penultimate(out).to(dtype)
            if self.use_ht_split:
                head_out = self.head_dropout(out[:,self.cfg.model.private_feature.head_start:self.cfg.model.private_feature.head_end])
                tail_out = self.tail_dropout(out[:,self.cfg.model.private_feature.tail_start:self.cfg.model.private_feature.tail_end])
                head_logits = self.head_layer(head_out) 
                tail_logits = self.tail_layer(tail_out) 
                out = torch.cat((head_logits, tail_logits), dim=1) 
            else:
                tail_out = self.tail_dropout(out[:,self.cfg.model.private_feature.tail_start:self.cfg.model.private_feature.tail_end])
                out = self.linear(tail_out)
            
        else:
            out = self.dropout(out)
            
            branch_out = self.group_branch(out) if self.cfg.model.auxiliary.use_meta_branch else None
            
            if self.cfg.model.penultimate.use_penultimate_layer:
                out = self.penultimate(out).to(dtype)
                
            if self.use_ht_split:
                head_logits = self.head_layer(out) 
                tail_logits = self.tail_layer(out) 
                out = torch.cat((head_logits, tail_logits), dim=1)  
            else:
                #out = self.linear(out)
                out = self.linear(out)
        
        
        return out, branch_out  
    

        
    def param_list(self):
        param_list, param_list_xmc = [], []
        if self.cfg.model.auxiliary.use_meta_branch:
            param_list.append({"params":self.group_branch.parameters(),"lr":self.cfg.training.optimization.meta_lr})
            
  
        optimizer_params_encoder = []
        for n, p in self.encoder.named_parameters():
            if p.requires_grad:
                optimizer_params_encoder.append((n, p))
        
        no_decay_params = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        param_list += [
                {'params': [p for n, p in optimizer_params_encoder if not any(nd in n for nd in no_decay_params)],
                    'weight_decay': self.cfg.training.optimization.wd_encoder, "lr":self.cfg.training.optimization.encoder_lr},
                {'params': [p for n, p in optimizer_params_encoder if any(nd in n for nd in no_decay_params)],
                    "lr":self.cfg.training.optimization.encoder_lr ,'weight_decay': 0.0}]
            
        if self.cfg.model.penultimate.use_penultimate_layer:
            param_list.append({"params":self.penultimate.parameters(),"lr":self.cfg.training.optimization.penultimate_lr})
        
        if self.use_ht_split:
            param_list_xmc.append({"params":self.head_layer.parameters(),"lr":self.cfg.training.optimization.lr,'weight_decay': self.cfg.training.optimization.wd})
            param_list_xmc.append({"params":self.tail_layer.parameters(),"lr":self.cfg.training.optimization.lr,'weight_decay': self.cfg.training.optimization.wd})
        else:
            param_list_xmc.append({"params":self.linear.parameters(),"lr":self.cfg.training.optimization.lr,'weight_decay': self.cfg.training.optimization.wd})
        
        return param_list, param_list_xmc
