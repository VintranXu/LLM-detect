from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize
from typing import List, Optional

transformers.logging.set_verbosity_error()

# 输入形状 (B , N, d_ff, patch_nums)
class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x) # (B , N, d_ff * patch_nums)
        x = self.linear(x)  # (B , N, pred_len)
        x = self.dropout(x)
        return x  # (B , N, pred_len)

# 输入形状 (B , N, d_ff, patch_nums)
class ClassificationHead(nn.Module):
    def __init__(self, n_vars, nf, num_classes, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear1 = nn.Linear(nf, nf // 2)
        self.linear2 = nn.Linear(nf // 2, num_classes)
        self.dropout = nn.Dropout(head_dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x) # (B , N, d_ff * patch_nums)
        x = x.mean(dim=1)   # (B , d_ff * patch_nums)
        x = self.linear1(x) # (B , d_ff * patch_nums //2)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x) # (B , num_classes)
        return x

# (B , N * pred_len)
# class classfier(nn.Module):
#     def __init__(self, n_vars, nf, target_window, head_dropout=0, num_classes=15):
#         super().__init__()
#         self.n_vars = n_vars
#         self.flatten = nn.Flatten(start_dim=-2)
#         self.linear = nn.Linear(nf, num_classes)
#         self.dropout = nn.Dropout(head_dropout)
#         self.softmax = nn.Softmax(dim=-1)
#     def forward(self, x):

#         x = self.flatten(x)  # (B , N, d_ff * patch_nums)

#         x = self.linear(x) # (B , N, d_ff * patch_nums)
#         x = self.softmax(x)
#         return x


class CustomPrototypeManager(nn.Module):
    """Improved custom prototype manager with subword averaging"""
    def __init__(self, llm_model, tokenizer, custom_prototypes: List[str], embedding_dim: int):
        super().__init__()
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.custom_prototypes = custom_prototypes
       
        # Get original embeddings
        self.original_embeddings = llm_model.get_input_embeddings()
        self.vocab_size = self.original_embeddings.weight.shape[0]
       
        # Process prototypes
        self.prototype_tokens, self.oov_prototypes = self._tokenize_prototypes()
       
        # Create learnable embeddings for OOV prototypes
        if self.oov_prototypes:
            self.oov_embeddings = nn.Parameter(
                torch.randn(len(self.oov_prototypes), embedding_dim) * 0.02
            )
            self._initialize_oov_embeddings()

    def _tokenize_prototypes(self):
        """Separate prototypes into known vocab and OOV"""
        prototype_tokens = {}
        oov_prototypes = []
       
        for prototype in self.custom_prototypes:
            tokens = self.tokenizer.encode(prototype, add_special_tokens=False)
            if len(tokens) == 1 and tokens[0] != self.tokenizer.unk_token_id:
                prototype_tokens[prototype] = tokens[0]
            else:
                oov_prototypes.append(prototype)
       
        return prototype_tokens, oov_prototypes

    def _initialize_oov_embeddings(self):
        """Initialize OOV embeddings using subword averaging"""
        with torch.no_grad():
            for i, prototype in enumerate(self.oov_prototypes):
                # 获取原型词的所有子词tokens
                tokens = self.tokenizer.encode(prototype, add_special_tokens=False)
                
                # 收集有效的子词嵌入
                valid_embeddings = []
                for token_id in tokens:
                    if (token_id != self.tokenizer.unk_token_id and 
                        token_id < self.vocab_size):
                        embedding = self.original_embeddings.weight[token_id]
                        valid_embeddings.append(embedding)
                
                if valid_embeddings:
                    # 使用子词嵌入的平均值作为初始化
                    self.oov_embeddings[i] = torch.stack(valid_embeddings).mean(dim=0)
                    print(f"Initialized '{prototype}' using {len(valid_embeddings)} subwords")
                else:
                    # 如果没有有效子词，使用备选策略
                    self._fallback_initialization(i, prototype)

    def _fallback_initialization(self, index: int, prototype: str):
        """备选初始化策略（当子词方法失败时）"""
        # 策略1: 使用字符级相似性
        char_similar_words = self._find_char_similar_words(prototype)
        if char_similar_words:
            embeddings = []
            for word in char_similar_words:
                tokens = self.tokenizer.encode(word, add_special_tokens=False)
                if len(tokens) == 1 and tokens[0] < self.vocab_size:
                    embedding = self.original_embeddings.weight[tokens[0]]
                    embeddings.append(embedding)
            
            if embeddings:
                self.oov_embeddings[index] = torch.stack(embeddings).mean(dim=0)
                print(f"Fallback: Initialized '{prototype}' using character similarity")
                return
        
        # 策略2: 使用随机相似单词的平均
        similar_words = ["trend", "pattern", "signal", "data", "value", "time"]
        embeddings = []
        for word in similar_words:
            tokens = self.tokenizer.encode(word, add_special_tokens=False)
            if len(tokens) == 1 and tokens[0] < self.vocab_size:
                embedding = self.original_embeddings.weight[tokens[0]]
                embeddings.append(embedding)
        
        if embeddings:
            self.oov_embeddings[index] = torch.stack(embeddings).mean(dim=0)
            print(f"Fallback: Initialized '{prototype}' using default similar words")

    def _find_char_similar_words(self, target_word: str, max_words: int = 5):
        """基于字符相似性找到词汇表中的相似词"""
        similar_words = []
        target_lower = target_word.lower()
        
        # 简单的字符匹配策略
        vocab = self.tokenizer.get_vocab()
        for word in vocab.keys():
            if (len(word) > 2 and 
                any(char in word.lower() for char in target_lower[:3]) and
                abs(len(word) - len(target_word)) <= 2):
                similar_words.append(word)
                if len(similar_words) >= max_words:
                    break
        
        return similar_words

    def get_prototype_embeddings(self):
        """Get all prototype embeddings"""
        prototype_embeddings = []
        prototype_names = []
       
        # Known vocab embeddings
        for prototype, token_id in self.prototype_tokens.items():
            embedding = self.original_embeddings.weight[token_id]
            prototype_embeddings.append(embedding)
            prototype_names.append(prototype)
       
        # OOV embeddings
        for i, prototype in enumerate(self.oov_prototypes):
            prototype_embeddings.append(self.oov_embeddings[i])
            prototype_names.append(prototype)
       
        if prototype_embeddings:
            return torch.stack(prototype_embeddings), prototype_names
        else:
            return torch.empty(0, self.embedding_dim), []

    def update_prototype_embeddings(self, learning_rate: float = 0.001):
        """可选：手动更新原型嵌入的方法"""
        if hasattr(self, 'oov_embeddings'):
            # 这里可以添加自定义的更新逻辑
            pass


class EnhancedReprogrammingLayer(nn.Module):
    """Enhanced reprogramming layer with custom prototypes"""
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1,       
                 custom_prototypes: Optional[List[str]] = None, llm_model=None, tokenizer=None):
        super().__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        self.d_model = d_model
        self.d_llm = d_llm
        self.n_heads = n_heads
        
        # Standard projections
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.dropout = nn.Dropout(attention_dropout)
        
        # Custom prototype support
        self.use_custom_prototypes = custom_prototypes is not None and len(custom_prototypes) > 0
        if self.use_custom_prototypes:
            self.prototype_manager = CustomPrototypeManager(
                llm_model=llm_model,
                tokenizer=tokenizer,
                custom_prototypes=custom_prototypes,
                embedding_dim=d_llm
            )
            # Prototype fusion layers
            self.prototype_fusion = nn.Linear(d_llm * 2, d_llm)
            self.prototype_gate = nn.Sequential(
                nn.Linear(d_llm, d_llm // 4),
                nn.ReLU(),
                nn.Linear(d_llm // 4, 1),
                nn.Sigmoid()
            )

    def forward(self, target_embedding, source_embedding, value_embedding):
        '''         enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        Standard reprogramming'''
        original_output = self._standard_reprogramming(target_embedding, source_embedding, value_embedding)
        
        if not self.use_custom_prototypes:
            return original_output
        
        # Enhanced with custom prototypes
        return self._prototype_enhanced_reprogramming(target_embedding, original_output)

    def _standard_reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads
        
        target_proj = self.query_projection(target_embedding).view(B, L, H, -1)
        source_proj = self.key_projection(source_embedding).view(S, H, -1)
        value_proj = self.value_projection(value_embedding).view(S, H, -1)
        
        out = self._reprogramming_attention(target_proj, source_proj, value_proj)
        out = out.reshape(B, L, -1)
        return self.out_projection(out)

    def _prototype_enhanced_reprogramming(self, target_embedding, original_output):
        B, L, _ = target_embedding.shape
        
        # Get prototype embeddings
        prototype_embeddings, prototype_names = self.prototype_manager.get_prototype_embeddings()
        if prototype_embeddings.numel() == 0:
            return original_output
        
        prototype_embeddings = prototype_embeddings.to(target_embedding.device)
        P, _ = prototype_embeddings.shape
        
        # Compute patch-prototype similarities
        target_flat = target_embedding.reshape(B * L, self.d_model)
        target_proj = self.query_projection(target_flat).view(B * L, self.n_heads, -1).mean(dim=1)
        proto_proj = self.key_projection(prototype_embeddings).view(P, self.n_heads, -1).mean(dim=1)
        
        similarity = torch.matmul(target_proj, proto_proj.T)
        attention_weights = F.softmax(similarity / sqrt(target_proj.shape[-1]), dim=-1)
        
        # Weighted prototype features
        prototype_features = torch.matmul(attention_weights, prototype_embeddings)
        prototype_features = prototype_features.view(B, L, self.d_llm)
        
        # Fuse original and prototype features
        combined = torch.cat([original_output, prototype_features], dim=-1)
        fused_output = self.prototype_fusion(combined)
        
        # Gate mechanism
        gate_weights = self.prototype_gate(fused_output)
        final_output = gate_weights * fused_output + (1 - gate_weights) * original_output
        
        return final_output

    def _reprogramming_attention(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape
        scale = 1. / sqrt(E)
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)
        return reprogramming_embedding

    def get_prototype_attention_weights(self, target_embedding):
        """Get prototype attention weights for analysis"""
        if not self.use_custom_prototypes:
            return None, None
        
        prototype_embeddings, prototype_names = self.prototype_manager.get_prototype_embeddings()
        if prototype_embeddings.numel() == 0:
            return None, None
        
        B, L, _ = target_embedding.shape
        prototype_embeddings = prototype_embeddings.to(target_embedding.device)
        
        target_flat = target_embedding.reshape(B * L, self.d_model)
        target_proj = self.query_projection(target_flat).view(B * L, self.n_heads, -1).mean(dim=1)
        proto_proj = self.key_projection(prototype_embeddings).view(-1, self.n_heads, -1).mean(dim=1)
        
        similarity = torch.matmul(target_proj, proto_proj.T)
        attention_weights = F.softmax(similarity / sqrt(target_proj.shape[-1]), dim=-1)
        
        return attention_weights.view(B, L, -1), prototype_names


class ReprogrammingLayer(nn.Module):
    """Standard reprogramming layer without custom prototypes"""
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super().__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)
        out = out.reshape(B, L, -1)
        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape
        scale = 1. / sqrt(E)
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)
        return reprogramming_embedding


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8, custom_prototypes: Optional[List[str]] = None):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        
        # Classification parameters
        self.num_classes = configs.num_classes
        self.is_classification = configs.is_classification

        # Load LLM model
        self._load_llm_model(configs)
        
        # Set up tokenizer
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        # Freeze LLM parameters
        for param in self.llm_model.parameters():
            param.requires_grad = False

        # Set description
        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)
        self.patch_embedding = PatchEmbedding(configs.d_model, self.patch_len, self.stride, configs.dropout)

        # Simplified embeddings handling
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        # Reprogramming layer with custom prototypes support
        if custom_prototypes is None:
            custom_prototypes = getattr(configs, 'custom_prototypes', None)

        if custom_prototypes and len(custom_prototypes) > 0:
            self.reprogramming_layer = EnhancedReprogrammingLayer(
                d_model=configs.d_model,
                n_heads=configs.n_heads,
                d_keys=self.d_ff,
                d_llm=self.d_llm,
                custom_prototypes=custom_prototypes,
                llm_model=self.llm_model,
                tokenizer=self.tokenizer,
                attention_dropout=configs.dropout,
            )
        else:
            self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        # Output heads
        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name in ['long_term_forecast', 'short_term_forecast']: 
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                               head_dropout=configs.dropout)
        elif self.task_name == 'classification' and self.is_classification:
            self.output_projection = ClassificationHead(configs.enc_in, self.head_nf, self.num_classes,
                                                       head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def _load_llm_model(self, configs):
        """Load the specified LLM model"""
        if configs.llm_model == 'LLAMA':
            local_model_path = "llama-2-7b-hf"
            self.llama_config = LlamaConfig.from_pretrained(local_model_path)
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    local_model_path, trust_remote_code=True, local_files_only=True, config=self.llama_config)
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    local_model_path, trust_remote_code=True, local_files_only=True)
            except Exception as e:
                print(f"Failed to load local model: {e}")
                raise
                
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2', trust_remote_code=True, local_files_only=True, config=self.gpt2_config)
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2', trust_remote_code=True, local_files_only=True)
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2', trust_remote_code=True, local_files_only=False, config=self.gpt2_config)
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2', trust_remote_code=True, local_files_only=False)
                    
        elif configs.llm_model == 'BERT':
            local_bert_path = "/home/kemove/.cache/modelscope/hub/models/AI-ModelScope/bert-base-uncased/"
            self.bert_config = BertConfig.from_pretrained(local_bert_path)
            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            
            try:
                self.llm_model = BertModel.from_pretrained(
                    local_bert_path, trust_remote_code=True, local_files_only=True, config=self.bert_config)
                self.tokenizer = BertTokenizer.from_pretrained(
                    local_bert_path, trust_remote_code=True, local_files_only=True)
            except Exception as e:
                print(f"Failed to load local BERT model: {e}")
                raise
        else:
            raise Exception('LLM model is not defined')

    def get_prototype_analysis(self, x_enc):
        """Analyze prototype usage (only when custom prototypes are available)"""
        if not hasattr(self.reprogramming_layer, 'get_prototype_attention_weights'):
            return None

        x_enc = self.normalize_layers(x_enc, 'norm')
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, _ = self.patch_embedding(x_enc)

        attention_weights, prototype_names = self.reprogramming_layer.get_prototype_attention_weights(enc_out)

        if attention_weights is not None:
            avg_attention = attention_weights.mean(dim=[0, 1]).detach().cpu().numpy()
            analysis = {
                'prototype_names': prototype_names,
                'average_attention': avg_attention,
                'attention_weights': attention_weights.detach().cpu().numpy(),
            }
            return analysis

        return None

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        elif self.task_name == 'classification' and self.is_classification:
            return self.classify(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size() # (B, T, N)
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1) # (B *  N ,T , 1)

        # 一些统计算子（如 median）对 bf16 不完全支持，临时用 fp32 计算
        stats_x = x_enc.float()
        min_values = torch.min(stats_x, dim=1)[0]
        max_values = torch.max(stats_x, dim=1)[0]
        medians = torch.median(stats_x, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = stats_x.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )
            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()  # (B, T, N)

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=256).input_ids # (B * N, prompt_len)
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device)) # (B * N, prompt_len, d_llm)

        # Simplified source embeddings
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)  # (num_tokens, d_llm)

        x_enc = x_enc.permute(0, 2, 1).contiguous()   # (B, N, T)
        enc_out, n_vars = self.patch_embedding(x_enc)  # (B * N, patch_nums, d_model)patch_nums 是切片的数量  n_vars = N
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)  # (B * N, patch_nums, d_llm)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1) # cat[ (B * N, prompt_len, d_llm) (B * N, patch_nums, d_llm) ] 
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state  # (B * N, prompt_len + patch_nums, d_llm)
        dec_out = dec_out[:, :, :self.d_ff]   # (B * N, prompt_len + patch_nums, d_ff)

        dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))# (B , N, prompt_len + patch_nums, d_ff)
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous() # (B , N, d_ff, prompt_len + patch_nums)

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:]) # (B , N, pred_len)
        # dec_out = dec_out.permute(0, 2, 1).contiguous() # (B, pred_len, N)

        dec_out = dec_out.reshape(B, -1) # (B , N * pred_len)

        return dec_out

    def classify(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """Classification task forward pass"""
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        stats_x = x_enc.float()
        min_values = torch.min(stats_x, dim=1)[0]
        max_values = torch.max(stats_x, dim=1)[0]
        medians = torch.median(stats_x, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = stats_x.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: classify the time series into {self.num_classes} categories based on the input sequence; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )
            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))

        # Simplified source embeddings
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        logits = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        return logits  # (Batchsize, numclasses)

    def calcute_lags(self, x_enc):
        # FFT 在部分设备对 bf16 支持有限，这里转为 fp32 进行计算
        x32 = x_enc.float()
        q_fft = torch.fft.rfft(x32.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x32.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags