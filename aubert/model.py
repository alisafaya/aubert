import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForMaskedLM
import transformers
import numpy as np


class AuBERT(nn.Module):
    def __init__(
    	self, 
    	text_model, 
    	audio_model, 
    	proj_dim=256,
    	dropout=0.1,
    	activation="relu"
    ):

        super(AuBERT, self).__init__()
        self.text_encoder = TextEncoder(text_model)
        self.text_projection = ProjectionHead(self.text_encoder.model.config.hidden_size, proj_dim, dropout, activation)
        self.audio_encoder = AudioEncoder(audio_model)
        self.audio_projection = ProjectionHead(self.audio_encoder.model.config.hidden_size, proj_dim, dropout, activation)
        
        # Source: https://github.com/openai/CLIP/blob/573315e83f07b53a61ff5098757e8fc885f1703e/clip/model.py#L291
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        

    def forward(self, text, audio, spans, *args, **kwargs):

        batch_size = spans.shape[0]

        # Get audio and text representations
        text_encoding, text_loss = self.text_encoder(spans=spans, **text, **kwargs)
        audio_encoding = self.audio_encoder(**audio, **kwargs)

        # Project to the same dimension
        text_features = self.text_projection(text_encoding)
        audio_features = self.audio_projection(audio_encoding)
        
        # Apply L2 normalization
        text_features = F.normalize(text_features, dim=-1)
        audio_features = F.normalize(audio_features, dim=-1)
        scale = self.logit_scale.exp()
        logits_per_text = scale * text_features @ audio_features.t() # B x B
        logits_per_audio = logits_per_text.t()

        # Simple range
        device = logits_per_text.device
        labels = torch.arange(batch_size, device=device)

        loss_t = F.cross_entropy(logits_per_text, labels)
        loss_a = F.cross_entropy(logits_per_audio, labels)

        loss = (loss_a + loss_t) / 2

        # Given the text, the accuracy of predicting the corresponding audio
        a_acc = (torch.argmax(logits_per_text, dim=-1).detach().cpu() == torch.arange(batch_size)).float().sum()

        # Given the audio, the accuracy of predicting the corresponding text
        t_acc = (torch.argmax(logits_per_audio, dim=-1).detach().cpu() == torch.arange(batch_size)).float().sum()

        result_dict = {}
        result_dict["loss_contrastive"] = loss
        result_dict["loss_text"] = text_loss
        result_dict["loss"] = text_loss + loss
        result_dict["audio_acc"] = a_acc
        result_dict["text_acc"] = t_acc

        return result_dict

    def _get_grad_norm(self):
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm 


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout,
        activation="gelu",
        linear=False
    ):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.linear = linear

        if not linear:
	        self.activation = transformers.activations.ACT2FN[activation]            

	        self.fc = nn.Linear(projection_dim, projection_dim)
	        self.dropout = nn.Dropout(dropout)
	        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        if self.linear:
        	return projected

        x = self.activation(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class TextEncoder(nn.Module):
    def __init__(
        self, 
        model,
        pooling="default"
    ):
        super(TextEncoder, self).__init__()
        if type(model) is str:
            model = AutoModelForMaskedLM.from_pretrained(model)
        self.model = model  
        
        if pooling == "default":
            self.pool = Pooler(self.model.config.hidden_size)
        elif pooling == "lstm":
            self.pool = LSTMPooler(self.model.config.hidden_size)
        elif pooling == "none":
            self.pool = None
        else:
            self.pool = Pooler(self.model.config.hidden_size, mode=pooling)
        
    def forward(self, spans=None, *args, **kwargs):
        model_output = self.model(*args,output_hidden_states=True, **kwargs)
        last_hiddens = model_output.hidden_states[-1] # output of the final layer of the text encoder model.
        loss = model_output.loss

        if spans is None:
            return last_hiddens, loss
        else:
            pooled_output = torch.cat([ self.pool(last_hiddens[i:i+1, b:e]) for i, (b, e) in enumerate(spans) ])
            return pooled_output, loss


class AudioEncoder(nn.Module):
    def __init__(
        self, 
        model, 
        pooling="default"
    ):
        super(AudioEncoder, self).__init__()
        if type(model) is str:
            model = AutoModel.from_pretrained(model)

        self.model = model
        
        if pooling == "default":
            self.pool = Pooler(self.model.config.hidden_size)
        elif pooling == "lstm":
            self.pool = LSTMPooler(self.model.config.hidden_size)
        elif pooling == "none":
            self.pool = None
        else:
            self.pool = Pooler(self.model.config.hidden_size, mode=pooling)
    
    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs).last_hidden_state
        pooled_output = self.pool(outputs)
        return pooled_output


class Pooler(nn.Module):

    def __init__(
        self, 
        hidden_size, 
        mode="all"
    ):
        super(Pooler, self).__init__()
        self.hidden_size = hidden_size
        
        if mode=="mean":
            self.pool_functions = [Pooler._avgpool]
        elif mode=="max":
            self.pool_functions = [Pooler._maxpool]
        elif mode=="all":
            self.pool_functions = [Pooler._avgpool, Pooler._maxpool]
        else:
            raise ValueError(f"mode argument {mode} unknown. Possible values: mean, max, all")
        
        self.out = nn.Linear(self.hidden_size * len(self.pool_functions) , self.hidden_size)

    def forward(self, x): 
        # x -> B, T, H
        assert x.ndim == 3
        conc = torch.cat([ f(x) for f in self.pool_functions], dim=-1) # -> B, len(self.pool_functions)*H
        conc = self.out(conc)  # -> B, H

        return conc
    
    def _maxpool(x):
        x, _ = torch.max(x, 1)
        return x
    
    def _avgpool(x):
        x = torch.mean(x, 1)
        return x


class LSTMPooler(nn.Module):
    def __init__(
        self, 
        hidden_size, 
        pooling_mode="all"
    ):
        super(LSTMPooler, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.pool = Pool(self.hidden_size*2, pooling_mode)

    def forward(self, x): 
        # x -> B, T, H
        h_lstm, _ = self.lstm(x) # -> B, 2*H
        conc = self.pool(h_lstm)  # -> B, H
        return conc