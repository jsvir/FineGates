"""
Author: Jonathan Svirsky, 2026
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init

SQRT_2 = math.sqrt(2)

class GatesVector(nn.Module):
    def __init__(self, size, sparsity_lambda=1, dtype=None, target_sparsity=1):
        super().__init__()
        self.dtype = dtype if dtype is not None else torch.float32
        self._gates_logits = torch.nn.Parameter(torch.ones(size).type(self.dtype))
        torch.nn.init.normal_(self._gates_logits, mean=1., std=0.1).type(self.dtype)
        self.sparsity_lambda = sparsity_lambda
        self.target_sparsity = target_sparsity
        self.noise_mean = 0.
        self.noise_std = 0.1
        self.register_buffer("noise", torch.empty_like(self._gates_logits))


    @property
    def gates(self):
        if self.training:
            self.noise.normal_(self.noise_mean, self.noise_std)
            return self.hard_sigmoid(self.mu + self.noise)
        else:
            return self.eval_gates

    @property 
    def eval_gates(self):
        return self.hard_sigmoid(self.mu)

    @property
    def mu(self):
        return torch.tanh(self._gates_logits)

    def sparsity_loss(self):
        return 0.5 - 0.5 * torch.erf((-0.5 - self.mu) / (0.5 * SQRT_2))

    @staticmethod
    def hard_sigmoid(x):
        return torch.clamp(x + 0.5, 0.0, 1.0)


class SparseLayer(nn.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            fan_in_fan_out: bool = False,
            original_layer=None,
            target_sparsity=0,
            dtype='float32',
            kurt=False,
            kurt_tau=100,
            **kwargs
    ):
        if original_layer == None:
            nn.Linear.__init__(self, in_features, out_features, **kwargs)
            nn.Linear.requires_grad = False
            self.flag_pretrained = False
        else:
            nn.Linear.__init__(self, in_features, out_features)
            self.weight.data = original_layer.weight.data.clone().contiguous()
            if original_layer.bias is not None:
                self.bias.data = original_layer.bias.data.clone().contiguous()
            else:
                self.bias = Parameter(torch.empty(out_features))
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.bias, -bound, bound)
            nn.Linear.requires_grad = False
            self.flag_pretrained = True
        self.fan_in_fan_out = fan_in_fan_out
        if 'dtype' in kwargs:
            self.weight_ = self.weight.data.type(getattr(torch, kwargs['dtype']))
        else:
            self.weight_ = self.weight.data
        self.weight = None
        self.target_sparsity = target_sparsity
        self.gates_columns = GatesVector(
            (1, self.weight_.size(1)),
            dtype=getattr(torch, dtype),
            target_sparsity=target_sparsity)
        self.gates_rows = GatesVector(
            (1, self.weight_.size(0)),
            dtype=getattr(torch, dtype),
            target_sparsity=target_sparsity)
        
        self.register_buffer(
            "target_keep",
            torch.tensor(1.0 - self.target_sparsity, dtype=self.weight_.dtype)
        )
        # we count only weight params and its reduction
        self.initial_params = torch.numel(self.weight_)
        if kurt:
            self.kurtosis_weights = torch.tensor(0)
            self.kurt_tau = kurt_tau
        else:
            self.kurtosis_weights = None
   
    def T(self, w):
        return w.transpose(0, 1) if self.fan_in_fan_out else w

    def number_compressed_parameters(self):
        gates_columns = self.gates_columns.eval_gates.reshape(-1)
        indices_columns = torch.nonzero(gates_columns > 0, as_tuple=True)[0].long()
        gates_rows = self.gates_rows.eval_gates.reshape(-1)
        indices_rows = torch.nonzero(gates_rows > 0, as_tuple=True)[0].long()
        current = torch.numel(torch.index_select(
            torch.index_select(self.weight_.to(indices_columns.device), 1, indices_columns),
            0,
            indices_rows))
        return self.initial_params - current

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.gates_columns.train(mode)
        self.gates_rows.train(mode)

    def forward(self, x: torch.Tensor): 
        if self.weight_.device != x.device:
            self.weight_ = self.weight_.to(x.device)

        if self.kurtosis_weights is not None:
            x_reduced = x.mean(0).mean(0).unsqueeze(0)
            rows_gates_eval = self.gates_rows.eval_gates.reshape(-1)
            cols_gates_eval = self.gates_columns.eval_gates.reshape(-1)
            positive_rows = torch.nonzero(rows_gates_eval > 0, as_tuple=True)[0]
            positive_columns = torch.nonzero(cols_gates_eval > 0, as_tuple=True)[0]

            activations = x_reduced * self.weight_
            # very slow
            # activations = x.mean(dim=1, keepdim = True) * gated_weight.unsqueeze(0)
            self.kurtosis_weights = (
                self.compute_kurtosis_weights(activations[positive_rows][:, positive_columns]),
                positive_rows,
                positive_columns
            )
        
        y = F.linear(x * self.gates_columns.gates, self.weight_, None)
        if self.bias is not None:
            y = y + self.bias                          
        return y * self.gates_rows.gates


    def prepare_for_inference(self, device='cpu'):
        with torch.no_grad():
            rows_gates_train = self.gates_rows.gates.to(device).reshape(-1)
            cols_gates_train = self.gates_columns.gates.to(device).reshape(-1)
            self.eval_rows_index = torch.nonzero(rows_gates_train > 0, as_tuple=True)[0].long()
            self.eval_cols_index = torch.nonzero(cols_gates_train > 0, as_tuple=True)[0].long()
            self.weight_eval = self.T(torch.index_select(rows_gates_train.reshape(-1, 1) * self.weight_.to(device) * cols_gates_train.reshape(1, -1), -1, self.eval_cols_index)).to(device)
            self.bias_eval = self.bias.to(device) * rows_gates_train

    def target_loss(self, val, target):
        return (val - target).abs()

    def sparsity_loss(self):
        loss_vec_rows = self.gates_rows.sparsity_loss().reshape(-1)
        loss_vec_cols = self.gates_columns.sparsity_loss().reshape(-1)
        if self.kurtosis_weights is not None:
            (sparse_weight_rows, sparse_weight_cols), positive_rows, positive_columns = self.kurtosis_weights
            means = torch.stack((
                    (loss_vec_rows[positive_rows] * sparse_weight_rows).sum(),
                    (loss_vec_cols[positive_columns] * sparse_weight_cols).sum(),
                ))
        else:
             means = torch.stack((loss_vec_rows.mean(), loss_vec_cols.mean()))
        return (means - self.target_keep).abs().mean()

    @torch.no_grad()
    def kurtosis(self, matrix, dim=0):
        std = torch.std(matrix, dim, keepdim=True)
        mu = torch.mean(matrix, dim, keepdim=True)
        # Compute the centered values
        centered = matrix - mu
        # Compute the zscore
        # Set zscores to 0 where std is 0
        zscores = centered / std
        zscores = torch.where(torch.isnan(zscores), torch.zeros_like(zscores), zscores)
        kurt = torch.mean(zscores.pow(4), dim)
        return kurt.sqrt()

    @torch.no_grad()
    def compute_kurtosis_weights(self, activations):
        """Compute the kurtosis (Pearson) of a distribution.
        Kurtosis is the fourth central moment divided by the square of the
        variance.
        """
        columns_kurt = self.kurtosis(activations, 0)
        columns_kurt_weights = torch.softmax(-columns_kurt / self.kurt_tau, dim=0)
        rows_kurt = self.kurtosis(activations.T, 0)
        rows_kurt_weights = torch.softmax(-rows_kurt / self.kurt_tau, dim=0)
        return rows_kurt_weights, columns_kurt_weights


class SparseLoRALayer(SparseLayer):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            lora_rank=4,
            lora_dropout=0,
            fan_in_fan_out: bool = False,
            original_layer=None,
            target_sparsity=0,
            dtype='float32',
            kurt=False,
            kurt_tau=100,
            **kwargs
    ):
        super().__init__(
            in_features,
            out_features,
            fan_in_fan_out,
            original_layer,
            target_sparsity,
            dtype,
            kurt,
            kurt_tau,
            **kwargs
        )
        self.fan_in_fan_out = fan_in_fan_out
        self.lora_A = nn.Parameter(self.weight_.new_zeros((lora_rank, in_features)))
        self.lora_B = nn.Parameter(self.weight_.new_zeros((out_features, lora_rank)))
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight_.data = self.weight_.data.transpose(0, 1)
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
    
    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.gates_columns.train(mode)
        self.gates_rows.train(mode)
        self.lora_dropout.train(mode)
        self.lora_A.train(mode)
        self.lora_B.train(mode)

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        y = F.linear(x * self.gates_columns.gates, self.weight_, None)
        if self.bias is not None:
            y = y + self.bias                          
        result = y * self.gates_rows.gates
        result += self.gates_columns.gates * (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.gates_rows.gates.reshape(-1,1)
        return result


class SparseLayerPretrain(nn.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            fan_in_fan_out: bool = False,
            original_layer=None,
            target_sparsity=0,
            dtype='float32',
            **kwargs
    ):
        if original_layer == None:
            nn.Linear.__init__(self, in_features, out_features, **kwargs)
            self.flag_pretrained = False
        else:
            # Create a new linear layer
            nn.Linear.__init__(self, in_features, out_features)
            # Copy weights and biases from the original layer to the new layer
            self.weight.data = original_layer.weight.data.clone()
            if original_layer.bias is not None:
                self.bias.data = original_layer.bias.data.clone()
            else:
                self.bias = Parameter(torch.empty(out_features))
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.bias, -bound, bound)
            self.flag_pretrained = True

        self.fan_in_fan_out = fan_in_fan_out
        self.target_sparsity = target_sparsity
        self.gates_columns = GatesVector(
            (1, self.weight.size(1)),
            dtype=getattr(torch, dtype),
            target_sparsity=target_sparsity)
        self.gates_rows = GatesVector(
            (1, self.weight.size(0)),
            dtype=getattr(torch, dtype),
            target_sparsity=target_sparsity)

        # we count only weight params and its reduction
        self.initial_params = torch.numel(self.weight)
        self.register_buffer(
            "target_keep",
            torch.tensor(1.0 - self.target_sparsity, dtype=self.weight.dtype)
        )

    def T(self, w):
        return w.transpose(0, 1) if self.fan_in_fan_out else w

    def number_compressed_parameters(self):
        gates_columns = self.gates_columns.eval_gates.reshape(-1)
        indices_columns = torch.nonzero(gates_columns > 0, as_tuple=True)[0].long()
        gates_rows = self.gates_rows.eval_gates.reshape(-1)
        indices_rows = torch.nonzero(gates_rows > 0, as_tuple=True)[0].long()
        current = torch.numel(
            torch.index_select(
                torch.index_select(
                    self.weight.to(indices_columns.device), 1, indices_columns),
                    0,
                    indices_rows))
        return self.initial_params - current

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.gates_columns.train(mode)
        self.gates_rows.train(mode)

    def forward(self, x: torch.Tensor):
        if self.weight.device != x.device:
            self.weight = self.weight.to(x.device)
        
        y = F.linear(x * self.gates_columns.gates, self.weight, None)
        if self.bias is not None:
            y = y + self.bias                          
        return y * self.gates_rows.gates


    def prepare_for_inference(self, device='cpu'):
        with torch.no_grad():
            rows_gates_train = self.gates_rows.gates.to(device).reshape(-1)
            cols_gates_train = self.gates_columns.gates.to(device).reshape(-1)
            self.eval_rows_index = torch.nonzero(rows_gates_train > 0, as_tuple=True)[0].long()
            self.eval_cols_index = torch.nonzero(cols_gates_train > 0, as_tuple=True)[0].long()
            self.weight_eval = torch.index_select(rows_gates_train.reshape(-1, 1) * self.weight.to(device) * cols_gates_train.reshape(1, -1), -1, self.eval_cols_index)
            self.bias_eval = self.bias.to(device) * rows_gates_train

    def target_loss(self, val, target):
        return (val - target).abs()

    def sparsity_loss(self):
        loss_vec_rows = self.gates_rows.sparsity_loss()
        loss_vec_cols = self.gates_columns.sparsity_loss()
        means = torch.stack((loss_vec_rows.mean(), loss_vec_cols.mean()))
        return (means - self.target_keep).abs().mean()
