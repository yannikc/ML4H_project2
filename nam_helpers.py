import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence
from typing import Tuple
import os
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from nam.trainer.metrics import accuracy, mae
from nam.types import Config


class Model(torch.nn.Module):

    def __init__(self, config, name):
        super(Model, self).__init__()
        self._config = config
        self._name = name

    def forward(self):
        raise NotImplementedError

    def __str__(self):
        return f'{self.__class__.__name__}(name={self._name})'

    @property
    def config(self):
        return self._config

    @property
    def name(self):
        return self._name
    
class ExU(torch.nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super(ExU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(in_features, out_features))
        self.bias = Parameter(torch.Tensor(in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        ## Page(4): initializing the weights using a normal distribution
        ##          N(x; 0:5) with x 2 [3; 4] works well in practice.
        torch.nn.init.trunc_normal_(self.weights, mean=4.0, std=0.5)
        torch.nn.init.trunc_normal_(self.bias, std=0.5)

    def forward(
        self,
        inputs: torch.Tensor,
        n: int = 1,
    ) -> torch.Tensor:
        output = (inputs - self.bias).matmul(torch.exp(self.weights))

        # ReLU activations capped at n (ReLU-n)
        output = F.relu(output)
        output = torch.clamp(output, 0, n)

        return output

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'
    
class LinReLU(torch.nn.Module):
    __constants__ = ['bias']

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super(LinReLU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(in_features, out_features))
        self.bias = Parameter(torch.Tensor(in_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weights)
        torch.nn.init.trunc_normal_(self.bias, std=0.5)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        output = (inputs - self.bias) @ self.weights
        output = F.relu(output)

        return output

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'


class FeatureNN(Model):
    """Neural Network model for each individual feature."""

    def __init__(
        self,
        config,
        name,
        *,
        input_shape: int,
        num_units: int,
        feature_num: int = 0,
    ) -> None:
        """Initializes FeatureNN hyperparameters.

        Args:
          num_units: Number of hidden units in first hidden layer.
          dropout: Coefficient for dropout regularization.
          feature_num: Feature Index used for naming the hidden layers.
        """
        super(FeatureNN, self).__init__(config, name)
        self._input_shape = input_shape
        self._num_units = num_units
        self._feature_num = feature_num
        self.dropout = nn.Dropout(p=self.config.dropout)

        hidden_sizes = [self._num_units] + self.config.hidden_sizes

        layers = []

        ## First layer is ExU
        if self.config.activation == "exu":
            layers.append(ExU(in_features=input_shape, out_features=num_units))
        else:
            layers.append(LinReLU(in_features=input_shape, out_features=num_units))

        ## Hidden Layers
        for in_features, out_features in zip(hidden_sizes, hidden_sizes[1:]):
            layers.append(LinReLU(in_features, out_features))

        ## Last Linear Layer
        layers.append(nn.Linear(in_features=hidden_sizes[-1], out_features=1))

        self.model = nn.ModuleList(layers)
        # self.apply(init_weights)

    def forward(self, inputs) -> torch.Tensor:
        """Computes FeatureNN output with either evaluation or training
        mode."""
        outputs = inputs.unsqueeze(1)
        for layer in self.model:
            outputs = self.dropout(layer(outputs))
        return outputs

def penalized_loss(config: Config, logits: torch.Tensor, targets: torch.Tensor, weights: torch.tensor,
                   fnn_out: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """Computes penalized loss with L2 regularization and output penalty.

    Args:
      config: Global config.
      model: Neural network model.
      inputs: Input values to be fed into the model for computing predictions.
      targets: Target values containing either real values or binary labels.

    Returns:
      The penalized loss.
    """

    def features_loss(per_feature_outputs: torch.Tensor) -> torch.Tensor:
        """Penalizes the L2 norm of the prediction of each feature net."""
        per_feature_norm = [  # L2 Regularization
            torch.mean(torch.square(outputs)) for outputs in per_feature_outputs
        ]
        return sum(per_feature_norm) / len(per_feature_norm)

    def weight_decay(model: nn.Module) -> torch.Tensor:
        """Penalizes the L2 norm of weights in each feature net."""
        num_networks = 1 if config.use_dnn else len(model.feature_nns)
        l2_losses = [(x**2).sum() for x in model.parameters()]
        return sum(l2_losses) / num_networks

    loss_func = mse_loss if config.regression else bce_loss
    loss = loss_func(logits, targets, weights)

    reg_loss = 0.0
    if config.output_regularization > 0:
        reg_loss += config.output_regularization * features_loss(fnn_out)

    if config.l2_regularization > 0:
        reg_loss += config.l2_regularization * weight_decay(model)

    return loss + reg_loss

def mae(logits, targets):
    return ((logits.view(-1) - targets.view(-1)).abs().sum() / logits.numel()).item()


def accuracy(logits, targets):
    return (((targets.view(-1) > 0) == (logits.view(-1) > 0.5)).sum() / targets.numel()).item()


class NAM(Model):


    def __init__(
        self,
        config,
        name,
        *,
        num_inputs: int,
        num_units: int,
    ) -> None:
        super(NAM, self).__init__(config, name)

        self._num_inputs = num_inputs
        self.dropout = nn.Dropout(p=self.config.dropout)

        if isinstance(num_units, list):
            assert len(num_units) == num_inputs
            self._num_units = num_units
        elif isinstance(num_units, int):
            self._num_units = [num_units for _ in range(self._num_inputs)]

        ## Builds the FeatureNNs on the first call.
        self.feature_nns = nn.ModuleList([
            FeatureNN(config=config, name=f'FeatureNN_{i}', input_shape=1, num_units=self._num_units[i], feature_num=i)
            for i in range(num_inputs)
        ])

        self._bias = torch.nn.Parameter(data=torch.zeros(1))

    def calc_outputs(self, inputs: torch.Tensor) -> Sequence[torch.Tensor]:
        """Returns the output computed by each feature net."""
        return [self.feature_nns[i](inputs[:, i]) for i in range(self._num_inputs)]

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        individual_outputs = self.calc_outputs(inputs)
        conc_out = torch.cat(individual_outputs, dim=-1)
        dropout_out = self.dropout(conc_out)

        out = torch.sum(dropout_out, dim=-1)
        return out + self._bias, dropout_out
    

class NAMTrainer:
    def __init__(self, config: Config, model: nn.Module, device=None):
        # Assumption: config has lr, decay_rate, regression attributes
        self.config = Config(**vars(config))
        self.model = model.to(device or torch.device("cpu"))
        self.device = device or torch.device("cpu")

        # wrap loss and metric functions
        self.criterion = lambda logits, targets, fnns_out: penalized_loss(
            self.config, logits, targets, None, fnns_out, self.model
        )
        self.metric_fn = mae if self.config.regression else accuracy
        self.metric_name = "MAE" if self.config.regression else "Accuracy"

        # optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.decay_rate
        )
        self.scheduler = StepLR(self.optimizer, gamma=0.995, step_size=1)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        total_metric = 0.0
        for features, targets in train_loader:
            features = features.to(self.device)
            targets  = targets.to(self.device)

            self.optimizer.zero_grad()
            logits, fnns_out = self.model(features)
            loss = self.criterion(logits, targets, fnns_out)
            metric = self.metric_fn(logits, targets)

            loss.backward()
            self.optimizer.step()

            total_loss   += loss.item() * features.size(0)
            total_metric += metric.item() * features.size(0)

        # step LR scheduler once per epoch
        self.scheduler.step()

        avg_loss   = total_loss / len(train_loader.dataset)
        avg_metric = total_metric / len(train_loader.dataset)
        print(f"[Train] Loss: {avg_loss:.4f}, {self.metric_name}: {avg_metric:.4f}")
        return avg_loss, avg_metric

    def validate_epoch(self, val_loader, split_name="Val"):
        self.model.eval()
        total_loss = 0.0
        total_metric = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(self.device)
                targets  = targets.to(self.device)

                logits, fnns_out = self.model(features)
                loss = self.criterion(logits, targets, fnns_out)
                metric = self.metric_fn(logits, targets)

                total_loss   += loss.item() * features.size(0)
                total_metric += metric.item() * features.size(0)

        avg_loss   = total_loss / len(val_loader.dataset)
        avg_metric = total_metric / len(val_loader.dataset)
        print(f"[{split_name}] Loss: {avg_loss:.4f}, {self.metric_name}: {avg_metric:.4f}")
        return avg_loss, avg_metric

    def fit(self, train_loader, val_loader=None, epochs=10):
        for epoch in range(1, epochs+1):
            print(f"=== Epoch {epoch}/{epochs} ===")
            self.train_epoch(train_loader)
            if val_loader is not None:
                self.validate_epoch(val_loader)

    def test(self, test_loader):
        return self.validate_epoch(test_loader, split_name="Test")

