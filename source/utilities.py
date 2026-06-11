import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(
        self,
        layers,
        output_activation="linear",
        input_mean=None,
        input_std=None,
        weight_norm=True,
        fourier_features=False,
        fourier_frequencies=(1, 2, 4),
        fourier_feature_scale=0.1,
    ):
        super().__init__()
        self.activation = nn.Tanh()
        self.output_activation = output_activation
        self.weight_norm = weight_norm
        self.fourier_features = fourier_features
        self.fourier_frequencies = tuple(fourier_frequencies)
        self.fourier_feature_scale = fourier_feature_scale
        if input_mean is None:
            input_mean = torch.zeros(1, layers[0], dtype=torch.float32)
        if input_std is None:
            input_std = torch.ones(1, layers[0], dtype=torch.float32)
        input_std = torch.clamp(torch.as_tensor(input_std, dtype=torch.float32), min=1e-12)
        self.register_buffer("input_mean", torch.as_tensor(input_mean, dtype=torch.float32))
        self.register_buffer("input_std", input_std)
        first_dim = layers[0]
        if self.fourier_features:
            first_dim = layers[0] * (1 + 2 * len(self.fourier_frequencies))
        internal_layers = [first_dim] + list(layers[1:])
        layer_list = []
        gamma_list = []
        for i in range(len(internal_layers) - 1):
            layer = nn.Linear(internal_layers[i], internal_layers[i + 1])
            nn.init.normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            layer_list.append(layer)
            gamma_list.append(nn.Parameter(torch.ones(1, internal_layers[i + 1])))
        self.layers = nn.ModuleList(layer_list)
        self.gammas = nn.ParameterList(gamma_list)

    def forward(self, x, t):
        X = torch.cat([x, t], dim=1)
        X = (X - self.input_mean) / self.input_std
        if self.fourier_features:
            features = [X]
            for frequency in self.fourier_frequencies:
                phase = 2.0 * np.pi * frequency * X
                features.append(self.fourier_feature_scale * torch.sin(phase))
                features.append(self.fourier_feature_scale * torch.cos(phase))
            X = torch.cat(features, dim=1)
        for i in range(len(self.layers) - 1):
            X = self._linear(i, X)
            X = self.activation(X)
        X = self._linear(len(self.layers) - 1, X)
        if self.output_activation == "softplus":
            X = F.softplus(X)
        elif self.output_activation == "relu":
            X = F.relu(X)
        elif self.output_activation in (None, "linear"):
            pass
        else:
            raise ValueError(f"Unsupported output activation: {self.output_activation}")
        return X

    def _linear(self, idx, x):
        layer = self.layers[idx]
        weight = layer.weight
        if self.weight_norm:
            weight = weight / torch.norm(weight, dim=1, keepdim=True).clamp_min(1e-12)
        # Match TF original: H = gamma * (x @ V) + bias. The bias is added AFTER the gamma
        # scaling, so it must not be multiplied by gamma (which F.linear(..., bias)*g would do).
        return F.linear(x, weight, None) * self.gammas[idx] + layer.bias


def set_random_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def pseudo_rate(iteration, max_iterations, schedule_type="fixed", q_min=0.02, q_max=0.1, warmup_ratio=0.6):
    if schedule_type == "fixed":
        return q_max
    if schedule_type != "linear_warmup":
        raise ValueError(f"Unsupported pseudo-label schedule: {schedule_type}")

    warmup_steps = max(1, int(max_iterations * warmup_ratio))
    progress = min(max(iteration, 0), warmup_steps) / warmup_steps
    return q_min + (q_max - q_min) * progress


def candidate_overlap(current_idx, previous_idx):
    if previous_idx is None or len(previous_idx) == 0 or len(current_idx) == 0:
        return 0.0
    current = set(np.asarray(current_idx, dtype=np.int64).tolist())
    previous = set(np.asarray(previous_idx, dtype=np.int64).tolist())
    return len(current.intersection(previous)) / max(1, len(current))


def normalize_score(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    v_min = values.min()
    v_max = values.max()
    return ((values - v_min) / max(v_max - v_min, 1e-12)).astype(np.float32)


def confidence_weights(scores, temperature=0.25):
    scores = normalize_score(scores)
    weights = np.exp(-scores / max(float(temperature), 1e-6)).astype(np.float32)
    return weights / max(float(weights.mean()), 1e-12)


def snapshot_model_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def restore_model_state(model, state):
    if state is not None:
        model.load_state_dict(state)


def mean_squared_error(pred, exact):
    if isinstance(pred, np.ndarray):
        pred = torch.tensor(pred, dtype=torch.float32)
    if isinstance(exact, np.ndarray):
        exact = torch.tensor(exact, dtype=torch.float32)
    exact = exact.to(pred.device)
    return torch.mean((pred - exact) ** 2)


def relative_error(pred, exact):
    if isinstance(pred, np.ndarray):
        pred = torch.tensor(pred, dtype=torch.float32)
    if isinstance(exact, np.ndarray):
        exact = torch.tensor(exact, dtype=torch.float32)
    exact = exact.to(pred.device)
    return (torch.norm(pred - exact)/ torch.norm(exact))
