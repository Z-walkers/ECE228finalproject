import time
import argparse
import copy
import numpy as np
# import tensorflow as tf
from pdes import Burgers1D
from utilities import (
    NeuralNet,
    candidate_overlap,
    confidence_weights,
    ensure_dir,
    mean_squared_error,
    normalize_score,
    pseudo_rate,
    relative_error,
    restore_model_state,
    set_random_seed,
    snapshot_model_state,
)
import torch
import torch.nn as nn
import torch.optim as optim

set_random_seed(1234)

class SelfTrainingPINN:
    def __init__(self,
                 x_init, t_init, u_init,
                 x_l, x_r, t_b,
                 x_sample, t_sample,
                 x_data, t_data, u_data,
                 x_test, t_test, u_test,
                 nu, batch_size, layers, log_path,
                 update_freq, max_rate, stab_coeff,
                 schedule_type="fixed", q_min=0.02, q_max=None, warmup_ratio=0.6,
                 adam_it=20000, teacher_decay=0.99, pseudo_loss_weight=1.0,
                 pseudo_uncertainty_weight=1.0, pseudo_temperature=0.25,
                 variant="ffustpinn"):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def to_tensor(value):
            if isinstance(value, torch.Tensor):
                return value.detach().clone().float().to(self.device)
            return torch.tensor(value, dtype=torch.float32, device=self.device)

        self.x_init = to_tensor(x_init)
        self.t_init = to_tensor(t_init)
        self.u_init = to_tensor(u_init)

        self.x_l = to_tensor(x_l)
        self.x_r = to_tensor(x_r)
        self.t_b = to_tensor(t_b)

        self.x_sample = to_tensor(x_sample)
        self.t_sample = to_tensor(t_sample)
        self.x_eqns = self.x_sample.clone()
        self.t_eqns = self.t_sample.clone()

        self.x_data = to_tensor(x_data)
        self.t_data = to_tensor(t_data)
        self.u_data = to_tensor(u_data)

        self.x_test = to_tensor(x_test)
        self.t_test = to_tensor(t_test)
        self.u_test = to_tensor(u_test)

        self.flag_pseudo = np.zeros(len(x_sample), dtype=np.int32)
        self.variant = variant

        input_stats = torch.cat([self.x_sample, self.t_sample], dim=1)
        self.net = NeuralNet(
            layers,
            output_activation="linear",
            input_mean=input_stats.mean(dim=0, keepdim=True).cpu(),
            input_std=input_stats.std(dim=0, keepdim=True).cpu(),
            fourier_features=self.variant == "ffustpinn",
        ).to(self.device)
        self.teacher_net = copy.deepcopy(self.net).to(self.device)
        for param in self.teacher_net.parameters():
            param.requires_grad_(False)
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

        self.nu = nu
        self.batch_size = batch_size

        self.update_freq = update_freq
        self.max_rate = max_rate
        self.q_max = max_rate if q_max is None else q_max
        self.q_min = q_min
        self.schedule_type = schedule_type
        self.warmup_ratio = warmup_ratio
        self.adam_it = adam_it
        self.current_rate = self.q_max
        self.stab_coeff = stab_coeff
        self.current_stab_coeff = stab_coeff
        self.teacher_decay = teacher_decay
        self.pseudo_loss_weight = pseudo_loss_weight
        self.pseudo_uncertainty_weight = pseudo_uncertainty_weight
        self.pseudo_temperature = pseudo_temperature
        self.previous_candidate_idx = None
        self.current_overlap = 0.0
        self.current_instability = 0.0
        self.current_mean_uncertainty = 0.0
        self.best_eval = float("inf")
        self.best_state = snapshot_model_state(self.net)
        self.best_it = -1

        self.log_path = log_path

        self.x_pseudo = None
        self.t_pseudo = None
        self.u_pseudo = None
        self.w_pseudo = None

    
    def forward(self, x, t):
        return self.net(x, t)

    def loss_pde(self, x, t):
        x.requires_grad_(True)
        t.requires_grad_(True)

        u = self.forward(x, t)
        f = Burgers1D(x, t, u, self.nu)
        return torch.mean(f ** 2)

    def loss_data(self):
        u_pred = self.forward(self.x_data, self.t_data)
        return torch.mean((u_pred - self.u_data) ** 2)

    def loss_init(self):
        u_pred = self.forward(self.x_init, self.t_init)
        return torch.mean((u_pred - self.u_init) ** 2)

    def loss_bound(self):
        u_l = self.forward(self.x_l, self.t_b)
        u_r = self.forward(self.x_r, self.t_b)
        return torch.mean((u_l - u_r) ** 2)

    def loss_pseudo(self):
        if (self.x_pseudo is None or self.x_pseudo.shape[0] == 0):
            return torch.tensor(0.0, device=self.device)
        u_pred = self.forward(self.x_pseudo, self.t_pseudo)
        if self.w_pseudo is None:
            return torch.mean((u_pred - self.u_pseudo) ** 2)
        return torch.mean(self.w_pseudo * (u_pred - self.u_pseudo) ** 2)

    def update_teacher(self):
        with torch.no_grad():
            for teacher_param, param in zip(self.teacher_net.parameters(), self.net.parameters()):
                teacher_param.mul_(self.teacher_decay).add_(param, alpha=1.0 - self.teacher_decay)
    
    def update_pseudo(self, iteration):
        x = self.x_sample.clone().detach().requires_grad_(True)
        t = self.t_sample.clone().detach().requires_grad_(True)
        u = self.forward(x, t)
        f = Burgers1D(x, t, u, self.nu)
        residual = torch.abs(f).detach().cpu().numpy().reshape(-1)
        if self.variant == "ffustpinn":
            with torch.no_grad():
                u_teacher = self.teacher_net(self.x_sample, self.t_sample)
                uncertainty = torch.abs(u.detach() - u_teacher).cpu().numpy().reshape(-1)
            score = normalize_score(residual) + self.pseudo_uncertainty_weight * normalize_score(uncertainty)
            self.current_mean_uncertainty = float(np.mean(uncertainty))
        else:
            score = residual
            self.current_mean_uncertainty = 0.0
        sample_size = len(score)
        self.current_rate = pseudo_rate(
            iteration, self.adam_it, self.schedule_type,
            q_min=self.q_min, q_max=self.q_max, warmup_ratio=self.warmup_ratio
        )
        pseudo_size = int(self.current_rate * sample_size)

        if pseudo_size > 0 and self.variant == "ffustpinn":
            residual_pool_size = min(sample_size, max(pseudo_size, 2 * pseudo_size))
            idx_pool = np.argpartition(residual, residual_pool_size - 1)[:residual_pool_size]
            local_uncertainty = uncertainty[idx_pool]
            idx_local = np.argpartition(local_uncertainty, pseudo_size - 1)[:pseudo_size]
            idx_pseudo = idx_pool[idx_local]
        elif pseudo_size > 0:
            idx_pseudo = np.argpartition(score, pseudo_size)[:pseudo_size]
        else:
            idx_pseudo = []

        idx_pseudo = np.asarray(idx_pseudo, dtype=np.int64)
        self.current_overlap = candidate_overlap(idx_pseudo, self.previous_candidate_idx)
        self.current_instability = 1.0 - self.current_overlap
        self.previous_candidate_idx = idx_pseudo.copy()
        self.current_stab_coeff = self.stab_coeff

        self.flag_pseudo[idx_pseudo] += 1
        mask = np.ones(sample_size, dtype=bool)
        mask[idx_pseudo] = False
        if self.variant == "ffustpinn":
            self.flag_pseudo[mask] = np.maximum(self.flag_pseudo[mask] - 1, 0)
        else:
            self.flag_pseudo[mask] = 0

        idx_stable = np.where(self.flag_pseudo > self.current_stab_coeff)[0]
        self.x_pseudo = self.x_sample[idx_stable].clone().detach()
        self.t_pseudo = self.t_sample[idx_stable].clone().detach()

        with torch.no_grad():
            label_net = self.teacher_net if self.variant == "ffustpinn" else self.net
            self.u_pseudo = label_net(self.x_pseudo, self.t_pseudo).detach()
        if self.variant == "ffustpinn" and len(idx_stable) > 0:
            pseudo_weights = confidence_weights(score[idx_stable], temperature=self.pseudo_temperature)
            self.w_pseudo = torch.tensor(pseudo_weights.reshape(-1, 1), dtype=torch.float32, device=self.device)
        elif self.variant == "ffustpinn":
            self.w_pseudo = torch.empty((0, 1), dtype=torch.float32, device=self.device)
        else:
            self.w_pseudo = None

        if self.variant == "ffustpinn":
            idx_eqns = np.arange(sample_size)
        else:
            mask = np.ones(sample_size, dtype=bool)
            mask[idx_stable] = False
            idx_eqns = np.where(mask)[0]
        self.x_eqns = self.x_sample[idx_eqns].clone().detach()
        self.t_eqns = self.t_sample[idx_eqns].clone().detach()
        self.logging(
            f"Pseudo Update: It={iteration}, Rate={self.current_rate:.4f}, "
            f"R={self.current_stab_coeff}, Overlap={self.current_overlap:.4f}, "
            f"Instability={self.current_instability:.4f}, "
            f"MeanUnc={self.current_mean_uncertainty:.3e}, "
            f"Pseudo Points={len(idx_stable)}, Eqns Points={len(idx_eqns)}, "
            f"MaxFlag={self.flag_pseudo.max()}"
        )

    def train(self, max_time, adam_it):
        start = time.time()
        total_time = 0
        self.update_pseudo(0)
        for it in range(adam_it):
            idx = np.random.choice(
                len(self.x_eqns),
                min(self.batch_size, len(self.x_eqns)),
                replace=False
            )

            x_eq = self.x_eqns[idx].clone().detach().requires_grad_(True)
            t_eq = self.t_eqns[idx].clone().detach().requires_grad_(True)

            init_loss = self.loss_init()
            data_loss = self.loss_data()
            bound_loss = self.loss_bound()
            eqns_loss = self.loss_pde(x_eq, t_eq)
            pseudo_loss = self.loss_pseudo()
            loss = init_loss + 100 * data_loss + bound_loss + eqns_loss + self.pseudo_loss_weight * pseudo_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.variant == "ffustpinn":
                self.update_teacher()

            if it % 10 == 0:
                elapsed = time.time() - start
                total_time += elapsed / 3600
                self.logging(
                    f"It: {it}, Rate={self.current_rate:.4f}, R={self.current_stab_coeff}, "
                    f"Init={init_loss.item():.3e}, "
                    f"Bound={bound_loss.item():.3e}, "
                    f"Eq={eqns_loss.item():.3e}, "
                    f"Data={data_loss.item():.3e}, "
                    f"Pseudo={pseudo_loss.item():.3e}, "
                    f"Total={loss.item():.3e}, Time={elapsed:.2f}s"
                )
                start = time.time()

            if it % 100 == 0:
                with torch.no_grad():
                    u_pred = self.forward(self.x_test, self.t_test)
                    l2 = relative_error(u_pred, self.u_test)
                    mse = mean_squared_error(u_pred, self.u_test)
                    self.logging(f"Eval: It={it}, L2={l2:.6e}, MSE={mse:.6e}")
                    l2_value = float(l2.detach().cpu())
                    if l2_value < self.best_eval:
                        self.best_eval = l2_value
                        self.best_it = it
                        self.best_state = snapshot_model_state(self.net)
                        self.logging(f"Best Eval: It={it}, L2={l2_value:.6e}")

            if it % self.update_freq == 0:
                self.update_pseudo(it)
        restore_model_state(self.net, self.best_state)
        self.logging(f"Restored best checkpoint: It={self.best_it}, L2={self.best_eval:.6e}")


    def predict(self, x, t):
        self.net.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            if isinstance(t, np.ndarray):
                t = torch.tensor(t, dtype=torch.float32)
            x = x.to(self.device)
            t = t.to(self.device)
            return self.net(x, t)
        
    def logging(self, log_item): 
        with open(self.log_path, 'a+') as log: 
            log.write(log_item + '\n') 
        print(log_item)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule-type", default="fixed", choices=["fixed", "linear_warmup"])
    parser.add_argument("--q-min", type=float, default=0.02)
    parser.add_argument("--q-max", type=float, default=0.5)
    parser.add_argument("--warmup-ratio", type=float, default=0.6)
    parser.add_argument("--adam-it", type=int, default=20000)
    parser.add_argument("--max-time", type=float, default=10)
    parser.add_argument("--teacher-decay", type=float, default=0.99)
    parser.add_argument("--pseudo-loss-weight", type=float, default=1.0)
    parser.add_argument("--pseudo-uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--pseudo-temperature", type=float, default=0.25)
    parser.add_argument("--variant", default="ffustpinn", choices=["stpinn", "ffustpinn"])
    args = parser.parse_args()

    ensure_dir("./output/log")
    ensure_dir("./output/prediction")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xL, xR = 0.0, 1.0
    nu = 0.01

    N_init = 1024
    N_bound = 512
    N_data = 1000
    N_test = 20000
    batch_size = 20000
    layers = [2] + 4 * [32] + [1]

    create_date = time.strftime("%Y-%m-%d-%H-%M-%S")
    if args.variant == "stpinn":
        method_name = "dynamic-stpinn" if args.schedule_type != "fixed" else "stpinn"
    else:
        method_name = "dynamic-ffu-stpinn" if args.schedule_type != "fixed" else "ffu-stpinn"
    log_path = f"./output/log/burgers1D-{method_name}-{create_date}.txt"

    update_freq = 100
    max_rate = args.q_max
    stab_coeff = 2
    data = np.load('./input/burgers1D.npy', allow_pickle=True).item()

    x = data['x']
    t = data['t']
    u = data['u']

    idx_init = np.where(t == 0.0)[0]
    x_init, t_init, u_init = x[idx_init], t[idx_init], u[idx_init]

    idx_bound = np.where(x == x[0, 0])[0]
    t_bound = t[idx_bound]
    x_l_bound = xL * np.ones_like(t_bound)
    x_r_bound = xR * np.ones_like(t_bound)

    x_sample = x
    t_sample = t
    u_sample = u

    idx_init = np.random.choice(len(x_init), min(N_init, len(x_init)), replace=False)
    x_init, t_init, u_init = x_init[idx_init], t_init[idx_init], u_init[idx_init]

    idx_bound = np.random.choice(len(t_bound), min(N_bound, len(t_bound)), replace=False)
    x_l_bound, x_r_bound, t_bound = x_l_bound[idx_bound], x_r_bound[idx_bound], t_bound[idx_bound]

    idx_data = np.random.choice(len(x), min(N_data, len(x)), replace=False)
    x_data, t_data, u_data = x[idx_data], t[idx_data], u[idx_data]

    idx_test = np.random.choice(len(x), min(N_test, len(x)), replace=False)
    x_test, t_test, u_test = x[idx_test], t[idx_test], u[idx_test]

    def to_tensor(x):
        return torch.tensor(x, dtype=torch.float32).to(device)

    x_init, t_init, u_init = to_tensor(x_init), to_tensor(t_init), to_tensor(u_init)
    x_l_bound, x_r_bound, t_bound = to_tensor(x_l_bound), to_tensor(x_r_bound), to_tensor(t_bound)
    x_sample, t_sample = to_tensor(x_sample), to_tensor(t_sample)
    x_data, t_data, u_data = to_tensor(x_data), to_tensor(t_data), to_tensor(u_data)
    x_test, t_test, u_test = to_tensor(x_test), to_tensor(t_test), to_tensor(u_test)

    model = SelfTrainingPINN(
        x_init, t_init, u_init,
        x_l_bound, x_r_bound, t_bound,
        x_sample, t_sample,
        x_data, t_data, u_data,
        x_test, t_test, u_test,
        nu, batch_size, layers, log_path,
        update_freq, max_rate, stab_coeff,
        schedule_type=args.schedule_type, q_min=args.q_min, q_max=args.q_max,
        warmup_ratio=args.warmup_ratio, adam_it=args.adam_it,
        teacher_decay=args.teacher_decay, pseudo_loss_weight=args.pseudo_loss_weight,
        pseudo_uncertainty_weight=args.pseudo_uncertainty_weight,
        pseudo_temperature=args.pseudo_temperature, variant=args.variant
    )


    model.train(max_time=args.max_time, adam_it=args.adam_it)
    u_pred = model.predict(x, t)
    u_true = torch.tensor(u, dtype=torch.float32).to(device)

    error_u = relative_error(u_pred, u_true)
    model.logging(f'L2 error u: {error_u:e}')
    error_u = mean_squared_error(u_pred, u_true)
    model.logging(f'MSE error u: {error_u:e}')

    data_output_path = f"./output/prediction/burgers1d-{method_name}-{create_date}.npy"
    np.save(data_output_path, {'u': u_pred.detach().cpu().numpy()})
