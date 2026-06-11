import time
import argparse
import numpy as np
from pdes import Diffusion_sorption, Boundary
from utilities import (
    NeuralNet,
    mean_squared_error,
    relative_error,
    restore_model_state,
    set_random_seed,
    snapshot_model_state,
    ensure_dir,
)

set_random_seed(1234)

import torch
import torch.nn as nn
import torch.optim as optim



class PhysicsInformedNN:
    def __init__(self, x_init, t_init, u_init, x_l_bound, x_r_bound, t_bound, x_eqns, t_eqns,
        x_data, t_data, u_data, x_test, t_test, u_test,
        batch_size, layers, log_path):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.float().to(self.device)
            return torch.tensor(x, dtype=torch.float32, device=self.device)
        
        self.x_init = to_tensor(x_init)
        self.t_init = to_tensor(t_init)
        self.u_init = to_tensor(u_init)

        self.x_l = to_tensor(x_l_bound)
        self.x_r = to_tensor(x_r_bound)
        self.t_b = to_tensor(t_bound)

        self.x_eqns = to_tensor(x_eqns)
        self.t_eqns = to_tensor(t_eqns)

        self.x_data = to_tensor(x_data)
        self.t_data = to_tensor(t_data)
        self.u_data = to_tensor(u_data)

        self.x_test = to_tensor(x_test)
        self.t_test = to_tensor(t_test)
        self.u_test = to_tensor(u_test)

        self.batch_size = batch_size
        self.log_path = log_path

        input_stats = torch.cat([self.x_eqns, self.t_eqns], dim=1)
        self.net = NeuralNet(
            layers,
            output_activation="relu",
            input_mean=input_stats.mean(dim=0, keepdim=True).cpu(),
            input_std=input_stats.std(dim=0, keepdim=True).cpu(),
        ).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)
        self.best_eval = float("inf")
        self.best_state = snapshot_model_state(self.net)
        self.best_it = -1

    def forward(self, x, t):
        return self.net(x, t)
    
    def loss_init(self):
        u_pred = self.forward(self.x_init, self.t_init)
        return torch.mean((u_pred - self.u_init) ** 2)

    def loss_data(self):
        u_pred = self.forward(self.x_data, self.t_data)
        return torch.mean((u_pred - self.u_data) ** 2)

    def loss_boundary(self):
        u_l = self.forward(self.x_l, self.t_b)
        loss_left = torch.mean((u_l - 1.0) ** 2)
        x_r = self.x_r.clone().detach().requires_grad_(True)
        u_r = self.forward(x_r, self.t_b)
        b = Boundary(x_r, u_r)
        loss_right = torch.mean((u_r - b) ** 2)
        return loss_left + loss_right

    def loss_pde(self, x, t):
        x.requires_grad_(True)
        t.requires_grad_(True)
        u = self.forward(x, t)
        f = Diffusion_sorption(x, t, u)
        return torch.mean(f ** 2)
    
    def train(self, max_time, adam_it):
        start_time = time.time()
        total_time = 0
        N_eqns = len(self.x_eqns)
        for it in range(adam_it):
            idx = np.random.choice(N_eqns, min(self.batch_size, N_eqns), replace=False)
            x_eq = self.x_eqns[idx].clone().detach()
            t_eq = self.t_eqns[idx].clone().detach()

            loss_init = self.loss_init()
            loss_bound = self.loss_boundary()
            loss_data = self.loss_data()
            loss_eq = self.loss_pde(x_eq, t_eq)

            loss = (loss_init + loss_bound + loss_eq + 5.0 * loss_data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if it % 10 == 0:
                elapsed = time.time() - start_time
                total_time += elapsed / 3600
                log_item = (
                    f"It: {it}, "
                    f"Loss={loss.item():.3e}, "
                    f"Init={loss_init.item():.3e}, "
                    f"Bound={loss_bound.item():.3e}, "
                    f"Eqns={loss_eq.item():.3e}, "
                    f"Data={loss_data.item():.3e}, "
                    f"Time={elapsed:.2f}s, "
                    f"Total={total_time:.2f}h"
                )

                self.logging(log_item)
                start_time = time.time()

            if it % 100 == 0:
                with torch.no_grad():
                    u_pred = self.forward(self.x_test, self.t_test)
                    err = relative_error(u_pred, self.u_test)
                self.logging(f"Error u: {err:e}")
                err_value = float(err.detach().cpu())
                if err_value < self.best_eval:
                    self.best_eval = err_value
                    self.best_it = it
                    self.best_state = snapshot_model_state(self.net)
                    self.logging(f"Best Eval: It={it}, L2={err_value:.6e}")
            if total_time > max_time:
                break
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
    parser.add_argument("--adam-it", type=int, default=20000)
    parser.add_argument("--max-time", type=float, default=10)
    args = parser.parse_args()

    ensure_dir("./output/log")
    ensure_dir("./output/prediction")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xL, xR = 0, 1
    N_init = 1024
    N_bound = 512
    N_data = 1000
    N_test = 20000
    batch_size = 20000
    layers = [2] + 4 * [32] + [1]
    create_date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    log_path = "./output/log/diffsorb1D-pinn-%s" % (create_date)

    ### load data
    data_path = r'./input/diffsorb1D.npy'
    data = np.load(data_path, allow_pickle=True)
    x = data.item()['x']
    t = data.item()['t']
    u = data.item()['u']

    # init
    idx_init = np.where(t == 0.0)[0]
    x_init = x[idx_init, :]
    t_init = t[idx_init, :]
    u_init = u[idx_init, :]

    # boundary
    idx_bound = np.where(x == x[0, 0])[0]
    t_bound = t[idx_bound, :]
    x_l_bound = xL * np.ones_like(t_bound)
    x_r_bound = xR * np.ones_like(t_bound)

    ### rearrange data
    # eqns
    x_eqns = x
    t_eqns = t

    # initail
    idx_init = np.random.choice(x_init.shape[0], min(N_init, x_init.shape[0]), replace=False)
    x_init = x_init[idx_init, :]
    t_init = t_init[idx_init, :]
    u_init = u_init[idx_init, :]

    # boundary
    idx_bound = np.random.choice(t_bound.shape[0], min(N_bound, t_bound.shape[0]), replace=False)
    x_l_bound = x_l_bound[idx_bound, :]
    x_r_bound = x_r_bound[idx_bound, :]
    t_bound = t_bound[idx_bound, :]

    # intre-domain
    idx_data = np.random.choice(x.shape[0], min(N_data, x.shape[0]), replace=False)
    x_data = x[idx_data, :]
    t_data = t[idx_data, :]
    u_data = u[idx_data, :]

    # test
    idx_test = np.random.choice(x.shape[0], min(N_test, x.shape[0]), replace=False)
    x_test = x[idx_test, :]
    t_test = t[idx_test, :]
    u_test = u[idx_test, :]

    model = PhysicsInformedNN(x_init, t_init, u_init, x_l_bound, x_r_bound, t_bound, x_eqns, t_eqns,
                              x_data, t_data, u_data, x_test, t_test, u_test, batch_size, layers, log_path)

    ### train
    model.train(max_time=args.max_time, adam_it=args.adam_it)

    ### test
    u_pred = model.predict(x, t)
    u = torch.tensor(u, dtype=torch.float32).to(device)

    error_u = relative_error(u_pred, u)
    model.logging('L2 error u: %e' % (error_u))

    u_pred = model.predict(x, t)
    error_u = mean_squared_error(u_pred, u)
    model.logging('MSE error u: %e' % (error_u))

    # save prediction
    data_output_path = "./output/prediction/diffsorb1D-pinn-%s.npy" % (create_date)
    data_output = {'u': u_pred.detach().cpu().numpy()}
    np.save(data_output_path, data_output)
