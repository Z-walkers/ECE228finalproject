import time
import argparse
import numpy as np
from pdes import diffusion_reaction_1d
from utilities import (
    NeuralNet,
    mean_squared_error,
    relative_error,
    restore_model_state,
    set_random_seed,
    snapshot_model_state,
    ensure_dir,
)

import torch
import torch.nn as nn
import torch.optim as optim

set_random_seed(1234)

class PhysicsInformedNN:
    def __init__(self,
                 x_init, t_init, u_init,
                 x_l, x_r, t_b,
                 x_eq, t_eq,
                 x_data, t_data, u_data,
                 x_test, t_test, u_test,
                 nu, rho, batch_size, layers, log_path, use_lbfgs=True):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.x_init = torch.tensor(x_init, dtype=torch.float32, device=self.device)
        self.t_init = torch.tensor(t_init, dtype=torch.float32, device=self.device)
        self.u_init = torch.tensor(u_init, dtype=torch.float32, device=self.device)

        self.x_l = torch.tensor(x_l, dtype=torch.float32, device=self.device)
        self.x_r = torch.tensor(x_r, dtype=torch.float32, device=self.device)
        self.t_b = torch.tensor(t_b, dtype=torch.float32, device=self.device)

        self.x_eq = torch.tensor(x_eq, dtype=torch.float32, device=self.device)
        self.t_eq = torch.tensor(t_eq, dtype=torch.float32, device=self.device)

        self.x_data = torch.tensor(x_data, dtype=torch.float32, device=self.device)
        self.t_data = torch.tensor(t_data, dtype=torch.float32, device=self.device)
        self.u_data = torch.tensor(u_data, dtype=torch.float32, device=self.device)

        self.x_test = torch.tensor(x_test, dtype=torch.float32, device=self.device)
        self.t_test = torch.tensor(t_test, dtype=torch.float32, device=self.device)
        self.u_test = torch.tensor(u_test, dtype=torch.float32, device=self.device)

        input_stats = torch.cat([self.x_eq, self.t_eq], dim=1)
        self.net = NeuralNet(
            layers,
            output_activation="linear",
            input_mean=input_stats.mean(dim=0, keepdim=True).cpu(),
            input_std=input_stats.std(dim=0, keepdim=True).cpu(),
        ).to(self.device)

        self.nu = nu
        self.rho = rho
        self.batch_size = batch_size

        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

        self.lbfgs = optim.LBFGS(self.net.parameters(),
                                 max_iter=5000,
                                 tolerance_grad=1e-12,
                                 tolerance_change=1e-12)

        self.log_path = log_path
        self.use_lbfgs = use_lbfgs
        self.best_eval = float("inf")
        self.best_state = snapshot_model_state(self.net)
        self.best_it = -1

    def forward(self, x, t):
        return self.net(x, t)

    def loss_init(self):
        u = self.forward(self.x_init, self.t_init)
        return torch.mean((u - self.u_init) ** 2)

    def loss_data(self):
        u = self.forward(self.x_data, self.t_data)
        return torch.mean((u - self.u_data) ** 2)

    def loss_bound(self):
        ul = self.forward(self.x_l, self.t_b)
        ur = self.forward(self.x_r, self.t_b)
        return torch.mean((ul - ur) ** 2)

    def loss_pde(self, x, t):
        x.requires_grad_(True)
        t.requires_grad_(True)

        u = self.forward(x, t)
        f = diffusion_reaction_1d(x, t, u, self.nu, self.rho)

        return torch.mean(f ** 2)


    def train(self, max_time, adam_it):
        start = time.time()
        for it in range(adam_it):
            idx = np.random.choice(len(self.x_eq),
                                   min(self.batch_size, len(self.x_eq)),
                                   replace=False)

            x_eq = self.x_eq[idx].clone().detach().requires_grad_(True)
            t_eq = self.t_eq[idx].clone().detach().requires_grad_(True)

            loss = (
                self.loss_init() +
                self.loss_bound() +
                self.loss_data() +
                self.loss_pde(x_eq, t_eq)
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if it % 10 == 0:
                self.logging(f"It: {it}, Loss: {loss.item():.3e}")
            if it % 100 == 0:
                with torch.no_grad():
                    l2 = relative_error(self.predict(self.x_test, self.t_test), self.u_test)
                l2_value = float(l2.detach().cpu())
                self.logging(f"Eval: It={it}, L2={l2_value:.6e}")
                if l2_value < self.best_eval:
                    self.best_eval = l2_value
                    self.best_it = it
                    self.best_state = snapshot_model_state(self.net)
                    self.logging(f"Best Eval: It={it}, L2={l2_value:.6e}")
            if time.time() - start > max_time * 3600:
                break

        lbfgs_step = [0]

        def closure():
            self.lbfgs.zero_grad()
            idx = np.random.choice(len(self.x_eq), self.batch_size, replace=False)
            x_eq = self.x_eq[idx].clone().detach().requires_grad_(True)
            t_eq = self.t_eq[idx].clone().detach().requires_grad_(True)
            loss = (
                self.loss_init() +
                self.loss_bound() +
                self.loss_data() +
                self.loss_pde(x_eq, t_eq)
            )
            loss.backward()
            lbfgs_step[0] += 1
            if lbfgs_step[0] % 10 == 0:
                current_it = adam_it + lbfgs_step[0]
                self.logging(f"It: {current_it}, Loss: {loss.item():.3e} (L-BFGS)")
                with torch.no_grad():
                    l2 = relative_error(self.predict(self.x_test, self.t_test), self.u_test)
                l2_value = float(l2.detach().cpu())
                self.logging(f"Eval: It={current_it}, L2={l2_value:.6e}")
                if l2_value < self.best_eval:
                    self.best_eval = l2_value
                    self.best_it = current_it
                    self.best_state = snapshot_model_state(self.net)
                    self.logging(f"Best Eval: It={current_it}, L2={l2_value:.6e}")
            return loss

        if self.use_lbfgs:
            self.lbfgs.step(closure)
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
    parser.add_argument("--skip-lbfgs", action="store_true")
    args = parser.parse_args()

    ensure_dir("./output/log")
    ensure_dir("./output/prediction")
    data = np.load('./input/diffreact1D.npy', allow_pickle=True).item()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    create_date = time.strftime("%Y-%m-%d-%H-%M-%S")   
    log_path = f"./output/log/diffreact1D-pinn-{create_date}.txt" 

    x = data['x']
    t = data['t']
    u = data['u']

    xL, xR = 0, 1
    nu = 0.5
    rho = 1.0
    N_init = 1024
    N_bound = 512
    N_data = 1000
    N_test = 20000
    batch_size = 20000
    layers = [2, 32, 32, 32, 32, 1]

    idx_init = np.where(t == 0)[0]
    x_init, t_init, u_init = x[idx_init], t[idx_init], u[idx_init]

    idx_b = np.where(x == x[0, 0])[0]
    t_b = t[idx_b]
    x_l = xL * np.ones_like(t_b)
    x_r = xR * np.ones_like(t_b)

    x_eq, t_eq = x, t

    idx_init = np.random.choice(len(x_init), min(N_init, len(x_init)), replace=False)
    x_init, t_init, u_init = x_init[idx_init], t_init[idx_init], u_init[idx_init]

    idx_bound = np.random.choice(len(t_b), min(N_bound, len(t_b)), replace=False)
    x_l, x_r, t_b = x_l[idx_bound], x_r[idx_bound], t_b[idx_bound]

    idx_data = np.random.choice(len(x), min(N_data, len(x)), replace=False)
    x_data, t_data, u_data = x[idx_data], t[idx_data], u[idx_data]

    idx_test = np.random.choice(len(x), min(N_test, len(x)), replace=False)
    x_test, t_test, u_test = x[idx_test], t[idx_test], u[idx_test]

    def T(x): 
        return torch.tensor(x, dtype=torch.float32)

    model = PhysicsInformedNN(
        T(x_init), T(t_init), T(u_init),
        T(x_l), T(x_r), T(t_b),
        T(x_eq), T(t_eq),
        T(x_data), T(t_data), T(u_data),
        T(x_test), T(t_test), T(u_test),
        nu, rho,
        batch_size=batch_size,
        layers=layers,
        log_path=log_path,
        use_lbfgs=not args.skip_lbfgs
    )

    model.train(max_time=args.max_time, adam_it=args.adam_it)

    u_pred = model.predict(x, t)
    u_true = torch.tensor(u, dtype=torch.float32).to(device)

    error_u = relative_error(u_pred,  u_true)
    model.logging('L2 error u: %e' % (error_u))

    u_pred = model.predict(x, t)
    error_u = mean_squared_error(u_pred, u_true)
    model.logging('MSE error u: %e' % (error_u))

    data_output_path = "./output/prediction/diffreact1d-pinn-%s.npy" % (create_date)
    data_output = {'u': u_pred.detach().cpu().numpy()}
    np.save(data_output_path, data_output)
