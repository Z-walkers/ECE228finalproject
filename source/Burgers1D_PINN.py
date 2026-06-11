import time
import argparse
import numpy as np

from pdes import Burgers1D
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
                 x_l_bound, x_r_bound, t_bound,
                 x_eqns, t_eqns, u_eqns,
                 x_data, t_data, u_data,
                 x_test, t_test, u_test,
                 nu, batch_size, layers, log_path):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.x_init = torch.tensor(x_init, dtype=torch.float32).to(self.device)
        self.t_init = torch.tensor(t_init, dtype=torch.float32).to(self.device)
        self.u_init = torch.tensor(u_init, dtype=torch.float32).to(self.device)

        self.x_l = torch.tensor(x_l_bound, dtype=torch.float32).to(self.device)
        self.x_r = torch.tensor(x_r_bound, dtype=torch.float32).to(self.device)
        self.t_b = torch.tensor(t_bound, dtype=torch.float32).to(self.device)

        self.x_data = torch.tensor(x_data, dtype=torch.float32).to(self.device)
        self.t_data = torch.tensor(t_data, dtype=torch.float32).to(self.device)
        self.u_data = torch.tensor(u_data, dtype=torch.float32).to(self.device)

        self.x_eq = torch.tensor(x_eqns, dtype=torch.float32).to(self.device)
        self.t_eq = torch.tensor(t_eqns, dtype=torch.float32).to(self.device)

        self.x_test = torch.tensor(x_test, dtype=torch.float32).to(self.device)
        self.t_test = torch.tensor(t_test, dtype=torch.float32).to(self.device)
        self.u_test = torch.tensor(u_test, dtype=torch.float32).to(self.device)

        self.nu = nu
        input_stats = torch.cat([self.x_eq, self.t_eq], dim=1)
        self.net = NeuralNet(
            layers,
            output_activation="linear",
            input_mean=input_stats.mean(dim=0, keepdim=True).cpu(),
            input_std=input_stats.std(dim=0, keepdim=True).cpu(),
        ).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)
        self.log_path = log_path
        self.batch_size = batch_size
        self.best_eval = float("inf")
        self.best_state = snapshot_model_state(self.net)
        self.best_it = -1

    def forward(self, x, t):
        return self.net(x, t)
    
    def compute_loss(self, x_eq, t_eq):
        x_eq = x_eq.clone().detach().requires_grad_(True)
        t_eq = t_eq.clone().detach().requires_grad_(True)
        u_eq = self.forward(x_eq, t_eq)
        f_eq = Burgers1D(x_eq, t_eq, u_eq, self.nu)
        return torch.mean(f_eq ** 2)

    def boundary_loss(self):
        u_l = self.forward(self.x_l, self.t_b)
        u_r = self.forward(self.x_r, self.t_b)
        return torch.mean((u_l - u_r) ** 2)

    def init_loss(self):
        u0 = self.forward(self.x_init, self.t_init)
        return torch.mean((u0 - self.u_init) ** 2)

    def data_loss(self):
        u_d = self.forward(self.x_data, self.t_data)
        return torch.mean((u_d - self.u_data) ** 2)
    
    def train(self, max_time, adam_it):
        N_eqns = self.t_eq.shape[0]
        start_time = time.time()
        total_time = 0

        for it in range(adam_it):
            idx = np.random.choice(N_eqns, min(self.batch_size, N_eqns), replace=False)
            idx = idx.astype(np.int64).tolist()

            x_eq_batch = self.x_eq[idx]
            t_eq_batch = self.t_eq[idx]

            loss_init = self.init_loss()
            loss_bnd = self.boundary_loss()
            loss_eq = self.compute_loss(x_eq_batch, t_eq_batch)
            loss_data = self.data_loss()
            loss = loss_init + 100 * loss_data + loss_eq + loss_bnd
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if it % 10 == 0:
                elapsed = time.time() - start_time
                total_time += elapsed / 3600.0

                log_item = (
                    f"It: {it}, Loss: {loss.item():.3e}, "
                    f"Init: {loss_init.item():.3e}, "
                    f"Bound: {loss_bnd.item():.3e}, "
                    f"Eq: {loss_eq.item():.3e}, "
                    f"Data: {loss_data.item():.3e}, "
                    f"Time: {elapsed:.2f}s"
                )

                self.logging(log_item)
                start_time = time.time()

            if it % 100 == 0:
                u_pred = self.predict(self.x_test, self.t_test)
                error = torch.mean((u_pred - self.u_test) ** 2).item()
                self.logging(f"Eval: It={it}, MSE={error:.6e}")
                if error < self.best_eval:
                    self.best_eval = error
                    self.best_it = it
                    self.best_state = snapshot_model_state(self.net)
                    self.logging(f"Best Eval: It={it}, MSE={error:.6e}")

            if total_time > max_time:
                break
        restore_model_state(self.net, self.best_state)
        self.logging(f"Restored best checkpoint: It={self.best_it}, MSE={self.best_eval:.6e}")

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adam-it", type=int, default=20000)
    parser.add_argument("--max-time", type=float, default=10)
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

    log_path = f"./output/log/burgers1D-pinn-{create_date}.txt"
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

    idx_init = np.random.choice(len(x_init), min(N_init, len(x_init)), replace=False)
    x_init, t_init, u_init = x_init[idx_init], t_init[idx_init], u_init[idx_init]

    idx_bound = np.random.choice(len(t_bound), min(N_bound, len(t_bound)), replace=False)
    x_l_bound, x_r_bound, t_bound = x_l_bound[idx_bound], x_r_bound[idx_bound], t_bound[idx_bound]

    idx_data = np.random.choice(len(x), min(N_data, len(x)), replace=False)
    x_data, t_data, u_data = x[idx_data], t[idx_data], u[idx_data]

    idx_test = np.random.choice(len(x), min(N_test, len(x)), replace=False)
    x_test, t_test, u_test = x[idx_test], t[idx_test], u[idx_test]

    model = PhysicsInformedNN(
        x_init, t_init, u_init,
        x_l_bound, x_r_bound, t_bound,
        x, t, u,
        x_data, t_data, u_data,
        x_test, t_test, u_test,
        nu, batch_size, layers, log_path
    )

    model.train(max_time=args.max_time, adam_it=args.adam_it)
    u_pred = model.predict(x, t)
    u = torch.tensor(u, dtype=torch.float32).to(device)

    error_u = relative_error(u_pred, u)
    model.logging('L2 error u: %e' % (error_u))

    u_pred = model.predict(x, t)
    error_u = mean_squared_error(u_pred, u)
    model.logging('MSE error u: %e' % (error_u))

    data_output_path = "./output/prediction/burgers1d-pinn-%s.npy" % (create_date)
    data_output = {'u': u_pred.detach().cpu().numpy()}
    np.save(data_output_path, data_output)
