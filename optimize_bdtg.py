import argparse
from pathlib import Path
from rdkit import Chem
import einops
import math
from objective import Objective, METRIC_MAXIMIZE
import wandb
from constants import FLOAT_TYPE, INT_TYPE
import torch.nn.functional as F
from utils import seed_everything
import utils
import torch
from tqdm import tqdm
import einops
from contextlib import contextmanager
from openbabel import openbabel
openbabel.obErrorLog.StopLogging()  # suppress OpenBabel messages

from lightning_modules import LigandPocketDDPM


import gpytorch
from torch import nn

def with_1d_support(transform_func):
    """Decorator to add 1D input support to transform methods."""
    def wrapper(self, data):
        is_1d = data.ndim == 1
        if is_1d:
            data = data.unsqueeze(0)
        
        result = transform_func(self, data)
        
        if is_1d:
            result = result.squeeze(0)
        
        return result
    return wrapper

class BaseScaler(nn.Module):
    """
    Base class for scalers. It's an "empty" scaler that does nothing.
    `fit`, `transform`, and `inverse_transform` can be overridden by subclasses.
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

    def fit(self, data):
        """Fits the scaler to the data. For this base class, it does nothing."""
        pass

    @with_1d_support
    def transform(self, data):
        """Transforms the data. For this base class, it returns the data as is."""
        return data

    @with_1d_support
    def inverse_transform(self, data):
        """Inverse transforms the data. For this base class, it returns the data as is."""
        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

class StandardScaler(BaseScaler):
    def __init__(self, feature_dim):
        super().__init__(feature_dim)
        self.register_buffer('mean', torch.zeros(feature_dim))
        self.register_buffer('std', torch.ones(feature_dim))

    def fit(self, data):
        """Computes the mean and standard deviation for scaling."""
        if data.ndim == 1:
            self.mean[:] = data.mean()
            self.std[:] = data.std().clamp(min=1e-8)
        else:
            self.mean[:] = data.mean(dim=0)
            self.std[:] = data.std(dim=0).clamp(min=1e-8)

    @with_1d_support
    def transform(self, data):
        """Standardizes the data."""
        device = data.device
        return (data - self.mean.to(device)) / self.std.to(device)

    @with_1d_support
    def inverse_transform(self, data):
        """Reverts the standardization."""
        device = data.device
        return data * self.std.to(device) + self.mean.to(device)

class MinMaxNegOneZeroScaler(BaseScaler):
    def __init__(self, feature_dim):
        super().__init__(feature_dim)
        self.register_buffer('min', torch.full((feature_dim,), float('inf')))
        self.register_buffer('max', torch.full((feature_dim,), float('-inf')))
    
    def fit(self, data):
        """Computes the min and max for scaling."""
        if data.ndim == 1:
            self.min[:] = data.min()
            self.max[:] = data.max()
        else:
            self.min[:] = data.min(dim=0).values
            self.max[:] = data.max(dim=0).values

    @with_1d_support
    def transform(self, data):
        """Scales the data to the range [-1, 0]."""
        device = data.device
        range = (self.max - self.min).clamp(min=1e-8)
        return -1 + (data - self.min.to(device)) / range.to(device)

    @with_1d_support
    def inverse_transform(self, data):
        """Reverts the scaling from [-1, 0] to the original range."""
        device = data.device
        range = (self.max - self.min)
        return (data + 1) * range.to(device) + self.min.to(device)

class ExactGpModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, x_dim):
        super(ExactGpModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module.base_kernel.lengthscale = (x_dim) ** 0.5

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ValueModel(nn.Module):
    def __init__(self, dimension, noise_level = 1e-4) -> None:
        super().__init__()
        self.dim = dimension
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.noise = noise_level
        self.likelihood.eval()

        model = ExactGpModel(None, None, self.likelihood, x_dim=dimension)

        self.model = model
        self.model.eval()
        self.model.requires_grad_(False)

        self.x_scaler = BaseScaler(dimension)
        self.y_scaler = BaseScaler(1)

        self.all_data = {
            "x": torch.empty(0, dimension, dtype=torch.float32, device='cpu'),
            "y": torch.empty(0, dtype=torch.float32, device='cpu')
        }

    def predict(self, x):
        device = x.device
        self.model.to(device)
        self.likelihood.to(device)
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            y_preds = self.likelihood(self.model(self.x_scaler.transform(x)))
        
        y_preds_mean = self.y_scaler.inverse_transform(y_preds.mean.to(device).unsqueeze(-1)).squeeze(-1)
        y_preds_var = y_preds.variance.to(device)

        return y_preds_mean.to(device), y_preds_var.to(device)

    # x: data points
    # y: lower is better
    def add_model_data(self, x, y):
        device = x.device

        self.all_data["x"] = torch.cat([self.all_data["x"], x.cpu()], dim=0)
        self.all_data["y"] = torch.cat([self.all_data["y"], y.cpu()], dim=0)

        self.x_scaler.fit(self.all_data["x"])
        self.y_scaler.fit(self.all_data["y"])
        
        self.model.set_train_data(
            inputs=self.x_scaler.transform(self.all_data['x']).to(device),
            targets=self.y_scaler.transform(self.all_data['y']).to(device),
            strict=False
        )

    def get_model_data(self):
        return self.all_data["x"], self.all_data["y"]

@contextmanager
def model_device_context(model: nn.Module, device: str):
    """
    A context manager to temporarily move a model to a specified device.

    Args:
        model (nn.Module): The model to move.
        device (str): The target device to move the model to (e.g., 'cuda', 'mps').
    """
    if 'cuda' in device and not torch.cuda.is_available():
        device = 'cpu'
        
    original_device = next(model.parameters()).device
    
    try:
        if original_device != device:
            model.to(device)
        yield
    finally:
        if next(model.parameters()).device != original_device:
            model.to(original_device)


def update_parameters(mu, sigma, noise, scores):
    ''' minimize score '''

    # noise: (T, B, D)
    # scores: (T, B)

    assert noise.shape[0] == scores.shape[0] == mu.shape[0] == sigma.shape[0]
    assert noise.shape[1] == scores.shape[1]

    T_dim = noise.shape[0]
    B_dim = noise.shape[1]
    D_dim = mu.shape[1]

    lr_mu = 1
    lr_sigma = 1/128
    

    mu = mu.clone()
    sigma = sigma.clone()

    for t in range(T_dim):
        
        scores_t = scores[t:].nanmean(0)
        valid_mask = ~torch.isnan(scores_t)
        scores_t = scores_t[valid_mask]
        z_t = noise[t][valid_mask]
        scores_t_normalized = (scores_t - scores_t.mean()) / (scores_t.std() + 1e-8)

        w = torch.exp( - scores_t_normalized) / torch.exp( - scores_t_normalized).sum()

        sigma[t] = 1 / (
            
            1/sigma[t] + lr_sigma * (
                (1/sigma[t])[None,:] * (z_t - mu[t,None,:]) * (z_t - mu[t,None,:]) * (1/sigma[t])[None,:] * \
                
                w[:,None]

            # sum over N
            ).sum(0)
        )
        
        mu[t] = mu[t] - lr_mu * (

            (z_t - mu[t][None,:]) * \
            
            scores_t_normalized[:,None]
        
        # mean over N
        ).mean(0)

    return mu, sigma

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=Path, default='checkpoints/crossdocked_fullatom_cond.ckpt')
    parser.add_argument('--pdbfile', type=str, default='example/5ndu.pdb')
    parser.add_argument('--pocket_pdbfile', type=str, default='example/5ndu_pocket.pdb')
    parser.add_argument('--ref_ligand', type=str, default='example/5ndu_C_8V2.sdf')
    parser.add_argument('--objective', type=str, default='qed')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--optimization_steps', type=int, default=1000)
    parser.add_argument('--diversify_from_timestep', type=int, default=100, help="diversify the [ref_ligand], lower timestep means closer to [ref_ligand], set -1 for no diversify (no reference ligand used).")

    parser.add_argument('--resi_list', type=str, nargs='+', default=None)
    parser.add_argument('--all_frags', action='store_true')
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--relax', action='store_true')
    parser.add_argument('--seed', type=int, default=0)


    args = parser.parse_args()
    seed = args.seed
    seed_everything(seed)

    if args.diversify_from_timestep == -1:
        args.diversify_from_timestep = None

    run = wandb.init(
        project=f"guide-sbdd",
        name=f"bdtg-gp-s{seed}-{args.objective}",
        config=args,
    )

    

    pdb_id = Path(args.pdbfile).stem

    device = "cuda"
    
    # Load model
    model = LigandPocketDDPM.load_from_checkpoint(
        args.checkpoint, map_location=device)
    model = model.to(device)

    batch_size = args.batch_size
    atom_dim = model.ddpm.n_dims + model.ddpm.atom_nf
    ref_mol = Chem.SDMolSupplier(args.ref_ligand)[0]
    ref_ligand = utils.prepare_ligands_from_mols([ref_mol]*batch_size, model.lig_type_encoder, device=model.device)
    num_atoms = ref_mol.GetNumAtoms()
    num_nodes_lig = torch.ones(batch_size, dtype=int) * num_atoms
    metrics = args.objective.split(";")
    objective_fn = Objective(metrics, args.pocket_pdbfile)
    
    dimension = num_atoms * atom_dim
    num_parameters = model.ddpm.T+2 if args.diversify_from_timestep is None else args.diversify_from_timestep+2
    mu = torch.zeros(num_parameters, dimension, dtype=torch.float32).to(device)
    sigma = torch.ones(num_parameters, dimension, dtype=torch.float32).to(device)
    optimization_steps = args.optimization_steps
    generator = torch.Generator(device=device).manual_seed(seed)

    value_model = ValueModel(
        dimension=num_atoms * atom_dim,
    )
    value_model.to(device)

    for optimization_idx in tqdm(range(optimization_steps), desc="Optimization Steps", leave=True):

        batch_mu = einops.repeat(mu, 'T D -> T B D', B=batch_size)
        batch_sigma = einops.repeat(sigma, 'T D -> T B D', B=batch_size)
        batch_noise = batch_mu + batch_sigma**0.5 * torch.randn(batch_mu.size(), generator=generator, device=device)
        batch_noise_norm = batch_noise.norm(dim=-1)
        batch_noise_projected = batch_noise / batch_noise_norm[:,:,None] * batch_noise_norm[:,:,None]
        given_noise_list = einops.rearrange(batch_noise_projected, 'T B (M N) -> T (B M) N', B=batch_size, M=num_atoms, N=atom_dim)

        with torch.inference_mode():
            molecules, pred_z0_lig_traj = model.generate_ligands(
                args.pdbfile, batch_size, args.resi_list, args.ref_ligand, ref_ligand,
                num_nodes_lig, args.sanitize, largest_frag=not args.all_frags,
                relax_iter=(200 if args.relax else 0),
                diversify_from_timestep=args.diversify_from_timestep,
                given_noise_list=given_noise_list,
            )
        assert len(molecules) == batch_size

        success_indices = []
        success_molecules = []
        for i, mol in enumerate(molecules):
            if mol is not None:
                success_indices.append(i)
                success_molecules.append(mol)
        batch_noise = batch_noise[:,success_indices,:]
        
        # Evaluate and save molecules
        objective_values = torch.zeros((num_parameters, len(success_indices)), dtype=torch.float32).to(device)
        objective_values_final , metrics_breakdown = objective_fn(success_molecules)
        
        # objective_values[:] = objective_values_final

        x = einops.rearrange(
            pred_z0_lig_traj[:,success_indices,:,:],
            "T B M N -> T B (M N)",
            M=num_atoms,
            N=atom_dim
        )

        value_model.add_model_data(x[-1],objective_values_final.to(device))
        for k, xk in enumerate(x[:-1]):
            estimated_values, variance = value_model.predict(xk)
            objective_values[k,:] = estimated_values
        objective_values[-1] = objective_values_final

        # want to minimize objective_values
        mu, sigma = update_parameters(mu, sigma, batch_noise, objective_values)

        # wandb log the score lower better
        log_dict = {
            "train/score_mean": objective_values[-1].mean(),
            "train/score_best": objective_values[-1].min(),
            "step": optimization_idx,
            "train/feasible_mol_rate": len(success_indices) / batch_size,
            "|mu|_2": mu.norm().item(),
            "|sigma|_2": sigma.sum(-1).mean().item(),
        }
        for k, v in metrics_breakdown.items():
            log_dict[f"train/{k}_mean"] = torch.tensor(v).mean()
            # Maximize the metric
            if METRIC_MAXIMIZE[k]:
                best = torch.tensor(v).max()
            else:
                best = torch.tensor(v).min()
            log_dict[f"train/{k}_best"] = best
        wandb.log(log_dict)
