import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from natsort import natsorted
import argparse
from model.networks import UNet3D
from utils import load_itk_image_new, mkdir, InnerTransformer
from TIB import TIB_refine



class TopoSculptRefiner:
    def __init__(
        self,
        model_path: str = None,
        input_path: str = None,
        output_path: str = None,
        device: str = "cuda:0",
        prior: dict = None,
        lr: float = 1e-3,
        mse_lambda: float = 1000.0,
        softcldice_lambda: float = 1000.0,
        topo_lambda_A: float = 1.0,
        topo_lambda_Z: float = 1.0,
        num_its: int = 100,
        construction: str = 'N',
        thresh: float = 0.5,
        parallel: bool = True,
        model_config: dict = None,
        fmaps_degree: int = 16,
        opt: torch.optim.Optimizer = torch.optim.Adam,
        warmup_full_ph_steps: int = 50,
        warmup_recompute_interval: int = 2,
        ph_recompute_interval: int= 5,
        topo_lambda_Z_magnitude: float= 10000.0,
        softcldice_lambda_magnitude: float = 0.1,
        mse_lambda_magnitude: float = 0.1,
    ):
        self.model_path = model_path
        self.input_path = input_path
        self.output_path = output_path
        self.device_name = device
        self.prior = prior
        self.lr = lr
        self.mse_lambda = mse_lambda
        self.softcldice_lambda = softcldice_lambda
        self.topo_lambda_A = topo_lambda_A
        self.topo_lambda_Z = topo_lambda_Z
        self.num_its = num_its
        self.construction = construction
        self.thresh = thresh
        self.parallel = parallel
        self.model_config = model_config
        self.fmaps_degree = fmaps_degree
        self.opt = opt
        self.warmup_full_ph_steps = warmup_full_ph_steps
        self.warmup_recompute_interval = warmup_recompute_interval
        self.ph_recompute_interval = ph_recompute_interval
        self.topo_lambda_Z_magnitude = topo_lambda_Z_magnitude
        self.softcldice_lambda_magnitude = softcldice_lambda_magnitude
        self.mse_lambda_magnitude = mse_lambda_magnitude

    def run(self):
        mkdir(self.output_path)
        
        if self.prior is None:
            self.prior = {(1,): (1,)}
            
        if self.model_config is None:
            self.model_config = {
                'in_channels': 1,
                'out_channels': 2,
                'finalsigmoid': 1,
                'fmaps_degree': self.fmaps_degree,
                'fmaps_layer_number': 4,
                'layer_order': 'cip'
            }

        device = torch.device(self.device_name) if torch.cuda.is_available() else torch.device("cpu")
        model = UNet3D(**self.model_config).to(device)
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=lambda storage, loc: storage.cuda(0) if device.type == "cuda" else "cpu")
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"model does not exist : {self.model_path}")
        
        if os.path.isdir(self.input_path):
            file_list = [os.path.join(self.input_path, f) for f in os.listdir(self.input_path) if f.endswith(".nii.gz")]
            file_list = natsorted(file_list)
        elif os.path.isfile(self.input_path):
            file_list = [self.input_path]
        else:
            raise FileNotFoundError(f"input path does not exist: {self.input_path}")

        for idx, one_path in enumerate(file_list, start=1):
            base = os.path.basename(one_path)
            filename = base[:-7] if base.endswith('.nii.gz') else os.path.splitext(base)[0]
            case_name = filename

            case_output_path = os.path.join(self.output_path, case_name)
            mkdir(case_output_path)
            model_save_path = os.path.join(case_output_path, "models")
            mkdir(model_save_path)

            image, origin, spacing, direction = load_itk_image_new(one_path)

            image_tensor = InnerTransformer.AddChannel(image)
            image_tensor = InnerTransformer.ToTensorFloat32(image_tensor)
            image_tensor = InnerTransformer.AddChannel(image_tensor)
            image_tensor = image_tensor.to(device)

            model_TP = TIB_refine(
                inputs=image_tensor, 
                model=model, 
                prior=self.prior,
                lr=self.lr, 
                mse_lambda=self.mse_lambda,
                softcldice_lambda=self.softcldice_lambda,
                topo_lambda_A=self.topo_lambda_A,
                topo_lambda_Z=self.topo_lambda_Z,
                output_path=case_output_path, 
                model_path=model_save_path,    
                filename=filename,
                origin=origin, 
                spacing=spacing, 
                direction=direction,
                opt=self.opt, 
                num_its=self.num_its, 
                construction=self.construction, 
                thresh=self.thresh, 
                parallel=self.parallel,
                warmup_full_ph_steps=self.warmup_full_ph_steps,
                warmup_recompute_interval=self.warmup_recompute_interval,
                ph_recompute_interval=self.ph_recompute_interval,
                topo_lambda_Z_magnitude=self.topo_lambda_Z_magnitude,
                softcldice_lambda_magnitude=self.softcldice_lambda_magnitude,
                mse_lambda_magnitude=self.mse_lambda_magnitude,
            )




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Betti refinement for topological post-processing')
    
    parser.add_argument('--model_path', type=str,
                        help='Path to the trained model')
    parser.add_argument('--input_path', type=str, 
                        help='Path to input images (.nii.gz)')
    parser.add_argument('--output_path', type=str,
                        help='Path to save output results')
    parser.add_argument('--device', type=str, 
                        help='Device to use (e.g., cuda:0, cpu)')
    parser.add_argument('--lr', type=float,
                        help='Learning rate')
    parser.add_argument('--mse_lambda', type=float, 
                        help='MSE loss weight')
    parser.add_argument('--softcldice_lambda', type=float,
                        help='Soft clDice loss weight')
    parser.add_argument('--topo_lambda_A', type=float, 
                        help='Topological loss weight for matching features')
    parser.add_argument('--topo_lambda_Z', type=float,
                        help='Topological loss weight for violating features')
    parser.add_argument('--num_its', type=int,  
                        help='Number of iterations')
    parser.add_argument('--construction', type=str,  
                        choices=['0', 'N'], help='Connectivity type (0 or N)')
    parser.add_argument('--thresh', type=float,  
                        help='Threshold for ROI definition')
    parser.add_argument('--parallel', action='store_true', 
                        help='Use parallel processing')
    parser.add_argument('--fmaps_degree', type=int,  
                        help='Feature maps degree')
    parser.add_argument('--warmup_full_ph_steps', type=int,  
                        help='Warmup steps for full phase')
    parser.add_argument('--warmup_recompute_interval', type=int,  
                        help='Warmup interval for recompute')
    parser.add_argument('--ph_recompute_interval', type=int,  
                        help='Interval for phase recompute')
    parser.add_argument('--topo_lambda_Z_magnitude', type=float, 
                        help='Magnitude for topological loss Z')
    parser.add_argument('--softcldice_lambda_magnitude', type=float,  
                        help='Magnitude for soft clDice loss')
    parser.add_argument('--mse_lambda_magnitude', type=float,  
                        help='Magnitude for MSE loss')
    args = parser.parse_args()

    config = {
        'model_path': args.model_path,
        'input_path': args.input_path,
        'output_path': args.output_path,
        'device': args.device,
        'lr': args.lr,
        'mse_lambda': args.mse_lambda,
        'softcldice_lambda': args.softcldice_lambda,
        'topo_lambda_A': args.topo_lambda_A,
        'topo_lambda_Z': args.topo_lambda_Z,
        'opt': torch.optim.AdamW,
        'num_its': args.num_its,
        'construction': args.construction,
        'thresh': args.thresh,
        'parallel': args.parallel,
        'fmaps_degree': args.fmaps_degree,
        'warmup_full_ph_steps': args.warmup_full_ph_steps,
        'warmup_recompute_interval': args.warmup_recompute_interval,
        'ph_recompute_interval': args.ph_recompute_interval,
        'topo_lambda_Z_magnitude': args.topo_lambda_Z_magnitude,
        'softcldice_lambda_magnitude': args.softcldice_lambda_magnitude,
        'mse_lambda_magnitude': args.mse_lambda_magnitude   
    }

    refiner = TopoSculptRefiner(**config)
    refiner.run()