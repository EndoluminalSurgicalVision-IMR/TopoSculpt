from multiprocessing import Pool
import cripser as crip
import tcripser as trip
import numpy as np
import torch
import torch.nn.functional as F
import copy
from torch.optim import SGD, Adam, AdamW
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import logging

from skimage import measure

def cal_betti0(image):
    _, b0 = measure.label(image, connectivity=3, return_num=True)
    return b0

def diag_tidy(diag, eps=1e-1):
    new_diag = []
    for _, x in diag:
        if np.abs(x[0] - x[1]) > eps:
            new_diag.append((_, x))
    return new_diag


def crip_wrapper(X, D):
    return crip.computePH(X, maxdim=D)


def trip_wrapper(X, D):
    return trip.computePH(X, maxdim=D)


def get_roi(X, thresh=0.01):
    true_points = torch.nonzero(X >= thresh)
    corner1 = true_points.min(dim=0)[0]
    corner2 = true_points.max(dim=0)[0]
    roi = [slice(None, None)] + [slice(c1, c2 + 1) for c1, c2 in zip(corner1, corner2)]
    return roi


def get_differentiable_barcode(tensor, barcode):
    inf = barcode[barcode[:, 2] == np.finfo(barcode.dtype).max]
    fin = barcode[barcode[:, 2] < np.finfo(barcode.dtype).max]
    inf_indices = inf[:, 3:3 + tensor.ndim].astype(np.int64)
    inf_birth = tensor[tuple(inf_indices.transpose())]
    birth_indices = fin[:, 3:3 + tensor.ndim].astype(np.int64)
    death_indices = fin[:, 6:6 + tensor.ndim].astype(np.int64)
    births = tensor[tuple(birth_indices.transpose())]
    deaths = tensor[tuple(death_indices.transpose())]
    delta_p = (deaths - births)
    delta_p = [delta_p[fin[:, 0] == d] for d in range(tensor.ndim)]
    delta_p = [torch.sort(d, descending=True)[0] for d in delta_p]
    return inf_birth, delta_p


def log_message_to_file(message,log_file):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    log_entry = f"{timestamp} - {message}\n"
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry)

def TIB_refine(
        inputs, model, prior,
        lr, mse_lambda, softcldice_lambda, topo_lambda_A, topo_lambda_Z,
        output_path, model_path, filename, origin, spacing, direction,
        opt=torch.optim.Adam, num_its=100, construction='0', thresh=None, parallel=True,
        topo_lambda_Z_magnitude=10000.0,
        softcldice_lambda_magnitude=0.1,
        mse_lambda_magnitude=0.1,
        warmup_full_ph_steps=10, 
        ph_recompute_interval=5, 
        warmup_recompute_interval=1 

):
    spatial_xyz = list(inputs.shape[2:])
    device = inputs.device
    
    model.eval()
    with torch.no_grad():
        pred_unet = torch.softmax(model(inputs), 1).detach().squeeze()

    if thresh:
        roi_cached = get_roi(pred_unet[1:].sum(0).squeeze(), thresh)
    else:
        roi_cached = [slice(None, None)] + [slice(None, None) for _ in range(len(spatial_xyz))]

    model_topo = copy.deepcopy(model)
    model_topo.eval()
    optimiser = opt(model_topo.parameters(), lr=lr)

    max_dims = [len(b) for b in prior.values()]
    prior = {torch.tensor(c): torch.tensor(b) for c, b in prior.items()}
    PH = {'0': crip_wrapper, 'N': trip_wrapper}
    
    cached_bcodes_arr = None
    last_recompute_it = -1

    interval = ph_recompute_interval
    interval = max(1, int(interval))
    warmup_full_ph_steps = max(0, int(warmup_full_ph_steps))
    warmup_interval = max(1, int(warmup_recompute_interval))

    for it in range(num_its):
        start_time_iter = time.time()
        optimiser.zero_grad()

        if it > 0:
            prev_model_path = os.path.join(model_path, 'model_it' + str(it - 1).zfill(6) + '.pth')
            if os.path.exists(prev_model_path):
                prev_model = copy.deepcopy(model_topo)
                prev_model.load_state_dict(torch.load(prev_model_path, map_location=device))
                prev_model.eval()
                with torch.no_grad():
                    pred_unet = torch.softmax(prev_model(inputs), 1).detach().squeeze()

        if cached_bcodes_arr is None:
            need_recompute = True
            reason = "init"
        elif it < warmup_full_ph_steps:
            need_recompute = (it - last_recompute_it) >= warmup_interval
            reason = f"warmup_interval_{warmup_interval}" if need_recompute else f"warmup_reuse_from_{last_recompute_it}"
        elif (it - last_recompute_it) >= interval:
            need_recompute = True
            reason = f"interval_{interval}"
        else:
            need_recompute = False
            reason = f"reuse_from_{last_recompute_it}"
            
        if need_recompute:
            if thresh:
                roi_cached = get_roi(pred_unet[1:].sum(0).squeeze(), thresh)
            else:
                roi_cached = [slice(None, None)] + [slice(None, None) for _ in range(len(spatial_xyz))]

        print(f"Iteration {it}: need_recompute={need_recompute} ({reason})")

        outputs = torch.softmax(model_topo(inputs), 1).squeeze()
        outputs_roi = outputs[roi_cached]
        
        combos = torch.stack([outputs_roi[c.T].sum(0) for c in prior.keys()])
        combos = 1 - combos
        combos_arr = combos.detach().cpu().numpy().astype(np.float64)

        if need_recompute:
            if parallel:
                with torch.no_grad():
                    with Pool(len(prior)) as p:
                        bcodes_arr = p.starmap(PH[construction], zip(combos_arr, max_dims))
            else:
                with torch.no_grad():
                    bcodes_arr = [PH[construction](combo, max_dim) for combo, max_dim in zip(combos_arr, max_dims)]
            cached_bcodes_arr = bcodes_arr
            last_recompute_it = it
            logger.info(f"Recomputed bcodes_arr at iter {it} ({reason})")
        else:
            bcodes_arr = cached_bcodes_arr

        max_features = max([bcode_arr.shape[0] for bcode_arr in bcodes_arr]) if len(bcodes_arr) > 0 else 0
        bcodes = torch.zeros([len(prior), max(max_dims), max_features], requires_grad=False, device=device)
        for c_idx, (combo, bcode_np) in enumerate(zip(combos, bcodes_arr)):
            _, fin = get_differentiable_barcode(combo, bcode_np)
            for dim in range(min(bcodes.shape[1], len(fin))):
                if len(fin[dim]) > 0:
                    bcodes[c_idx, dim, :len(fin[dim])] = fin[dim]

        stacked_prior = torch.stack(list(prior.values()))
        stacked_prior.T[0] -= 1
        matching = torch.zeros_like(bcodes).detach().bool()
        for c_idx, combo_req in enumerate(stacked_prior):
            for dim in range(len(combo_req)):
                matching[c_idx, dim, slice(None, stacked_prior[c_idx, dim])] = True

        ## Define the full loss TIB loss function
        A = (1 - bcodes[matching]).sum()
        Z = bcodes[~matching].sum()
        mse = F.mse_loss(outputs, pred_unet)
        cldicecriterion = SoftlocalclDiceLossV2()
        cldice_loss = cldicecriterion(outputs, pred_unet)
        if it == warmup_full_ph_steps:
            topo_lambda_Z *= topo_lambda_Z_magnitude
            softcldice_lambda *= softcldice_lambda_magnitude
            mse_lambda *= mse_lambda_magnitude
        loss = topo_lambda_A * A + topo_lambda_Z * Z + softcldice_lambda * cldice_loss + mse_lambda * mse
        
        loss.backward()
        optimiser.step()
        
    return model_topo
