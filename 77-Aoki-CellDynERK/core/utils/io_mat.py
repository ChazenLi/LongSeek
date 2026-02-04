from pathlib import Path
import scipy.io as sio
import numpy as np
from typing import Tuple, Dict, Any, Union, Optional, List

def load_mat(path: str | Path) -> Tuple[Union[Dict, Any], str]:
    path = Path(path)
    try:
        data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        data = {k: v for k, v in data.items() if not k.startswith("__")}
        return data, "v5"
    except NotImplementedError:
        import h5py
        f = h5py.File(path, "r")
        return f, "v7.3"

def peek_mat(path: str | Path):
    data, kind = load_mat(path)
    print(f"File: {path}")
    print(f"Format: {kind}")
    
    if kind == "v5":
        print("Keys:", list(data.keys()))
        for k, v in data.items():
            try:
                shape = getattr(v, "shape", None)
                dtype = getattr(v, "dtype", type(v))
                print(f"  {k}: shape={shape}, dtype={dtype}")
                
                # Show sample values for small arrays
                if hasattr(v, "shape") and len(v.shape) <= 2 and np.prod(v.shape) <= 20:
                    print(f"    values: {v}")
                elif hasattr(v, "dtype") and v.dtype == np.uint8 and len(v.shape) == 2:
                    unique_vals = np.unique(v)
                    print(f"    unique values: {unique_vals}")
            except Exception as e:
                print(f"  {k}: type={type(v)}, error={e}")
    else:
        print("H5 keys:", list(data.keys()))
    
    return data, kind

def load_binary_mask(path: str | Path) -> np.ndarray:
    data, _ = load_mat(path)
    
    # Handle different key names
    mask_keys = ["manualRoi", "multiCellSegRoi", "topmanRoi", "tscratchRoi"]
    
    for key in mask_keys:
        if key in data:
            mask = data[key]
            if mask.dtype == np.uint8:
                return mask.astype(np.float32)
    
    raise ValueError(f"No valid binary mask found in {path}")

def load_measures(path: str | Path) -> Dict[str, Any]:
    data, _ = load_mat(path)
    
    result = {
        "num_frames": len(data["stats"]),
        "frames": []
    }
    
    for i, frame_stat in enumerate(data["stats"]):
        frame_info = {
            "frame_idx": i,
            "name": frame_stat.name,
            "msc": {
                "fmeasure": float(frame_stat.msc.fmeasure),
                "precision": float(frame_stat.msc.precision),
                "recall": float(frame_stat.msc.recall),
                "tp": int(frame_stat.msc.tp),
                "tn": int(frame_stat.msc.tn),
                "fp": int(frame_stat.msc.fp),
                "fn": int(frame_stat.msc.fn),
            },
            "tsc": {
                "fmeasure": float(frame_stat.tsc.fmeasure),
                "precision": float(frame_stat.tsc.precision),
                "recall": float(frame_stat.tsc.recall),
                "tp": int(frame_stat.tsc.tp),
                "tn": int(frame_stat.tsc.tn),
                "fp": int(frame_stat.tsc.fp),
                "fn": int(frame_stat.tsc.fn),
            },
            "top": {
                "fmeasure": float(frame_stat.top.fmeasure),
                "precision": float(frame_stat.top.precision),
                "recall": float(frame_stat.top.recall),
                "tp": int(frame_stat.top.tp),
                "tn": int(frame_stat.top.tn),
                "fp": int(frame_stat.top.fp),
                "fn": int(frame_stat.top.fn),
            }
        }
        result["frames"].append(frame_info)
    
    return result

def calculate_frame_weights(measures: Dict[str, Any], source: str = "msc", min_weight: float = 0.2) -> np.ndarray:
    """
    Calculate frame weights based on segmentation quality (F-measure).
    
    Frames with poor segmentation quality get lower weights in the loss function.
    w_t = clip(F_t, min_weight, 1.0)
    
    Args:
        measures: Measures dictionary from load_measures()
        source: Segmentation source ("msc", "tsc", or "top")
        min_weight: Minimum weight (default 0.2 to avoid completely ignoring frames)
        
    Returns:
        Array of weights for each frame
    """
    weights = []
    for frame in measures["frames"]:
        fmeasure = frame[source]["fmeasure"]
        weight = np.clip(fmeasure, min_weight, 1.0)
        weights.append(weight)
    
    return np.array(weights, dtype=np.float32)

def calculate_noise_model(measures: Dict[str, Any], source: str = "msc", scale_factor: float = 1e-6) -> np.ndarray:
    """
    Calculate noise level for each frame based on segmentation errors.
    
    Noise model: σ_t ∝ (FP_t + FN_t)
    More false positives/negatives → higher uncertainty → larger σ
    
    Args:
        measures: Measures dictionary from load_measures()
        source: Segmentation source ("msc", "tsc", or "top")
        scale_factor: Scaling factor to convert error count to noise level
        
    Returns:
        Array of noise levels (σ_t) for each frame
    """
    noise_levels = []
    for frame in measures["frames"]:
        fp = frame[source]["fp"]
        fn = frame[source]["fn"]
        total_error = fp + fn
        sigma = scale_factor * total_error
        noise_levels.append(sigma)
    
    return np.array(noise_levels, dtype=np.float32)

def integrate_multi_source_observations(observations_list: list, weights: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Integrate observations from multiple segmentation sources.
    
    Different segmentation algorithms (manual, MultiCellSeg, TScratch, Topman)
    may produce different wound area estimates. We combine them using
    either simple averaging or quality-weighted averaging.
    
    Args:
        observations_list: List of observation dictionaries from different sources
        weights: Optional weights for each source (e.g., based on quality). 
                 If None, use equal weights.
                 
    Returns:
        Integrated observation dictionary
    """
    if len(observations_list) == 0:
        raise ValueError("observations_list cannot be empty")
    
    n_sources = len(observations_list)
    
    if weights is None:
        weights_np = np.ones(n_sources) / n_sources
    else:
        weights_np = np.array(weights)
        weights_np = weights_np / weights_np.sum()
    
    integrated = {}
    
    for key in observations_list[0].keys():
        values = []
        for obs in observations_list:
            if key in obs:
                values.append(obs[key])
        
        if len(values) > 0:
            if isinstance(values[0], (int, float)):
                integrated[key] = float(np.average(values, weights=weights_np[:len(values)]))
            elif isinstance(values[0], np.ndarray):
                stacked = np.array(values)
                integrated[key] = np.average(stacked, weights=weights_np[:len(values)], axis=0)
            else:
                integrated[key] = values[0]
    
    return integrated

if __name__ == "__main__":
    measures_path = Path("CA/DATA/SN15/SN15/measures.mat")
    measures = load_measures(measures_path)
    print(f"Loaded {measures['num_frames']} frames")
    print(f"\nFirst frame:")
    print(f"  Name: {measures['frames'][0]['name']}")
    print(f"  MSC TP: {measures['frames'][0]['msc']['tp']}, FP: {measures['frames'][0]['msc']['fp']}")
    print(f"  TSC TP: {measures['frames'][0]['tsc']['tp']}, FP: {measures['frames'][0]['tsc']['fp']}")
    print(f"  TOP TP: {measures['frames'][0]['top']['tp']}, FP: {measures['frames'][0]['top']['fp']}")
