"""
Stat-level analysis functions for demo data
"""
from typing import Dict, List, Callable

def analyze_pkl_data(pkl_data: dict,
                     analysis_fn: Callable,
                     label_fns: List[Callable],
                     **kwargs) -> Dict[str, List]:
    """Analyze pkl data using specified analysis function.
    
    Args:
        pkl_data: Dictionary containing pkl data
        analysis_fn: Analysis function to use
        **kwargs: Additional arguments for analysis function
        
    Returns:
        Analysis results
    """
    demo_files = pkl_data["rollouts"]
    results = []
    
    for demo_idx, demo_file in demo_files.items():
        result = analysis_fn(demo_file, **kwargs)
        for label_fn in label_fns:
            result.update(label_fn(demo_idx, pkl_data))
        results.append(result)
        
    return results

def label_success(demo_idx: str, pkl_data: dict) -> Dict[str, List]:
    demo_idx = int(demo_idx.split("_")[1])
    return {"success": pkl_data["stats"]["Success_Rate"][demo_idx]}
