import os
import pickle
import re
from datetime import datetime
from typing import Dict, Any, List, Callable, Union, Tuple, Literal
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import combinations
import glob
import robomimic.dev.hparam.metrics.metric_fn as Metrics
import robomimic.dev.hparam.utils.pkl_utils as pkl_utils

def parse_hyperparameters(filename: str) -> Dict[str, Any]:
    """
    Parse hyperparameters from a pkl filename.
    
    Example filename formats:
    square_image_action_horizon_16_cls_f_ts_20_eta_0_nres_0_loss_mse_wg_10_nsmc_1_stdmc_0_iter_0_[timestamp].pkl
    square_image_action_horizon_16_inp_t_ts_20_eta_1_nres_0_iter_0_[timestamp].pkl
    
    Note: loss types are either 'mse' or 'weighted_mse'
    """
    base_name = filename.split('_2025-')[0]
    
    params = {}
    param_patterns = {
        'ts': r'ts_(\d+)',
        'eta': r'eta_(\d+)',
        'nres': r'nres_(\d+)',
        'iter': r'iter_(\d+)'
    }
    
    # Parse numeric parameters
    for param_name, pattern in param_patterns.items():
        match = re.search(pattern, base_name)
        if match:
            value = match.group(1)
            params[param_name] = int(value)
    
    # Parse loss type
    if 'weighted_mse' in base_name:
        params['loss'] = 'weighted_mse'
    elif 'mse' in base_name:
        params['loss'] = 'mse'
    
    # Parse additional parameters if loss is weighted_mse
    if params.get('loss') in ["mse", "weighted_mse"]:
        wg_match = re.search(r'wg_(\d+)', base_name)
        nsmc_match = re.search(r'nsmc_(\d+)', base_name)
        stdmc_match = re.search(r'stdmc_(\d+)', base_name)
        
        if wg_match:
            params['wg'] = int(wg_match.group(1))
        if nsmc_match:
            params['nsmc'] = int(nsmc_match.group(1))
        if stdmc_match:
            params['stdmc'] = int(stdmc_match.group(1))
    
    # Handle boolean flags
    params['cls'] = 'cls_f' in base_name
    params['square_image'] = 'square_image' in base_name
    params['inp'] = 'inp_t' in base_name
    
    return params

class HparamParser:
    """
    Parse hyperparameters from a pkl filename.
    """
    def __init__(self, param_patterns: Dict[str, str]):
        self.param_patterns = param_patterns

    def __call__(self, filename: str) -> Dict[str, Any]:
        params = {}
        for param_name, pattern in self.param_patterns.items():
            match = re.search(pattern, filename)
            if match:
                value = match.group(1)
                params[param_name] = float(value)

        return params

class DemoEvalSuite:
    """
    Evaluation Suite for a set of pkl files.
    
    This class processes a list of pickle files containing demonstration data and evaluates
    them using a provided evaluation function. It creates a DataFrame containing the evaluation
    results for each demonstration in each pickle file.

    Args:
        pkl_files (List[str]): List of paths to pickle files to evaluate
        eval_fn (Callable): Function that takes a demo dictionary and returns a metric value
        metric_name (str): Name to use for the metric column in the output DataFrame
    """
    def __init__(self, pkl_files: List[str], eval_fn: Callable, metric_name: str = "metric_value"):
        if not pkl_files:
            raise ValueError("pkl_files list cannot be empty")
        if not callable(eval_fn):
            raise TypeError("eval_fn must be callable")
        
        self.pkl_files = pkl_files
        self.eval_fn = eval_fn
        self.metric_name = metric_name
        self.df = self._process_results()
        self.metric_names = []

    def _process_results(self) -> pd.DataFrame:
        """
        Process all pkl files in the directory
        
        Returns:
            pd.DataFrame: DataFrame containing evaluation results with columns:
                - result_idx: Index of the pickle file
                - demo_idx: Index of the demonstration within the file  
                - metric_name: Value of the evaluation metric for this demo
        """
        rows = []

        for (result_idx, pkl_file) in enumerate(tqdm(self.pkl_files)):
            result_dict = pkl_utils.load_result_pkl_file(pkl_file)
            demo_result_dicts = result_dict["rollouts"]
            
            try:
                for demo_idx, demo_dict in demo_result_dicts.items():
                    row = {
                        'result_idx': result_idx,
                        'demo_idx': demo_idx,
                        self.metric_name: self.eval_fn(demo_dict)
                    }
                    rows.append(row)
            
            except Exception as e:
                self.eval_fn(demo_dict)
                print(f"Error processing {pkl_file} demo {demo_idx}: {str(e)}")
        
        return pd.DataFrame(rows)
    
    def add_metric(self, metric_name: str, eval_fn: Callable[[Dict[str, Any]], Any]) -> None:
        """
        Add a new metric to the DataFrame
        
        Args:
            metric_name: Name for the new metric column
            eval_fn: Function that computes the metric from a demo dictionary
        """
        print("Adding metric: ", metric_name)
        if metric_name not in self.df.columns:
            dummy_df = DemoEvalSuite(self.pkl_files, eval_fn, metric_name).df
            self.df = self.df.merge(dummy_df, on = ["result_idx", "demo_idx"])
            self.metric_names.append(metric_name)
        else:
            # Update existing column with new values
            dummy_df = DemoEvalSuite(self.pkl_files, eval_fn, metric_name).df
            self.df[metric_name] = dummy_df[metric_name]
        
    def aggregate_metric(self, metric_name: str, agg_suffix_name: str, agg_fn: Callable[[np.ndarray], float]):
        """
        Aggregate a metric across all demonstrations
        """
        self.df[f"{metric_name}_{agg_suffix_name}"] = self.df[metric_name].apply(agg_fn)

    def plot_metric_distributions(self, metrics: str | List[str], figsize: tuple[int, int] = (10, 6)) -> None:
        """
        Plot distributions of all metrics in the DataFrame using kernel density estimation.
        
        Args:
            figsize: Size of the figure (width, height)
        """
        if isinstance(metrics, str):
            metrics = [metrics]
        
        n_metrics = len(metrics)
        
        plt.figure(figsize=figsize)
        for i, metric in enumerate(metrics, 1):
            plt.subplot(1, n_metrics, i)
            sns.kdeplot(data=self.df[metric], fill=True)
            plt.title(f'{metric} Distribution')
            plt.xlabel(metric)
        plt.tight_layout()

    def plot_metric_correlation(self, figsize: tuple[int, int] = (8, 6)) -> None:
        """
        Plot correlation matrix between all metrics.
        
        Args:
            figsize: Size of the figure (width, height)
        """
        metrics = [self.metric_name] + self.metric_names
        if len(metrics) < 2:
            raise ValueError("Need at least 2 metrics to plot correlations")
        
        corr = self.df[metrics].corr()
        
        plt.figure(figsize=figsize)
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Metric Correlations')
        plt.tight_layout()

    def plot_metric_by_result(self, metric: str | None = None, 
                             kind: Literal['box', 'violin'] = 'box',
                             figsize: tuple[int, int] = (12, 6)) -> None:
        """
        Plot metric values grouped by result_idx using box or violin plots.
        
        Args:
            metric: Name of metric to plot. If None, uses first metric
            kind: Type of plot ('box' or 'violin')
            figsize: Size of the figure (width, height)
        """
        if metric is None:
            metric = self.metric_name
        elif metric not in self.df.columns:
            raise ValueError(f"Metric {metric} not found in DataFrame")
        
        plt.figure(figsize=figsize)
        if kind == 'box':
            sns.boxplot(data=self.df, x='result_idx', y=metric)
        else:
            sns.violinplot(data=self.df, x='result_idx', y=metric)
        plt.title(f'{metric} by Result')
        plt.xlabel('Result Index')
        plt.xticks(rotation=45)
        plt.tight_layout()

    def plot_metric_scatter(self, x_metric: str, y_metric: str, 
                             hue: str | None = None,
                             figsize: tuple[int, int] = (10, 6)) -> None:
        """
        Create scatter plot comparing two metrics.
        
        Args:
            x_metric: Name of metric to plot on x-axis
            y_metric: Name of metric to plot on y-axis
            hue: Column to use for point colors
            figsize: Size of the figure (width, height)
        """
        if x_metric not in self.df.columns or y_metric not in self.df.columns:
            raise ValueError("Specified metrics not found in DataFrame")
        
        plt.figure(figsize=figsize)
        sns.scatterplot(data=self.df, x=x_metric, y=y_metric, hue=hue, alpha=0.6)
        plt.title(f'{y_metric} vs {x_metric}')
        plt.tight_layout()

    def get_metric_stats(self, metric: str | None = None) -> pd.DataFrame:
        """
        Get summary statistics for specified metric grouped by result_idx.
        
        Args:
            metric: Name of metric to analyze. If None, uses first metric
            
        Returns:
            DataFrame with summary statistics for each result
        """
        if metric is None:
            metric = self.metric_name
        elif metric not in self.df.columns:
            raise ValueError(f"Metric {metric} not found in DataFrame")
        
        stats = self.df.groupby('result_idx')[metric].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(3)
        
        return stats
    
    def plot_metric_histogram_by_success(self, metric: str, figsize: tuple[int, int] = (15, 5), xlim: tuple[float, float] | None = None) -> None:
        """
        Plot histogram of specified metric grouped by success/failure.
        
        Args:
            metric: Name of metric to plot
            figsize: Size of the figure (width, height)
            xlim: Tuple of (min, max) for the x-axis
        """
        assert "success" in self.df.columns, "Success column not found in DataFrame"

        success_mask = self.df['success'] > 0.5
        plt.figure(figsize=figsize)
        plt.hist(self.df[metric][success_mask], bins=30, alpha=0.5, label='Success', density=True)
        plt.hist(self.df[metric][~success_mask], bins=30, alpha=0.5, label='Failure', density=True)
        plt.title(f'{metric} Distribution')
        plt.xlabel(metric)
        plt.ylabel('Density')
        plt.legend()
        if xlim:
            plt.xlim(xlim)
        plt.tight_layout()
        plt.show()
    
    def plot_success_rate_per_bin(self, metric: str, bins: int = 15, ax: plt.Axes | None = None, title: str | None = None, figsize: tuple[int, int] = (10, 6)) -> None:
        """
        Plot success rate per bin for a given metric.

        Args:
            metric: Name of metric to analyze
            bins: Number of bins to divide the metric range into
            ax: Optional matplotlib axes to plot on. If None, current axes will be used
            title: Optional title for the plot
            figsize: Size of the figure (width, height)
        """
        assert "success" in self.df.columns, "Success column not found in DataFrame"
        if metric not in self.df.columns:
            raise ValueError(f"Metric {metric} not found in DataFrame")

        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.gca()

        data = self.df[metric].values
        success_mask = self.df['success'].values > 0.5

        # Calculate bin edges
        bin_edges = np.linspace(np.min(data), np.max(data), bins+1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Initialize arrays to store success rates and sample counts
        success_rates = []
        sample_counts = []
        
        # Calculate success rate for each bin
        for i in range(len(bin_edges)-1):
            mask = (data >= bin_edges[i]) & (data < bin_edges[i+1])
            if np.sum(mask) > 0:  # Only include bins with samples
                success_rate = np.mean(success_mask[mask])
                success_rates.append(success_rate)
                sample_counts.append(np.sum(mask))
            else:
                success_rates.append(0)
                sample_counts.append(0)
        
        # Convert to numpy arrays
        success_rates = np.array(success_rates)
        sample_counts = np.array(sample_counts)
        
        # Plot success rate
        ax.bar(bin_centers, success_rates, width=(bin_edges[1]-bin_edges[0])*0.8)
        
        # Add sample count as text above each bar
        for x, y, count in zip(bin_centers, success_rates, sample_counts):
            if count > 0:  # Only show text for bins with samples
                ax.text(x, y + 0.05, str(count), ha='center', va='bottom')
        
        ax.set_ylim(0, 1.2)  # Leave room for sample count text
        ax.set_ylabel('Success Rate')
        ax.set_xlabel(metric)
        if title:
            ax.set_title(title)
        
        # Add horizontal line at mean success rate
        mean_success = np.mean(success_mask)
        ax.axhline(y=mean_success, color='r', linestyle='--', alpha=0.5, 
                   label=f'Mean Success Rate: {mean_success:.2f}')
        ax.legend()

    def plot_stacked_success_histogram(self, metric: str, bins: int = 15, ax: plt.Axes | None = None, 
                                     title: str | None = None, figsize: tuple[int, int] = (10, 6)) -> None:
        """
        Create a stacked histogram showing success/failure counts and rates for each bin.
        
        Args:
            metric: Name of metric to analyze
            bins: Number of bins to divide the metric range into
            ax: Optional matplotlib axes to plot on. If None, current axes will be used
            title: Optional title for the plot
            figsize: Size of the figure (width, height)
        """
        assert "success" in self.df.columns, "Success column not found in DataFrame"
        if metric not in self.df.columns:
            raise ValueError(f"Metric {metric} not found in DataFrame")
        
        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.gca()
        
        data = self.df[metric].values
        success_mask = self.df['success'].values > 0.5
        
        # Calculate bin edges
        bin_edges = np.linspace(np.min(data), np.max(data), bins+1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Initialize arrays to store success and failure counts
        success_counts = []
        failure_counts = []
        
        # Calculate counts for each bin
        for i in range(len(bin_edges)-1):
            mask = (data >= bin_edges[i]) & (data < bin_edges[i+1])
            success_count = np.sum(mask & success_mask)
            failure_count = np.sum(mask & ~success_mask)
            success_counts.append(success_count)
            failure_counts.append(failure_count)
        
        # Convert to numpy arrays
        success_counts = np.array(success_counts)
        failure_counts = np.array(failure_counts)
        
        # Create stacked bar plot
        width = (bin_edges[1]-bin_edges[0])*0.8
        ax.bar(bin_centers, success_counts, width=width, color='blue', label='Success')
        ax.bar(bin_centers, failure_counts, width=width, bottom=success_counts, color='red', label='Failure')
        
        # Add total count as text above each bar
        for x, s, f in zip(bin_centers, success_counts, failure_counts):
            total = s + f
            if total > 0:  # Only show text for bins with samples
                ax.text(x, s + f + 0.5, str(int(total)), ha='center', va='bottom')
                if s > 0:  # Show success rate if there are successes
                    success_rate = s / total
                    ax.text(x, s/2, f'{success_rate:.2f}', ha='center', va='center', color='white')
        
        ax.set_ylabel('Count')
        ax.set_xlabel(metric)
        if title:
            ax.set_title(title)
        ax.legend()

class HparamAnalyzer:
    """
    Analyzer for hyperparameter optimization results.
    
    Processes directories containing pickle files with experiment results and provides
    utilities for analyzing and visualizing hyperparameter effects.
    
    Attributes:
        result_dir (str): Directory containing result files
        eval_fn (Callable): Function to evaluate results
        level (str): Analysis level ('demo' or 'stat')
        filter (str): Filter string for result files
        metric_name (str): Name of the metric being analyzed
        results (List[Dict[str, Any]]): Processed results
        df (pd.DataFrame): DataFrame containing analyzed results
    """
    PLOT_TYPES = {'scatter', 'box', 'violin'}
    
    def __init__(self, result_dir: str, eval_fn: Callable, level: str, hparam_parser: HparamParser, filter: str, metric_name: str = "metric_value"):
        self.result_dir = result_dir
        self.eval_fn    = eval_fn
        self.level      = level
        self.filter     = filter
        self.metric_name = metric_name
        self.hparam_parser = hparam_parser

        self.results    = self._process_directory()
        self.df         = self._create_dataframe()

    def _process_directory(self) -> List[Dict[str, Any]]:
        """Process all pkl files in the directory"""
        print(f"Processing {self.result_dir} with filter {self.filter}")
        
        results = []
        if self.filter is not None:
            pkl_files = glob.glob(os.path.join(self.result_dir, f"*{self.filter}*.pkl"))
        else:
            pkl_files = glob.glob(os.path.join(self.result_dir, f"*.pkl"))
        
        for pkl_file in tqdm(pkl_files):
            pkl_path = os.path.join(self.result_dir, pkl_file)
            try:
                result = self.evaluate_result(pkl_path, self.eval_fn, self.level)
                results.append(result)
            except Exception as e:
                print(f"Error processing {pkl_file}: {str(e)}")
        
        return results
    
    def _create_dataframe(self) -> pd.DataFrame:
        """
        Create a DataFrame from the results, handling vector metrics by creating
        multiple rows for each vector element
        """
        rows = []
        for result in self.results:
            base_row = result['hyperparameters'].copy()
            metric_value = result['metric_value']
            
            if isinstance(metric_value, (np.ndarray, list)):
                # Create a row for each element in the vector
                for i, value in enumerate(metric_value):
                    row = base_row.copy()
                    row[self.metric_name] = value
                    row['metric_index'] = i
                    rows.append(row)
            else:
                base_row[self.metric_name] = metric_value
                base_row['metric_index'] = 0
                rows.append(base_row)
                
        return pd.DataFrame(rows)
    
    def evaluate_result(self, 
                        pkl_path: str, 
                        eval_fn: Callable[[Dict[str, Any]], Union[float, List[float]]], 
                        level: str = 'demo') -> Dict[str, Any]:
        """
        Evaluate a pkl file using the provided evaluation function.
        
        Args:
            pkl_path: Path to the pkl file
            eval_fn: Function that takes a demo_dict and returns a metric value or vector
        
        Returns:
            Dictionary containing evaluation results and metadata
        """
        assert level in ["demo", "stat"]

        # Load the pkl file
        with open(pkl_path, 'rb') as f:
            result_dict = pickle.load(f)
        
        # Get hyperparameters
        filename = os.path.basename(pkl_path)
        hyperparams = self.hparam_parser(filename)

        if level == "demo":
            demo_result_dicts = result_dict["rollouts"]
            metric_values     = [eval_fn(demo_dict) for (k, demo_dict) in demo_result_dicts.items()]
            metric_values     = np.array(metric_values)
        
        elif level == "stat":
            stat_dict = result_dict["stats"]
            metric_values = eval_fn(stat_dict)

        result = {
            'hyperparameters': hyperparams,
            'metric_value': metric_values,
            'filename': filename,
            'filepath': pkl_path
        }
        
        return result

    def plot_hparam_comparison(self, 
                            x_param: str, 
                            y_param: str | None = None, 
                            fixed_params: Dict[str, Any] | None = None,
                            plot_type: Literal['scatter', 'box', 'violin'] = 'scatter',
                            figsize: tuple[int, int] = (12, 6),
                            ylim: tuple[float, float] | None = None) -> pd.DataFrame:
        """
        Plot comparison of hyperparameters with support for vector metrics.

        Args:
            x_param: Hyperparameter to plot on x-axis.
            y_param: Hyperparameter to plot on y-axis (optional).
            fixed_params: Dictionary of hyperparameters to fix.
            plot_type: One of ['scatter', 'box', 'violin'] for vector metrics.
            figsize: Tuple specifying figure size (width, height).
            ax: Matplotlib Axes object for subplot (optional).
        """
        if plot_type not in self.PLOT_TYPES:
            raise ValueError(f"Invalid plot type. Must be one of: {self.PLOT_TYPES}")

        df = self.df.copy()

        if fixed_params:
            for param, value in fixed_params.items():
                df = df[df[param] == value]

        fig, ax = plt.subplots(figsize=figsize)

        if y_param is None:
            if plot_type == 'scatter':
                x_values = df[x_param].unique()
                for x_val in x_values:
                    subset = df[df[x_param] == x_val]
                    x_jittered = np.random.normal(x_val, 0.1, size=len(subset))
                    ax.scatter(x_jittered, subset[self.metric_name], alpha=0.5)
                ax.set_xlabel(x_param)
                ax.set_ylabel(self.metric_name)
            elif plot_type == 'box':
                sns.boxplot(data=df, x=x_param, y=self.metric_name, ax=ax)
            elif plot_type == 'violin':
                sns.violinplot(data=df, x=x_param, y=self.metric_name, ax=ax)
                sns.stripplot(data=df, x=x_param, y=self.metric_name, ax=ax, color="k", alpha=0.3)
            
            ax.set_ylabel(self.metric_name)
            ax.grid(True)
        else:
            pivot_df = df.pivot_table(index=x_param, columns=y_param, values=self.metric_name, aggfunc='mean')
            sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="viridis", ax=ax)

        ax.set_title(f"{self.metric_name} vs {x_param}")
        
        
        if ylim is not None:
            ax.set_ylim(ylim)

        return df
    
    def get_best_config(self, maximize: bool = True, aggregation: str = 'mean') -> Dict[str, Any]:
        """
        Get the hyperparameter configuration with the best metric value.
        
        Args:
            maximize: If True, find maximum value; if False, find minimum
            aggregation: How to aggregate vector metrics ('mean', 'median', 'max', 'min')
            
        Returns:
            Dictionary mapping parameter names to their optimal values
            
        Raises:
            ValueError: If dataframe is empty or invalid aggregation method is provided
        """
        if self.df.empty:
            raise ValueError("No data available to find best configuration")
            
        if aggregation not in ['mean', 'median', 'max', 'min']:
            raise ValueError(f"Invalid aggregation method: {aggregation}")
        
        df_grouped = self.df.groupby([col for col in self.df.columns 
                                    if col not in [self.metric_name, 'metric_index']])
        
        if aggregation == 'mean':
            agg_values = df_grouped[self.metric_name].mean()
        elif aggregation == 'median':
            agg_values = df_grouped[self.metric_name].median()
        elif aggregation == 'max':
            agg_values = df_grouped[self.metric_name].max()
        elif aggregation == 'min':
            agg_values = df_grouped[self.metric_name].min()
        
        idx = agg_values.idxmax() if maximize else agg_values.idxmin()
        return dict(zip(agg_values.index.names, idx))
    

if __name__ == "__main__":
    # result_dir = '/home/wjung85/Repo/projects/FastIL/logs/hparam'
    # success_analyzer = HparamAnalyzer(result_dir, eval_fn = Metrics.success_eval_fn, level="stat", filter = "inp")
    # success_analyzer.plot_hparam_comparison(x_param='nres', plot_type='violin', fixed_params={"inp": True, "eta": 0})

    cfg_dir = "/home/wjung85/Repo/projects/FastIL/logs/hparam_cfg/"
    hparam_parser = HparamParser(param_patterns = {
        'eta': r'eta_(\d+)',
        'weight': r'weight_(\d+\.?\d*)',
        'iter': r'iter_(\d+)'
    })
    hparam_analyzer_sparc_abs = HparamAnalyzer(result_dir = cfg_dir, hparam_parser = hparam_parser, filter="weight", metric_name="smoothness", 
                                        eval_fn = lambda x: np.quantile(Metrics.sparc_absolute_sliding_window_eval_fn(x, sliding_window_size=4), 0.5), 
                                        level="demo")
    