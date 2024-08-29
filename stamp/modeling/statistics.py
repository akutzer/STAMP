import sys
import argparse
from pathlib import Path
import os
from typing import Sequence
import pandas as pd
from matplotlib import pyplot as plt
import torch
import seaborn as sns
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.fitters.coxph_fitter import CoxPHFitter

from .marugoto.stats.categorical import categorical_aggregated_
from .marugoto.visualizations.roc import plot_multiple_decorated_roc_curves, plot_single_decorated_roc_curve
from .marugoto.visualizations.prc import plot_precision_recall_curves_, plot_single_decorated_prc_curve
from .marugoto.transformer.metric import ConcordanceIndex

def add_roc_curve_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "pred_csvs",
        metavar="PREDS_CSV",
        nargs="*",
        type=Path,
        help="Predictions to create ROC curves for.",
        default=[sys.stdin],
    )
    parser.add_argument(
        "--target-label",
        metavar="LABEL",
        required=True,
        type=str,
        help="The target label to calculate the ROC/PRC for.",
    )
    parser.add_argument(
        "--true-class",
        metavar="CLASS",
        required=True,
        type=str,
        help="The class to consider as positive for the ROC/PRC.",
    )
    parser.add_argument(
        "-o",
        "--outpath",
        metavar="PATH",
        required=True,
        type=Path,
        help=(
            "Path to save the statistics to."
        ),
    )

    parser.add_argument(
        "--n-bootstrap-samples",
        metavar="N",
        type=int,
        required=False,
        help="Number of bootstrapping samples to take for confidence interval generation.",
        default=1000
    )

    parser.add_argument(
        "--figure-width",
        metavar="INCHES",
        type=float,
        required=False,
        help="Width of the figure in inches.",
        default=3.8,
    )
    
    parser.add_argument(
        "--threshold-cmap",
        metavar="COLORMAP",
        type=plt.get_cmap,
        required=False,
        help="Draw Curve with threshold color.",
    )

    return parser


def read_table(file) -> pd.DataFrame:
    """Loads a dataframe from a file."""
    if isinstance(file, Path) and file.suffix == ".xlsx":
        return pd.read_excel(file)
    else:
        return pd.read_csv(file)


def compute_stats(pred_csvs: Sequence[Path], output_dir: Path, method: str):
    # read all the patient preds
    # and transform their true / preds columns into np arrays

    pred_df = pd.concat([pd.read_csv(p) for p in pred_csvs])

    if method == "cox":
        event_time, event, estimate = pred_df["follow_up_years"].to_numpy(), pred_df["event"].to_numpy(), pred_df["relative_risk"].to_numpy()
        event_time, event, estimate = torch.from_numpy(event_time), torch.from_numpy(event), torch.from_numpy(estimate)
        ci_fn = ConcordanceIndex("cox")
        ci = ci_fn(event_time, event, estimate)
        print(f"Concordance Index: {ci.item():.4f}")
    elif method == "logistic-hazard":
        event_time, event = pred_df["follow_up_years"].to_numpy(), pred_df["event"].to_numpy()
        event_time, event = torch.from_numpy(event_time), torch.from_numpy(event)

        mrl, isurv = pred_df["mean_residual_lifetime"].to_numpy(), pred_df["integrated_survival"].to_numpy()
        mrl, isurv = torch.from_numpy(mrl), torch.from_numpy(isurv)

        ci_mrl = ConcordanceIndex("mrl").calc_ci(event_time, event, mrl)
        ci_isurv = ConcordanceIndex("isurv").calc_ci(event_time, event, isurv)
        print(f"Concordance Index (MRL): {ci_mrl.item():.4f}")
        print(f"Concordance Index (ISURV): {ci_isurv.item():.4f}")
    exit(0)
    y_trues = [df[target_label] == true_class for df in preds_dfs]
    y_preds = [
        pd.to_numeric(df[f"{target_label}_{true_class}"]) for df in preds_dfs
    ]
    
    n_bootstrap_samples = 1000
    figure_width = 3.8 # inches
    threshold_cmap= plt.get_cmap()
    
    roc_curve_figure_aspect_ratio = 1.08
    fig, ax = plt.subplots(
        figsize=(figure_width, figure_width * roc_curve_figure_aspect_ratio),
        dpi=300,
    )

    if len(preds_dfs) == 1:
        plot_single_decorated_roc_curve(
                ax,
                y_trues[0],
                y_preds[0],
                title=f"{target_label} = {true_class}",
                n_bootstrap_samples=n_bootstrap_samples,
                threshold_cmap=threshold_cmap,
            )

    else:
        plot_multiple_decorated_roc_curves(
            ax,
            y_trues,
            y_preds,
            title=f"{target_label} = {true_class}",
            n_bootstrap_samples=None,
        )

    fig.tight_layout()
    stats_dir=(output_dir/"model_statistics")
    if not stats_dir.exists():
        stats_dir.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(stats_dir/f"AUROC_{target_label}={true_class}.svg")
    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(figure_width, figure_width * roc_curve_figure_aspect_ratio),
        dpi=300,
    )
    if len(preds_dfs) == 1:
        plot_single_decorated_prc_curve(
                ax,
                y_trues[0],
                y_preds[0],
                title=f"{target_label} = {true_class}",
                n_bootstrap_samples=n_bootstrap_samples
        )

    else:
        plot_precision_recall_curves_(
            ax,
            pred_csvs,
            target_label=target_label,
            true_label=true_class,
            outpath=stats_dir
        )

    fig.tight_layout()
    fig.savefig(stats_dir/f"AUPRC_{target_label}={true_class}.svg")
    plt.close(fig)

    categorical_aggregated_(pred_csvs,
                            target_label=target_label,
                            outpath=stats_dir)


def visualize_results(csv_files: Sequence[Path], output_dir: Path):
    """
    Function to visualize and save plots from 5-fold cross-validation CSV files.

    Parameters:
    csv_files (list): List of file paths to the CSV files containing the test results.
    output_dir (str): Directory path to save the output plots.
    """
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    
    data_frames = [pd.read_csv(f) for f in csv_files]
    combined_df = pd.concat(data_frames, ignore_index=True)

    # box plot
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='event', y='relative_risk', data=combined_df)
    plt.title('Relative Risk Distribution by Event Occurrence')
    plt.xlabel('Event Occurred')
    plt.ylabel('Relative Risk')
    risk_boxplot_path = os.path.join(output_dir, 'risk_distribution_boxplot.png')
    plt.savefig(risk_boxplot_path)
    plt.close()

    # km curve
    kmf = KaplanMeierFitter()
    median_risk = combined_df['relative_risk'].median()
    high_risk = combined_df[combined_df['relative_risk'] > median_risk]
    low_risk = combined_df[combined_df['relative_risk'] <= median_risk]

    plt.figure(figsize=(10, 6))
    for label, group in zip(["High Risk", "Low Risk"], [high_risk, low_risk]):
        kmf.fit(group['follow_up_years'], group['event'], label=label)
        kmf.plot_survival_function()

    plt.title('Survival Curves by Risk Group')
    plt.xlabel('Years')
    plt.ylabel('Survival Probability')
    plt.legend()
    survival_curve_path = os.path.join(output_dir, 'survival_curves.png')
    plt.savefig(survival_curve_path)
    plt.close()

    # Log-rank Test
    results = logrank_test(high_risk['follow_up_years'], low_risk['follow_up_years'], 
                           event_observed_A=high_risk['event'], event_observed_B=low_risk['event'])
    print(f"Log-rank Test p-value: {results.p_value:.4f}")

    # Save the test results in a text file
    with open(os.path.join(output_dir, 'log_rank_test_results.txt'), 'w') as f:
        f.write(f"Log-rank Test p-value: {results.p_value:.4f}")

    print(f"Plots and results saved to {output_dir}")

def KM_plot(output_dir, csv_file, time_col, event_col, group_col, groups=["high_risk","low_risk"]):
    KM_df = pd.read_csv(csv_file)

    group_A = KM_df[KM_df[group_col] == groups[0]]
    group_B = KM_df[KM_df[group_col] == groups[1]]

    kmf = KaplanMeierFitter()
    kmf.fit(group_A[time_col], event_observed=group_A[event_col], label=groups[0])
    ax = kmf.plot()

    kmf.fit(group_B[time_col], event_observed=group_B[event_col], label=groups[1])
    kmf.plot(ax=ax)
    KM_df['group_numeric'] = KM_df[group_col].apply(lambda x: 1 if x == groups[0] else 0)
    cph = CoxPHFitter()
    cph.fit(KM_df[[time_col, event_col, 'group_numeric']], duration_col=time_col, event_col=event_col)
    hazard_ratio = cph.hazard_ratios_['group_numeric'] 
    print(cph.confidence_intervals_)
    ci_lower = np.exp(cph.confidence_intervals_.loc['group_numeric', '95% lower-bound'])
    ci_upper = np.exp(cph.confidence_intervals_.loc['group_numeric', '95% upper-bound'])

    plt.xlabel('Time [years]')
    plt.ylabel("Biochemical recurrence free survival probability")
    results = logrank_test(group_A[time_col], group_B[time_col], event_observed_A=group_A[event_col], event_observed_B=group_B[event_col])
    p_value = results.p_value
    plt.annotate(f'Log-rank p-value: {p_value:.4f}\nHR (High vs Low): {hazard_ratio:.2f} [{ci_lower:.2f} - {ci_upper:.2f}]',
                 xy=(0.3, 0.2), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="white"))
    survival_curve_path = os.path.join(output_dir, 'survival_curves_s.png')
    plt.savefig(survival_curve_path)
    plt.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a ROC Curve.")
    args = add_roc_curve_args(parser).parse_args()
    compute_stats(pred_csvs=args.pred_csvs,
                  target_label=args.target_label,
                  true_class=args.true_class,
                  output_dir=args.outpath,
                  n_bootstrap_samples=args.n_bootstrap_samples,
                  figure_width=args.figure_width,
                  threshold_cmap=args.threshold_cmap)