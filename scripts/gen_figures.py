import seaborn as sns
import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def PlotUtilityVsEpsilon(data, outdir):
    """Plot accuracy vs epsilon for multiple experiments."""
    if "epsilon" not in data.columns or "accuracy" not in data.columns:
        return
    if "experiment_name" not in data.columns:
        data = data.copy()
        data["experiment_name"] = "exp"

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=data, x="epsilon", y="accuracy", hue="experiment_name", marker="o")
    plt.title("Model Accuracy vs Privacy Budget (ε)")
    plt.xlabel("Epsilon (ε)")
    plt.ylabel("Test Accuracy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Experiment")
    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    figpath = os.path.join(outdir, "utility_vs_epsilon.png")
    plt.savefig(figpath, dpi=200)
    plt.close()

def LoadResults(resultsdir):
    """Load all metrics CSVs from a results tree."""
    pattern = os.path.join(resultsdir, "**", "*.csv")
    filelist = sorted(glob.glob(pattern, recursive=True))
    datalist = []

    for path in filelist:
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue

        if "experiment_name" in frame.columns:
            expname = str(frame["experiment_name"].iloc[0])
        else:
            base = os.path.basename(os.path.dirname(path))
            expname = base if base else os.path.basename(path)

        lastrow = frame.tail(1).copy()
        lastrow["experiment_name"] = expname
        datalist.append(lastrow)

    if not datalist:
        return pd.DataFrame()

    return pd.concat(datalist, ignore_index=True)

def Main(resultsdir):
    """Entry point for figure generation."""
    data = LoadResults(resultsdir)
    if data.empty:
        print(f"No CSV results found under {resultsdir}")
        return

    outdir = os.path.join(resultsdir, "summary_plots")
    PlotUtilityVsEpsilon(data, outdir)
    print(f"Saved plots to {outdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        dest="resultsdir",
        type=str,
        default="results",
        help="Directory containing experiment results.",
    )
    args = parser.parse_args()
    Main(args.resultsdir)