import argparse

from DataAnalysis.src.utils import load_all_prices, formation_pipeline
from DataAnalysis.config import DATA_PATH, REFERENCE_ASSET, ADF_SIGNIFICANCE_LEVEL, SYMBOLS

def main():
    parser = argparse.ArgumentParser(description="Formation-step pipeline (reference-asset-based cointegration)")
    parser.add_argument("--data-path", type=str, default=DATA_PATH, help="Folder with <SYMBOL>.csv files (with a 'close' column).")
    parser.add_argument("--out-dir", type=str, default="./artifacts", help="Where to write formation outputs.")
    parser.add_argument("--reference", type=str, default=REFERENCE_ASSET, help="Reference asset symbol (default from config).")
    parser.add_argument("--adf-level", type=float, default=ADF_SIGNIFICANCE_LEVEL, help="ADF significance level (e.g. 0.10).")
    parser.add_argument("--min-obs", type=int, default=200, help="Minimum observations required to keep a series.")
    parser.add_argument("--top-k", type=int, default=6, help="Take top-K accepted coins by |Kendall τ| for candidate pairing.")
    args = parser.parse_args()

    prices = load_all_prices(args.data_path, SYMBOLS)
    artifacts = formation_pipeline(
        prices=prices,
        reference_symbol=args.reference,
        adf_level=args.adf_level,
        min_obs=args.min_obs,
        top_k=args.top_k,
        out_dir=args.out_dir,
    )
    print("Artifacts written:")
    for k, v in artifacts.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()