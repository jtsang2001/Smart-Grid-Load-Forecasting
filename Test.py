"""
test.py
"""

import json
import logging
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pytorch_forecasting import TimeSeriesDataSet

from data_utils import prepare_data, create_forecasting_dataset
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import GNNEnhancedTFT, build_gnn_tft
from gnn_module import build_full_adjacency, get_encoder_cont_size
# Config
BASE_DIR  = Path(__file__).resolve().parent
LOGS_DIR  = BASE_DIR / "logsgnnnoise"

TARGETS   = ["ontario_demand", "market_demand"]
DATA_PATH = str(BASE_DIR / "MergedDataset.csv")

BATCH_SIZE = 64

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

#Add a small value to avoid divide by zero error
def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.abs(y_true) > 1e-6
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

#Prediction using the median quantile
def predict_with_model(
    model: GNNEnhancedTFT,
    dataloader,
) -> tuple[np.ndarray, np.ndarray]:

    model.eval()
    output = model.predict(
        dataloader,
        mode="prediction",          
        return_x=False,
        return_y=True,              
        trainer_kwargs=dict(
            accelerator="auto",
            devices=1,
            enable_progress_bar=False,
            logger=False,
        ),
    )

    #Log predicted vs actual
    preds   = output.output.cpu().numpy()    
    actuals = output.y[0].cpu().numpy()     
    #Flatten to 1D and return
    return preds.reshape(-1), actuals.reshape(-1)



# Main
if __name__ == "__main__":
    try:
        torch.manual_seed(42)

        all_metrics = []

        for TARGET_COL in TARGETS:
            logger.info(f"\n{'='*60}")
            logger.info(f"  Evaluating: {TARGET_COL}")
            logger.info(f"{'='*60}\n")


            # Load data from training
            meta_path = BASE_DIR / f"model_meta_{TARGET_COL}.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                MAX_ENCODER_LENGTH    = meta["max_encoder_length"]
                MAX_PREDICTION_LENGTH = meta["max_prediction_length"]
                test_start_idx        = meta["test_start_idx"]
                unknown_reals         = meta["unknown_reals"]
                known_reals           = meta["known_reals"]
                # GNN config (with safe defaults for backward compat)
                gnn_hidden  = meta.get("gnn_hidden",  32)
                gnn_heads   = meta.get("gnn_heads",    4)
                gnn_layers  = meta.get("gnn_layers",   2)
                gnn_dropout = meta.get("gnn_dropout", 0.1)
                logger.info(f"Loaded meta from {meta_path}")
            else:
                MAX_ENCODER_LENGTH    = 168
                MAX_PREDICTION_LENGTH = 24
                test_start_idx        = None
                unknown_reals         = None
                known_reals           = None
                gnn_hidden, gnn_heads, gnn_layers, gnn_dropout = 32, 4, 2, 0.1
                logger.warning(f"No meta file at {meta_path} – using defaults.")

            # Load + featurise data
            df, _unk, _kno = prepare_data(DATA_PATH, TARGET_COL)

            if unknown_reals is None:
                unknown_reals = _unk
            if known_reals is None:
                known_reals = _kno
            if test_start_idx is None:
                test_start_idx = df["time_idx"].max() - 720 + 1


            # Rebuilt the training dataset up to test set start so statistics from training set can be captured
            train_df = df[df["time_idx"] < test_start_idx].copy()

            training_dataset = create_forecasting_dataset(
                train_df,
                TARGET_COL,
                unknown_reals,
                known_reals,
                MAX_ENCODER_LENGTH,
                MAX_PREDICTION_LENGTH,
            )

            # Build test dataset, the last 336 hours are captured
            ctx_start  = test_start_idx - MAX_ENCODER_LENGTH
            test_df    = df[df["time_idx"] >= ctx_start].copy()

            test_dataset = TimeSeriesDataSet.from_dataset(
                training_dataset,
                test_df,
                #Create one sample per window
                predict=True,
                #Ensure chronological order
                stop_randomization=True,
            )
            logger.info(f"Test samples: {len(test_dataset)}")
            #Set this to not learn
            test_dl = test_dataset.to_dataloader(
                train=False, batch_size=BATCH_SIZE, num_workers=4
            )

            # Load best model checkpoint from training
            model_path_file = BASE_DIR / f"best_model_{TARGET_COL}.txt"
            with open(model_path_file) as f:
                best_model_path = f.read().strip()

            logger.info(f"Loading: {best_model_path}")

            # Reconstruct NxN adjacency matrix 
            n_nodes = get_encoder_cont_size(training_dataset)
            adj_matrix = build_full_adjacency(n_nodes)
            #Grab the values defined earlier
            model = GNNEnhancedTFT.load_from_checkpoint(
                best_model_path,
                adj_matrix  = adj_matrix,
                gnn_hidden  = gnn_hidden,
                gnn_heads   = gnn_heads,
                gnn_layers  = gnn_layers,
                gnn_dropout = gnn_dropout,
            )


            # Predictions
            preds_flat, actuals_flat = predict_with_model(model, test_dl)
            n_samples  = len(preds_flat) // MAX_PREDICTION_LENGTH
            if n_samples == 0:
                n_samples = 1
            preds_2d   = preds_flat.reshape(n_samples, MAX_PREDICTION_LENGTH)
            actuals_2d = actuals_flat.reshape(n_samples, MAX_PREDICTION_LENGTH)

            # Non-overlapping 24h windows
            stride    = MAX_PREDICTION_LENGTH
            idx_no    = list(range(0, n_samples, stride))
            p_no      = preds_2d[idx_no].reshape(-1)
            a_no      = actuals_2d[idx_no].reshape(-1)
            min_len   = min(len(p_no), len(a_no))
            p_no, a_no = p_no[:min_len], a_no[:min_len]

            # Metrics for non-overlapping windows
            mae      = mean_absolute_error(a_no, p_no)
            rmse     = np.sqrt(mean_squared_error(a_no, p_no))
            r2       = r2_score(a_no, p_no)
            mape_val = mape(a_no, p_no)
            # Log the information
            logger.info(f"\nMetrics – {TARGET_COL}  (non-overlapping 24h windows)")
            logger.info(f"  MAE  : {mae:.2f} MW")
            logger.info(f"  RMSE : {rmse:.2f} MW")
            logger.info(f"  MAPE : {mape_val:.2f} %")
            logger.info(f"  R²   : {r2:.4f}")

            horizon_maes = [
                mean_absolute_error(actuals_2d[:, h], preds_2d[:, h])
                for h in range(MAX_PREDICTION_LENGTH)
            ]
            logger.info(f"  Per-horizon MAE (h=1..24): "
                        f"min={min(horizon_maes):.1f}  max={max(horizon_maes):.1f}  "
                        f"h1={horizon_maes[0]:.1f}  h24={horizon_maes[-1]:.1f}")

            all_metrics.append(
                {
                    "target":   TARGET_COL,
                    "MAE_MW":   round(mae, 2),
                    "RMSE_MW":  round(rmse, 2),
                    "MAPE_pct": round(mape_val, 3),
                    "R2":       round(r2, 4),
                    "MAE_h1":   round(horizon_maes[0], 2),
                    "MAE_h24":  round(horizon_maes[-1], 2),
                }
            )


            # Plots 
            LOGS_DIR.mkdir(parents=True, exist_ok=True)

            # Forecast overlay
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(a_no, label="Actual",    linewidth=1.2)
            ax.plot(p_no, label="Predicted", linewidth=1.2, linestyle="--")
            ax.set_title(f"{TARGET_COL} – Forecast vs Actual (non-overlapping 24h windows)", fontsize=18)
            ax.set_xlabel("Hour", fontsize=16)
            ax.set_ylabel("Demand (MW)", fontsize=16)
            ax.legend(fontsize=14)
            ax.tick_params(axis="both", labelsize=14)
            fig.tight_layout()
            out = LOGS_DIR / f"{TARGET_COL}_forecast.png"
            fig.savefig(out, dpi=200)
            plt.close(fig)
            logger.info(f"Forecast plot → {out}")

            # Residual histogram
            residuals = p_no - a_no
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(residuals, bins=60, edgecolor="white", linewidth=0.4)
            ax.axvline(0, color="red", linestyle="--")
            ax.set_title(f"{TARGET_COL} – Residual distribution", fontsize=18)
            ax.set_xlabel("Residual (MW)", fontsize=16)
            ax.set_ylabel("Count", fontsize=16)
            fig.tight_layout()
            out = LOGS_DIR / f"{TARGET_COL}_residuals.png"
            fig.savefig(out, dpi=200)
            plt.close(fig)
            logger.info(f"Residuals plot → {out}")

            # Actual vs Predicted scatter
            fig, ax = plt.subplots(figsize=(7, 7))
            lims = [min(a_no.min(), p_no.min()), max(a_no.max(), p_no.max())]
            ax.scatter(a_no, p_no, alpha=0.3, s=4)
            ax.plot(lims, lims, "r--", linewidth=1)
            ax.set_xlabel("Actual (MW)", fontsize=16)
            ax.set_ylabel("Predicted (MW)", fontsize=16)
            ax.set_title(f"{TARGET_COL} – Scatter (R²={r2:.3f})", fontsize=18)
            fig.tight_layout()
            out = LOGS_DIR / f"{TARGET_COL}_scatter.png"
            fig.savefig(out, dpi=200)
            plt.close(fig)
            logger.info(f"Scatter plot   → {out}")

            # Per-horizon MAE curve
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(range(1, MAX_PREDICTION_LENGTH + 1), horizon_maes,
                    marker="o", markersize=4, linewidth=1.5)
            ax.set_xlabel("Forecast horizon (h)", fontsize=16)
            ax.set_ylabel("MAE (MW)", fontsize=16)
            ax.set_title(f"{TARGET_COL} – MAE by forecast horizon", fontsize=18)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            out = LOGS_DIR / f"{TARGET_COL}_horizon_mae.png"
            fig.savefig(out, dpi=200)
            plt.close(fig)
            logger.info(f"Horizon MAE    → {out}")

        # Save metrics
        metrics_df = pd.DataFrame(all_metrics)
        out = LOGS_DIR / "test_metrics.csv"
        metrics_df.to_csv(out, index=False)
        logger.info(f"\nAll metrics written → {out}")
        logger.info(f"\n{metrics_df.to_string(index=False)}")

    except Exception:
        logger.error(f"TEST – CRITICAL FAILURE:\n{traceback.format_exc()}")
        raise
