"""
train.py
"""

import os
import json
import logging
import traceback
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.strategies import DDPStrategy

from data_utils import prepare_data, create_forecasting_dataset
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import GNNEnhancedTFT, build_gnn_tft
from gnn_module import build_full_adjacency, get_encoder_cont_size

# Config
BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logsgnn_noise"

MAX_ENCODER_LENGTH    = 336  
MAX_PREDICTION_LENGTH = 24  

BATCH_SIZE   = 64
NUM_WORKERS  = 12
SEED         = 42

DATASET_PATH = str(BASE_DIR / "MergedDataset.csv")
TARGETS      = ["ontario_demand", "market_demand"]

# Cross-validation
N_CV_FOLDS         = 5        
VAL_HORIZON_HOURS  = 720      

TEST_HORIZON_HOURS = 720 

# Logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


# Gaussian noise augmentation callback
class GaussianNoiseCallback(pl.Callback):
    # Adds Gaussian noise to encoder during training batches.
    def __init__(self, std: float = 0.01):
        self.std = std

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        x, _ = batch
        if isinstance(x, dict) and "encoder_cont" in x:
            x["encoder_cont"] = (
                x["encoder_cont"]
                + torch.randn_like(x["encoder_cont"]) * self.std
            )


# Custom callback: record loss per epoch to plot later
def _make_val_x(n_train: int, n_val: int) -> list:

    if n_val > 0 and n_train > 0:
        val_step = n_train / n_val
        return [val_step * (i + 1) for i in range(n_val)]
    return list(range(1, n_val + 1))

#Loss callback cont.
class LossCurveCallback(pl.Callback):

    def __init__(self, fold_idx, target_col: str, save_dir: Path = None):
        self.fold_idx  = fold_idx
        self.target    = target_col
        self.save_dir  = Path(save_dir) if save_dir else LOGS_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)
        #Save the losses in a list
        self.train_losses: list = []
        self.val_losses:   list = []

    #After epoch, grab training and validation metrics
    def on_train_epoch_end(self, trainer, pl_module):
        v = trainer.callback_metrics.get("train_loss_epoch")
        if v is not None:
            self.train_losses.append(float(v))

    def on_validation_epoch_end(self, trainer, pl_module):
        v = trainer.callback_metrics.get("val_loss")
        if v is not None:
            self.val_losses.append(float(v))

    #Runs this at the end of training
    def on_fit_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        self._plot_and_save()

    #The plotting function
    def _plot_and_save(self):
        fig, ax = plt.subplots(figsize=(10, 6))

        n_train = len(self.train_losses)
        n_val   = len(self.val_losses)
        train_x = list(range(1, n_train + 1))
        val_x   = _make_val_x(n_train, n_val)

        ax.plot(train_x, self.train_losses,
                label="Train loss", linewidth=2,
                color="#1f77b4", marker="o", markersize=5, markevery=1)

        ax.plot(val_x, self.val_losses,
                label="Val loss", linewidth=2, linestyle="--",
                color="#ff7f0e", marker="s", markersize=5, markevery=1)

        #Mark best epoch
        if self.val_losses:
            best_vi    = int(min(range(len(self.val_losses)),
                               key=lambda i: self.val_losses[i]))
            best_epoch = val_x[best_vi]
            best_loss  = self.val_losses[best_vi]
            best_epoch = round(best_epoch)
            ax.axvline(best_epoch, color="green", linestyle=":", linewidth=1.8,
                       label=f"Best epoch ({round(best_epoch)}, loss={best_loss:.2f})")
            ax.scatter([best_epoch], [best_loss],
                       color="green", zorder=5, s=80, marker="*")

        #Axis and titles
        ax.set_xlabel("Epoch", fontsize=20)
        ax.set_ylabel("Quantile loss", fontsize=20)
        ax.set_title(f"{self.target} – Fold {self.fold_idx} loss curves",
                     fontsize=20)
        ax.tick_params(axis="both", labelsize=16)
        ax.legend(fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        fig.tight_layout()
        #Save file
        out = self.save_dir / f"{self.target}_fold{self.fold_idx}_loss.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        logger.info(f"Loss curve saved → {out}")
    #Grab fold, epoch, and losses data
    def get_history(self) -> dict:
        return {
            "fold":         self.fold_idx,
            "train_losses": self.train_losses,
            "val_losses":   self.val_losses,
        }


# Combined multi-fold loss plot
#Run after fold training is completed
def plot_all_folds(histories: list, target_col: str, save_dir: Path = None):
    save_dir = Path(save_dir) if save_dir else LOGS_DIR
    n        = len(histories)
    fig, axes = plt.subplots(n, 1, figsize=(10, 6 * n), sharex=False)
    if n == 1:
        axes = [axes]
    #Create subplots and combine them together
    for ax, h in zip(axes, histories):
        fold    = h["fold"]
        n_train = len(h["train_losses"])
        n_val   = len(h["val_losses"])
        train_x = list(range(1, n_train + 1))
        val_x   = _make_val_x(n_train, n_val)

        ax.plot(train_x, h["train_losses"],
                label="Train loss", linewidth=2, color="#1f77b4",
                marker="o", markersize=4, markevery=1)
        ax.plot(val_x, h["val_losses"],
                label="Val loss", linewidth=2, linestyle="--", color="#ff7f0e",
                marker="s", markersize=4, markevery=1)

        # Best epoch
        if h["val_losses"]:
            best_vi    = int(min(range(len(h["val_losses"])),
                               key=lambda i: h["val_losses"][i]))
            best_epoch = val_x[best_vi]
            best_loss  = h["val_losses"][best_vi]
            best_epoch = round(best_epoch)
            ax.axvline(best_epoch, color="green", linestyle=":", linewidth=1.6,
                       label=f"Best ({round(best_epoch)}, {best_loss:.2f})")
            ax.scatter([best_epoch], [best_loss],
                       color="green", zorder=5, s=60, marker="*")

        ax.set_title(f"Fold {fold}", fontsize=19)
        ax.set_xlabel("Epoch", fontsize=20)
        ax.set_ylabel("Quantile loss", fontsize=20)
        ax.tick_params(axis="both", labelsize=16)
        ax.legend(fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.suptitle(f"{target_col} – all folds", fontsize=18, y=1.01)
    fig.tight_layout()
    #Save and output
    out = save_dir / f"{target_col}_all_folds_loss.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Combined fold plot → {out}")


# GAT config
GNN_CFG = dict(
    gnn_hidden  = 32,
    gnn_heads   = 4,
    gnn_layers  = 2,
    gnn_dropout = 0.1,
)

# Build GAT-enhanced TFT
def build_model(train_ds):
    #calls the function from model.py
    return build_gnn_tft(
        train_ds,
        **GNN_CFG,
        learning_rate              = 3e-4,
        hidden_size                = 128,
        attention_head_size        = 4,
        dropout                    = 0.2,
        hidden_continuous_size     = 64,
        reduce_on_plateau_patience = 8,
        weight_decay               = 1e-3,
    )

#Print model summary
def log_model_summary(model, target_col: str, train_ds=None, fold_label: str = "fold1"):

    import os
    if int(os.environ.get("LOCAL_RANK", 0)) != 0:
        return

    summary = ModelSummary(model, max_depth=4)
    lines   = str(summary).splitlines()

    #Count the number of parameters in each component
    gnn_params = sum(
        p.numel() for n, p in model.named_parameters()
        if n.startswith("gat.") and p.requires_grad
    )
    tft_params = sum(
        p.numel() for n, p in model.named_parameters()
        if not n.startswith("gat.") and p.requires_grad
    )
    total_params = gnn_params + tft_params

    # FLOP count
    flop_str = "n/a"
    if train_ds is not None:
        try:
            import warnings
            FlopCounterMode = None
            for _mod in [
                "torch.utils.flop_counter",
                "torch.utils.flop_count",
                "torch.utils._flop_count",
            ]:
                try:
                    import importlib
                    _m = importlib.import_module(_mod)
                    # Try known class names across PyTorch versions
                    for _cls in ["FlopCounterMode", "FlopCounter", "flop_counter"]:
                        if hasattr(_m, _cls):
                            FlopCounterMode = getattr(_m, _cls)
                            logger.info(f"FlopCounterMode found: {_mod}.{_cls}")
                            break
                    if FlopCounterMode is not None:
                        break
                except (ImportError, AttributeError):
                    continue

            if FlopCounterMode is None:
                raise ImportError("FlopCounterMode not found — run: python -c \"import torch.utils.flop_counter; print(dir(torch.utils.flop_counter))\"")

            dl = train_ds.to_dataloader(train=False, batch_size=1, num_workers=0)
            batch_x, _ = next(iter(dl))
            device = next(model.parameters()).device
            batch_x = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in batch_x.items()}

            model.eval()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with FlopCounterMode(display=False) as fcm:
                    with torch.no_grad():
                        model(batch_x)
            total_flops = fcm.get_total_flops()
            flop_str = f"{total_flops:,}  ({total_flops / 1e9:.2f} GFLOPs)"
        except Exception as e:
            flop_str = f"unavailable ({e})"

    header = [
        "=" * 70,
        f"  Model summary — {target_col}",
        "=" * 70,
        f"  Total trainable params : {total_params:,}",
        f"  FeatureGATEncoder      : {gnn_params:,}  ({gnn_params/total_params*100:.1f}%)",
        f"  TFT backbone           : {tft_params:,}  ({tft_params/total_params*100:.1f}%)",
        f"  FLOPs per forward pass : {flop_str}",
        "-" * 70,
    ]

    full_text = "\n".join(header + lines)
    logger.info("\n" + full_text)

    out = LOGS_DIR / f"model_summary_{target_col}.txt"
    out.write_text(full_text)
    logger.info(f"Model summary written → {out}")


#Build training model
def build_trainer(
    *,
    target_col: str,
    run_name: str,
    checkpoint_dir: Path,
    max_epochs: int = 100,
    extra_callbacks: list = None,
):
    #Capture the best weights and restore back to best weights after early stopping
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        monitor="val_loss",
        mode="min",
        filename=f"{target_col}-{{epoch:03d}}-{{val_loss:.4f}}",
        save_top_k=1,
    )

    #Early stoppping with a patience of 10
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, min_delta=1e-5),
        LearningRateMonitor(logging_interval="epoch"),
        checkpoint_cb,
    ] + (extra_callbacks or [])

    #Create a trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=4,
        num_nodes=1,
        strategy=DDPStrategy(
            timeout=timedelta(seconds=3600),
            find_unused_parameters=True,
        ),
        gradient_clip_val=0.1,
        val_check_interval=0.5,
        logger=CSVLogger(str(LOGS_DIR), name=run_name),
        callbacks=callbacks,
        enable_progress_bar=True,
    )
    return trainer, checkpoint_cb

#Dataloader, load in the dataset with the specified parameters from above
def make_dataloaders(ds, train: bool, batch_size: int):
    return ds.to_dataloader(
        train=train,
        batch_size=batch_size if train else batch_size * 2,
        num_workers=NUM_WORKERS,
        shuffle=train,
        persistent_workers=(NUM_WORKERS > 0),
    )

# Main
if __name__ == "__main__":
    try:
        pl.seed_everything(SEED, workers=True)
        torch.set_float32_matmul_precision("high")

        # Create output directories
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"BASE_DIR : {BASE_DIR}")
        logger.info(f"LOGS_DIR : {LOGS_DIR}")
        logger.info(f"DATASET  : {DATASET_PATH}")

        for TARGET_COL in TARGETS:
            logger.info(f"\n{'='*60}")
            logger.info(f"  Target: {TARGET_COL}")
            logger.info(f"{'='*60}\n")

            # Load data
            df, unknown_reals, known_reals = prepare_data(DATASET_PATH, TARGET_COL)

            max_time_idx = df["time_idx"].max()

            # Exclude testing data from set to prevent the model from touching
            test_start_idx = max_time_idx - TEST_HORIZON_HOURS + 1
            cv_df = df[df["time_idx"] < test_start_idx].copy()
            cv_max = cv_df["time_idx"].max()

            logger.info(
                f"CV data range: idx 0 – {cv_max}  ({len(cv_df)} rows)"
            )
            logger.info(
                f"Test holdout : idx {test_start_idx} – {max_time_idx}  "
                f"({TEST_HORIZON_HOURS} rows)"
            )

            # Purging cross-validation
            # Divide into 6 segments, training + 5 validation
            usable_start = MAX_ENCODER_LENGTH + MAX_PREDICTION_LENGTH
            usable_end   = cv_max - VAL_HORIZON_HOURS - MAX_PREDICTION_LENGTH
            segment_size = (usable_end - usable_start) // (N_CV_FOLDS + 1)
            
            fold_starts = []
            for k in range(1, N_CV_FOLDS + 1):
                anchor    = usable_start + k * segment_size
                val_start = anchor - VAL_HORIZON_HOURS // 2
                # Nudge forward if too close to the start
                val_start = max(val_start, usable_start)
                fold_starts.append(int(val_start))

            # Log the windows so the user can verify distribution
            logger.info("CV validation windows (distributed across dataset):")
            for k, vs in enumerate(fold_starts, 1):
                ve        = vs + VAL_HORIZON_HOURS - 1
                ts_s      = df["timestamp"].iloc[vs].date()
                ts_e      = df["timestamp"].iloc[ve].date()
                vm        = df[TARGET_COL].iloc[vs:ve+1].mean()
                tm        = df[TARGET_COL].iloc[:vs].mean()
                shift_pct = (vm / tm - 1) * 100
                logger.info(
                    f"  Fold {k}: {ts_s} → {ts_e}  "
                    f"val_mean={vm:.0f}  train_mean={tm:.0f}  "
                    f"shift={shift_pct:+.1f}%"
                )

            # Model summary
            _summary_ds = create_forecasting_dataset(
                cv_df[cv_df["time_idx"] < fold_starts[0]].copy(),
                TARGET_COL, unknown_reals, known_reals,
                MAX_ENCODER_LENGTH, MAX_PREDICTION_LENGTH,
            )
            #Calls the function defined above
            log_model_summary(build_model(_summary_ds), TARGET_COL, train_ds=_summary_ds)
            del _summary_ds
            #Stores the results in the lists per fold
            cv_results     = []
            fold_histories = []
            
            for fold_idx, val_start in enumerate(fold_starts, start=1):
                val_end = val_start + VAL_HORIZON_HOURS - 1

                logger.info(f"\n--- CV Fold {fold_idx}/{N_CV_FOLDS} ---")
                logger.info(f"  Train: idx 0 – {val_start - 1}")
                logger.info(f"  Val:   idx {val_start} – {val_end}")

                fold_train_df = cv_df[cv_df["time_idx"] < val_start].copy()

                ctx_start = val_start - MAX_ENCODER_LENGTH
                fold_val_df = cv_df[
                    (cv_df["time_idx"] >= ctx_start) &
                    (cv_df["time_idx"] <= val_end)
                ].copy()
                #Catch if there is not enough training data
                if len(fold_train_df) < MAX_ENCODER_LENGTH + MAX_PREDICTION_LENGTH:
                    logger.warning(f"Fold {fold_idx}: training data too small, skipping.")
                    continue

                fold_train_ds = create_forecasting_dataset(
                    fold_train_df, TARGET_COL, unknown_reals, known_reals,
                    MAX_ENCODER_LENGTH, MAX_PREDICTION_LENGTH,
                )
                fold_val_ds = create_forecasting_dataset(
                    fold_val_df, TARGET_COL, unknown_reals, known_reals,
                    MAX_ENCODER_LENGTH, MAX_PREDICTION_LENGTH,
                    training_dataset=fold_train_ds,
                )

                logger.info(
                    f"  Train samples: {len(fold_train_ds)}  "
                    f"Val samples: {len(fold_val_ds)}"
                )
                #Build model
                tft      = build_model(fold_train_ds)
                loss_cb  = LossCurveCallback(fold_idx, TARGET_COL)
                noise_cb = GaussianNoiseCallback(std=0.01)
                ckpt_dir = LOGS_DIR / f"checkpoints_{TARGET_COL}_fold{fold_idx}"

                trainer, checkpoint_cb = build_trainer(
                    target_col=TARGET_COL,
                    run_name=f"{TARGET_COL}_fold{fold_idx}",
                    checkpoint_dir=ckpt_dir,
                    max_epochs=100,
                    extra_callbacks=[loss_cb, noise_cb],
                )
                #Train the model, pass the training and validation sets
                trainer.fit(
                    tft,
                    train_dataloaders=make_dataloaders(fold_train_ds, True,  BATCH_SIZE),
                    val_dataloaders  =make_dataloaders(fold_val_ds,   False, BATCH_SIZE),
                )
                #Grab the best validation checkpoint
                best_val = float(checkpoint_cb.best_model_score or float("inf"))
                logger.info(f"  Fold {fold_idx} best val_loss: {best_val:.4f}")
                logger.info(f"  Fold {fold_idx} checkpoint   : {checkpoint_cb.best_model_path}")

                cv_results.append({
                    "fold":          fold_idx,
                    "val_start_idx": val_start,
                    "val_end_idx":   val_end,
                    "best_val_loss": best_val,
                    "best_ckpt":     checkpoint_cb.best_model_path,
                })
                fold_histories.append(loss_cb.get_history())

            # Summarise CV results
            if cv_results:
                #Log per fold losses and checkpoints, as well as the mean and std of loss
                cv_df_results = pd.DataFrame(cv_results)
                cv_path = LOGS_DIR / f"cv_results_{TARGET_COL}.csv"
                cv_df_results.to_csv(cv_path, index=False)
                logger.info(f"CV results → {cv_path}")
                logger.info(
                    f"\nCV summary for {TARGET_COL}:\n"
                    f"  Mean val loss: {cv_df_results['best_val_loss'].mean():.4f} "
                    f"± {cv_df_results['best_val_loss'].std():.4f}\n"
                )
                plot_all_folds(fold_histories, TARGET_COL)

            # Final model: train on ALL non-test data
            logger.info(f"\n--- Final model training: {TARGET_COL} ---")
            #
            final_val_start = test_start_idx - VAL_HORIZON_HOURS * 5
            best_fvs = final_val_start
            best_shift = 999
            for offset in range(0, VAL_HORIZON_HOURS * 4, 24):
                candidate = final_val_start - offset
                if candidate < MAX_ENCODER_LENGTH + VAL_HORIZON_HOURS:
                    break
                tr_mean  = df[TARGET_COL].iloc[:candidate].mean()
                val_mean = df[TARGET_COL].iloc[candidate:candidate + VAL_HORIZON_HOURS].mean()
                shift    = abs(val_mean / tr_mean - 1) * 100
                if shift < best_shift:
                    best_shift = shift
                    best_fvs   = candidate
                if shift < 2.0:
                    break
            final_val_start = best_fvs
            logger.info(
                f"Final model val window: idx {final_val_start} – "
                f"{final_val_start + VAL_HORIZON_HOURS - 1}  "
                f"({df['timestamp'].iloc[final_val_start].date()} → "
                f"{df['timestamp'].iloc[final_val_start + VAL_HORIZON_HOURS - 1].date()})  "
                f"shift={best_shift:.1f}%"
            )

            actual_final_train_df = df[df["time_idx"] < final_val_start].copy()

            ctx_start_final = final_val_start - MAX_ENCODER_LENGTH
            final_val_df = df[
                (df["time_idx"] >= ctx_start_final) &
                (df["time_idx"] <  final_val_start + VAL_HORIZON_HOURS)
            ].copy()

            final_train_ds = create_forecasting_dataset(
                actual_final_train_df, TARGET_COL, unknown_reals, known_reals,
                MAX_ENCODER_LENGTH, MAX_PREDICTION_LENGTH,
            )
            final_val_ds = create_forecasting_dataset(
                final_val_df, TARGET_COL, unknown_reals, known_reals,
                MAX_ENCODER_LENGTH, MAX_PREDICTION_LENGTH,
                training_dataset=final_train_ds,
            )
            #Rebuild the model
            tft_final      = build_model(final_train_ds)
            loss_cb_final  = LossCurveCallback(fold_idx="final", target_col=TARGET_COL)
            noise_cb_final = GaussianNoiseCallback(std=0.01)
            ckpt_dir_final = LOGS_DIR / f"checkpoints_{TARGET_COL}_final"

            trainer_final, ckpt_cb_final = build_trainer(
                target_col=TARGET_COL,
                run_name=f"{TARGET_COL}_final",
                checkpoint_dir=ckpt_dir_final,
                max_epochs=150,
                extra_callbacks=[loss_cb_final, noise_cb_final],
            )

            trainer_final.fit(
                tft_final,
                train_dataloaders=make_dataloaders(final_train_ds, True,  BATCH_SIZE),
                val_dataloaders  =make_dataloaders(final_val_ds,   False, BATCH_SIZE),
            )
            #Load in the best weights
            best_ckpt = ckpt_cb_final.best_model_path
            if best_ckpt and Path(best_ckpt).exists():
                from gnn_module import build_full_adjacency, get_encoder_cont_size
                n_nodes = get_encoder_cont_size(final_train_ds)
                tft_final = GNNEnhancedTFT.load_from_checkpoint(
                    best_ckpt,
                    adj_matrix  = build_full_adjacency(n_nodes),
                    gnn_hidden  = GNN_CFG["gnn_hidden"],
                    gnn_heads   = GNN_CFG["gnn_heads"],
                    gnn_layers  = GNN_CFG["gnn_layers"],
                    gnn_dropout = GNN_CFG["gnn_dropout"],
                )
                logger.info(f"Reverted to best checkpoint: {best_ckpt}")
            else:
                logger.warning(
                    f"Best checkpoint not found at '{best_ckpt}' — "
                    "keeping last-epoch weights. "
                    "This can happen if EarlyStopping fired before the first "
                    "ModelCheckpoint save."
                )

            if trainer_final.global_rank == 0:
                best_path = ckpt_cb_final.best_model_path

                # Absolute paths for both output files
                txt_path  = BASE_DIR / f"best_model_{TARGET_COL}.txt"
                meta_path = BASE_DIR / f"model_meta_{TARGET_COL}.json"

                txt_path.write_text(best_path)
                logger.info(f"Best checkpoint : {best_path}")
                logger.info(f"Checkpoint path written → {txt_path}")

                meta = {
                    "target_col":             TARGET_COL,
                    "max_encoder_length":     MAX_ENCODER_LENGTH,
                    "max_prediction_length":  MAX_PREDICTION_LENGTH,
                    "test_start_idx":         int(test_start_idx),
                    "unknown_reals":          unknown_reals,
                    "known_reals":            known_reals,
                    # GNN config needed to reconstruct model at test time
                    "gnn_hidden":             GNN_CFG["gnn_hidden"],
                    "gnn_heads":              GNN_CFG["gnn_heads"],
                    "gnn_layers":             GNN_CFG["gnn_layers"],
                    "gnn_dropout":            GNN_CFG["gnn_dropout"],
                }
                meta_path.write_text(json.dumps(meta, indent=2))
                logger.info(f"Model meta written      → {meta_path}")

                logger.info(f"\nAll outputs for {TARGET_COL} are under: {LOGS_DIR}")

    except Exception:
        logger.error(f"CRITICAL FAILURE:\n{traceback.format_exc()}")
        raise
