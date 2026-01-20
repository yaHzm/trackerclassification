import wandb
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class WandBRunPlotter:
    def __init__(
        self,
        run_path: str,
        train_loss_key: str,
        eval_acc_key: str,
        eval_recall_key: str,
        output_dir: str = "wandb_plots",
        # Styling
    ) -> None:
        self.run_path = run_path
        self.train_loss_key = train_loss_key
        self.eval_acc_key = eval_acc_key
        self.eval_recall_key = eval_recall_key

        text_color: str = "#FAF5FC"
        loss_color: str = "#C192BE"
        acc_color: str = "#A6D9E7"
        recall_color: str = "#F0E1A6"

        self.text_color = text_color
        self.loss_color = loss_color
        self.acc_color = acc_color
        self.recall_color = recall_color

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        api = wandb.Api()
        self.run = api.run(run_path)

    def _get_history(self) -> pd.DataFrame:
        df = self.run.history(pandas=True)
        df = df.dropna(how="all")
        if "_step" not in df.columns:
            # W&B usually provides _step; fallback if needed
            df["_step"] = range(len(df))
        return df

    def _apply_axis_style(self, ax) -> None:
        # Transparent plot area
        ax.set_facecolor("none")

        # Text colors (labels, title, ticks)
        ax.title.set_color(self.text_color)
        ax.xaxis.label.set_color(self.text_color)
        ax.yaxis.label.set_color(self.text_color)
        ax.tick_params(colors=self.text_color)

        # Axis spines
        for spine in ax.spines.values():
            spine.set_color(self.text_color)

        # Legend text color (if legend exists)
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                t.set_color(self.text_color)
            leg.get_frame().set_facecolor("none")
            leg.get_frame().set_edgecolor(self.text_color)

    def plot_training_loss(self, filename: str = "training_loss.png") -> Path:
        df = self._get_history()
        if self.train_loss_key not in df.columns:
            raise KeyError(f"'{self.train_loss_key}' not found in W&B history. Available: {list(df.columns)}")

        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_alpha(0.0)  # transparent figure background

        ax.plot(df["_step"], df[self.train_loss_key], color=self.loss_color, linewidth=2.0)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Training loss")
        ax.set_title("Training Loss")

        self._apply_axis_style(ax)
        fig.tight_layout()

        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300, transparent=True)
        plt.close(fig)
        return out_path

    def plot_eval_metrics(self, filename: str = "eval_precision_recall.png") -> Path:
        df = self._get_history()

        missing = [k for k in (self.eval_acc_key, self.eval_recall_key) if k not in df.columns]
        if missing:
            raise KeyError(f"Missing keys in W&B history: {missing}. Available: {list(df.columns)}")

        # Keep only rows where at least one eval metric is present
        df_eval = df.loc[
            df[[self.eval_acc_key, self.eval_recall_key]].notna().any(axis=1),
            ["_step", self.eval_acc_key, self.eval_recall_key]
        ].copy()

        if df_eval.empty:
            raise ValueError("No non-NaN evaluation points found for the selected metrics.")

        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_alpha(0.0)

        ax.plot(df_eval["_step"], df_eval[self.eval_acc_key],
                color=self.acc_color, linewidth=2.0, label="Precision")
        ax.plot(df_eval["_step"], df_eval[self.eval_recall_key],
                color=self.recall_color, linewidth=2.0, label="Recall")

        ax.set_xlabel("Training step")
        ax.set_ylabel("Evaluation metric")
        ax.set_title("Evaluation: Precision & Recall")
        ax.legend(loc="best")

        self._apply_axis_style(ax)
        fig.tight_layout()

        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300, transparent=True)
        plt.close(fig)
        return out_path

    def plot_all(self) -> dict[str, Path]:
        return {
            "training_loss": self.plot_training_loss(),
            "eval_metrics": self.plot_eval_metrics(),
        }
