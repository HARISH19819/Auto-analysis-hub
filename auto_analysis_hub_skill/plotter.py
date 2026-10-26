"""
plotter.py: Dynamic matplotlib/seaborn 4-panel digital chart generator.
Renders high-contrast, dark-mode PNG charts in memory without saving to disk.
Returns base64 strings for frontend rendering and raw bytes for Gemma 4 multimodal analysis.
"""

from __future__ import annotations
import base64
import io
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns


# Dark Theme Styling Constants
THEME = {
    "fig_face": "#0b0f19",      # Deep dark navy
    "ax_face": "#131b2e",       # Subtle card dark
    "grid": "#1e293b",          # Slate grid
    "text": "#f1f5f9",          # Bright slate text
    "subtext": "#94a3b8",       # Slate 400
    "accent_cyan": "#38bdf8",   # Primary glow
    "accent_indigo": "#818cf8", # Secondary purple
    "accent_emerald": "#34d399",# Health green
    "accent_rose": "#f43f5e",   # Anomaly / Outlier red
    "accent_amber": "#fbbf24",  # Warning amber
}


class VisualPlotter:
    """Dynamic 4-panel visual chart generator for tabular datasets."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._classify_columns()

    def _classify_columns(self):
        """Identify numerical, categorical, and datetime candidate columns."""
        df = self.df
        self.num_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.date_col: Optional[str] = None

        for col in df.columns:
            dtype_str = str(df[col].dtype).lower()
            if "datetime" in dtype_str or pd.api.types.is_datetime64_any_dtype(df[col]):
                if self.date_col is None:
                    self.date_col = col
            elif pd.api.types.is_numeric_dtype(df[col]):
                if df[col].dropna().nunique() > 1:
                    self.num_cols.append(col)
            else:
                # Check for datetime parsing
                sample = df[col].dropna().head(10)
                if len(sample) > 0 and self.date_col is None:
                    try:
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            pd.to_datetime(sample, errors="raise")
                        self.date_col = col
                        continue
                    except Exception:
                        pass
                self.cat_cols.append(col)

    def _apply_theme(self, ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = ""):
        """Apply sleek dark glassmorphism styling to a matplotlib axis."""
        ax.set_facecolor(THEME["ax_face"])
        ax.set_title(title, fontsize=12, fontweight="bold", color=THEME["text"], pad=14)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=10, color=THEME["subtext"], labelpad=8)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=10, color=THEME["subtext"], labelpad=8)

        ax.tick_params(colors=THEME["subtext"], labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(THEME["grid"])
            spine.set_linewidth(1.0)
        ax.grid(True, linestyle="--", linewidth=0.6, color=THEME["grid"], alpha=0.7)

    def _fig_to_bytes_and_base64(self, fig: plt.Figure) -> Tuple[bytes, str]:
        """Convert matplotlib Figure to PNG bytes and base64 data URI."""
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=110,
            bbox_inches="tight",
            facecolor=THEME["fig_face"],
            edgecolor="none"
        )
        buf.seek(0)
        png_bytes = buf.getvalue()
        b64_str = f"data:image/png;base64,{base64.b64encode(png_bytes).decode('utf-8')}"
        buf.close()
        plt.close(fig)
        return png_bytes, b64_str

    def generate_chart_1_trend(self) -> Dict[str, Any]:
        """Chart 1: Time-Series Progression or Sequential Index Line Plot."""
        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=THEME["fig_face"])

        if self.num_cols:
            primary_metric = self.num_cols[0]
            secondary_metric = self.num_cols[1] if len(self.num_cols) > 1 else None

            # Sort by date if present
            plot_df = self.df.copy()
            x_label = "Record Sequence (Index)"
            if self.date_col:
                try:
                    plot_df[self.date_col] = pd.to_datetime(plot_df[self.date_col], errors="coerce")
                    plot_df = plot_df.sort_values(self.date_col).dropna(subset=[self.date_col])
                    x_vals = plot_df[self.date_col]
                    x_label = f"Timeline ({self.date_col})"
                except Exception:
                    x_vals = np.arange(len(plot_df))
            else:
                x_vals = np.arange(len(plot_df))

            # Limit to 300 points for smooth rendering
            if len(plot_df) > 300:
                step = len(plot_df) // 300
                plot_df = plot_df.iloc[::step]
                if isinstance(x_vals, np.ndarray):
                    x_vals = x_vals[::step]
                else:
                    x_vals = x_vals.iloc[::step]

            y_vals = plot_df[primary_metric].ffill().bfill().values
            ax.plot(x_vals, y_vals, color=THEME["accent_cyan"], linewidth=2.2, label=f"{primary_metric}")
            ax.fill_between(x_vals, y_vals, alpha=0.15, color=THEME["accent_cyan"])

            if secondary_metric:
                y2_vals = plot_df[secondary_metric].ffill().bfill().values
                ax2 = ax.twinx()
                ax2.plot(x_vals, y2_vals, color=THEME["accent_amber"], linewidth=1.8, linestyle="--", label=f"{secondary_metric}")
                ax2.tick_params(colors=THEME["subtext"], labelsize=9)
                ax2.spines["right"].set_color(THEME["grid"])
                for sp in ["top", "bottom", "left"]:
                    ax2.spines[sp].set_visible(False)
                ax2.set_ylabel(f"{secondary_metric}", color=THEME["accent_amber"], fontsize=9)

            title = f"Progression & Trendline: {primary_metric}"
            desc = f"Sequential trajectory curve showing trends, peaks, and troughs for {primary_metric}."
            ax.legend(facecolor=THEME["ax_face"], edgecolor=THEME["grid"], labelcolor=THEME["text"], loc="upper left")
        else:
            # Categorical frequency trend
            counts = self.df.index.to_series()
            ax.plot(counts, color=THEME["accent_cyan"], linewidth=2)
            title = "Row Sequence Distribution"
            desc = "Sequential index profile across entries."
            x_label = "Row Index"

        self._apply_theme(ax, title, xlabel=x_label, ylabel="Metric Value")
        png_bytes, b64_str = self._fig_to_bytes_and_base64(fig)
        return {
            "title": title,
            "description": desc,
            "bytes": png_bytes,
            "base64": b64_str,
            "type": "trend_line"
        }

    def generate_chart_2_category(self) -> Dict[str, Any]:
        """Chart 2: Categorical Distribution or Group Aggregation Bar Plot."""
        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=THEME["fig_face"])

        target_cat = None
        for col in self.cat_cols:
            n_unique = self.df[col].dropna().nunique()
            if 2 <= n_unique <= 20:
                target_cat = col
                break
        if not target_cat and self.cat_cols:
            target_cat = self.cat_cols[0]

        if target_cat:
            if self.num_cols:
                metric = self.num_cols[0]
                grouped = self.df.groupby(target_cat)[metric].mean().dropna().sort_values(ascending=False).head(8)
                categories = [str(x)[:16] for x in grouped.index]
                values = grouped.values
                title = f"Top Mean {metric} by {target_cat}"
                ylabel = f"Mean {metric}"
                desc = f"Grouped comparative breakdown of {metric} across prominent categories in {target_cat}."
            else:
                counts = self.df[target_cat].value_counts().head(8)
                categories = [str(x)[:16] for x in counts.index]
                values = counts.values
                title = f"Frequency Distribution: {target_cat}"
                ylabel = "Record Count"
                desc = f"Distribution of dominant classes within {target_cat}."

            colors = [THEME["accent_cyan"], THEME["accent_indigo"], THEME["accent_emerald"],
                      THEME["accent_amber"], THEME["accent_rose"]]
            bar_colors = [colors[i % len(colors)] for i in range(len(categories))]

            bars = ax.bar(categories, values, color=bar_colors, width=0.55, edgecolor=THEME["grid"], alpha=0.9)
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height:,.1f}" if height < 10000 else f"{height:,.0f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, color=THEME["text"], fontweight="bold")

            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(categories, rotation=25, ha="right")
        elif self.num_cols:
            # Fallback to histogram if no categorical columns
            metric = self.num_cols[0]
            counts, bins, _ = ax.hist(self.df[metric].dropna(), bins=12, color=THEME["accent_indigo"],
                                      edgecolor=THEME["grid"], alpha=0.85)
            title = f"Frequency Histogram: {metric}"
            ylabel = "Frequency Count"
            desc = f"Binned distribution density for continuous variable {metric}."
        else:
            ax.text(0.5, 0.5, "Insufficient Categorical Data", ha="center", va="center", color=THEME["subtext"])
            title = "Categorical Distribution"
            ylabel = "Value"
            desc = "No categorical dimensions found in dataset."

        self._apply_theme(ax, title, xlabel="Category Segment", ylabel=ylabel)
        png_bytes, b64_str = self._fig_to_bytes_and_base64(fig)
        return {
            "title": title,
            "description": desc,
            "bytes": png_bytes,
            "base64": b64_str,
            "type": "categorical_bar"
        }

    def generate_chart_3_correlation(self) -> Dict[str, Any]:
        """Chart 3: Numerical Correlation Matrix Heatmap or 2D Density."""
        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=THEME["fig_face"])

        if len(self.num_cols) >= 2:
            cols_to_use = self.num_cols[:7]  # Limit to 7 for readable heatmap
            corr = self.df[cols_to_use].corr()

            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            sns.heatmap(
                corr,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                ax=ax,
                cbar=True,
                cbar_kws={"shrink": 0.8},
                linewidths=0.5,
                linecolor=THEME["fig_face"],
                annot_kws={"size": 8, "weight": "bold", "color": "#ffffff"}
            )
            title = "Multivariate Correlation Matrix"
            desc = f"Heatmap tracking linear Pearson relationships across top numerical indicators ({', '.join(cols_to_use[:3])})."
            ax.set_xticklabels([str(c)[:12] for c in cols_to_use], rotation=30, ha="right", color=THEME["subtext"], fontsize=8)
            ax.set_yticklabels([str(c)[:12] for c in cols_to_use], rotation=0, color=THEME["subtext"], fontsize=8)
        elif len(self.num_cols) == 1 and self.cat_cols:
            # Cross-tab heatmap
            cat = self.cat_cols[0]
            num = self.num_cols[0]
            quantiles = pd.qcut(self.df[num].dropna(), q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
            ct = pd.crosstab(self.df[cat].head(6), quantiles)
            sns.heatmap(ct, annot=True, fmt="d", cmap="magma", ax=ax, linewidths=0.5, linecolor=THEME["fig_face"])
            title = f"Cross-Tabulation Heatmap: {cat} vs {num}"
            desc = f"Bivariate relationship matrix between categorical {cat} and quartiles of {num}."
        else:
            ax.text(0.5, 0.5, "Single Variable Dataset\nCorrelation Matrix N/A", ha="center", va="center", color=THEME["subtext"], fontsize=11)
            title = "Correlation Heatmap"
            desc = "Requires at least 2 numerical dimensions."

        self._apply_theme(ax, title)
        png_bytes, b64_str = self._fig_to_bytes_and_base64(fig)
        return {
            "title": title,
            "description": desc,
            "bytes": png_bytes,
            "base64": b64_str,
            "type": "correlation_heatmap"
        }

    def generate_chart_4_outlier(self) -> Dict[str, Any]:
        """Chart 4: Anomaly Radar & Outlier Dispersion Plot (Box Plot + Jitter Scatter)."""
        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=THEME["fig_face"])

        if self.num_cols:
            target_metric = self.num_cols[0]
            series = self.df[target_metric].dropna()

            # Compute IQR bounds
            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr

            # Categorize normal vs outliers
            is_outlier = (series < lower_bound) | (series > upper_bound)
            outlier_count = int(is_outlier.sum())

            # Draw sleek horizontal boxplot
            bp = ax.boxplot(
                series,
                vert=False,
                patch_artist=True,
                showfliers=False,
                widths=0.35,
                positions=[0],
                boxprops=dict(facecolor=THEME["accent_indigo"], color=THEME["accent_cyan"], linewidth=1.5, alpha=0.6),
                whiskerprops=dict(color=THEME["accent_cyan"], linewidth=1.5),
                capprops=dict(color=THEME["accent_cyan"], linewidth=1.5),
                medianprops=dict(color=THEME["accent_amber"], linewidth=2.0)
            )

            # Scatter jitter points on top
            rng = np.random.default_rng(42)
            y_jitter = rng.uniform(-0.15, 0.15, size=len(series))

            # Limit scatter points if dataset is massive
            if len(series) > 500:
                idx_sample = np.random.choice(len(series), 500, replace=False)
                sample_series = series.iloc[idx_sample]
                sample_outlier = is_outlier.iloc[idx_sample]
                sample_jitter = y_jitter[idx_sample]
            else:
                sample_series = series
                sample_outlier = is_outlier
                sample_jitter = y_jitter

            # Plot normal points in cyan
            normal_mask = ~sample_outlier
            ax.scatter(sample_series[normal_mask], sample_jitter[normal_mask],
                       color=THEME["accent_cyan"], alpha=0.45, s=20, label="Nominal Data")

            # Plot outlier points in glowing neon rose
            if outlier_count > 0:
                ax.scatter(sample_series[sample_outlier], sample_jitter[sample_outlier],
                           color=THEME["accent_rose"], alpha=0.9, s=48, edgecolors="#ffffff",
                           linewidths=0.8, label=f"Outliers: {outlier_count} pts", zorder=5)

            # Mark threshold lines
            ax.axvline(lower_bound, color=THEME["accent_rose"], linestyle=":", linewidth=1.2, alpha=0.7)
            ax.axvline(upper_bound, color=THEME["accent_rose"], linestyle=":", linewidth=1.2, alpha=0.7)

            title = f"Anomaly Radar & Outlier Dispersion: {target_metric}"
            desc = f"Identified {outlier_count} visual anomalies outside 1.5x IQR boundaries [{lower_bound:,.1f}, {upper_bound:,.1f}]."
            ax.set_yticks([])
            ax.legend(facecolor=THEME["ax_face"], edgecolor=THEME["grid"], labelcolor=THEME["text"], loc="upper right")
        else:
            ax.text(0.5, 0.5, "Numerical Data Required for Outlier Boxplot", ha="center", va="center", color=THEME["subtext"])
            title = "Anomaly Radar"
            desc = "No numerical columns detected."

        self._apply_theme(ax, title, xlabel="Value Range")
        png_bytes, b64_str = self._fig_to_bytes_and_base64(fig)
        return {
            "title": title,
            "description": desc,
            "bytes": png_bytes,
            "base64": b64_str,
            "type": "outlier_scatter"
        }

    def generate_all_charts(self) -> Dict[str, Any]:
        """Generate all 4 diagnostic charts and return structured payloads."""
        c1 = self.generate_chart_1_trend()
        c2 = self.generate_chart_2_category()
        c3 = self.generate_chart_3_correlation()
        c4 = self.generate_chart_4_outlier()

        return {
            "chart_1": c1,
            "chart_2": c2,
            "chart_3": c3,
            "chart_4": c4,
            "all_bytes": [c1["bytes"], c2["bytes"], c3["bytes"], c4["bytes"]],
            "gallery": [c1, c2, c3, c4]
        }


def generate_diagnostic_charts(df: pd.DataFrame) -> Dict[str, Any]:
    """Convenience helper to generate all 4 diagnostic charts from DataFrame."""
    plotter = VisualPlotter(df)
    return plotter.generate_all_charts()
