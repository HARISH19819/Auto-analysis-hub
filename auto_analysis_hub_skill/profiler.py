"""
profiler.py: Dynamic pandas data health & statistical text summarizer.
Profiles any user-uploaded CSV dataset with zero static mock data.
"""

from __future__ import annotations
import io
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats


class DataProfiler:
    """Dynamic Profiler that computes dataset health, statistical metrics,
    and structured textual summaries for downstream multimodal AI agents.
    """

    def __init__(self, df: pd.DataFrame, source_name: str = "dataset.csv"):
        self.df = df.copy()
        self.source_name = source_name
        self._profile_data: Optional[Dict[str, Any]] = None

    @classmethod
    def from_csv_bytes(
        cls,
        csv_bytes: bytes,
        source_name: str = "uploaded_file.csv"
    ) -> "DataProfiler":
        """Instantiate DataProfiler from raw CSV bytes, auto-detecting encoding."""
        encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]
        df = None
        for enc in encodings_to_try:
            try:
                df = pd.read_csv(io.BytesIO(csv_bytes), encoding=enc)
                break
            except Exception:
                continue

        if df is None:
            # Fallback to python engine with replace errors
            csv_str = csv_bytes.decode("utf-8", errors="replace")
            df = pd.read_csv(io.StringIO(csv_str), engine="python")

        return cls(df, source_name=source_name)

    @classmethod
    def from_csv_string(
        cls,
        csv_string: str,
        source_name: str = "dataset.csv"
    ) -> "DataProfiler":
        """Instantiate DataProfiler from CSV string."""
        df = pd.read_csv(io.StringIO(csv_string))
        return cls(df, source_name=source_name)

    def profile(self) -> Dict[str, Any]:
        """Compute comprehensive statistical metrics, health scores, and metadata."""
        if self._profile_data is not None:
            return self._profile_data

        df = self.df
        total_rows, total_cols = df.shape
        total_cells = max(1, total_rows * total_cols)

        # 1. Null & Health Metrics
        null_counts = df.isnull().sum()
        total_nulls = int(null_counts.sum())
        data_health_score = round(max(0.0, (1.0 - (total_nulls / total_cells))) * 100, 2)

        # 2. Duplicate Rows
        duplicate_rows_count = int(df.duplicated().sum())
        duplicate_percentage = round((duplicate_rows_count / max(1, total_rows)) * 100, 2)

        # 3. Detect Column Types
        col_types: Dict[str, List[str]] = {
            "numerical": [],
            "categorical": [],
            "datetime": [],
            "boolean": [],
        }

        for col in df.columns:
            dtype_str = str(df[col].dtype).lower()
            if "datetime" in dtype_str:
                col_types["datetime"].append(col)
            elif "bool" in dtype_str:
                col_types["boolean"].append(col)
            elif np.issubdtype(df[col].dtype, np.number):
                # Ensure it's not all NaNs
                col_types["numerical"].append(col)
            else:
                # Check if it might be parseable as datetime
                try_date = False
                if df[col].dropna().shape[0] > 0 and isinstance(df[col].dropna().iloc[0], str):
                    sample = df[col].dropna().head(10)
                    if any(c in sample.iloc[0] for c in ["-", "/", ":"]):
                        try:
                            pd.to_datetime(sample, errors="raise")
                            try_date = True
                        except Exception:
                            try_date = False
                if try_date:
                    col_types["datetime"].append(col)
                else:
                    col_types["categorical"].append(col)

        # 4. Missing value breakdown per column
        missing_breakdown: List[Dict[str, Any]] = []
        for col in df.columns:
            m_count = int(null_counts[col])
            m_pct = round((m_count / max(1, total_rows)) * 100, 2)
            missing_breakdown.append({
                "column": str(col),
                "dtype": str(df[col].dtype),
                "missing_count": m_count,
                "missing_percentage": m_pct,
                "valid_count": total_rows - m_count,
            })

        # 5. Numerical Statistics
        num_stats: Dict[str, Dict[str, Any]] = {}
        total_outliers_count = 0
        for col in col_types["numerical"]:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outlier_mask = (series < lower_bound) | (series > upper_bound)
            outliers_count = int(outlier_mask.sum())
            total_outliers_count += outliers_count

            # Skewness and kurtosis
            try:
                skew_val = float(stats.skew(series)) if len(series) > 2 else 0.0
            except Exception:
                skew_val = 0.0

            try:
                kurt_val = float(stats.kurtosis(series)) if len(series) > 3 else 0.0
            except Exception:
                kurt_val = 0.0

            num_stats[str(col)] = {
                "mean": round(float(series.mean()), 4),
                "median": round(float(series.median()), 4),
                "std": round(float(series.std()), 4) if len(series) > 1 else 0.0,
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
                "q25": round(q25, 4),
                "q75": round(q75, 4),
                "iqr": round(iqr, 4),
                "skewness": round(skew_val, 4),
                "kurtosis": round(kurt_val, 4),
                "zero_count": int((series == 0).sum()),
                "outliers_count": outliers_count,
                "outliers_percentage": round((outliers_count / len(series)) * 100, 2),
            }

        # 6. Categorical Statistics
        cat_stats: Dict[str, Dict[str, Any]] = {}
        for col in col_types["categorical"]:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            unique_count = int(series.nunique())
            val_counts = series.value_counts()
            top_val = str(val_counts.index[0]) if len(val_counts) > 0 else "None"
            top_freq = int(val_counts.iloc[0]) if len(val_counts) > 0 else 0
            top_5_dist = {str(k): int(v) for k, v in val_counts.head(5).items()}

            cat_stats[str(col)] = {
                "unique_count": unique_count,
                "top_value": top_val,
                "top_frequency": top_freq,
                "top_percentage": round((top_freq / len(series)) * 100, 2),
                "distribution_top5": top_5_dist,
            }

        # 7. Correlation Analysis (Top Correlations)
        top_correlations: List[Dict[str, Any]] = []
        if len(col_types["numerical"]) >= 2:
            try:
                corr_matrix = df[col_types["numerical"]].corr()
                pairs = []
                num_cols = col_types["numerical"]
                for i in range(len(num_cols)):
                    for j in range(i + 1, len(num_cols)):
                        val = corr_matrix.iloc[i, j]
                        if not np.isnan(val):
                            pairs.append((num_cols[i], num_cols[j], float(val)))
                # Sort by absolute correlation
                pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                for c1, c2, r in pairs[:6]:
                    top_correlations.append({
                        "col1": str(c1),
                        "col2": str(c2),
                        "coefficient": round(r, 4),
                        "strength": "Strong" if abs(r) >= 0.7 else "Moderate" if abs(r) >= 0.4 else "Weak",
                        "direction": "Positive" if r > 0 else "Negative"
                    })
            except Exception:
                pass

        # 8. Memory Usage & General Dimensions
        memory_usage_kb = round(df.memory_usage(deep=True).sum() / 1024, 2)

        # 9. Format Statistical Text Summary for LLM Context
        summary_text = self._build_text_summary(
            source_name=self.source_name,
            total_rows=total_rows,
            total_cols=total_cols,
            health_score=data_health_score,
            total_nulls=total_nulls,
            duplicate_rows=duplicate_rows_count,
            col_types=col_types,
            num_stats=num_stats,
            cat_stats=cat_stats,
            top_correlations=top_correlations,
            total_outliers_count=total_outliers_count
        )

        self._profile_data = {
            "source_name": self.source_name,
            "dimensions": {
                "total_rows": total_rows,
                "total_columns": total_cols,
                "total_cells": total_cells,
                "memory_kb": memory_usage_kb,
            },
            "health": {
                "score": data_health_score,
                "rating": "Excellent" if data_health_score >= 95 else "Good" if data_health_score >= 80 else "Needs Cleaning",
                "total_null_cells": total_nulls,
                "null_percentage": round((total_nulls / total_cells) * 100, 2),
                "duplicate_rows": duplicate_rows_count,
                "duplicate_percentage": duplicate_percentage,
                "total_outliers_detected": total_outliers_count,
            },
            "column_types": col_types,
            "missing_breakdown": missing_breakdown,
            "numerical_metrics": num_stats,
            "categorical_metrics": cat_stats,
            "top_correlations": top_correlations,
            "text_summary": summary_text,
            "preview_rows": df.head(10).fillna("").to_dict(orient="records"),
        }

        return self._profile_data

    def _build_text_summary(
        self,
        source_name: str,
        total_rows: int,
        total_cols: int,
        health_score: float,
        total_nulls: int,
        duplicate_rows: int,
        col_types: Dict[str, List[str]],
        num_stats: Dict[str, Dict[str, Any]],
        cat_stats: Dict[str, Dict[str, Any]],
        top_correlations: List[Dict[str, Any]],
        total_outliers_count: int
    ) -> str:
        """Create a dense, structured Markdown summary string optimal for multimodal LLM context."""
        lines = [
            f"# DATASET STATISTICAL PROFILE: `{source_name}`",
            f"- **Dimensions**: {total_rows:,} rows × {total_cols} columns",
            f"- **Data Health Score**: {health_score}% | Total Null Cells: {total_nulls:,} | Duplicate Rows: {duplicate_rows:,}",
            f"- **Total Visual Outliers Detected (IQR 1.5x)**: {total_outliers_count:,}",
            f"- **Column Types**: {len(col_types['numerical'])} Numerical, {len(col_types['categorical'])} Categorical, {len(col_types['datetime'])} Datetime",
            "",
            "## Numerical Column Distributions:"
        ]

        for col, s in list(num_stats.items())[:8]:
            lines.append(
                f"- **{col}**: Mean={s['mean']}, Median={s['median']}, Std={s['std']}, "
                f"Range=[{s['min']} to {s['max']}], IQR={s['iqr']}, Skewness={s['skewness']}, "
                f"Outliers={s['outliers_count']} ({s['outliers_percentage']}%)"
            )

        if cat_stats:
            lines.append("\n## Categorical Column Distributions:")
            for col, s in list(cat_stats.items())[:6]:
                top_items = ", ".join([f"'{k}': {v}" for k, v in list(s['distribution_top5'].items())[:4]])
                lines.append(
                    f"- **{col}**: {s['unique_count']} unique classes | Top='{s['top_value']}' ({s['top_percentage']}%) | Sample: [{top_items}]"
                )

        if top_correlations:
            lines.append("\n## Top Statistical Correlations:")
            for c in top_correlations:
                lines.append(f"- `{c['col1']}` ↔ `{c['col2']}`: r = {c['coefficient']} ({c['strength']} {c['direction']})")

        return "\n".join(lines)


def profile_dataset(
    data: Union[bytes, str, pd.DataFrame],
    source_name: str = "dataset.csv"
) -> Dict[str, Any]:
    """Convenience helper to profile dataset from bytes, string, or DataFrame."""
    if isinstance(data, pd.DataFrame):
        profiler = DataProfiler(data, source_name=source_name)
    elif isinstance(data, bytes):
        profiler = DataProfiler.from_csv_bytes(data, source_name=source_name)
    else:
        profiler = DataProfiler.from_csv_string(data, source_name=source_name)
    return profiler.profile()
