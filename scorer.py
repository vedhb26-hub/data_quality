# scorer.py
import pandas as pd
import numpy as np
from typing import Any

def score_dataframe(df: pd.DataFrame) -> dict:
    report = {"overall_score": 0, "columns": {}, "suggestions": [], "summary": {}}
    scores = []

    for col in df.columns:
        series = df[col]
        col_report = {}

        # Completeness
        missing_pct = round(series.isna().mean() * 100, 1)
        col_report["missing_pct"] = missing_pct
        completeness = round(100 - missing_pct, 1)

        # Uniqueness
        duplicate_pct = round((1 - series.nunique() / len(series)) * 100, 1)
        col_report["duplicate_pct"] = duplicate_pct

        # Outliers (numeric only)
        outlier_count = 0
        if pd.api.types.is_numeric_dtype(series):
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            outliers = series[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)]
            outlier_count = len(outliers)
            col_report["outlier_count"] = outlier_count

        # Validity — detect mixed types
        if series.dtype == object:
            numeric_like = pd.to_numeric(series.dropna(), errors='coerce').notna().mean()
            if 0.1 < numeric_like < 0.9:
                col_report["mixed_types"] = True
                report["suggestions"].append(
                    f"Column '{col}' has mixed numeric and text values — consider splitting or standardizing."
                )

        col_score = round(completeness * 0.6 + (100 - duplicate_pct) * 0.3 + max(0, 100 - outlier_count * 5) * 0.1, 1)
        col_report["column_score"] = col_score
        scores.append(col_score)

        # Plain-English suggestions
        if missing_pct > 20:
            report["suggestions"].append(f"'{col}' has {missing_pct}% missing — consider imputing or dropping.")
        if duplicate_pct > 50 and series.dtype == object:
            report["suggestions"].append(f"'{col}' is {duplicate_pct}% duplicated — might be a categorical column.")
        if outlier_count > 5:
            report["suggestions"].append(f"'{col}' has {outlier_count} outliers — review or cap with IQR method.")

        report["columns"][col] = col_report

    report["overall_score"] = round(np.mean(scores), 1)
    report["summary"] = {
        "rows": len(df),
        "columns": len(df.columns),
        "total_missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    return report