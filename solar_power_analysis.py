from __future__ import annotations

from pathlib import Path

import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"
OUTPUT_DIR = BASE_DIR / "outputs"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_dataset_path(filename: str) -> Path:
    dataset_path = DATA_DIR / filename
    if dataset_path.exists():
        return dataset_path
    fallback_path = BASE_DIR / filename
    if fallback_path.exists():
        return fallback_path
    raise FileNotFoundError(f"Khong tim thay dataset: {filename}")


def parse_generation_datetime(series: pd.Series, plant_no: int) -> pd.Series:
    if plant_no == 1:
        return pd.to_datetime(series, format="%d-%m-%Y %H:%M", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def load_plant_data(plant_no: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    generation = pd.read_csv(resolve_dataset_path(f"Plant_{plant_no}_Generation_Data.csv"))
    weather = pd.read_csv(resolve_dataset_path(f"Plant_{plant_no}_Weather_Sensor_Data.csv"))

    generation["DATE_TIME"] = parse_generation_datetime(generation["DATE_TIME"], plant_no)
    weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"], errors="coerce")

    generation["PLANT_NO"] = plant_no
    weather["PLANT_NO"] = plant_no

    if generation["DATE_TIME"].isna().any() or weather["DATE_TIME"].isna().any():
        raise ValueError(f"Loi parse DATE_TIME o Plant {plant_no}")

    generation = generation.sort_values(["DATE_TIME", "SOURCE_KEY"]).reset_index(drop=True)
    weather = weather.sort_values(["DATE_TIME"]).reset_index(drop=True)
    return generation, weather


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["HOUR"] = out["DATE_TIME"].dt.hour
    out["MINUTE"] = out["DATE_TIME"].dt.minute
    out["DAY"] = out["DATE_TIME"].dt.day
    out["DAYOFWEEK"] = out["DATE_TIME"].dt.dayofweek
    out["WEEKOFYEAR"] = out["DATE_TIME"].dt.isocalendar().week.astype(int)
    out["MONTH"] = out["DATE_TIME"].dt.month
    out["DAYOFYEAR"] = out["DATE_TIME"].dt.dayofyear
    hour_fraction = out["HOUR"] + out["MINUTE"] / 60.0
    out["HOUR_SIN"] = np.sin(2 * np.pi * hour_fraction / 24.0)
    out["HOUR_COS"] = np.cos(2 * np.pi * hour_fraction / 24.0)
    out["IS_DAYLIGHT"] = (out["HOUR"].between(6, 18)).astype(int)
    return out


def aggregate_generation(generation: pd.DataFrame) -> pd.DataFrame:
    aggregated = (
        generation.groupby(["PLANT_ID", "PLANT_NO", "DATE_TIME"], as_index=False)
        .agg(
            DC_POWER_TOTAL=("DC_POWER", "sum"),
            AC_POWER_TOTAL=("AC_POWER", "sum"),
            DAILY_YIELD_TOTAL=("DAILY_YIELD", "sum"),
            TOTAL_YIELD_TOTAL=("TOTAL_YIELD", "sum"),
            ACTIVE_SOURCE_COUNT=("SOURCE_KEY", "nunique"),
            ZERO_AC_COUNT=("AC_POWER", lambda s: int((s == 0).sum())),
        )
    )
    aggregated["ZERO_AC_RATIO"] = (
        aggregated["ZERO_AC_COUNT"] / aggregated["ACTIVE_SOURCE_COUNT"].replace(0, np.nan)
    )
    return aggregated


def merge_generation_weather(generation_agg: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    weather_renamed = weather.rename(columns={"SOURCE_KEY": "WEATHER_SENSOR_KEY"})
    merged = generation_agg.merge(
        weather_renamed,
        on=["PLANT_ID", "PLANT_NO", "DATE_TIME"],
        how="left",
        validate="one_to_one",
    )
    merged = add_time_features(merged)
    return merged


def summarize_generation(generation: pd.DataFrame, plant_no: int) -> dict:
    daily = generation.copy()
    daily["DATE"] = daily["DATE_TIME"].dt.date
    daily_summary = (
        daily.groupby("DATE", as_index=False)
        .agg(
            AC_POWER_TOTAL=("AC_POWER", "sum"),
            DC_POWER_TOTAL=("DC_POWER", "sum"),
            DAILY_YIELD_TOTAL=("DAILY_YIELD", "sum"),
        )
    )
    return {
        "plant_no": plant_no,
        "rows": int(len(generation)),
        "timestamps": int(generation["DATE_TIME"].nunique()),
        "source_keys": int(generation["SOURCE_KEY"].nunique()),
        "ac_power_mean": float(generation["AC_POWER"].mean()),
        "dc_power_mean": float(generation["DC_POWER"].mean()),
        "zero_ac_ratio": float((generation["AC_POWER"] == 0).mean()),
        "total_daily_yield_sum": float(generation["DAILY_YIELD"].sum()),
        "best_day_ac_power_total": float(daily_summary["AC_POWER_TOTAL"].max()),
    }


def summarize_weather(weather: pd.DataFrame, plant_no: int) -> dict:
    return {
        "plant_no": plant_no,
        "rows": int(len(weather)),
        "timestamps": int(weather["DATE_TIME"].nunique()),
        "ambient_temperature_mean": float(weather["AMBIENT_TEMPERATURE"].mean()),
        "module_temperature_mean": float(weather["MODULE_TEMPERATURE"].mean()),
        "irradiation_mean": float(weather["IRRADIATION"].mean()),
        "irradiation_max": float(weather["IRRADIATION"].max()),
    }


def compute_metrics(actual: pd.Series, predicted: np.ndarray) -> dict:
    rmse = math.sqrt(mean_squared_error(actual, predicted))
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(rmse),
        "R2": float(r2_score(actual, predicted)),
    }


def plot_daily_generation(combined_generation: pd.DataFrame) -> None:
    daily = combined_generation.copy()
    daily["DATE"] = daily["DATE_TIME"].dt.floor("D")
    summary = (
        daily.groupby(["PLANT_NO", "DATE"], as_index=False)
        .agg(AC_POWER_TOTAL=("AC_POWER", "sum"), DC_POWER_TOTAL=("DC_POWER", "sum"))
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    for plant_no in sorted(summary["PLANT_NO"].unique()):
        plant_data = summary[summary["PLANT_NO"] == plant_no]
        ax.plot(
            plant_data["DATE"],
            plant_data["AC_POWER_TOTAL"],
            marker="o",
            linewidth=1.5,
            label=f"Plant {plant_no} - AC",
        )
    ax.set_title("Tong AC Power theo ngay")
    ax.set_xlabel("Ngay")
    ax.set_ylabel("Tong AC Power")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "daily_generation_comparison.png", dpi=150)
    plt.close(fig)


def plot_daily_weather(combined_weather: pd.DataFrame) -> None:
    daily = combined_weather.copy()
    daily["DATE"] = daily["DATE_TIME"].dt.floor("D")
    summary = (
        daily.groupby(["PLANT_NO", "DATE"], as_index=False)
        .agg(
            IRRADIATION_MEAN=("IRRADIATION", "mean"),
            AMBIENT_TEMPERATURE_MEAN=("AMBIENT_TEMPERATURE", "mean"),
            MODULE_TEMPERATURE_MEAN=("MODULE_TEMPERATURE", "mean"),
        )
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for plant_no in sorted(summary["PLANT_NO"].unique()):
        plant_data = summary[summary["PLANT_NO"] == plant_no]
        axes[0].plot(
            plant_data["DATE"],
            plant_data["IRRADIATION_MEAN"],
            marker="o",
            linewidth=1.5,
            label=f"Plant {plant_no}",
        )
        axes[1].plot(
            plant_data["DATE"],
            plant_data["MODULE_TEMPERATURE_MEAN"],
            marker="o",
            linewidth=1.5,
            label=f"Plant {plant_no}",
        )
    axes[0].set_title("Irradiation trung binh theo ngay")
    axes[0].set_ylabel("Irradiation")
    axes[1].set_title("Module temperature trung binh theo ngay")
    axes[1].set_ylabel("Nhiet do")
    axes[1].set_xlabel("Ngay")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "daily_weather_comparison.png", dpi=150)
    plt.close(fig)


def plot_heatmap(merged: pd.DataFrame, plant_no: int) -> None:
    numeric_cols = [
        "AC_POWER_TOTAL",
        "DC_POWER_TOTAL",
        "DAILY_YIELD_TOTAL",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "IRRADIATION",
        "ACTIVE_SOURCE_COUNT",
        "ZERO_AC_RATIO",
        "HOUR",
    ]
    corr = merged[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.values, cmap="YlOrRd", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    ax.set_title(f"Tuong quan dac trung - Plant {plant_no}")
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"plant_{plant_no}_correlation_heatmap.png", dpi=150)
    plt.close(fig)


def plot_predictions(test_df: pd.DataFrame, plant_no: int) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test_df["DATE_TIME"], test_df["AC_POWER_TOTAL"], label="Thuc te", linewidth=1.5)
    ax.plot(test_df["DATE_TIME"], test_df["PREDICTED_AC_POWER"], label="Du doan", linewidth=1.5)
    ax.set_title(f"Du doan AC Power tren tap test - Plant {plant_no}")
    ax.set_xlabel("DATE_TIME")
    ax.set_ylabel("AC Power tong")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"plant_{plant_no}_predictions.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(model: Pipeline, feature_columns: list[str]) -> None:
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["model"]

    transformed_feature_names = preprocessor.get_feature_names_out(feature_columns)
    importance = pd.DataFrame(
        {
            "feature": transformed_feature_names,
            "importance": regressor.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    top_importance = importance.head(12).sort_values("importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_importance["feature"], top_importance["importance"], color="#c4632d")
    ax.set_title("Top 12 dac trung quan trong cua Random Forest")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)


def train_model(merged_all: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, dict]:
    model_df = merged_all.dropna(subset=["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]).copy()
    model_df = model_df.sort_values("DATE_TIME").reset_index(drop=True)

    feature_columns = [
        "PLANT_NO",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "IRRADIATION",
        "HOUR",
        "MINUTE",
        "DAYOFWEEK",
        "DAYOFYEAR",
        "MONTH",
        "HOUR_SIN",
        "HOUR_COS",
        "IS_DAYLIGHT",
    ]
    target_column = "AC_POWER_TOTAL"

    split_index = int(len(model_df) * 0.8)
    train_df = model_df.iloc[:split_index].copy()
    test_df = model_df.iloc[split_index:].copy()

    numeric_features = [
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "IRRADIATION",
        "HOUR",
        "MINUTE",
        "DAYOFWEEK",
        "DAYOFYEAR",
        "MONTH",
        "HOUR_SIN",
        "HOUR_COS",
        "IS_DAYLIGHT",
    ]
    categorical_features = ["PLANT_NO"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                 RandomForestRegressor(
                     n_estimators=300,
                     max_depth=14,
                     min_samples_leaf=2,
                     random_state=42,
                    n_jobs=1,
                 ),
             ),
         ]
     )

    model.fit(train_df[feature_columns], train_df[target_column])
    test_df["PREDICTED_AC_POWER"] = model.predict(test_df[feature_columns])

    metrics = {"overall": compute_metrics(test_df[target_column], test_df["PREDICTED_AC_POWER"])}
    for plant_no in sorted(test_df["PLANT_NO"].unique()):
        plant_test = test_df[test_df["PLANT_NO"] == plant_no]
        metrics[f"plant_{plant_no}"] = compute_metrics(
            plant_test[target_column], plant_test["PREDICTED_AC_POWER"]
        )

    plot_feature_importance(model, feature_columns)
    return model, test_df, metrics


def save_summary_report(
    generation_summaries: list[dict],
    weather_summaries: list[dict],
    merged_summaries: list[dict],
    metrics: dict,
) -> None:
    generation_text = pd.DataFrame(generation_summaries).to_string(index=False)
    weather_text = pd.DataFrame(weather_summaries).to_string(index=False)
    merged_text = pd.DataFrame(merged_summaries).to_string(index=False)
    metrics_text = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "scope"}).to_string(index=False)

    report_lines = [
        "# Solar Power Analysis Report",
        "",
        "## Generation Summary",
        "```text",
        generation_text,
        "```",
        "",
        "## Weather Summary",
        "```text",
        weather_text,
        "```",
        "",
        "## Merged Summary",
        "```text",
        merged_text,
        "```",
        "",
        "## Model Metrics",
        "```text",
        metrics_text,
        "```",
    ]
    (OUTPUT_DIR / "analysis_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    ensure_output_dir()

    generation_frames = []
    weather_frames = []
    merged_frames = []
    generation_summaries = []
    weather_summaries = []
    merged_summaries = []

    for plant_no in [1, 2]:
        generation, weather = load_plant_data(plant_no)
        generation_frames.append(generation)
        weather_frames.append(weather)

        generation_summaries.append(summarize_generation(generation, plant_no))
        weather_summaries.append(summarize_weather(weather, plant_no))

        aggregated_generation = aggregate_generation(generation)
        merged = merge_generation_weather(aggregated_generation, weather)
        merged_frames.append(merged)

        merged_summary = {
            "plant_no": plant_no,
            "merged_rows": int(len(merged)),
            "missing_weather_rows": int(merged["AMBIENT_TEMPERATURE"].isna().sum()),
            "date_time_min": str(merged["DATE_TIME"].min()),
            "date_time_max": str(merged["DATE_TIME"].max()),
            "target_mean_ac_power": float(merged["AC_POWER_TOTAL"].mean()),
        }
        merged_summaries.append(merged_summary)

        merged.to_csv(OUTPUT_DIR / f"plant_{plant_no}_merged.csv", index=False)
        plot_heatmap(merged.dropna(subset=["AMBIENT_TEMPERATURE", "IRRADIATION"]), plant_no)

    combined_generation = pd.concat(generation_frames, ignore_index=True)
    combined_weather = pd.concat(weather_frames, ignore_index=True)
    merged_all = pd.concat(merged_frames, ignore_index=True).sort_values("DATE_TIME").reset_index(drop=True)
    merged_all.to_csv(OUTPUT_DIR / "all_plants_merged.csv", index=False)

    plot_daily_generation(combined_generation)
    plot_daily_weather(combined_weather)

    _, test_df, metrics = train_model(merged_all)
    test_df.to_csv(OUTPUT_DIR / "model_test_predictions.csv", index=False)

    for plant_no in [1, 2]:
        plot_predictions(test_df[test_df["PLANT_NO"] == plant_no], plant_no)

    save_summary_report(generation_summaries, weather_summaries, merged_summaries, metrics)

    print("Analysis completed. Outputs saved in:", OUTPUT_DIR)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
