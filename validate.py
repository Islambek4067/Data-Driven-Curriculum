"""Сверяет выводы модели с тем, что заложено в синтетических данных.

Запуск: python validate.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import generate_data as gen
from core.analysis import ALL, Bundle, run_analysis
from core.data import load_dir


def build_report(bundle: Bundle, truth: dict) -> dict:
    broken = set(truth["broken_items"])
    flagged = set(bundle.quality.loc[bundle.quality["is_bad"], "item_id"])

    diagnosis = bundle.diagnosis.set_index("topic_id")
    expected = {t: truth["statuses"].get(t, "ok") for t in diagnosis.index}
    status_rows = pd.DataFrame(
        {
            "topic_id": list(expected),
            "expected": list(expected.values()),
            "found": [diagnosis.loc[t, "status"] for t in expected],
        }
    )
    status_rows["match"] = status_rows["expected"] == status_rows["found"]

    contrasts = bundle.textbooks["contrasts"]
    topic_rows = contrasts[contrasts["scope"] != ALL].copy()
    def true_value(row) -> float:
        key = f"{row['first']}-{row['second']}"
        if key in truth["contrasts"]:
            return truth["contrasts"][key][row["scope"]]
        return -truth["contrasts"][f"{row['second']}-{row['first']}"][row["scope"]]

    topic_rows["truth"] = topic_rows.apply(true_value, axis=1)
    topic_rows["covered"] = (topic_rows["low"] <= topic_rows["truth"]) & (topic_rows["truth"] <= topic_rows["high"])
    real = topic_rows["truth"].abs() >= 1.0
    overall = contrasts[contrasts["scope"] == ALL].copy()
    overall["truth"] = overall.apply(
        lambda r: float(
            np.mean(
                [
                    truth["contrasts"][f"{r['first']}-{r['second']}"][t]
                    if f"{r['first']}-{r['second']}" in truth["contrasts"]
                    else -truth["contrasts"][f"{r['second']}-{r['first']}"][t]
                    for t in diagnosis.index
                ]
            )
        ),
        axis=1,
    )

    world = gen.build_world()
    planned = bundle.hours.planned
    base = gen.true_scores(world, plan=planned)
    best = gen.true_scores(world, plan=bundle.best_plan)
    true_gain = float(np.mean([best[t] - base[t] for t in base]))

    return {
        "items": {"broken": len(broken), "flagged": len(flagged), "correct": len(broken & flagged)},
        "statuses": status_rows,
        "contrasts": topic_rows,
        "overall": overall,
        "mae": float((topic_rows["diff"] - topic_rows["truth"]).abs().mean()),
        "coverage": float(topic_rows["covered"].mean()),
        "real_found": int((topic_rows["proven"] & real).sum()),
        "real_total": int(real.sum()),
        "false_alarms": int((topic_rows["proven"] & ~real).sum()),
        "null_total": int((~real).sum()),
        "plan": {
            "promised": bundle.best_forecast["gain"],
            "low": bundle.best_forecast["low"],
            "high": bundle.best_forecast["high"],
            "true": true_gain,
        },
        "frozen_ok": set(bundle.hours.frozen) == set(truth["no_variation"]),
    }


def main() -> None:
    dataset = load_dir(gen.DATA_DIR, is_demo=True)
    bundle = run_analysis(dataset)
    truth = json.loads((gen.DATA_DIR / "truth.json").read_text(encoding="utf-8"))
    report = build_report(bundle, truth)
    pd.set_option("display.width", 200, "display.max_columns", 20)

    print("Вопросы:", report["items"])
    print(bundle.diagnosis[["topic_id", "raw", "clean", "n_bad", "status", "root", "with_base", "without_base"]].round(1))
    print(report["statuses"][~report["statuses"]["match"]])
    means = bundle.textbooks["means"]
    print(means[means["scope"] == ALL].round(1))
    print(bundle.textbooks["profile"].round(1))
    print(report["overall"][["first", "second", "diff", "low", "high", "truth"]].round(2))
    print(f"Средняя ошибка оценки эффекта: {report['mae']:.2f} п.п.; интервал накрыл правду: {report['coverage']:.0%}")
    print(f"Реальных эффектов найдено: {report['real_found']}/{report['real_total']}; ложных: {report['false_alarms']}/{report['null_total']}")
    print("План:", {k: round(v, 2) for k, v in report["plan"].items()}, "замороженные темы верны:", report["frozen_ok"])
    table = bundle.best_forecast["table"].copy()
    table["reliable"] = table["topic_id"].map(bundle.hours.is_reliable)
    print(table.round(1))


if __name__ == "__main__":
    main()
