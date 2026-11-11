"""Статистика платформы: качество вопросов, диагноз тем, учебники, часы."""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import optimize, stats
from statsmodels.stats.multitest import multipletests

from core.data import Dataset

BAD_ITEM_R = 0.10
WEAK_MASTERY = 60.0
MASTERED_STUDENT = 60.0
PREREQ_LINK = 0.25
MIN_HOURS_SPREAD = 0.03
ALPHA = 0.05
Z95 = 1.96
FORECAST_DRAWS = 2000
ALL = "ALL"


# --- Вопросы и баллы ------------------------------------------------------------------


def item_quality(ds: Dataset) -> pd.DataFrame:
    """Брак: вопрос не отделяет сильных учеников от слабых (r с остальным тестом < 0,10)."""
    wide = ds.responses.pivot_table(index="student_id", columns="item_id", values="correct", aggfunc="mean")
    total = wide.sum(axis=1)
    count = wide.notna().sum(axis=1)
    rows = []
    for item in wide.columns:
        answers = wide[item]
        valid = answers.notna() & (count > 1)
        # Доля верных по остальным вопросам ученика: в TIMSS каждый решает только свой вариант.
        rest = (total[valid] - answers[valid]) / (count[valid] - 1)
        r = float(np.corrcoef(answers[valid], rest)[0, 1]) if answers[valid].std() > 0 else 0.0
        rows.append({"item_id": item, "p_correct": 100 * float(answers.mean()), "r_rest": r})
    quality = ds.items.merge(pd.DataFrame(rows), on="item_id", how="left")
    quality["is_bad"] = quality["r_rest"].fillna(0) < BAD_ITEM_R
    # Если сломаны все вопросы темы, убирать нечего: тема останется с исходным баллом.
    all_bad = quality.groupby("topic_id")["is_bad"].transform("all")
    quality.loc[all_bad, "is_bad"] = False
    return quality


def student_scores(ds: Dataset, quality: pd.DataFrame) -> pd.DataFrame:
    """Строка на ученика и тему: raw по всем вопросам, clean без бракованных."""
    merged = ds.responses.merge(quality[["item_id", "topic_id", "is_bad"]], on="item_id")
    grouped = merged.groupby(["student_id", "topic_id"])
    raw = grouped["correct"].mean().mul(100).rename("raw")
    good = merged[~merged["is_bad"]].groupby(["student_id", "topic_id"])["correct"].mean().mul(100).rename("clean")
    return pd.concat([raw, good], axis=1).reset_index()


def prereq_map(ds: Dataset) -> dict[str, list[str]]:
    return {r.topic_id: [p for p in r.prereqs.split(";") if p] for r in ds.topics.itertuples()}


# --- Диагноз тем ----------------------------------------------------------------------


def _partial_slope(y: np.ndarray, x: np.ndarray, control: np.ndarray) -> float:
    """Насколько балл темы растёт с баллом базы при равном входном уровне."""
    design = np.column_stack([np.ones_like(x), x, control])
    return float(np.linalg.lstsq(design, y, rcond=None)[0][1])


def weak_threshold(ds: Dataset, scores: pd.DataFrame) -> float:
    """Порог слабой темы. Для демо 60%, для реальных данных — на 5 п.п. ниже среднего по темам.

    Средний балл по TIMSS заметно ниже 60%, и абсолютный порог назвал бы слабыми все темы.
    """
    if not ds.is_timss:
        return WEAK_MASTERY
    return float(np.floor(scores.groupby("topic_id")["clean"].mean().mean() - 5))


def diagnose_topics(
    ds: Dataset, scores: pd.DataFrame, quality: pd.DataFrame, student_ids=None, weak: float = WEAK_MASTERY
) -> pd.DataFrame:
    """Причина слабого результата: брак вопросов, провал в базе или трудная тема."""
    data = scores if student_ids is None else scores[scores["student_id"].isin(student_ids)]
    means = data.groupby("topic_id")[["raw", "clean"]].mean()
    bad_counts = quality.groupby("topic_id")["is_bad"].sum()
    wide = data.pivot(index="student_id", columns="topic_id", values="clean")
    pretest = ds.students.set_index("student_id")["pretest"]
    pretest = pretest.fillna(pretest.mean())
    prereqs = prereq_map(ds)

    rows = []
    for topic in ds.topics.itertuples():
        tid = topic.topic_id
        clean, raw = float(means.loc[tid, "clean"]), float(means.loc[tid, "raw"])
        status, root, with_base, without_base = "ok", None, None, None
        if clean < weak:
            status = "hard"
            candidates = [p for p in prereqs[tid] if means.loc[p, "clean"] < weak]
            for base in sorted(candidates, key=lambda p: means.loc[p, "clean"]):
                pair = wide[[tid, base]].dropna()
                control = pretest.reindex(pair.index).to_numpy()
                if _partial_slope(pair[tid].to_numpy(), pair[base].to_numpy(), control) >= PREREQ_LINK:
                    status, root = "prereq", base
                    mastered = pair[base] >= (MASTERED_STUDENT if weak == WEAK_MASTERY else weak)
                    with_base = float(pair.loc[mastered, tid].mean())
                    without_base = float(pair.loc[~mastered, tid].mean())
                    break
        elif raw < weak and bad_counts.get(tid, 0) > 0:
            status = "bad_items"
        rows.append(
            {
                "topic_id": tid,
                "name": topic.name_ru,
                "raw": raw,
                "clean": clean,
                "n_bad": int(bad_counts.get(tid, 0)),
                "status": status,
                "root": root,
                "with_base": with_base,
                "without_base": without_base,
                "has_prereqs": bool(prereqs[tid]),
            }
        )
    return pd.DataFrame(rows)


# --- Модели ---------------------------------------------------------------------------


@dataclass
class Fit:
    columns: list[str]
    beta: np.ndarray
    cov: np.ndarray

    def index(self, name: str) -> int:
        return self.columns.index(name)

    def coef(self, name: str) -> float:
        return float(self.beta[self.index(name)])


def _fit_mixed(y: pd.Series, design: pd.DataFrame, groups: pd.Series) -> Fit:
    """REML со случайным эффектом школы: ученики одной школы не считаются независимыми.

    При одном случайном эффекте всё сводится к поиску одного числа, отношения
    дисперсии школ к дисперсии учеников. Прямой поиск устойчивее общего оптимизатора
    statsmodels, который на этих данных падает с вырожденной матрицей.
    """
    x, target = design.to_numpy(dtype=float), y.to_numpy(dtype=float)
    _, codes = np.unique(groups.to_numpy(), return_inverse=True)
    sizes = np.bincount(codes).astype(float)
    p, total = x.shape[1], len(target)

    xx, xy, yy = x.T @ x, x.T @ target, float(target @ target)
    x_sum = np.zeros((len(sizes), p))
    np.add.at(x_sum, codes, x)
    y_sum = np.bincount(codes, weights=target)

    def solve(ratio: float):
        shrink = ratio / (1 + sizes * ratio)
        a = xx - (x_sum.T * shrink) @ x_sum
        b = xy - (x_sum.T * shrink) @ y_sum
        beta = np.linalg.solve(a, b)
        rss = yy - float((shrink * y_sum) @ y_sum) - float(beta @ b)
        return a, beta, rss

    def reml(log_ratio: float) -> float:
        ratio = float(np.exp(log_ratio))
        a, _, rss = solve(ratio)
        return (total - p) * np.log(rss / (total - p)) + np.log1p(sizes * ratio).sum() + np.linalg.slogdet(a)[1]

    best = optimize.minimize_scalar(reml, bounds=(-14.0, 5.0), method="bounded")
    a, beta, rss = solve(float(np.exp(best.x)))
    return Fit(list(design.columns), beta, rss / (total - p) * np.linalg.inv(a))


def _covariates(ds: Dataset, books: list[str]) -> pd.DataFrame:
    """Входной балл ученика, средний входной балл школы, тип школы, язык и учебник.

    Средний балл школы нужен потому, что индивидуальная диагностика шумная и одна
    не убирает разницу между сильными и слабыми школами.
    """
    frame = ds.students.merge(ds.schools, on="school_id").set_index("student_id")
    pretest = frame["pretest"].fillna(frame["pretest"].mean())
    school_mean = pretest.groupby(frame["school_id"]).transform("mean")
    design = pd.DataFrame(index=frame.index)
    design["const"] = 1.0
    design["pretest"] = (pretest - pretest.mean()) / pretest.std()
    design["school_pretest"] = (school_mean - school_mean.mean()) / school_mean.std()
    for column in ["school_type", "language"]:
        dummies = pd.get_dummies(frame[column], prefix=column, drop_first=True, dtype=float)
        design = design.join(dummies)
    for book in books[1:]:
        design[f"book::{book}"] = (frame["textbook"] == book).astype(float)
    varying = (design.nunique() > 1) | (design.columns == "const")
    return design.loc[:, varying]


def _contrast(fit: Fit, first: str, second: str, reference: str) -> tuple[float, float]:
    vector = np.zeros(len(fit.columns))
    if first != reference:
        vector[fit.index(f"book::{first}")] += 1
    if second != reference:
        vector[fit.index(f"book::{second}")] -= 1
    return float(vector @ fit.beta), float(np.sqrt(vector @ fit.cov @ vector))


# --- Учебники -------------------------------------------------------------------------


def _adjusted_means(fit: Fit, y: pd.Series, books: list[str], shares: pd.Series) -> dict[str, float]:
    """Средний балл, если бы каждый учебник достался одним и тем же школам."""
    effects = {b: (0.0 if b == books[0] else fit.coef(f"book::{b}")) for b in books}
    center = sum(effects[b] * shares[b] for b in books)
    return {b: float(y.mean() + effects[b] - center) for b in books}


def compare_textbooks(ds: Dataset, scores: pd.DataFrame, fits: dict[str, Fit], books: list[str]) -> dict:
    wide = scores.pivot(index="student_id", columns="topic_id", values="clean")
    overall = wide.mean(axis=1).rename(ALL)
    book_of = ds.students.merge(ds.schools, on="school_id").set_index("student_id")["textbook"]
    shares = book_of.value_counts(normalize=True)

    means_rows, contrast_rows = [], []
    for scope in [ALL, *ds.topics["topic_id"]]:
        y = overall if scope == ALL else wide[scope].dropna()
        naive = y.groupby(book_of.reindex(y.index)).mean()
        adjusted = _adjusted_means(fits[scope], y, books, shares)
        for book in books:
            means_rows.append(
                {"scope": scope, "textbook": book, "naive": float(naive[book]), "adjusted": adjusted[book]}
            )
        for first, second in combinations(books, 2):
            diff, se = _contrast(fits[scope], first, second, books[0])
            contrast_rows.append(
                {
                    "scope": scope,
                    "first": first,
                    "second": second,
                    "diff": diff,
                    "low": diff - Z95 * se,
                    "high": diff + Z95 * se,
                    "p": float(2 * stats.norm.sf(abs(diff / se))),
                }
            )
    contrasts = pd.DataFrame(contrast_rows)

    # Тем и пар много, поэтому порог значимости поправлен на множественные сравнения.
    is_topic = contrasts["scope"] != ALL
    contrasts["p_adj"] = contrasts["p"]
    contrasts.loc[is_topic, "p_adj"] = multipletests(contrasts.loc[is_topic, "p"], method="fdr_bh")[1]
    contrasts["proven"] = contrasts["p_adj"] < ALPHA

    profile = (
        ds.students.merge(ds.schools, on="school_id")
        .groupby("textbook")
        .agg(
            schools=("school_id", "nunique"),
            pretest=("pretest", "mean"),
            gymnasium=("school_type", lambda s: 100 * (s == "Гимназия").mean()),
            rural=("school_type", lambda s: 100 * (s == "Сельская").mean()),
        )
        .reset_index()
    )
    return {"means": pd.DataFrame(means_rows), "contrasts": contrasts, "profile": profile, "books": books}


def explain_contrast(ds: Dataset, topic_id: str, first: str, second: str, lang: str = "ru") -> str | None:
    """Почему учебник сильнее: сколько упражнений и примеров на тему в каждом."""
    if ds.coverage is None:
        return None
    table = ds.coverage[ds.coverage["topic_id"] == topic_id].set_index("textbook")
    if first not in table.index or second not in table.index:
        return None
    a, b = table.loc[first], table.loc[second]
    exercises, examples = ("Жаттығулар", "Талданған мысалдар") if lang == "kz" else ("Упражнений", "Разобранных примеров")
    return (
        f"{exercises}: {first} — {int(a.exercises)}, {second} — {int(b.exercises)}. "
        f"{examples}: {first} — {int(a.examples)}, {second} — {int(b.examples)}."
    )


# --- Часы -----------------------------------------------------------------------------


@dataclass
class HoursModel:
    topics: list[str]
    planned: dict[str, int]
    low: dict[str, int]
    high: dict[str, int]
    frozen: list[str]
    terms: dict[str, list[tuple[str, int]]]  # тема -> (чьи часы, позиция в векторе)
    beta: dict[str, np.ndarray]
    cov: dict[str, np.ndarray]

    def effect(self, target: str, source: str) -> float:
        for name, position in self.terms[target]:
            if name == source:
                return float(self.beta[target][position])
        return 0.0

    def value_per_log_hour(self, source: str) -> float:
        """Отдача от часов темы: она сама плюс темы, которые на неё опираются."""
        return sum(max(self.effect(t, source), 0.0) for t in self.topics)

    def is_reliable(self, source: str) -> bool:
        for target in self.topics:
            for name, position in self.terms[target]:
                if name == source:
                    se = np.sqrt(self.cov[target][position, position])
                    if self.beta[target][position] - Z95 * se > 0:
                        return True
        return False


@dataclass
class _ModelInputs:
    base: pd.DataFrame
    wide: pd.DataFrame
    student_school: pd.Series
    log_hours: pd.DataFrame
    prereqs: dict[str, list[str]]
    frozen: list[str]


def _fit_topics(inputs: _ModelInputs, schools=None) -> tuple[dict[str, Fit], dict, dict, dict]:
    """Модель на каждую тему; schools ограничивает расчёт частью школ."""
    keep = inputs.student_school.isin(schools) if schools is not None else inputs.student_school.notna()
    base = inputs.base[keep]
    base = base.loc[:, (base.nunique() > 1) | (base.columns == "const")]
    student_school = inputs.student_school[keep]
    fits, terms, beta, cov = {}, {}, {}, {}
    for tid in inputs.log_hours.columns:
        design = base.copy()
        sources = [s for s in [tid, *inputs.prereqs[tid]] if s not in inputs.frozen]
        for source in sources:
            design[f"hours::{source}"] = inputs.log_hours[source].reindex(student_school).to_numpy()
        y = inputs.wide[tid].reindex(base.index).dropna()
        fit = _fit_mixed(y, design.loc[y.index], student_school.loc[y.index])
        fits[tid] = fit
        positions = [fit.index(f"hours::{s}") for s in sources]
        terms[tid] = [(source, i) for i, source in enumerate(sources)]
        beta[tid] = fit.beta[positions]
        cov[tid] = fit.cov[np.ix_(positions, positions)]
    return fits, terms, beta, cov


def fit_models(ds: Dataset, scores: pd.DataFrame, books: list[str]) -> tuple[dict[str, Fit], HoursModel, _ModelInputs]:
    base = _covariates(ds, books)
    wide = scores.pivot(index="student_id", columns="topic_id", values="clean")
    groups = ds.students.set_index("student_id")["school_id"]
    planned = ds.topics.set_index("topic_id")["planned_hours"]
    hours = ds.hours.pivot(index="school_id", columns="topic_id", values="hours")[list(planned.index)]
    log_hours = np.log(hours / planned)
    frozen = [t for t in planned.index if log_hours[t].std() < MIN_HOURS_SPREAD]
    student_school = groups.reindex(base.index)
    inputs = _ModelInputs(base, wide, student_school, log_hours, prereq_map(ds), frozen)

    fits, terms, beta, cov = _fit_topics(inputs)
    fits[ALL] = _fit_mixed(wide.mean(axis=1).reindex(base.index), base, student_school)

    low, high = {}, {}
    for tid in planned.index:
        plan = int(planned[tid])
        if tid in frozen:
            low[tid] = high[tid] = plan
        else:
            # Границы задают наблюдаемые часы: модель не экстраполирует.
            low[tid] = min(plan, int(np.ceil(hours[tid].quantile(0.05))))
            high[tid] = max(plan, int(np.floor(hours[tid].quantile(0.95))))
    model = HoursModel(
        topics=list(planned.index),
        planned={t: int(planned[t]) for t in planned.index},
        low=low,
        high=high,
        frozen=frozen,
        terms=terms,
        beta=beta,
        cov=cov,
    )
    return fits, model, inputs


def plan_realism(model: HoursModel, inputs: _ModelInputs, seed: int = 11) -> float:
    """Какая доля обещанного прироста подтверждается на школах вне обучения.

    Оптимизатор несёт часы туда, где оценка случайно завышена, и поэтому обещает
    больше, чем будет на деле. План выбирается на одной половине школ и проверяется
    на другой. На половине данных шум вдвое сильнее, поэтому берём половину разрыва.
    """
    schools = np.random.default_rng(seed).permutation(sorted(inputs.student_school.unique()))
    halves = [schools[: len(schools) // 2], schools[len(schools) // 2 :]]
    models = []
    for half in halves:
        _, terms, beta, cov = _fit_topics(inputs, half)
        models.append(replace(model, terms=terms, beta=beta, cov=cov))
    promised, confirmed = 0.0, 0.0
    for trained, held_out in [(models[0], models[1]), (models[1], models[0])]:
        plan = optimize_hours(trained)
        promised += forecast(trained, plan)["gain"]
        confirmed += forecast(held_out, plan)["gain"]
    if promised <= 0:
        return 1.0
    optimism = max(promised - confirmed, 0.0) / promised
    return float(max(1 - optimism / 2, 0.0))


def discount_forecast(result: dict, factor: float) -> dict:
    table = result["table"].copy()
    table[["gain", "low", "high"]] *= factor
    return {
        "table": table,
        "gain": result["gain"] * factor,
        "low": result["low"] * factor,
        "high": result["high"] * factor,
        "realism": factor,
    }


def optimize_hours(model: HoursModel) -> dict[str, int]:
    """Часы раздаются по одному туда, где следующий час полезнее. Сумма не меняется.

    Отдача вогнутая и суммируется по темам, поэтому такой перебор даёт точный оптимум.
    """
    budget = sum(model.planned.values())
    plan = dict(model.low)
    values = {t: model.value_per_log_hour(t) for t in model.topics}
    if all(v == 0 for v in values.values()):
        return dict(model.planned)
    for _ in range(budget - sum(plan.values())):
        candidates = [t for t in model.topics if plan[t] < model.high[t]]
        best = max(candidates, key=lambda t: values[t] * np.log((plan[t] + 1) / plan[t]))
        plan[best] += 1
    return plan


def forecast(model: HoursModel, plan: dict[str, int], seed: int = 7) -> dict:
    """Прогноз прироста в процентных пунктах относительно программы, с 95% интервалом."""
    shift = {t: float(np.log(plan[t] / model.planned[t])) for t in model.topics}
    rng = np.random.default_rng(seed)
    point, samples = {}, {}
    for target in model.topics:
        shifts = np.array([shift[name] for name, _ in model.terms[target]])
        if shifts.size == 0:
            point[target] = 0.0
            samples[target] = np.zeros(FORECAST_DRAWS)
            continue
        point[target] = float(model.beta[target] @ shifts)
        draws = rng.multivariate_normal(model.beta[target], model.cov[target], FORECAST_DRAWS)
        samples[target] = draws @ shifts
    matrix = np.column_stack([samples[t] for t in model.topics])
    total = matrix.mean(axis=1)
    rows = []
    for position, target in enumerate(model.topics):
        rows.append(
            {
                "topic_id": target,
                "planned": model.planned[target],
                "new": plan[target],
                "gain": point[target],
                "low": float(np.quantile(matrix[:, position], 0.025)),
                "high": float(np.quantile(matrix[:, position], 0.975)),
            }
        )
    return {
        "table": pd.DataFrame(rows),
        "gain": float(np.mean(list(point.values()))),
        "low": float(np.quantile(total, 0.025)),
        "high": float(np.quantile(total, 0.975)),
    }


# --- Сравнение групп школ (реальные данные) --------------------------------------------

GROUPS = [
    # (название, колонка, первая группа, вторая группа)
    ("Местность", "school_type", "Городская", "Сельская"),
    ("Язык обучения", "language", "казахский", "русский"),
]


def compare_groups(ds: Dataset, scores: pd.DataFrame) -> pd.DataFrame:
    """Город и село, казахский и русский язык: как есть и при равных домашних ресурсах учеников.

    Обе группы входят в одну модель, поэтому разница по языку учитывает и местность, и наоборот.
    """
    frame = ds.students.merge(ds.schools[["school_id", "school_type"]], on="school_id")
    if "language" not in frame.columns:
        frame = frame.merge(ds.schools[["school_id", "language"]], on="school_id")
    frame = frame.set_index("student_id")
    keep = pd.Series(True, index=frame.index)
    for _, column, first, second in GROUPS:
        keep &= frame[column].isin([first, second])
    frame = frame[keep]

    covariate = frame["pretest"].fillna(frame["pretest"].mean())
    school_mean = covariate.groupby(frame["school_id"]).transform("mean")
    design = pd.DataFrame(index=frame.index)
    design["const"] = 1.0
    design["covariate"] = (covariate - covariate.mean()) / covariate.std()
    design["school_covariate"] = (school_mean - school_mean.mean()) / school_mean.std()
    for name, column, first, _ in GROUPS:
        design[name] = (frame[column] == first).astype(float)

    wide = scores.pivot(index="student_id", columns="topic_id", values="clean")
    rows = []
    for scope in [ALL, *ds.topics["topic_id"]]:
        y = (wide.mean(axis=1) if scope == ALL else wide[scope]).reindex(frame.index).dropna()
        fit = _fit_mixed(y, design.loc[y.index], frame.loc[y.index, "school_id"])
        for name, column, first, second in GROUPS:
            position = fit.index(name)
            diff, se = float(fit.beta[position]), float(np.sqrt(fit.cov[position, position]))
            groups = frame.loc[y.index, column]
            rows.append(
                {
                    "scope": scope,
                    "factor": name,
                    "first": first,
                    "second": second,
                    "naive_first": float(y[groups == first].mean()),
                    "naive_second": float(y[groups == second].mean()),
                    "n_first": int((groups == first).sum()),
                    "n_second": int((groups == second).sum()),
                    "diff": diff,
                    "low": diff - Z95 * se,
                    "high": diff + Z95 * se,
                    "p": float(2 * stats.norm.sf(abs(diff / se))),
                }
            )
    table = pd.DataFrame(rows)
    is_topic = table["scope"] != ALL
    table["p_adj"] = table["p"]
    table.loc[is_topic, "p_adj"] = multipletests(table.loc[is_topic, "p"], method="fdr_bh")[1]
    table["proven"] = table["p_adj"] < ALPHA
    return table


# --- Всё вместе -----------------------------------------------------------------------


@dataclass
class Bundle:
    quality: pd.DataFrame
    scores: pd.DataFrame
    diagnosis: pd.DataFrame
    weak: float
    textbooks: dict | None = None
    hours: HoursModel | None = None
    best_plan: dict[str, int] | None = None
    best_forecast: dict | None = None
    groups: pd.DataFrame | None = None


def run_analysis(ds: Dataset) -> Bundle:
    quality = item_quality(ds)
    scores = student_scores(ds, quality)
    weak = weak_threshold(ds, scores)
    diagnosis = diagnose_topics(ds, scores, quality, weak=weak)
    if ds.is_timss:
        return Bundle(quality, scores, diagnosis, weak, groups=compare_groups(ds, scores))
    books = sorted(ds.schools["textbook"].unique())
    fits, hours, inputs = fit_models(ds, scores, books)
    textbooks = compare_textbooks(ds, scores, fits, books)
    plan = optimize_hours(hours)
    honest = discount_forecast(forecast(hours, plan), plan_realism(hours, inputs))
    return Bundle(quality, scores, diagnosis, weak, textbooks, hours, plan, honest)
