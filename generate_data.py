"""Генератор синтетических данных с заранее известной правдой.

Анализ (core/analysis.py) эту правду не видит. Она нужна только validate.py и
вкладке «Проверка модели», чтобы доказать: модель находит то, что заложено.

Заложено:
  * сильные школы чаще выбирают учебник А, хотя по сути лучше учебник Б;
  * в «Линейной функции» два сломанных вопроса, тема только кажется трудной;
  * «Алгебраические дроби» проседают из-за провала в «Разложении на множители»;
  * «Текстовые задачи» действительно трудные сами по себе;
  * часы влияют на темы по-разному, с убывающей отдачей;
  * у «Линейных неравенств» часы везде одинаковые, вывод о них сделать нельзя.

Запуск: python generate_data.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

SEED = 20261109
DATA_DIR = Path(__file__).parent / "data" / "demo"

N_SCHOOLS = 120
CLASSES_PER_SCHOOL = 2
STUDENTS_PER_CLASS = 24
ITEMS_PER_TOPIC = 5
TOTAL_HOURS = 102
GUESS = 0.15
TEXTBOOKS = ["А", "Б", "В"]

# id, русское название, казахское название, часы по программе, лёгкость, отдача от часов
TOPICS = [
    ("T01", "Степень с натуральным показателем", "Натурал көрсеткішті дәреже", 9, 1.30, 0.10),
    ("T02", "Одночлены", "Бірмүшелер", 7, 0.95, 0.10),
    ("T03", "Сложение и вычитание многочленов", "Көпмүшелерді қосу және азайту", 8, 0.95, 0.15),
    ("T04", "Умножение многочленов", "Көпмүшелерді көбейту", 9, 0.75, 0.20),
    ("T05", "Формулы сокращённого умножения", "Қысқаша көбейту формулалары", 10, 0.60, 0.50),
    ("T06", "Разложение на множители", "Көпмүшені көбейткіштерге жіктеу", 6, -0.55, 0.95),
    ("T07", "Алгебраические дроби", "Алгебралық бөлшектер", 8, -0.05, 0.35),
    ("T08", "Линейные уравнения", "Сызықтық теңдеулер", 10, 1.25, 0.15),
    ("T09", "Линейная функция", "Сызықтық функция", 9, 0.60, 0.30),
    ("T10", "Системы линейных уравнений", "Сызықтық теңдеулер жүйелері", 8, 0.70, 0.40),
    ("T11", "Текстовые задачи", "Мәтіндік есептер", 12, -0.65, 0.60),
    ("T12", "Линейные неравенства", "Сызықтық теңсіздіктер", 6, 0.70, 0.30),
]

# тема -> {базовая тема: сила передачи}
PREREQS = {
    "T02": {"T01": 0.20},
    "T03": {"T02": 0.20},
    "T04": {"T03": 0.20},
    "T05": {"T04": 0.20},
    "T06": {"T05": 0.15},
    "T07": {"T06": 1.00},
    "T09": {"T08": 0.20},
    "T10": {"T08": 0.20, "T09": 0.10},
    "T11": {"T10": 0.15},
    "T12": {"T08": 0.20},
}

# Истинный вклад учебников в логитах относительно учебника А
TEXTBOOK_EFFECT = {
    "Б": {"T05": 0.50, "T06": 0.55, "T07": 0.35, "T09": -0.40, "T11": 0.50},
    "В": {"T10": -0.40},
}

# Упражнения и разобранные примеры: в полной версии извлекаются из PDF учебника
BASE_EXERCISES = {t[0]: 14 for t in TOPICS}
COVERAGE_OVERRIDE = {
    ("А", "T05"): (9, 3), ("Б", "T05"): (22, 8), ("В", "T05"): (13, 4),
    ("А", "T06"): (5, 2), ("Б", "T06"): (21, 9), ("В", "T06"): (12, 4),
    ("А", "T07"): (8, 3), ("Б", "T07"): (17, 6), ("В", "T07"): (12, 4),
    ("А", "T09"): (24, 9), ("Б", "T09"): (10, 3), ("В", "T09"): (16, 5),
    ("А", "T10"): (15, 5), ("Б", "T10"): (15, 5), ("В", "T10"): (6, 2),
    ("А", "T11"): (11, 4), ("Б", "T11"): (26, 10), ("В", "T11"): (13, 4),
}

BAD_ITEMS_PER_TOPIC = {"T09": 2, "T03": 1, "T12": 1}
FIXED_HOURS_TOPIC = "T12"
SCHOOL_TYPES = {"Гимназия": 0.65, "Городская": 0.10, "Сельская": -0.40}
TOPIC_IDS = [t[0] for t in TOPICS]


@dataclass
class World:
    """Всё, что нужно, чтобы пересчитать ответы при других учебниках и часах."""

    schools: pd.DataFrame
    students: pd.DataFrame
    items: pd.DataFrame
    hours: pd.DataFrame
    topics: pd.DataFrame
    theta: np.ndarray
    noise: dict[str, np.ndarray]
    school_index: np.ndarray


def _item_probability(mastery: np.ndarray, a: float, d: float) -> np.ndarray:
    return GUESS + (1 - GUESS) * expit(a * (mastery - d))


def mastery_by_topic(
    world: World,
    textbook_by_school: np.ndarray,
    hours_by_topic: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Латентное усвоение каждой темы при заданных учебниках и часах."""
    idx = world.school_index
    planned = world.topics.set_index("topic_id")["planned_hours"]
    ease = {t[0]: t[4] for t in TOPICS}
    gamma = {t[0]: t[5] for t in TOPICS}
    mastery: dict[str, np.ndarray] = {}
    for tid in TOPIC_IDS:  # темы идут так, что база всегда раньше
        value = world.theta + ease[tid] + world.noise[tid]
        book = textbook_by_school[idx]
        for name, effects in TEXTBOOK_EFFECT.items():
            if tid in effects:
                value = value + (book == name) * effects[tid]
        value = value + gamma[tid] * np.log(hours_by_topic[tid][idx] / planned[tid])
        for prereq, rho in PREREQS.get(tid, {}).items():
            # Передаётся только то, насколько база отстала от общих сил ученика.
            value = value + rho * (mastery[prereq] - world.theta)
        mastery[tid] = value
    return mastery


def expected_good_item_score(world: World, mastery: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Ожидаемый процент правильных по исправным вопросам темы, для каждого ученика."""
    good = world.items[~world.items["is_broken"]]
    scores = {}
    for tid, group in good.groupby("topic_id"):
        probabilities = [
            _item_probability(mastery[tid], row.a, row.d) for row in group.itertuples()
        ]
        scores[tid] = 100 * np.mean(probabilities, axis=0)
    return scores


def build_world(seed: int = SEED) -> World:
    rng = np.random.default_rng(seed)
    topics = pd.DataFrame(
        [{"topic_id": t[0], "name_ru": t[1], "name_kz": t[2], "planned_hours": t[3]} for t in TOPICS]
    )
    topics["prereqs"] = topics["topic_id"].map(lambda t: ";".join(PREREQS.get(t, {})))

    types = rng.choice(list(SCHOOL_TYPES), N_SCHOOLS, p=[0.22, 0.43, 0.35])
    strength = np.array([SCHOOL_TYPES[t] for t in types]) + rng.normal(0, 0.28, N_SCHOOLS)
    # Перекос сильный, но группы пересекаются: иначе сравнивать было бы не с кем.
    utility = np.column_stack([1.7 * strength, -1.3 * strength, np.zeros(N_SCHOOLS)])
    utility = utility + rng.gumbel(0, 0.75, utility.shape)
    textbooks = np.array(TEXTBOOKS)[utility.argmax(axis=1)]
    schools = pd.DataFrame(
        {
            "school_id": [f"S{i + 1:03d}" for i in range(N_SCHOOLS)],
            "school_type": types,
            "language": rng.choice(["русский", "казахский"], N_SCHOOLS, p=[0.45, 0.55]),
            "district": rng.choice([f"Район {i}" for i in range(1, 7)], N_SCHOOLS),
            "textbook": textbooks,
        }
    )

    n_students = N_SCHOOLS * CLASSES_PER_SCHOOL * STUDENTS_PER_CLASS
    school_index = np.repeat(np.arange(N_SCHOOLS), CLASSES_PER_SCHOOL * STUDENTS_PER_CLASS)
    class_number = np.tile(
        np.repeat(np.arange(CLASSES_PER_SCHOOL), STUDENTS_PER_CLASS), N_SCHOOLS
    )
    class_effect = rng.normal(0, 0.15, (N_SCHOOLS, CLASSES_PER_SCHOOL))
    theta = (
        strength[school_index]
        + class_effect[school_index, class_number]
        + rng.normal(0, 0.75, n_students)
    )
    pretest = np.clip(52 + 17 * (theta + rng.normal(0, 0.45, n_students)), 0, 100).round(1)
    students = pd.DataFrame(
        {
            "student_id": [f"U{i + 1:05d}" for i in range(n_students)],
            "school_id": schools["school_id"].to_numpy()[school_index],
            "class_id": [
                f"{schools['school_id'].iloc[s]}-{c + 1}" for s, c in zip(school_index, class_number)
            ],
            "pretest": pretest,
        }
    )

    rows = []
    for tid in TOPIC_IDS:
        broken = BAD_ITEMS_PER_TOPIC.get(tid, 0)
        for j in range(ITEMS_PER_TOPIC):
            is_broken = j < broken
            rows.append(
                {
                    "item_id": f"{tid}-Q{j + 1}",
                    "topic_id": tid,
                    "is_broken": is_broken,
                    # Сломанный вопрос: почти не зависит от знаний, правильно отвечают редко.
                    "a": 0.02 if is_broken else float(rng.uniform(1.1, 1.6)),
                    "d": float(rng.uniform(60, 90)) if is_broken else float(rng.normal(0, 0.3)),
                }
            )
    items = pd.DataFrame(rows)

    planned = topics.set_index("topic_id")["planned_hours"]
    hour_rows = []
    hours_matrix = np.zeros((N_SCHOOLS, len(TOPIC_IDS)))
    free_topics = [t for t in TOPIC_IDS if t != FIXED_HOURS_TOPIC]
    free_positions = [TOPIC_IDS.index(t) for t in free_topics]
    for s in range(N_SCHOOLS):
        raw = planned[free_topics].to_numpy() * np.exp(rng.normal(0, 0.24, len(free_topics)))
        budget = TOTAL_HOURS - planned[FIXED_HOURS_TOPIC]
        rounded = np.maximum(2, np.round(raw * budget / raw.sum()))
        # Округление сдвигает сумму; добираем до годового бюджета по одному часу.
        step = 0
        while rounded.sum() != budget:
            position = step % len(rounded)
            delta = np.sign(budget - rounded.sum())
            if rounded[position] + delta >= 2:
                rounded[position] += delta
            step += 1
        hours_matrix[s, free_positions] = rounded
        hours_matrix[s, TOPIC_IDS.index(FIXED_HOURS_TOPIC)] = planned[FIXED_HOURS_TOPIC]
        for j, tid in enumerate(TOPIC_IDS):
            hour_rows.append(
                {
                    "school_id": schools["school_id"].iloc[s],
                    "topic_id": tid,
                    "hours": int(hours_matrix[s, j]),
                }
            )
    hours = pd.DataFrame(hour_rows)

    noise = {tid: rng.normal(0, 0.35, n_students) for tid in TOPIC_IDS}
    return World(schools, students, items, hours, topics, theta, noise, school_index)


def hours_arrays(world: World, plan: dict[str, float] | None = None) -> dict[str, np.ndarray]:
    """Часы по школам: фактические или единый план для всех школ."""
    n = len(world.schools)
    if plan is not None:
        return {tid: np.full(n, float(plan[tid])) for tid in TOPIC_IDS}
    wide = world.hours.pivot(index="school_id", columns="topic_id", values="hours")
    wide = wide.loc[world.schools["school_id"]]
    return {tid: wide[tid].to_numpy(dtype=float) for tid in TOPIC_IDS}


def true_scores(
    world: World,
    textbook: str | None = None,
    plan: dict[str, float] | None = None,
) -> dict[str, float]:
    """Средний истинный балл по темам при заданном учебнике и плане часов."""
    books = world.schools["textbook"].to_numpy()
    if textbook is not None:
        books = np.full(len(world.schools), textbook)
    mastery = mastery_by_topic(world, books, hours_arrays(world, plan))
    return {t: float(v.mean()) for t, v in expected_good_item_score(world, mastery).items()}


def true_textbook_contrasts(world: World) -> dict[str, dict[str, float]]:
    scores = {book: true_scores(world, textbook=book) for book in TEXTBOOKS}
    out: dict[str, dict[str, float]] = {}
    for first, second in [("Б", "А"), ("В", "А"), ("Б", "В")]:
        out[f"{first}-{second}"] = {t: scores[first][t] - scores[second][t] for t in TOPIC_IDS}
    return out


def sample_responses(world: World, seed: int = SEED + 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mastery = mastery_by_topic(world, world.schools["textbook"].to_numpy(), hours_arrays(world))
    student_ids = world.students["student_id"].to_numpy()
    frames = []
    for item in world.items.itertuples():
        probability = _item_probability(mastery[item.topic_id], item.a, item.d)
        frames.append(
            pd.DataFrame(
                {
                    "student_id": student_ids,
                    "item_id": item.item_id,
                    "correct": (rng.random(len(student_ids)) < probability).astype(int),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def coverage_table() -> pd.DataFrame:
    rows = []
    for book in TEXTBOOKS:
        for tid in TOPIC_IDS:
            exercises, examples = COVERAGE_OVERRIDE.get((book, tid), (BASE_EXERCISES[tid], 5))
            rows.append({"textbook": book, "topic_id": tid, "exercises": exercises, "examples": examples})
    return pd.DataFrame(rows)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    world = build_world()
    world.topics.to_csv(DATA_DIR / "topics.csv", index=False)
    world.schools.to_csv(DATA_DIR / "schools.csv", index=False)
    world.students.to_csv(DATA_DIR / "students.csv", index=False)
    world.items[["item_id", "topic_id"]].to_csv(DATA_DIR / "items.csv", index=False)
    world.hours.to_csv(DATA_DIR / "hours.csv", index=False)
    coverage_table().to_csv(DATA_DIR / "coverage.csv", index=False)
    responses = sample_responses(world)
    responses.to_csv(DATA_DIR / "responses.csv", index=False)

    truth = {
        "broken_items": world.items.loc[world.items["is_broken"], "item_id"].tolist(),
        "statuses": {"T06": "hard", "T07": "prereq", "T09": "bad_items", "T11": "hard"},
        "root_cause": {"T07": "T06"},
        "contrasts": true_textbook_contrasts(world),
        "no_variation": [FIXED_HOURS_TOPIC],
    }
    (DATA_DIR / "truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Школ: {len(world.schools)}, учеников: {len(world.students)}, ответов: {len(responses)}")
    print(world.schools.groupby(["textbook", "school_type"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
