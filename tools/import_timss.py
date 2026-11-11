"""Переводит данные TIMSS 2023 (8 класс, Казахстан, математика) в формат проекта.

Источник: TIMSS 2023 International Database, IEA, https://timss2023.org/data
Использование только в некоммерческих учебных и исследовательских целях.

Запуск:
  python tools/import_timss.py data/timss_raw/T23_Data_SPSS_G8.zip data/timss_raw/T23_ItemInformation_G8.xlsx

Результат: data/timss_kz/*.csv и meta.json. Из архива читаются только файлы Казахстана,
они распаковываются под фиксированными именами в data/timss_raw/kaz/.
"""
from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "timss_kz"
COUNTRY = "KAZ"
FILES = {"BSA": "achievement", "BSG": "student", "BCG": "school"}
LANGUAGES = {39: "казахский", 29: "русский"}
# BCBG05B: 1 город, 2 пригород, 3 средний город, 4 малый город или село, 5 отдалённое село
LOCATION = {1: "Городская", 2: "Городская", 3: "Городская", 4: "Сельская", 5: "Сельская"}

TOPICS = {
    "Integers": ("T1", "Целые числа", "Бүтін сандар"),
    "Fractions and Decimals": ("T2", "Дроби и десятичные дроби", "Жай және ондық бөлшектер"),
    "Ratio, Proportion, and Percent": ("T3", "Отношения, пропорции и проценты", "Қатынас, пропорция және пайыз"),
    "Expressions, Operations, and Equations": ("T4", "Выражения, действия и уравнения", "Өрнектер, амалдар және теңдеулер"),
    "Relationships and Functions": ("T5", "Зависимости и функции", "Тәуелділіктер және функциялар"),
    "Geometry and Measurement": ("T6", "Геометрия и измерения", "Геометрия және өлшеу"),
    # В бумажной версии теста та же тема называется иначе.
    "Geometric Shapes and Measurements": ("T6", "Геометрия и измерения", "Геометрия және өлшеу"),
    "Data": ("T7", "Данные", "Деректер"),
    "Probability": ("T8", "Вероятность", "Ықтималдық"),
}
# Предпосылки между темами: допущение команды по логике школьной программы, не данные TIMSS.
PREREQS = {"T3": "T2", "T4": "T1", "T5": "T4"}


def extract_country_files(archive: Path) -> dict[str, Path]:
    target = archive.parent / "kaz"
    target.mkdir(exist_ok=True)
    found = {}
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            match = re.search(rf"(BSA|BSG|BCG){COUNTRY}M8\.sav$", Path(name).name, flags=re.I)
            if not match:
                continue
            kind = match.group(1).upper()
            path = target / f"{kind}{COUNTRY}M8.sav"  # своё имя, а не путь из архива
            with zf.open(name) as source, open(path, "wb") as sink:
                sink.write(source.read())
            found[kind] = path
    missing = set(FILES) - set(found)
    if missing:
        raise SystemExit(f"В архиве нет файлов Казахстана: {', '.join(sorted(missing))}")
    return found


def math_items(item_info: Path, columns: set[str]) -> pd.DataFrame:
    frames = [pd.read_excel(item_info, sheet_name=s) for s in pd.ExcelFile(item_info).sheet_names]
    items = pd.concat(frames, ignore_index=True)
    items.columns = [c.strip() for c in items.columns]
    items = items[(items["Subject"] == "M") & (items["Scaling Status"] == "Yes")]
    items = items[items["Item ID"].isin(columns)].drop_duplicates("Item ID")
    unknown = set(items["Topic Area"]) - set(TOPICS)
    if unknown:
        raise SystemExit(f"Неизвестные темы TIMSS: {unknown}")
    return items


def full_credit_codes(labels: dict, item: pd.Series) -> set:
    """Коды полного балла по меткам SPSS.

    Кодировка разная: у вопросов прошлых циклов верный ответ — 10 или 20, у новых — 1,
    у вопросов с выбором верный вариант помечен звёздочкой.
    """
    texts = {code: str(text).strip().lower() for code, text in labels.items()}
    if item["Item Type"] == "MC":
        return {code for code, text in texts.items() if text.endswith("*")}
    correct = {
        code for code, text in texts.items()
        if "correct" in text and "incorrect" not in text and "partial" not in text
    }
    if int(item["Maximum Points"]) == 2 and correct:
        return {max(correct)}  # у двухбалльных полный балл — старший код
    return correct


def score(values: pd.Series, labels: dict, item: pd.Series) -> pd.Series:
    """1 — полный балл, 0 — неверно или пропущено, NaN — не дошёл до вопроса или не получал его."""
    texts = {code: str(text).lower() for code, text in labels.items()}
    not_reached = {code for code, text in texts.items() if "not reached" in text or "not admin" in text}
    full = full_credit_codes(labels, item)
    if not full:
        raise SystemExit(f"Не нашёл код верного ответа для {item['Item ID']}: {labels}")
    result = pd.Series(np.nan, index=values.index)
    given = values.notna() & ~values.isin(not_reached)
    result[given] = values[given].isin(full).astype(float)  # пропуск «omitted» считается неверным
    return result


def main(archive: Path, item_info: Path) -> None:
    files = extract_country_files(archive)
    achievement, meta_a = pyreadstat.read_sav(files["BSA"], user_missing=True)
    student, _ = pyreadstat.read_sav(files["BSG"], user_missing=True, usecols=["IDSTUD", "BSBGHER"])
    school, _ = pyreadstat.read_sav(files["BCG"], user_missing=True, usecols=["IDSCHOOL", "BCBG05B"])

    items = math_items(item_info, set(achievement.columns))
    achievement = achievement[achievement["IDBOOK"] > 0]

    responses = []
    for _, item in items.iterrows():
        scored = score(achievement[item["Item ID"]], meta_a.variable_value_labels.get(item["Item ID"], {}), item)
        part = pd.DataFrame({"student_id": achievement["IDSTUD"], "item_id": item["Item ID"], "correct": scored})
        responses.append(part.dropna())
    responses = pd.concat(responses, ignore_index=True)
    responses["student_id"] = responses["student_id"].astype(int).map(lambda s: f"U{s}")
    responses["correct"] = responses["correct"].astype(int)

    topics = pd.DataFrame(
        [
            {"topic_id": tid, "name_ru": ru, "name_kz": kz, "planned_hours": 1, "prereqs": PREREQS.get(tid, "")}
            for tid, ru, kz in TOPICS.values()
        ]
    ).drop_duplicates("topic_id")
    items_out = pd.DataFrame({"item_id": items["Item ID"], "topic_id": items["Topic Area"].map(lambda t: TOPICS[t][0])})
    # Вопросы бумажной версии в Казахстане не давали: у них нет ответов.
    items_out = items_out[items_out["item_id"].isin(responses["item_id"])]
    topics = topics[topics["topic_id"].isin(items_out["topic_id"])]

    language = achievement.groupby("IDSCHOOL")["ITLANG_SA"].agg(lambda s: s.mode().iat[0])
    schools = school.set_index("IDSCHOOL").reindex(language.index)
    schools_out = pd.DataFrame(
        {
            "school_id": [f"S{int(s)}" for s in language.index],
            "school_type": schools["BCBG05B"].map(LOCATION).fillna("Нет данных").to_numpy(),
            "language": language.map(LANGUAGES).fillna("другой").to_numpy(),
            "textbook": "—",
        }
    )

    her = student.set_index("IDSTUD")["BSBGHER"]
    her = her.where(her.between(0, 20))  # коды пропусков выше шкалы
    students = pd.DataFrame(
        {
            "student_id": achievement["IDSTUD"].astype(int).map(lambda s: f"U{s}"),
            "school_id": achievement["IDSCHOOL"].astype(int).map(lambda s: f"S{s}"),
            "class_id": achievement["IDCLASS"].astype(int).map(lambda s: f"C{s}"),
            "pretest": achievement["IDSTUD"].map(her).to_numpy(),
            "language": achievement["ITLANG_SA"].map(LANGUAGES).fillna("другой").to_numpy(),
        }
    )
    students = students[students["student_id"].isin(responses["student_id"])]

    OUT.mkdir(parents=True, exist_ok=True)
    topics.to_csv(OUT / "topics.csv", index=False)
    schools_out.to_csv(OUT / "schools.csv", index=False)
    students.to_csv(OUT / "students.csv", index=False)
    items_out.to_csv(OUT / "items.csv", index=False)
    responses.to_csv(OUT / "responses.csv", index=False)
    meta = {
        "kind": "timss",
        "title": "TIMSS 2023 · Казахстан · математика, 8 класс",
        "source": "TIMSS 2023 International Database, IEA (timss2023.org/data). Некоммерческое учебное использование.",
        "covariate": "домашние образовательные ресурсы (шкала TIMSS)",
        "prereqs_note": "Связи между темами заданы командой по логике программы, это не данные TIMSS.",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Школ: {len(schools_out)}, учеников: {len(students)}, вопросов: {len(items_out)}, ответов: {len(responses)}")
    print(schools_out.groupby(["school_type", "language"]).size())
    print(items_out["topic_id"].value_counts().sort_index())
    print("Доля верных:", round(responses["correct"].mean(), 3), "| пропусков ресурсов:", int(students["pretest"].isna().sum()))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
