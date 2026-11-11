"""Загрузка и проверка данных: папка с CSV или zip-архив с теми же файлами."""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

SCHEMA = {
    "topics": ["topic_id", "name_ru", "name_kz", "planned_hours", "prereqs"],
    "schools": ["school_id", "school_type", "language", "textbook"],
    "students": ["student_id", "school_id", "pretest"],
    "items": ["item_id", "topic_id"],
    "responses": ["student_id", "item_id", "correct"],
    "hours": ["school_id", "topic_id", "hours"],
}
OPTIONAL = {"coverage": ["textbook", "topic_id", "exercises", "examples"]}


class DataError(ValueError):
    """Понятное методисту сообщение о том, что не так с файлами."""


@dataclass
class Dataset:
    topics: pd.DataFrame
    schools: pd.DataFrame
    students: pd.DataFrame
    items: pd.DataFrame
    responses: pd.DataFrame
    hours: pd.DataFrame
    coverage: pd.DataFrame | None
    key: str
    is_demo: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def is_timss(self) -> bool:
        """Реальные данные TIMSS: нет учебников и часов по темам."""
        return self.meta.get("kind") == "timss"


def _check_columns(name: str, frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise DataError(f"В файле {name}.csv нет колонок: {', '.join(missing)}")


def _check_references(frames: dict[str, pd.DataFrame], is_timss: bool = False) -> None:
    pairs = [
        ("students", "school_id", "schools", "school_id"),
        ("items", "topic_id", "topics", "topic_id"),
        ("responses", "student_id", "students", "student_id"),
        ("responses", "item_id", "items", "item_id"),
    ]
    if not is_timss:
        pairs += [("hours", "school_id", "schools", "school_id"), ("hours", "topic_id", "topics", "topic_id")]
    for child, child_col, parent, parent_col in pairs:
        unknown = set(frames[child][child_col]) - set(frames[parent][parent_col])
        if unknown:
            sample = ", ".join(map(str, sorted(unknown)[:3]))
            raise DataError(
                f"В {child}.csv есть {child_col}, которых нет в {parent}.csv (например: {sample})"
            )
    if not set(frames["responses"]["correct"].unique()) <= {0, 1}:
        raise DataError("В responses.csv колонка correct может содержать только 0 и 1")
    if not is_timss:
        if frames["schools"]["textbook"].nunique() < 2:
            raise DataError("Для сравнения в schools.csv нужно минимум два разных учебника")
        expected = len(frames["schools"]) * len(frames["topics"])
        if frames["hours"].drop_duplicates(["school_id", "topic_id"]).shape[0] != expected:
            raise DataError("В hours.csv часы должны быть указаны для каждой пары «школа и тема»")
        if (frames["hours"]["hours"] <= 0).any() or (frames["topics"]["planned_hours"] <= 0).any():
            raise DataError("Часы в hours.csv и topics.csv должны быть больше нуля")
    known = set(frames["topics"]["topic_id"])
    for raw in frames["topics"]["prereqs"]:
        unknown_prereqs = {p for p in str(raw).split(";") if p} - known
        if unknown_prereqs:
            raise DataError(f"В topics.csv указаны неизвестные темы: {', '.join(sorted(unknown_prereqs))}")


def _build(frames: dict[str, pd.DataFrame], key: str, is_demo: bool, meta: dict | None = None) -> Dataset:
    meta = meta or {}
    is_timss = meta.get("kind") == "timss"
    if is_timss:
        frames.setdefault("hours", pd.DataFrame(columns=SCHEMA["hours"]))
    for name, columns in SCHEMA.items():
        if name not in frames:
            raise DataError(f"Не хватает файла {name}.csv")
        _check_columns(name, frames[name], columns)
    frames["topics"]["prereqs"] = frames["topics"]["prereqs"].fillna("").astype(str)
    for column in ["student_id", "school_id"]:
        frames["students"][column] = frames["students"][column].astype(str)
    _check_references(frames, is_timss)
    coverage = frames.get("coverage")
    if coverage is not None:
        _check_columns("coverage", coverage, OPTIONAL["coverage"])
    return Dataset(
        topics=frames["topics"],
        schools=frames["schools"],
        students=frames["students"],
        items=frames["items"],
        responses=frames["responses"],
        hours=frames["hours"],
        coverage=coverage,
        key=key,
        is_demo=is_demo,
        meta=meta,
    )


def load_dir(path: Path, is_demo: bool = False) -> Dataset:
    frames = {}
    digest = hashlib.sha256()
    for name in [*SCHEMA, *OPTIONAL]:
        file = path / f"{name}.csv"
        if file.exists():
            digest.update(file.read_bytes())
            frames[name] = pd.read_csv(file, dtype={"prereqs": str, "student_id": str, "school_id": str})
    meta_file = path / "meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    return _build(frames, digest.hexdigest(), is_demo, meta)


def load_zip(content: bytes) -> Dataset:
    """Читает из архива только ожидаемые CSV, сами файлы на диск не распаковываются."""
    frames = {}
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            by_name = {Path(n).name: n for n in archive.namelist() if not n.endswith("/")}
            for name in [*SCHEMA, *OPTIONAL]:
                entry = by_name.get(f"{name}.csv")
                if entry is not None:
                    with archive.open(entry) as handle:
                        frames[name] = pd.read_csv(handle, dtype={"prereqs": str})
    except zipfile.BadZipFile as error:
        raise DataError("Файл не похож на zip-архив") from error
    return _build(frames, hashlib.sha256(content).hexdigest(), False)
