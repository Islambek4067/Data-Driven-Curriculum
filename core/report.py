"""Документ для методического совета на русском и казахском."""
from __future__ import annotations

import io
from dataclasses import dataclass

from docx import Document
from docx.shared import Pt

from core.analysis import ALL, Bundle
from core.data import Dataset

STRINGS = {
    "ru": {
        "title": "Обоснование решения для методического совета",
        "scope": "Данные: {schools} школ, {students} учеников, {items} вопросов теста.",
        "demo": "Документ сформирован на синтетических демо-данных.",
        "book_h": "Учебник",
        "book": "Рекомендуется учебник {best}. С учётом уровня школ его средний результат {x}%, у учебника {other} {y}% (разница {d} п.п., 95% интервал {l}–{h}).",
        "book_unproven": "Различие между учебниками {best} и {other} не доказано: разница {d} п.п., 95% интервал {l}–{h}. Менять учебник только на основании этих данных не следует.",
        "naive": "Простое сравнение средних показало бы лидером учебник {book} ({value}%), но его выбирают школы с более высоким входным уровнем.",
        "topic_other": "По теме «{topic}» сильнее учебник {book} (+{d} п.п.). При выборе учебника {best} эту тему стоит усилить дополнительными материалами.",
        "hard_h": "Сложные разделы",
        "prereq": "«{topic}» ({clean}%): причина в базовой теме «{root}» ({root_score}%). Усвоившие базу решают тему на {with_base}%, не усвоившие на {without}%.",
        "hard": "«{topic}» ({clean}%): база усвоена, тема трудна сама по себе.",
        "bad": "«{topic}»: низкий балл вызван бракованными вопросами ({n}). Без них результат {clean}%, вопросы нужно заменить.",
        "none": "Слабых разделов не выявлено.",
        "hours_h": "Учебные часы",
        "hours": "Общий объём {total} ч не меняется. Ожидаемый прирост среднего балла: {g} п.п. (95% интервал {l}–{h}).",
        "move": "«{topic}»: {old} → {new} ч{weak}.",
        "weak": " (эффект часов не доказан)",
        "frozen": "По теме «{topic}» часы во всех школах одинаковы, поэтому вывод сделать нельзя. Нужен пилот.",
        "limit_h": "Ограничения",
        "limit": "Выводы основаны на наблюдательных данных: учебники и часы распределялись не случайно. Модель учитывает входной балл, тип школы и язык обучения, но не может учесть всё. Перед массовым внедрением рекомендуется пилот.",
    },
    "kz": {
        "title": "Әдістемелік кеңеске арналған шешім негіздемесі",
        "scope": "Деректер: {schools} мектеп, {students} оқушы, {items} тест сұрағы.",
        "demo": "Құжат синтетикалық демо-деректер негізінде жасалды.",
        "book_h": "Оқулық",
        "book": "{best} оқулығы ұсынылады. Мектептердің деңгейі ескерілгенде оның орташа нәтижесі {x}%, {other} оқулығында {y}% (айырмашылық {d} п.п., 95% сенімділік аралығы {l}–{h}).",
        "book_unproven": "{best} және {other} оқулықтарының айырмашылығы дәлелденбеді: айырмашылық {d} п.п., 95% сенімділік аралығы {l}–{h}. Тек осы деректерге сүйеніп оқулықты ауыстыруға болмайды.",
        "naive": "Орташа балдарды қарапайым салыстырғанда {book} оқулығы алда болып көрінер еді ({value}%), алайда оны кіріс деңгейі жоғары мектептер таңдайды.",
        "topic_other": "«{topic}» тақырыбында {book} оқулығы күштірек (+{d} п.п.). {best} оқулығын таңдағанда бұл тақырыпты қосымша материалдармен күшейту қажет.",
        "hard_h": "Қиын меңгерілетін бөлімдер",
        "prereq": "«{topic}» ({clean}%): себебі «{root}» негізгі тақырыбында ({root_score}%). Негізгі тақырыпты меңгергендердің нәтижесі {with_base}%, меңгермегендердікі {without}%.",
        "hard": "«{topic}» ({clean}%): негізгі тақырыптар меңгерілген, тақырыптың өзі қиын.",
        "bad": "«{topic}»: төмен балдың себебі сапасыз сұрақтар ({n}). Оларсыз нәтиже {clean}%, сұрақтарды ауыстыру қажет.",
        "none": "Әлсіз бөлімдер анықталған жоқ.",
        "hours_h": "Оқу сағаттары",
        "hours": "Жалпы көлем ({total} сағат) өзгермейді. Орташа балдың күтілетін өсімі: {g} п.п. (95% сенімділік аралығы {l}–{h}).",
        "move": "«{topic}»: {old} → {new} сағат{weak}.",
        "weak": " (сағат әсері дәлелденбеді)",
        "frozen": "«{topic}» тақырыбы бойынша сағат саны барлық мектепте бірдей, сондықтан қорытынды жасау мүмкін емес. Пилоттық сынақ қажет.",
        "limit_h": "Шектеулер",
        "limit": "Қорытындылар бақылау деректеріне негізделген: оқулықтар мен сағаттар кездейсоқ бөлінбеген. Модель кіріс балын, мектеп түрін және оқыту тілін ескереді, бірақ барлық факторды ескере алмайды. Жаппай енгізер алдында пилоттық сынақ өткізу ұсынылады.",
    },
}


def fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",").replace("-", "−")


@dataclass
class Report:
    title: str
    lead: list[str]
    sections: list[tuple[str, list[str]]]


def _pair(contrasts, first: str, second: str, scope: str):
    """Контраст «первый минус второй», в какую бы сторону он ни был посчитан."""
    table = contrasts[contrasts["scope"] == scope]
    row = table[(table["first"] == first) & (table["second"] == second)]
    if not row.empty:
        r = row.iloc[0]
        return r["diff"], r["low"], r["high"], bool(r["proven"])
    r = table[(table["first"] == second) & (table["second"] == first)].iloc[0]
    return -r["diff"], -r["high"], -r["low"], bool(r["proven"])


def _book_section(ds: Dataset, bundle: Bundle, s: dict, names: dict[str, str]) -> list[str]:
    means = bundle.textbooks["means"]
    overall = means[means["scope"] == ALL].set_index("textbook")
    ranking = overall["adjusted"].sort_values(ascending=False)
    best, other = ranking.index[0], ranking.index[1]
    contrasts = bundle.textbooks["contrasts"]
    diff, low, high, _ = _pair(contrasts, best, other, ALL)
    proven = low > 0 or high < 0
    values = {"best": best, "other": other, "d": fmt(diff), "l": fmt(low), "h": fmt(high)}
    if proven:
        lines = [s["book"].format(x=fmt(ranking[best]), y=fmt(ranking[other]), **values)]
    else:
        return [s["book_unproven"].format(**values)]

    naive_best = overall["naive"].idxmax()
    profile = bundle.textbooks["profile"].set_index("textbook")["pretest"]
    if naive_best != best and profile[naive_best] > profile[best]:
        lines.append(s["naive"].format(book=naive_best, value=fmt(overall.loc[naive_best, "naive"])))

    for book in ranking.index[1:]:
        for tid in ds.topics["topic_id"]:
            topic_diff, _, _, topic_proven = _pair(contrasts, book, best, tid)
            if topic_proven and topic_diff > 0:
                lines.append(s["topic_other"].format(topic=names[tid], book=book, d=fmt(topic_diff), best=best))
    return lines


def _hard_section(bundle: Bundle, s: dict, names: dict[str, str]) -> list[str]:
    table = bundle.diagnosis.set_index("topic_id")
    lines = []
    for tid, row in table[table["status"] != "ok"].sort_values("clean").iterrows():
        if row["status"] == "prereq":
            lines.append(
                s["prereq"].format(
                    topic=names[tid],
                    clean=fmt(row["clean"], 0),
                    root=names[row["root"]],
                    root_score=fmt(table.loc[row["root"], "clean"], 0),
                    with_base=fmt(row["with_base"], 0),
                    without=fmt(row["without_base"], 0),
                )
            )
        elif row["status"] == "hard":
            lines.append(s["hard"].format(topic=names[tid], clean=fmt(row["clean"], 0)))
        else:
            lines.append(s["bad"].format(topic=names[tid], n=row["n_bad"], clean=fmt(row["clean"], 0)))
    return lines or [s["none"]]


def _hours_section(bundle: Bundle, s: dict, names: dict[str, str]) -> list[str]:
    model, plan, result = bundle.hours, bundle.best_plan, bundle.best_forecast
    lines = [
        s["hours"].format(
            total=sum(model.planned.values()),
            g=fmt(result["gain"]),
            l=fmt(result["low"]),
            h=fmt(result["high"]),
        )
    ]
    for tid in model.topics:
        if plan[tid] != model.planned[tid]:
            weak = "" if model.is_reliable(tid) else s["weak"]
            lines.append(s["move"].format(topic=names[tid], old=model.planned[tid], new=plan[tid], weak=weak))
    lines.extend(s["frozen"].format(topic=names[tid]) for tid in model.frozen)
    return lines


def build_report(ds: Dataset, bundle: Bundle, lang: str) -> Report:
    s = STRINGS[lang]
    names = ds.topics.set_index("topic_id")[f"name_{lang}"].to_dict()
    lead = [s["scope"].format(schools=len(ds.schools), students=len(ds.students), items=len(ds.items))]
    if ds.is_demo:
        lead.append(s["demo"])
    return Report(
        title=s["title"],
        lead=lead,
        sections=[
            (s["book_h"], _book_section(ds, bundle, s, names)),
            (s["hard_h"], _hard_section(bundle, s, names)),
            (s["hours_h"], _hours_section(bundle, s, names)),
            (s["limit_h"], [s["limit"]]),
        ],
    )


def to_docx(report: Report) -> bytes:
    document = Document()
    document.styles["Normal"].font.name = "Arial"
    document.styles["Normal"].font.size = Pt(11)
    document.add_heading(report.title, level=1)
    for line in report.lead:
        document.add_paragraph(line)
    for heading, lines in report.sections:
        document.add_heading(heading, level=2)
        for line in lines:
            document.add_paragraph(line)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()
