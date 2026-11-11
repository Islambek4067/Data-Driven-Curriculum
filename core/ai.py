"""ИИ-помощник: отвечает на вопросы методиста только по посчитанным результатам.

Цифры считает статистика (core/analysis.py). Модель получает готовую сводку и
пересказывает её. Каждое число в ответе сверяется со сводкой, выдуманные числа
помечаются в интерфейсе.

Провайдер задаётся ключом в .streamlit/secrets.toml или в переменных окружения:
  XKIRO_API_KEY      ключ xKiro (посредник, Anthropic-совместимый API)
  ANTHROPIC_API_KEY  ключ Anthropic напрямую
  AI_BASE_URL        необязательно: другой адрес API
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import anthropic

from core.analysis import ALL, Bundle
from core.data import Dataset
from core.i18n import term
from core.report import fmt

XKIRO_URL = "https://api.xkiro.com"
MAX_TOKENS = 2000
HISTORY_TURNS = 8
SMALL_INTEGER = 5

# Модели, которые можно выбрать в интерфейсе: (подпись, id у провайдера). Первая выбрана по умолчанию.
XKIRO_MODELS = [
    ("Mistral Large · бесплатно", "mistralai/mistral-large-2512"),
    ("Claude Opus 5.5 · платно", "anthropic/claude-opus-5.5"),
    ("Claude Haiku 5.5 · платно, дёшево", "anthropic/claude-haiku-5.5"),
]
ANTHROPIC_MODELS = [
    ("Claude Opus 5.5", "claude-opus-5-5"),
    ("Claude Haiku 5.5", "claude-haiku-5-5"),
]

RULES = """Ты помощник методиста в платформе Data-Driven Curriculum. Отвечай только на основе результатов анализа, которые приведены ниже.

Правила:
1. Используй только числа, которые есть в результатах, и переписывай их как есть. Не вычисляй новые числа: никаких разностей, сумм, долей и процентов от процентов.
2. Если ответа нет в результатах, прямо скажи, что таких данных нет.
3. Различай «доказано» и «не доказано». Если различие не доказано, не называй один учебник лучше другого.
4. Не придумывай причин, которых нет в результатах. Практический совет методисту давать можно, если он прямо следует из результатов.
5. Отвечай коротко: 3–6 предложений, без таблиц. Отвечай на языке вопроса: по-русски или по-казахски.
6. Если в результатах сказано, что данные демонстрационные, не выдавай выводы за факты о реальных школах."""

STATUS_TEXT = {
    "ok": "норма",
    "hard": "трудная тема: базовые темы усвоены, тема трудна сама по себе",
    "bad_items": "брак вопросов: низкий балл вызван плохими вопросами теста",
}


@dataclass
class Settings:
    provider: str
    api_key: str
    base_url: str | None
    models: list[tuple[str, str]]


def _setting(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st

        value = st.secrets.get(name)
    except Exception:  # нет файла secrets.toml или запуск вне Streamlit
        return None
    return str(value) if value else None


def load_settings() -> Settings | None:
    """Какой провайдер настроен. None, если ключа нет."""
    base_url = _setting("AI_BASE_URL")
    if key := _setting("XKIRO_API_KEY"):
        return Settings("xKiro", key, base_url or XKIRO_URL, XKIRO_MODELS)
    if key := _setting("ANTHROPIC_API_KEY"):
        return Settings("Anthropic", key, base_url, ANTHROPIC_MODELS)
    return None


# --- Сводка результатов для модели --------------------------------------------------


def _signed(value: float) -> str:
    return ("+" if value > 0 else "") + fmt(value)


def _interval(low: float, high: float) -> str:
    return f"95% интервал от {fmt(low)} до {fmt(high)}"


def _group_lines(ds: Dataset, bundle: Bundle, names: dict[str, str], lang: str) -> list[str]:
    """Сравнение групп учеников для реальных данных."""
    groups = bundle.groups
    lines = [
        "\n# Сравнение групп учеников (средняя доля верных по всем темам)",
        f"«При равных условиях» = при одинаковых ресурсах: {ds.meta.get('covariate', 'входной уровень')}, "
        "а также с учётом местности и языка. Это связь, а не причина.",
    ]
    for row in groups[groups["scope"] == ALL].itertuples():
        verdict = "доказано" if row.proven else "не доказано"
        first, second, factor = term(row.first, lang), term(row.second, lang), term(row.factor, lang)
        lines.append(
            f"- {factor}: как есть {first} {fmt(row.naive_first)}%, {second} {fmt(row.naive_second)}% "
            f"(учеников {row.n_first} и {row.n_second}); при равных условиях {first} минус {second}: "
            f"{_signed(row.diff)} п.п., {_interval(row.low, row.high)}, {verdict}"
        )
    lines.append("\n# Разница групп по темам при равных условиях (только доказанные)")
    proven = groups[(groups["scope"] != ALL) & groups["proven"]]
    for row in proven.itertuples():
        first, second, factor = term(row.first, lang), term(row.second, lang), term(row.factor, lang)
        lines.append(
            f"- «{names[row.scope]}», {factor}: {first} минус {second} {_signed(row.diff)} п.п., "
            f"{_interval(row.low, row.high)}; как есть {fmt(row.naive_first)}% и {fmt(row.naive_second)}%"
        )
    rest = len(groups[groups["scope"] != ALL]) - len(proven)
    lines.append(f"По остальным {rest} сравнениям по темам разница не доказана.")
    return lines


def build_context(ds: Dataset, bundle: Bundle, lang: str = "ru") -> str:
    """Сводка для модели. В lang="kz" названия тем и групп казахские, чтобы ответ шёл на казахском."""
    names = ds.topics.set_index("topic_id")[f"name_{lang}"].to_dict()
    lines: list[str] = []

    lines.append("# Данные")
    if lang == "kz":
        lines.append("Интерфейс на казахском: названия тем даны по-казахски, отвечай на языке вопроса.")
    if ds.is_demo:
        lines.append("Данные демонстрационные (синтетические), не реальные школы.")
    if ds.is_timss:
        lines.append(f"Реальные данные: {ds.meta.get('title', 'TIMSS')}. Источник: {ds.meta.get('source', '')}")
        lines.append(
            "Это доля верных ответов без выборочных весов и правдоподобных значений IEA, поэтому это не "
            "официальная статистика TIMSS. Учебников и часов по темам в этих данных нет."
        )
        if note := ds.meta.get("prereqs_note"):
            lines.append(note)
    lines.append(f"Школ: {len(ds.schools)}, учеников: {len(ds.students)}, вопросов теста: {len(ds.items)}.")

    lines.append(f"\n# Усвоение тем (доля верных по исправным вопросам; порог слабой темы {fmt(bundle.weak, 0)}%)")
    diagnosis = bundle.diagnosis.set_index("topic_id")
    for tid, row in diagnosis.iterrows():
        if row["status"] == "prereq":
            status = (
                f"провал в базе: причина в теме «{names[row['root']]}»; кто усвоил базовую тему, "
                f"решает эту тему на {fmt(row['with_base'])}%, кто не усвоил — на {fmt(row['without_base'])}%"
            )
        elif row["status"] == "hard" and not row["has_prereqs"]:
            status = "слабая тема: результат ниже порога, базовые темы для неё не заданы"
        else:
            status = STATUS_TEXT[row["status"]]
        extra = f", с бракованными вопросами {fmt(row['raw'])}%" if row["n_bad"] else ""
        lines.append(f"- «{names[tid]}»: {fmt(row['clean'])}%{extra}; {status}")

    bad = bundle.quality[bundle.quality["is_bad"]]
    if not bad.empty:
        items = ", ".join(f"{r.item_id} («{names[r.topic_id]}»)" for r in bad.itertuples())
        lines.append(f"Бракованные вопросы (сильные ученики ошибаются в них не реже слабых): {items}.")

    if bundle.textbooks is None:
        lines.extend(_group_lines(ds, bundle, names, lang))
        return "\n".join(lines)

    textbooks = bundle.textbooks
    means = textbooks["means"]
    overall = means[means["scope"] == ALL].set_index("textbook")
    profile = textbooks["profile"].set_index("textbook")
    lines.append("\n# Учебники: средний балл по всем темам")
    for book in textbooks["books"]:
        p = profile.loc[book]
        lines.append(
            f"- Учебник {book}: без поправки {fmt(overall.loc[book, 'naive'])}%, "
            f"с учётом уровня школ {fmt(overall.loc[book, 'adjusted'])}%; школ {int(p['schools'])}, "
            f"средний входной балл учеников {fmt(p['pretest'])}, гимназий {fmt(p['gymnasium'], 0)}%, "
            f"сельских школ {fmt(p['rural'], 0)}%"
        )
    lines.append("«С учётом уровня школ» = сравнение при равном входном балле, типе школы и языке обучения.")
    lines.append(
        "Поправка одна для всех школ: отдельного сравнения учебников внутри гимназий, городских или "
        "сельских школ нет, поэтому лучший учебник с учётом уровня школ — лучший и для каждого типа школ, "
        "насколько позволяет модель."
    )

    contrasts = textbooks["contrasts"]
    lines.append("\n# Разница учебников с учётом уровня школ (первый минус второй)")
    for row in contrasts[contrasts["scope"] == ALL].itertuples():
        verdict = "доказано" if row.low > 0 or row.high < 0 else "не доказано"
        lines.append(
            f"- {row.first} − {row.second}: {_signed(row.diff)} п.п., {_interval(row.low, row.high)}, {verdict}"
        )

    by_topic = contrasts[contrasts["scope"] != ALL]
    proven = by_topic[by_topic["proven"]]
    lines.append("\n# Доказанные различия учебников по темам (с поправкой на множественные сравнения)")
    for row in proven.itertuples():
        winner, loser = (row.first, row.second) if row.diff > 0 else (row.second, row.first)
        line = (
            f"- «{names[row.scope]}»: учебник {winner} сильнее учебника {loser} на {fmt(abs(row.diff))} п.п., "
            f"{_interval(min(abs(row.low), abs(row.high)), max(abs(row.low), abs(row.high)))}"
        )
        if ds.coverage is not None:
            table = ds.coverage[ds.coverage["topic_id"] == row.scope].set_index("textbook")
            if winner in table.index and loser in table.index:
                line += (
                    f"; упражнений в учебнике {winner}: {int(table.loc[winner, 'exercises'])}, "
                    f"в учебнике {loser}: {int(table.loc[loser, 'exercises'])}"
                )
        lines.append(line)
    lines.append(f"По остальным {len(by_topic) - len(proven)} сравнениям тем различие не доказано.")
    if ds.coverage is not None and ds.is_demo:
        lines.append("Число упражнений в учебниках в демо задано таблицей вручную.")

    hours, plan, result = bundle.hours, bundle.best_plan, bundle.best_forecast
    lines.append("\n# Учебные часы: оптимальный план")
    lines.append(
        f"Всего часов: {sum(hours.planned.values())}, сумма не меняется. Прогноз среднего балла: "
        f"{_signed(result['gain'])} п.п., {_interval(result['low'], result['high'])}."
    )
    if "realism" in result:
        lines.append(
            f"Прогноз уже снижен до {fmt(100 * result['realism'], 0)}% расчётного: "
            "план подбирали на одной половине школ и проверяли на другой."
        )
    table = result["table"].set_index("topic_id")
    for tid in hours.topics:
        row = table.loc[tid]
        if tid in hours.frozen:
            basis = "часы во всех школах одинаковы, вывод сделать нельзя, нужен пилот"
        elif plan[tid] == hours.planned[tid]:
            basis = "без изменений"
        else:
            basis = "эффект часов доказан" if hours.is_reliable(tid) else "эффект часов не доказан"
        lines.append(
            f"- «{names[tid]}»: по программе {hours.planned[tid]} ч, в плане {plan[tid]} ч; "
            f"прогноз балла темы {_signed(row['gain'])} п.п.; {basis}"
        )
    lines.append("Часы меняются только в пределах, которые уже встречаются в школах.")
    return "\n".join(lines)


# --- Сверка чисел -------------------------------------------------------------------

NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def _numbers(text: str) -> list[tuple[str, float]]:
    # Номера вопросов вроде T09-Q1 — не числа из расчётов.
    cleaned = re.sub(r"\b[A-ZА-Я]+\d+(?:-[A-ZА-Я]+\d+)?\b", " ", text)
    return [(raw, float(raw.replace(",", "."))) for raw in NUMBER.findall(cleaned)]


def unverified_numbers(answer: str, context: str) -> list[str]:
    """Числа из ответа, которых нет в сводке (с учётом округления)."""
    known = [value for _, value in _numbers(context)]
    suspicious = []
    for raw, value in _numbers(answer):
        if value.is_integer() and value <= SMALL_INTEGER:
            continue
        tolerance = 0.5 if value.is_integer() else 0.051
        if not any(abs(value - other) <= tolerance for other in known) and raw not in suspicious:
            suspicious.append(raw)
    return suspicious


# --- Запрос к модели ----------------------------------------------------------------


def client(settings: Settings) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=settings.api_key, base_url=settings.base_url, timeout=120, max_retries=1)


def stream_answer(settings: Settings, model: str, context: str, history: list[dict]):
    """Контекст-менеджер потока ответа. history — сообщения {"role", "content"}, последнее от пользователя."""
    return client(settings).messages.stream(
        model=model,
        max_tokens=MAX_TOKENS,
        system=f"{RULES}\n\n# Результаты анализа\n{context}",
        messages=history[-HISTORY_TURNS:],
    )


ERRORS = {
    "auth": {
        "ru": "Ключ {p} не подошёл. Проверь его в файле .streamlit/secrets.toml.",
        "kz": "{p} кілті жарамайды. Оны .streamlit/secrets.toml файлынан тексеріңіз.",
    },
    "paid": {
        "ru": "Эта модель у {p} платная: нужен тариф или пополненный баланс. Выбери бесплатную модель или пополни баланс.",
        "kz": "Бұл модель {p} сервисінде ақылы: тариф немесе толтырылған баланс керек. Тегін модельді таңдаңыз.",
    },
    "denied": {"ru": "{p} отказал в доступе к модели.", "kz": "{p} модельге қол жеткізуге рұқсат бермеді."},
    "not_found": {"ru": "{p} не знает такую модель. Выбери другую.", "kz": "{p} мұндай модельді білмейді. Басқасын таңдаңыз."},
    "rate": {
        "ru": "Слишком много запросов подряд. Подожди минуту и спроси снова.",
        "kz": "Сұрақтар тым көп. Бір минут күтіп, қайта сұраңыз.",
    },
    "offline": {
        "ru": "Нет связи с {p}. Проверь интернет. Остальная платформа работает без ИИ.",
        "kz": "{p} сервисімен байланыс жоқ. Интернетті тексеріңіз. Платформаның қалған бөлігі ЖИ-сіз жұмыс істейді.",
    },
    "status": {
        "ru": "{p} вернул ошибку {code}. Попробуй ещё раз или выбери другую модель.",
        "kz": "{p} {code} қатесін қайтарды. Қайта көріңіз немесе басқа модельді таңдаңыз.",
    },
    "other": {"ru": "Не удалось получить ответ: {text}", "kz": "Жауап алу мүмкін болмады: {text}"},
}


def explain_error(error: Exception, settings: Settings, lang: str = "ru") -> str:
    """Понятное методисту сообщение вместо технической ошибки."""
    text = str(error).replace(settings.api_key, "***")
    p = settings.provider
    if isinstance(error, anthropic.AuthenticationError):
        key = "auth"
    elif isinstance(error, anthropic.PermissionDeniedError):
        key = "paid" if ("premium" in text.lower() or "paid" in text.lower()) else "denied"
    elif isinstance(error, anthropic.NotFoundError):
        key = "not_found"
    elif isinstance(error, anthropic.RateLimitError):
        key = "rate"
    elif isinstance(error, anthropic.APIConnectionError):
        key = "offline"
    elif isinstance(error, anthropic.APIStatusError):
        return ERRORS["status"][lang].format(p=p, code=error.status_code)
    else:
        return ERRORS["other"][lang].format(text=text[:200])
    return ERRORS[key][lang].format(p=p)
