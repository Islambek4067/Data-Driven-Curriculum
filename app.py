"""Data-Driven Curriculum. Запуск: streamlit run app.py

Интерфейс на казахском и русском: тексты в core/i18n.py, язык в st.session_state["lang"].
"""
from __future__ import annotations

import json

import anthropic
import numpy as np
import pandas as pd
import streamlit as st

import generate_data as gen
import validate
from core import ai, charts
from core.analysis import ALL, Bundle, diagnose_topics, explain_contrast, forecast, run_analysis
from core.data import DataError, Dataset, load_dir, load_zip
from core.i18n import LANGS, lang, model_label, t, term
from core.report import build_report, fmt, to_docx

st.set_page_config(page_title="Data-Driven Curriculum", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem; max-width: 1240px;}
    h1 {font-size: 1.9rem !important; font-weight: 650 !important; letter-spacing: -0.01em;}
    .lede {color: #52514e; margin: -0.5rem 0 0.6rem 0; font-size: 1.02rem;}
    .pill {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.82rem;
           background: #f0efec; color: #52514e; margin-left: 8px;}
    .headline {font-size: 1.55rem; font-weight: 650; line-height: 1.25; color: #0b0b0b; margin: 0.2rem 0 0.35rem 0;}
    .sub {color: #52514e; font-size: 1rem; line-height: 1.45;}
    .tag {display: inline-block; padding: 1px 9px; border-radius: 6px; font-size: 0.8rem;
          background: #cde2fb; color: #104281; margin-bottom: 4px;}
    .tag.bad {background: #f0efec; color: #52514e;}
    [data-testid="stMetricValue"] {font-size: 1.7rem;}
    button[data-baseweb="tab"] p {font-size: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def demo_dataset() -> Dataset:
    if not (gen.DATA_DIR / "responses.csv").exists():
        gen.main()
    return load_dir(gen.DATA_DIR, is_demo=True)


TIMSS_DIR = gen.DATA_DIR.parent / "timss_kz"


@st.cache_resource(show_spinner=False)
def timss_dataset() -> Dataset:
    return load_dir(TIMSS_DIR)


@st.cache_resource(show_spinner=False, max_entries=3)
def analyse(key: str, _dataset: Dataset) -> Bundle:
    return run_analysis(_dataset)


@st.cache_data(show_spinner=False)
def model_check(key: str, _bundle: Bundle) -> dict:
    truth = json.loads((gen.DATA_DIR / "truth.json").read_text(encoding="utf-8"))
    return validate.build_report(_bundle, truth)


def plot(figure) -> None:
    st.plotly_chart(figure, width="stretch", config=charts.CONFIG)


def spaced(number: int) -> str:
    return f"{number:,}".replace(",", " ")


def html(text: str, css: str) -> None:
    st.markdown(f'<div class="{css}">{text}</div>', unsafe_allow_html=True)


# --- Шапка, язык и данные ---------------------------------------------------------------

st.session_state.setdefault("lang", "kz")
st.session_state.setdefault("lang_last", st.session_state["lang"])


def keep_language() -> None:
    """Повторный клик по выбранному языку снимает выбор: возвращаем прежний язык."""
    if st.session_state.get("lang") in LANGS:
        st.session_state["lang_last"] = st.session_state["lang"]
    else:
        st.session_state["lang"] = st.session_state["lang_last"]


sources = ["demo"]
if (TIMSS_DIR / "responses.csv").exists():
    sources.append("timss")
if st.session_state.get("uploaded") is not None:
    sources.append("upload")

title_col, side_col = st.columns([5, 1.6], vertical_alignment="bottom")
with side_col:
    st.segmented_control(t("lang_label"), list(LANGS), format_func=LANGS.get, key="lang", on_change=keep_language)
with title_col:
    st.title("Data-Driven Curriculum")
    lede = st.empty()
source = st.segmented_control(
    t("source_label"), sources, default="demo", key="source", format_func=lambda s: t(f"source_{s}")
) or "demo"
if source == "timss":
    ds: Dataset = timss_dataset()
elif source == "upload":
    ds = st.session_state["uploaded"]
else:
    ds = demo_dataset()

if ds.is_timss:
    intro, pill = t("intro_timss"), f'<span class="pill">{t("pill_timss")}</span>'
else:
    intro = t("intro_demo")
    pill = f'<span class="pill">{t("pill_demo")}</span>' if ds.is_demo else ""
lede.markdown(
    f'<div class="lede">{intro} · {t("n_schools", n=len(ds.schools))} · '
    f"{t('n_students', n=spaced(len(ds.students)))}{pill}</div>",
    unsafe_allow_html=True,
)
with side_col:
    with st.popover(t("upload_button"), width="stretch"):
        st.markdown(t("upload_help"))
        file = st.file_uploader(t("upload_label"), type="zip", label_visibility="collapsed")
        if file is not None:
            try:
                uploaded = load_zip(file.getvalue())
            except DataError as error:
                st.error(str(error))
            else:
                previous = st.session_state.get("uploaded")
                if previous is None or uploaded.key != previous.key:
                    st.session_state["uploaded"] = uploaded
                    st.session_state["source"] = "upload"
                    st.rerun()

try:
    with st.spinner(t("computing")):
        bundle = analyse(ds.key, ds)
except np.linalg.LinAlgError:
    st.error(t("model_failed"))
    st.stop()

if (ds.is_demo or ds.is_timss) and not st.session_state.get("intro_closed"):
    kind = "timss" if ds.is_timss else "demo"
    with st.container(border=True):
        text_col, button_col = st.columns([6, 1], vertical_alignment="center")
        with text_col:
            st.markdown(f"**{t('intro_title')}** {t(f'intro_{kind}_body')}")
            st.markdown(f"**{t('intro_start')}** {t(f'intro_{kind}_start')}")
        if button_col.button(t("intro_ok"), width="stretch"):
            st.session_state["intro_closed"] = True
            st.rerun()

names = ds.topics.set_index("topic_id")[f"name_{lang()}"].to_dict()
books: list[str] = bundle.textbooks["books"] if bundle.textbooks else []
if ds.is_timss:
    tab_ids = ["topics", "groups", "ai"]
else:
    tab_ids = ["topics", "textbooks", "hours", "decision", "ai"]
    if ds.is_demo:
        tab_ids.append("check")
tab_map = dict(zip(tab_ids, st.tabs([t(f"tab_{tab}") for tab in tab_ids])))


# --- Темы -------------------------------------------------------------------------------

with tab_map["topics"]:
    filter_a, filter_b, _ = st.columns([1, 1, 2])
    school_type = filter_a.selectbox(
        t("school_type"), ["Все", *sorted(ds.schools["school_type"].unique())], format_func=term
    )
    book_filter = filter_b.selectbox(t("textbook"), ["Все", *books], format_func=term) if books else "Все"
    schools = ds.schools
    if school_type != "Все":
        schools = schools[schools["school_type"] == school_type]
    if book_filter != "Все":
        schools = schools[schools["textbook"] == book_filter]
    students = ds.students[ds.students["school_id"].isin(schools["school_id"])]

    if students.empty:
        st.info(t("empty_sample"))
    else:
        diagnosis = diagnose_topics(ds, bundle.scores, bundle.quality, students["student_id"], weak=bundle.weak)
        diagnosis["name"] = diagnosis["topic_id"].map(names)
        problems = diagnosis[diagnosis["status"] != "ok"].sort_values("clean")
        tile_1, tile_2, tile_3, tile_4 = st.columns(4)
        tile_1.metric(t("m_avg"), f"{fmt(diagnosis['clean'].mean())}%")
        tile_2.metric(t("m_weak"), t("of", a=(diagnosis["clean"] < bundle.weak).sum(), b=len(diagnosis)))
        tile_3.metric(t("m_bad"), t("of", a=int(bundle.quality["is_bad"].sum()), b=len(bundle.quality)))
        tile_4.metric(t("m_students"), spaced(len(students)))

        chart_col, card_col = st.columns([1.5, 1], gap="large")
        with chart_col:
            st.subheader(t("mastery_title"))
            st.caption(t("cap_timss" if ds.is_timss else "cap_demo", w=int(bundle.weak)))
            plot(charts.topic_mastery(diagnosis, bundle.weak))
        with card_col:
            st.subheader(t("why_title"))
            by_id = diagnosis.set_index("topic_id")
            if problems.empty:
                st.write(t("no_weak"))
            for row in problems.itertuples():
                with st.container(border=True):
                    tag_class = "tag bad" if row.status == "bad_items" else "tag"
                    st.markdown(
                        f'<span class="{tag_class}">{charts.status_label(row.status).capitalize()}</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**{row.name}** · {fmt(row.clean, 0)}%")
                    if row.status == "prereq":
                        st.write(
                            t(
                                "prereq_text",
                                root=names[row.root],
                                rs=fmt(by_id.loc[row.root, "clean"], 0),
                                w=fmt(row.with_base, 0),
                                wo=fmt(row.without_base, 0),
                            )
                        )
                        st.caption(t("prereq_do"))
                    elif row.status == "hard":
                        st.write(t("hard_base" if row.has_prereqs else "hard_nobase"))
                        st.caption(t("hard_do"))
                    else:
                        bad = bundle.quality[(bundle.quality["topic_id"] == row.topic_id) & bundle.quality["is_bad"]]
                        st.write(t("bad_text", raw=fmt(row.raw, 0)))
                        st.caption(t("bad_do", items=", ".join(bad["item_id"])))

        with st.expander(t("quality_title")):
            table = bundle.quality.copy()
            view = pd.DataFrame(
                {
                    t("col_item"): table["item_id"],
                    t("col_topic"): table["topic_id"].map(names),
                    t("col_correct"): table["p_correct"].round(1),
                    t("col_disc"): table["r_rest"].round(2),
                    t("col_status"): np.where(table["is_bad"], t("status_bad_item"), t("status_ok_item")),
                }
            )
            st.dataframe(view.sort_values(t("col_disc")), hide_index=True, width="stretch", height=320)


# --- Учебники ---------------------------------------------------------------------------

if "textbooks" in tab_map:
    with tab_map["textbooks"]:
        means = bundle.textbooks["means"]
        contrasts = bundle.textbooks["contrasts"]
        overall = means[means["scope"] == ALL].set_index("textbook")
        profile = bundle.textbooks["profile"].set_index("textbook")

        adjusted = st.toggle(t("adjust_toggle"), value=False, key="adjusted")
        column = "adjusted" if adjusted else "naive"
        leader = overall[column].idxmax()
        left, right = st.columns([1.4, 1], gap="large")
        with left:
            html(t("leader", b=leader, v=fmt(overall.loc[leader, column])), "headline")
            if adjusted:
                naive_leader = overall["naive"].idxmax()
                row = contrasts[
                    (contrasts["scope"] == ALL)
                    & contrasts["first"].isin([leader, naive_leader])
                    & contrasts["second"].isin([leader, naive_leader])
                ]
                if naive_leader != leader and not row.empty:
                    r = row.iloc[0]
                    sign = 1 if r["first"] == leader else -1
                    bounds = sorted([sign * r["low"], sign * r["high"]])
                    text = t(
                        "flip_text", l=leader, n=naive_leader, d=fmt(sign * r["diff"]),
                        lo=fmt(bounds[0]), hi=fmt(bounds[1]),
                    )
                else:
                    text = t("same_leader")
                html(f"{text} {t('hollow')}", "sub")
            else:
                html(t("naive_text"), "sub")
            plot(charts.textbook_ranking(means, books, adjusted))
        with right:
            st.subheader(t("who_title"))
            profile_view = pd.DataFrame(
                {
                    t("textbook"): profile.index,
                    t("pc_schools"): profile["schools"].to_numpy(),
                    t("pc_pretest"): [fmt(v) for v in profile["pretest"]],
                    t("pc_gym"): [f"{fmt(v, 0)}%" for v in profile["gymnasium"]],
                    t("pc_rural"): [f"{fmt(v, 0)}%" for v in profile["rural"]],
                }
            )
            st.table(profile_view.set_index(t("textbook")))
            st.caption(t("adjust_note"))

        st.divider()
        pairs = [(r.first, r.second) for r in contrasts[contrasts["scope"] == ALL].itertuples()]
        default_pair = next((i for i, p in enumerate(pairs) if set(p) == set(books[:2])), 0)
        pair_col, _ = st.columns([1, 3])
        pair = pair_col.selectbox(
            t("compare_label"), pairs, index=default_pair, format_func=lambda p: t("pair", a=p[0], b=p[1])
        )
        first, second = pair
        topic_rows = contrasts[
            (contrasts["scope"] != ALL) & (contrasts["first"] == first) & (contrasts["second"] == second)
        ].set_index("scope").loc[list(ds.topics["topic_id"])].reset_index()
        forest_col, why_col = st.columns([1.5, 1], gap="large")
        with forest_col:
            st.subheader(t("diff_title"))
            st.caption(t("forest_cap", a=first, b=second))
            plot(charts.contrast_forest(topic_rows, names, first, second, books))
        with why_col:
            st.subheader(t("proven_title"))
            proven = topic_rows[topic_rows["proven"]].sort_values("diff", key=abs, ascending=False)
            if proven.empty:
                st.write(t("none_proven"))
            for row in proven.itertuples():
                winner = first if row.diff > 0 else second
                with st.container(border=True):
                    st.markdown(t("stronger", topic=names[row.scope], w=winner, d=fmt(abs(row.diff))))
                    reason = explain_contrast(ds, row.scope, first, second, lang())
                    if reason:
                        st.caption(reason)
            if ds.is_demo and not proven.empty:
                st.caption(t("demo_coverage"))


# --- Часы -------------------------------------------------------------------------------

if "hours" in tab_map:
    with tab_map["hours"]:
        model = bundle.hours
        total_hours = sum(model.planned.values())
        mode = st.segmented_control(
            t("mode"), ["best", "custom"], default="best", key="hours_mode", format_func=lambda m: t(f"mode_{m}")
        ) or "best"

        if mode == "best":
            plan = bundle.best_plan
            result = bundle.best_forecast
        else:
            movable = [tid for tid in model.topics if tid not in model.frozen]
            col_from, col_to, col_hours = st.columns(3)
            source_topic = col_from.selectbox(
                t("from_topic"), [tid for tid in movable if model.planned[tid] > model.low[tid]], format_func=names.get
            )
            targets = [tid for tid in movable if tid != source_topic and model.high[tid] > model.planned[tid]]
            default_target = max(targets, key=model.value_per_log_hour)
            target = col_to.selectbox(
                t("to_topic"), targets, index=targets.index(default_target), format_func=names.get
            )
            limit = min(
                model.planned[source_topic] - model.low[source_topic], model.high[target] - model.planned[target]
            )
            if limit > 1:
                moved = col_hours.slider(t("how_many"), 1, limit, min(2, limit))
            else:
                moved = 1
                col_hours.markdown(t("only_one"))
            plan = dict(model.planned)
            plan[source_topic] -= moved
            plan[target] += moved
            result = forecast(model, plan)

        headline, note = st.columns([1.4, 1], gap="large")
        with headline:
            gain = ("+" if result["gain"] > 0 else "") + fmt(result["gain"])
            html(t("forecast_head", g=gain), "headline")
            html(t("forecast_sub", lo=fmt(result["low"]), hi=fmt(result["high"]), h=total_hours), "sub")
        with note:
            if mode == "best" and "realism" in result:
                st.caption(t("realism", r=fmt(100 * result["realism"], 0)))
            st.caption(t("bounds"))

        plot(charts.hours_plan(result["table"], names))

        def basis(topic: str) -> str:
            if topic in model.frozen:
                return t("basis_frozen")
            if plan[topic] == model.planned[topic]:
                return t("basis_same")
            return t("basis_yes") if model.is_reliable(topic) else t("basis_no")

        detail = result["table"].copy()
        detail_view = pd.DataFrame(
            {
                t("col_topic"): detail["topic_id"].map(names),
                t("col_planned"): detail["planned"],
                t("col_new"): detail["new"],
                t("col_gain"): [f"{g:+.1f}".replace(".", ",").replace("-", "−") for g in detail["gain"]],
                t("col_basis"): detail["topic_id"].map(basis),
            }
        )
        st.table(detail_view.set_index(t("col_topic")))


# --- Решение ----------------------------------------------------------------------------

if "decision" in tab_map:
    with tab_map["decision"]:
        doc_lang = st.segmented_control(
            t("doc_lang"), list(LANGS), default=lang(), key=f"doc_lang_{lang()}", format_func=LANGS.get
        ) or lang()
        report = build_report(ds, bundle, doc_lang)
        st.download_button(
            "Word жүктеу" if doc_lang == "kz" else "Скачать Word",
            to_docx(report),
            file_name=f"methodsovet_{doc_lang}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            type="primary",
        )
        with st.container(border=True):
            st.subheader(report.title)
            for line in report.lead:
                st.caption(line)
            for heading, lines in report.sections:
                st.markdown(f"**{heading}**")
                for line in lines:
                    st.write(line)


# --- Город и село, язык -----------------------------------------------------------------

if "groups" in tab_map:
    with tab_map["groups"]:
        groups = bundle.groups
        html(t("groups_head"), "headline")
        html(t("groups_sub"), "sub")
        overall = groups[groups["scope"] == ALL]
        for column, row in zip(st.columns(len(overall), gap="large"), overall.itertuples()):
            first, second = term(row.first), term(row.second)
            with column, st.container(border=True):
                st.markdown(t("group_title", f=term(row.factor), a=first, b=second))
                st.write(
                    t(
                        "as_is", a=first, x=fmt(row.naive_first), b=second, y=fmt(row.naive_second),
                        na=spaced(row.n_first), nb=spaced(row.n_second),
                    )
                )
                st.write(
                    t(
                        "gap_sentence",
                        l=first if row.diff > 0 else second,
                        d=fmt(abs(row.diff)),
                        lo=fmt(row.low),
                        hi=fmt(row.high),
                        verdict=t("verdict_yes" if row.proven else "verdict_no"),
                    )
                )

        factors = list(overall["factor"])
        factor = st.segmented_control(
            t("diff_title"), factors, default=factors[-1], key="group_factor", format_func=term
        ) or factors[-1]
        chosen = overall[overall["factor"] == factor].iloc[0]
        first, second = term(chosen["first"]), term(chosen["second"])
        st.caption(t("gap_cap", a=first, b=second))
        plot(charts.group_gaps(groups, names, factor, first, second))
        st.info(t("timss_info"))
        st.caption(ds.meta.get("source", ""))


# --- Спросить ИИ ------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def ai_context(key: str, language: str, _ds: Dataset, _bundle: Bundle) -> str:
    return ai.build_context(_ds, _bundle, language)


def show_check(unverified: list[str]) -> None:
    if unverified:
        st.warning(t("check_bad", n=", ".join(unverified)))
    else:
        st.caption(t("check_ok"))


with tab_map["ai"]:
    settings = ai.load_settings()
    if settings is None:
        st.info(t("ai_not_configured"))
    else:
        context = ai_context(ds.key, lang(), ds, bundle)
        intro_col, picker = st.columns([2.6, 1.2], vertical_alignment="bottom", gap="large")
        with intro_col:
            html(t("ai_head"), "headline")
            html(t("ai_sub"), "sub")
        with picker:
            model_ids = [model_id for _, model_id in settings.models]
            labels = dict((model_id, label) for label, model_id in settings.models)
            model_id = st.selectbox(
                t("ai_model", p=settings.provider), model_ids, key="ai_model_id",
                format_func=lambda m: model_label(labels[m]),
            )

        chat_key = f"ai_chat_{ds.key}"
        history: list[dict] = st.session_state.setdefault(chat_key, [])

        picked = None
        prefix = "ex_timss" if ds.is_timss else "ex_demo"
        examples = [t(f"{prefix}_{i}") for i in (1, 2, 3)]
        for column, text in zip(st.columns(len(examples)), examples):
            if column.button(text, width="stretch"):
                picked = text

        for message in history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    show_check(message["unverified"])

        question = st.chat_input(t("ai_placeholder")) or picked
        if question:
            history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                request = [{"role": m["role"], "content": m["content"]} for m in history]
                try:
                    with ai.stream_answer(settings, model_id, context, request) as stream:
                        answer = st.write_stream(stream.text_stream)
                        final = stream.get_final_message()
                except anthropic.APIError as error:
                    history.pop()
                    st.error(ai.explain_error(error, settings, lang()))
                else:
                    if final.stop_reason == "refusal":
                        answer = t("refusal")
                        st.markdown(answer)
                    unverified = ai.unverified_numbers(answer, context)
                    show_check(unverified)
                    history.append({"role": "assistant", "content": answer, "unverified": unverified})

        bottom_left, bottom_right = st.columns([3, 1])
        with bottom_left:
            with st.expander(t("ai_sees")):
                st.code(context, language=None, wrap_lines=True)
        with bottom_right:
            if history and st.button(t("clear_chat"), width="stretch"):
                st.session_state[chat_key] = []
                st.rerun()


# --- Проверка ---------------------------------------------------------------------------

if "check" in tab_map:
    with tab_map["check"]:
        check = model_check(ds.key, bundle)
        html(t("check_head"), "headline")
        html(t("check_sub"), "sub")
        tile_1, tile_2, tile_3, tile_4 = st.columns(4)
        items = check["items"]
        tile_1.metric(t("c_bad"), t("of", a=items["correct"], b=items["broken"]))
        tile_2.metric(t("c_real"), t("of", a=check["real_found"], b=check["real_total"]))
        tile_3.metric(t("c_false"), t("of", a=check["false_alarms"], b=check["null_total"]))
        tile_4.metric(t("c_diag"), t("of", a=int(check["statuses"]["match"].sum()), b=len(check["statuses"])))

        left, right = st.columns([1.2, 1], gap="large")
        with left:
            st.subheader(t("c_left"))
            st.caption(t("c_left_cap", e=fmt(check["mae"], 2)))
            plot(charts.truth_scatter(check["contrasts"]))
        with right:
            st.subheader(t("c_right"))
            plan_check = check["plan"]
            st.metric(t("c_promised"), f"{fmt(plan_check['promised'], 2)} п.п.")
            st.metric(t("c_true"), f"{fmt(plan_check['true'], 2)} п.п.")
            st.caption(
                t(
                    "c_right_cap", lo=fmt(plan_check["low"], 2), hi=fmt(plan_check["high"], 2),
                    c=fmt(100 * check["coverage"], 0),
                )
            )
