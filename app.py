"""
🌸 Study Buddy – AI student study planner (Streamlit + Gemini)

Run:
    pip install -r requirements.txt
    export GEMINI_API_KEY="your-key"      # or paste it in the sidebar
    streamlit run app.py
"""

import datetime as dt
import json
import os
import random
import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from fpdf import FPDF
from google import genai
from google.genai import types

st.set_page_config(page_title="Study Buddy", page_icon="🌸", layout="wide")

MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
DATA_FILE = Path(__file__).with_name("study_data.json")
PAGES = ["🏠 Home", "📝 Notes", "🃏 Flashcards", "🧠 Quiz", "📅 Planner"]
HOME, NOTES, CARDS, QUIZ, PLANNER = PAGES
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
NUDGES = [
    "💧 Time for a sip of water and a stretch!",
    "🧠 Quick recall: close your notes and write 3 things you just learned.",
    "🃏 Got 5 minutes? Flip through a few flashcards.",
    "🌸 Small steps count. Keep going, you're doing great!",
    "👀 Rest your eyes for 20 seconds, look at something far away.",
    "📅 Check your planner, is today's task done?",
]


# ══════════════════════════════════════════════════════════════════════════
# Cream & pink theme (Streamlit's own theme system, no HTML/CSS)
# On first launch the app writes .streamlit/config.toml for you.
# ══════════════════════════════════════════════════════════════════════════
THEME = """[theme]
base = "light"
primaryColor = "#E58AA3"
backgroundColor = "#FFF8EE"
secondaryBackgroundColor = "#FCE4EA"
textColor = "#5A3A41"
"""


def ensure_theme():
    cfg = Path.cwd() / ".streamlit" / "config.toml"
    if cfg.exists():
        return False
    try:
        cfg.parent.mkdir(exist_ok=True)
        cfg.write_text(THEME)
        return True
    except Exception:
        return False


THEME_CREATED = ensure_theme()


# ══════════════════════════════════════════════════════════════════════════
# State (saved to study_data.json so topics survive a restart)
# ══════════════════════════════════════════════════════════════════════════
def load_state():
    if st.session_state.get("loaded"):
        return
    data = {}
    if DATA_FILE.exists():
        try:
            data = json.loads(DATA_FILE.read_text())
        except Exception:
            data = {}
    ss = st.session_state
    ss.topics = data.get("topics", {})
    ss.quiz_history = data.get("quiz_history", [])
    ss.quick_notes = data.get("quick_notes", "")
    ss.current = data.get("current")
    ss.page = HOME
    ss.tutor_msgs = []
    ss.quiz = None
    ss.fc_idx = {}
    ss.fc_show = False
    ss.loaded = True


def save_state():
    try:
        DATA_FILE.write_text(
            json.dumps(
                {
                    "topics": st.session_state.topics,
                    "quiz_history": st.session_state.quiz_history,
                    "quick_notes": st.session_state.get("quick_notes", ""),
                    "current": st.session_state.current,
                },
                indent=2,
            )
        )
    except Exception:
        pass


def go(page, topic=None):
    st.session_state.page = page
    if topic:
        st.session_state.current = topic


def delete_topic(name):
    ss = st.session_state
    ss.topics.pop(name, None)
    if ss.current == name:
        ss.current = next(iter(ss.topics), None)
    ss.quiz = None


def clear_everything():
    ss = st.session_state
    ss.topics, ss.quiz_history, ss.current = {}, [], None
    ss.quiz, ss.tutor_msgs, ss.fc_idx = None, [], {}


def topic_picker(label="Choose a topic"):
    names = list(st.session_state.topics)
    cur = st.session_state.current if st.session_state.current in names else names[0]
    choice = st.selectbox(label, names, index=names.index(cur), key=f"pick_{st.session_state.page}")
    st.session_state.current = choice
    return choice


# ══════════════════════════════════════════════════════════════════════════
# Gemini helpers
# ══════════════════════════════════════════════════════════════════════════
class AIError(Exception):
    pass


def env_key():
    k = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not k:
        try:
            k = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            k = None
    return k


def api_key():
    return st.session_state.get("api_key_input") or env_key()


def pick_model(client):
    """If a model is retired, find a current Flash model that this key can use."""
    try:
        names = [m.name.replace("models/", "") for m in client.models.list()
                 if "generateContent" in (m.supported_actions or [])]
    except Exception:
        return None
    bad = ("lite", "image", "tts", "live", "audio", "embedding", "robotics")
    flash = [n for n in names if "flash" in n and not any(x in n for x in bad)]
    return sorted(flash)[-1] if flash else None


FALLBACK_MODELS = ["gemini-flash-latest", "gemini-3.8-flash", "gemini-flash-lite-latest", "gemini-2.5-flash"]


def ask(prompt, system=None, as_json=False):
    key = api_key()
    if not key:
        raise AIError("Please add your Gemini API key in the sidebar first. 🔑")
    cfg = types.GenerateContentConfig(
        system_instruction=system,
        temperature=0.6,
        response_mime_type="application/json" if as_json else "text/plain",
    )
    client = genai.Client(api_key=key.strip())
    first = st.session_state.get("model", MODEL)
    candidates = [first] + [m for m in FALLBACK_MODELS if m != first]
    last_error = None
    for model in candidates:
        for attempt in range(3):  # retry busy-server errors with a short pause
            try:
                resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
                st.session_state.model = model  # remember the one that worked
                return (resp.text or "").strip()
            except Exception as e:
                last_error = e
                msg = str(e)
                busy = any(c in msg for c in ("503", "UNAVAILABLE", "500", "INTERNAL", "overloaded"))
                if busy and attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                break
        msg = str(last_error)
        # try the next model only if this one is missing, retired or too busy
        if not any(c in msg for c in ("404", "NOT_FOUND", "503", "UNAVAILABLE")):
            break
    if any(c in str(last_error) for c in ("503", "UNAVAILABLE")):
        raise AIError("Google's servers are busy right now. Please wait a minute and try again. 🌸")
    raise AIError(f"Gemini error: {last_error}")


def ask_json(prompt, system=None):
    text = ask(prompt, system=system, as_json=True)
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    raise AIError("I couldn't read the AI's answer. Please try again.")


def notes_prompt(topic):
    return (
        f'You are an expert teacher writing study notes for a student on the topic: "{topic}".\n'
        "Write the ACTUAL subject content: real explanations, definitions, facts, formulas, dates, "
        "and examples. Do NOT write an outline of what the notes 'would cover' and do NOT add "
        "meta-commentary.\n\n"
        "Use Markdown with these sections:\n"
        "## Overview (what it is and why it matters)\n"
        "## Key Concepts (each as a ### sub-heading with a clear 3-6 sentence explanation and an example)\n"
        "## Important Terms (bullets: **Term**: meaning)\n"
        "## Worked Example / Real-World Application\n"
        "## Common Mistakes & Misconceptions\n"
        "## Quick Revision Points\n\n"
        "Use simple, student-friendly language and be accurate. Aim for 800-1100 words."
    )


def summary_prompt(notes):
    return (
        "Read these study notes and return JSON with exactly two keys:\n"
        '"summary": a clear summary of about 120-170 words in plain text,\n'
        '"key_topics": a list of 5-8 short strings (2-6 words each), the main sub-topics a student '
        "must study, in a sensible learning order.\n\nNOTES:\n" + notes[:12000]
    )


def create_topic(name, notes=None):
    """Generate (or accept) notes, summarise them, and store the topic."""
    name = name.strip()
    if notes is None:
        notes = ask(notes_prompt(name))
    data = ask_json(summary_prompt(notes))
    t = st.session_state.topics.setdefault(
        name, {"flashcards": [], "plan": [], "created": dt.date.today().isoformat()}
    )
    t["notes"] = notes
    t["summary"] = str(data.get("summary", "")).strip()
    t["key_topics"] = [str(x) for x in data.get("key_topics", [])][:10] or [name]
    st.session_state.current = name


def gen_flashcards(topic, n, focus=""):
    t = st.session_state.topics[topic]
    existing = [c["front"] for c in t["flashcards"]]
    prompt = (
        f"Create {n} NEW flashcards for a student studying '{topic}', based on the summary and notes below. "
        'Return a JSON list of objects with keys "front" (a question or term) and "back" '
        "(a short, clear answer, max 35 words). "
        + (f"Focus especially on: {focus}. " if focus else "")
        + f"Do not repeat any of these existing fronts: {existing}.\n\n"
        f"SUMMARY:\n{t['summary']}\n\nNOTES:\n{t['notes'][:8000]}"
    )
    data = ask_json(prompt)
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), [])
    new = [
        {"front": str(c["front"]), "back": str(c["back"]), "status": "new"}
        for c in data
        if isinstance(c, dict) and c.get("front") and c.get("back") and c["front"] not in existing
    ]
    t["flashcards"].extend(new)
    return len(new)


def gen_quiz(topic, n, level):
    t = st.session_state.topics[topic]
    prompt = (
        f"Write a {level} multiple-choice quiz with {n} questions on '{topic}' using ONLY the notes below. "
        'Return a JSON list; each item has "question", "options" (exactly 4 strings), '
        '"answer_index" (0-3, the correct option) and "explanation" (one sentence). '
        "Vary which position holds the correct answer.\n\n"
        f"NOTES:\n{t['notes'][:9000]}"
    )
    data = ask_json(prompt)
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), [])
    qs = []
    for q in data:
        try:
            opts = [str(o) for o in q["options"]]
            a = int(q["answer_index"])
            if len(opts) >= 2 and 0 <= a < len(opts):
                qs.append({"question": str(q["question"]), "options": opts, "answer": a,
                            "explanation": str(q.get("explanation", ""))})
        except Exception:
            continue
    if not qs:
        raise AIError("The quiz came back empty. Please try again.")
    st.session_state.quiz = {"id": int(time.time()), "topic": topic, "questions": qs, "result": None}


# ══════════════════════════════════════════════════════════════════════════
# PDF export
# ══════════════════════════════════════════════════════════════════════════
def clean(s):
    rep = {"–": "-", "—": "-", "‘": "'", "’": "'", "“": '"', "”": '"', "•": "-", "→": "->",
           "…": "...", "≈": "~", "≤": "<=", "≥": ">=", "×": "x", "−": "-"}
    for k, v in rep.items():
        s = s.replace(k, v)
    s = s.replace("**", "").replace("__", "").replace("`", "")
    return s.encode("latin-1", "ignore").decode("latin-1")


def make_pdf(title, blocks):
    """blocks = [(heading, markdown_text), ...] -> PDF bytes"""
    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()

    def mc(text, h=6):
        pdf.multi_cell(0, h, clean(text), new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(181, 84, 111)
    mc(title, 10)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(130, 110, 115)
    mc(f"Made with Study Buddy on {dt.date.today():%d %b %Y}", 5)
    pdf.ln(3)

    for heading, text in blocks:
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(181, 84, 111)
        mc(heading, 8)
        pdf.ln(1)
        pdf.set_text_color(60, 40, 45)
        for line in text.splitlines():
            s = line.strip()
            if not s:
                pdf.ln(2)
            elif s.startswith("#"):
                pdf.set_font("Helvetica", "B", 12)
                pdf.set_text_color(181, 84, 111)
                mc(s.lstrip("#").strip(), 7)
                pdf.set_text_color(60, 40, 45)
            elif s[:2] in ("- ", "* "):
                pdf.set_font("Helvetica", "", 11)
                pdf.set_x(pdf.l_margin + 4)
                mc("- " + s[2:])
            else:
                pdf.set_font("Helvetica", "", 11)
                mc(s)
        pdf.ln(4)
    return bytes(pdf.output())


def pdf_button(label, filename, title, blocks, key):
    try:
        st.download_button(label, make_pdf(title, blocks), file_name=filename,
                           mime="application/pdf", key=key)
    except Exception as e:
        st.warning(f"Couldn't build the PDF: {e}")


# ══════════════════════════════════════════════════════════════════════════
# Progress + reminders
# ══════════════════════════════════════════════════════════════════════════
def work_rows(t):
    return [r for r in t.get("plan", []) if r.get("Minutes", 0) > 0]


def progress(name):
    t = st.session_state.topics[name]
    parts = {}
    work = work_rows(t)
    if work:
        parts["Planner"] = sum(bool(r["Done"]) for r in work) / len(work)
    cards = t["flashcards"]
    if cards:
        parts["Flashcards"] = sum(c["status"] == "know" for c in cards) / len(cards)
    hist = [h for h in st.session_state.quiz_history if h["topic"] == name]
    if hist:
        parts["Quiz"] = max(h["score"] / h["total"] for h in hist)
    overall = sum(parts.values()) / len(parts) if parts else 0.0
    return parts, overall


def get_reminders():
    today = dt.date.today().isoformat()
    msgs = []
    for name, t in st.session_state.topics.items():
        work = work_rows(t)
        due = [r for r in work if r["Date"] == today and not r["Done"]]
        late = [r for r in work if r["Date"] < today and not r["Done"]]
        if due:
            more = f" (+{len(due) - 1} more)" if len(due) > 1 else ""
            msgs.append(f"📌 Today in {name}: {due[0]['Task']}{more}")
        if late:
            msgs.append(f"⏰ {len(late)} overdue task(s) in {name}, move them in the Planner")
        if not t["plan"]:
            msgs.append(f"📅 You haven't made a study plan for {name} yet")
        left = sum(c["status"] != "know" for c in t["flashcards"])
        if left:
            msgs.append(f"🃏 {left} flashcard(s) still to master in {name}")
        if not any(h["topic"] == name for h in st.session_state.quiz_history):
            msgs.append(f"🧠 Try a quiz on {name} to test yourself")
    return msgs


def show_reminders(limit=4):
    for m in get_reminders()[:limit]:
        st.info(m)


def reminder_ticker():
    """Pops up a friendly toast every N minutes while the app is open."""
    mins = st.session_state.get("remind_every", 0)
    if not mins:
        return

    @st.fragment(run_every=mins * 60)
    def _tick():
        now = time.time()
        last = st.session_state.get("last_remind")
        if last is None:
            st.session_state.last_remind = now
            return
        if now - last >= mins * 60 - 2:
            st.session_state.last_remind = now
            extra = get_reminders()
            msg = random.choice(NUDGES)
            if extra:
                msg += "\n\n" + extra[0]
            st.toast(msg, icon="🔔")

    _tick()


# ══════════════════════════════════════════════════════════════════════════
# AI tutor (bottom chat box)
# ══════════════════════════════════════════════════════════════════════════
def tutor(topic):
    ss = st.session_state
    q = st.chat_input("💬 Ask the AI tutor about anything confusing…")
    if q:
        ss.tutor_msgs.append({"role": "user", "content": q})
        t = ss.topics.get(topic, {})
        system = (
            "You are a warm, clear AI tutor helping a student. Give quick clarifications in simple "
            "language, under 150 words unless asked for more. Use a short example when helpful.\n"
            f"The student is studying '{topic}'. Their notes for context:\n{t.get('notes', '')[:6000]}"
        )
        history = "\n".join(f"{m['role'].title()}: {m['content']}" for m in ss.tutor_msgs[-8:])
        try:
            with st.spinner("Thinking…"):
                reply = ask(history + "\nTutor:", system=system)
        except AIError as e:
            reply = str(e)
        ss.tutor_msgs.append({"role": "assistant", "content": reply})

    st.divider()
    with st.expander("💬 AI Tutor – your quick-clarification chat", expanded=bool(ss.tutor_msgs)):
        if not ss.tutor_msgs:
            st.caption("Type a question in the chat box at the bottom of the page.")
        for m in ss.tutor_msgs:
            with st.chat_message(m["role"], avatar="🌸" if m["role"] == "assistant" else None):
                st.markdown(m["content"])
        if ss.tutor_msgs and st.button("Clear chat", key="clear_tutor"):
            ss.tutor_msgs = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# Pages
# ══════════════════════════════════════════════════════════════════════════
def page_home():
    ss = st.session_state
    st.title("🌸 Study Buddy")
    st.write("Type a topic in the chat box at the bottom, and I'll make your notes. "
             "Everything you study shows up here.")

    show_reminders()

    if not ss.topics:
        st.info("No topics yet. Try typing something like **Photosynthesis** or **French Revolution** below ⬇️")
    for name in list(ss.topics):
        parts, overall = progress(name)
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"### {name}")
            c2.markdown(f"**{int(overall * 100)}% complete**")
            st.progress(overall)
            if parts:
                st.caption("  ·  ".join(f"{k}: {int(v * 100)}%" for k, v in parts.items()))
            else:
                st.caption("Not started yet. Open the notes, then try flashcards, a quiz or a plan.")
            b = st.columns(5)
            b[0].button("📝 Notes", key=f"hn_{name}", on_click=go, args=(NOTES, name))
            b[1].button("🃏 Flashcards", key=f"hf_{name}", on_click=go, args=(CARDS, name))
            b[2].button("🧠 Quiz", key=f"hq_{name}", on_click=go, args=(QUIZ, name))
            b[3].button("📅 Planner", key=f"hp_{name}", on_click=go, args=(PLANNER, name))
            b[4].button("🗑️", key=f"hd_{name}", on_click=delete_topic, args=(name,), help="Remove this topic")

    if ss.topics:
        with st.expander("🧹 Clear dashboard / start with a new topic"):
            st.caption("This removes all topics, flashcards, plans and quiz scores.")
            ok = st.checkbox("Yes, clear everything", key="confirm_clear")
            if st.button("Clear dashboard", disabled=not ok):
                clear_everything()
                st.rerun()

    topic_in = st.chat_input("What do you want to study? e.g. Photosynthesis, Newton's Laws…")
    if topic_in:
        try:
            with st.spinner(f"Writing your notes on “{topic_in}”… ✍️"):
                create_topic(topic_in)
            save_state()
            st.toast(f"Notes for {topic_in} are ready!", icon="🌸")
            st.session_state.pending_page = NOTES  # applied at the start of the next run
            st.rerun()
        except AIError as e:
            st.error(str(e))


def page_notes():
    ss = st.session_state
    st.title("📝 Notes")

    with st.expander("➕ Create notes on a new topic", expanded=not ss.topics):
        tab_ai, tab_own = st.tabs(["Generate with AI", "Summarise my own notes"])
        with tab_ai:
            with st.form("new_topic"):
                name = st.text_input("Topic", placeholder="e.g. The Water Cycle")
                go_btn = st.form_submit_button("Generate notes")
            if go_btn and name.strip():
                try:
                    with st.spinner("Writing notes…"):
                        create_topic(name)
                    st.rerun()
                except AIError as e:
                    st.error(str(e))
        with tab_own:
            with st.form("own_notes"):
                name2 = st.text_input("Topic name", key="own_name")
                text = st.text_area("Paste your notes here", height=180)
                go2 = st.form_submit_button("Summarise & save")
            if go2 and name2.strip() and text.strip():
                try:
                    with st.spinner("Summarising…"):
                        create_topic(name2, notes=text)
                    st.rerun()
                except AIError as e:
                    st.error(str(e))

    if not ss.topics:
        st.info("Create your first topic above, or type one on the Home page.")
        return

    topic = topic_picker("Your topics")
    t = ss.topics[topic]

    tab_notes, tab_sum = st.tabs(["📖 Full notes", "✨ Summary & key topics"])
    with tab_notes:
        st.markdown(t["notes"])
        c1, c2 = st.columns(2)
        with c1:
            pdf_button("⬇️ Download full notes (PDF)", f"{topic}_notes.pdf", topic,
                       [("Summary", t["summary"]), ("Notes", t["notes"])], key="pdf_full")
        with c2:
            if st.button("🔄 Regenerate these notes"):
                try:
                    with st.spinner("Rewriting…"):
                        create_topic(topic)
                    st.rerun()
                except AIError as e:
                    st.error(str(e))
    with tab_sum:
        st.subheader("Summary")
        st.write(t["summary"])
        st.subheader("Key topics to study")
        for i, k in enumerate(t["key_topics"], 1):
            st.markdown(f"{i}. {k}")
        c1, c2 = st.columns(2)
        with c1:
            pdf_button("⬇️ Export summary as PDF", f"{topic}_summary.pdf", f"{topic} - Summary",
                       [("Summary", t["summary"]),
                        ("Key topics to study", "\n".join(f"- {k}" for k in t["key_topics"]))],
                       key="pdf_sum")
        with c2:
            st.button("📅 Plan my study time", on_click=go, args=(PLANNER, topic))

    tutor(topic)


def fc_move(topic, delta, n):
    ss = st.session_state
    ss.fc_idx[topic] = (ss.fc_idx.get(topic, 0) + delta) % n
    ss.fc_show = False


def fc_mark(topic, status, n):
    ss = st.session_state
    i = ss.fc_idx.get(topic, 0) % n
    ss.topics[topic]["flashcards"][i]["status"] = status
    fc_move(topic, 1, n)


def fc_flip():
    st.session_state.fc_show = not st.session_state.fc_show


def fc_reset(topic):
    for c in st.session_state.topics[topic]["flashcards"]:
        c["status"] = "new"
    st.session_state.fc_idx[topic] = 0
    st.session_state.fc_show = False


def page_flashcards():
    ss = st.session_state
    st.title("🃏 Flashcards")
    if not ss.topics:
        st.info("Create a topic first, then come back for flashcards.")
        return
    topic = topic_picker()
    t = ss.topics[topic]
    cards = t["flashcards"]

    with st.expander("✨ Generate new flashcards", expanded=not cards):
        with st.form("gen_cards"):
            c1, c2 = st.columns([1, 2])
            n = c1.slider("How many?", 3, 20, 8)
            focus = c2.text_input("Focus on (optional)", placeholder="e.g. definitions, dates, formulas")
            gen = st.form_submit_button("Generate cards")
        if gen:
            try:
                with st.spinner("Making cards…"):
                    added = gen_flashcards(topic, n, focus)
                st.toast(f"Added {added} new cards!", icon="🃏")
                st.rerun()
            except AIError as e:
                st.error(str(e))
        with st.form("manual_card", clear_on_submit=True):
            f = st.text_input("Or add your own: front")
            b = st.text_input("Back")
            if st.form_submit_button("Add card") and f.strip() and b.strip():
                cards.append({"front": f.strip(), "back": b.strip(), "status": "new"})
                st.rerun()

    if not cards:
        st.info("No cards yet. Generate some above ⬆️")
        tutor(topic)
        return

    n = len(cards)
    idx = ss.fc_idx.get(topic, 0) % n
    card = cards[idx]
    know = sum(c["status"] == "know" for c in cards)
    st.progress(know / n, text=f"Mastered {know} of {n} cards")

    showing_back = ss.fc_show
    with st.container(border=True):
        if showing_back:
            st.caption("ANSWER")
            st.subheader(card["back"])
        else:
            st.caption(f"QUESTION · card {idx + 1} of {n}")
            st.header(card["front"])
        st.write("")
    b = st.columns(5)
    b[0].button("⬅️ Prev", on_click=fc_move, args=(topic, -1, n), key="fc_prev")
    b[1].button("🔄 Flip", on_click=fc_flip, key="fc_flip")
    b[2].button("Next ➡️", on_click=fc_move, args=(topic, 1, n), key="fc_next")
    b[3].button("✅ Got it", on_click=fc_mark, args=(topic, "know", n), key="fc_know")
    b[4].button("🔁 Review again", on_click=fc_mark, args=(topic, "review", n), key="fc_rev")

    with st.expander("See all cards"):
        st.dataframe(pd.DataFrame(cards).rename(columns={"front": "Front", "back": "Back", "status": "Status"}),
                     hide_index=True)
        st.button("Reset progress", on_click=fc_reset, args=(topic,), key="fc_reset")

    tutor(topic)


def page_quiz():
    ss = st.session_state
    st.title("🧠 Quiz")
    if not ss.topics:
        st.info("Create a topic first, then test yourself here.")
        return
    topic = topic_picker("Quiz me on")

    with st.form("quiz_setup"):
        c1, c2 = st.columns(2)
        n = c1.slider("Number of questions", 3, 15, 5)
        level = c2.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)
        start = st.form_submit_button("Start a new quiz")
    if start:
        try:
            with st.spinner("Writing your questions…"):
                gen_quiz(topic, n, level)
            st.rerun()
        except AIError as e:
            st.error(str(e))

    quiz = ss.quiz
    if quiz and quiz["topic"] == topic:
        qs = quiz["questions"]
        if quiz["result"] is None:
            with st.form("quiz_form"):
                answers = []
                for i, q in enumerate(qs):
                    st.markdown(f"**Q{i + 1}. {q['question']}**")
                    answers.append(st.radio("Answer", q["options"], index=None,
                                            key=f"q_{quiz['id']}_{i}", label_visibility="collapsed"))
                submit = st.form_submit_button("Submit answers")
            if submit:
                if any(a is None for a in answers):
                    st.warning("Please answer every question before submitting.")
                else:
                    score = sum(a == q["options"][q["answer"]] for a, q in zip(answers, qs))
                    quiz["result"] = {"score": score, "answers": answers}
                    ss.quiz_history.append({"topic": topic, "score": score, "total": len(qs),
                                            "date": dt.datetime.now().strftime("%d %b %H:%M")})
                    st.rerun()
        else:
            r = quiz["result"]
            pct = int(100 * r["score"] / len(qs))
            (st.balloons if pct >= 80 else (lambda: None))()
            st.success(f"You scored **{r['score']} / {len(qs)}** ({pct}%)")
            for i, (q, a) in enumerate(zip(qs, r["answers"]), 1):
                right = q["options"][q["answer"]]
                icon = "✅" if a == right else "❌"
                with st.container(border=True):
                    st.markdown(f"{icon} **Q{i}. {q['question']}**")
                    st.write(f"Your answer: {a}")
                    if a != right:
                        st.write(f"Correct answer: **{right}**")
                    if q["explanation"]:
                        st.caption(q["explanation"])

    # score tracking
    st.subheader("📈 Your scores")
    hist = [h for h in ss.quiz_history if h["topic"] == topic]
    if hist:
        pcts = [round(100 * h["score"] / h["total"]) for h in hist]
        m = st.columns(3)
        m[0].metric("Attempts", len(hist))
        m[1].metric("Best", f"{max(pcts)}%")
        m[2].metric("Average", f"{round(sum(pcts) / len(pcts))}%")
        if len(pcts) > 1:
            st.line_chart(pd.DataFrame({"Score %": pcts}, index=range(1, len(pcts) + 1)))
    else:
        st.caption("No attempts for this topic yet. Your scores will appear here.")
    if ss.quiz_history:
        with st.expander("All quiz attempts (every topic)"):
            df = pd.DataFrame(ss.quiz_history)
            df["Score"] = df["score"].astype(str) + "/" + df["total"].astype(str)
            st.dataframe(df[["date", "topic", "Score"]].iloc[::-1], hide_index=True)

    tutor(topic)


def build_plan(topic, start, end, weekdays, minutes):
    kts = st.session_state.topics[topic]["key_topics"] or [topic]
    all_days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    avail = [d for d in all_days if d.weekday() in weekdays]
    if not avail:
        return []
    final = avail[-2:] if len(avail) >= 4 else []
    study = avail[:-2] if final else avail
    n, m = len(kts), len(study)
    tasks = {}
    for d, day in enumerate(study):
        if n > m:
            tasks[day] = "Learn: " + "; ".join(kts[d * n // m:(d + 1) * n // m])
        elif d < n:
            tasks[day] = f"Learn: {kts[d]}"
        else:
            tasks[day] = f"Revise: {kts[(d - n) % n]} + flashcards"
    if final:
        tasks[final[0]] = "Flashcards review + practice quiz"
        tasks[final[1]] = "Final revision: skim summary & notes, retake quiz"
    rows = []
    for day in all_days:
        rows.append({"Date": day.isoformat(), "Day": WEEKDAYS[day.weekday()],
                     "Task": tasks.get(day, "🌿 Rest day"),
                     "Minutes": minutes if day in tasks else 0, "Done": False})
    return rows


def toggle_task(topic, i, key):
    st.session_state.topics[topic]["plan"][i]["Done"] = bool(st.session_state[key])


def move_missed(topic):
    today = dt.date.today().isoformat()
    for r in st.session_state.topics[topic]["plan"]:
        if r["Minutes"] > 0 and not r["Done"] and r["Date"] < today:
            r["Date"] = today
            r["Day"] = WEEKDAYS[dt.date.today().weekday()]


def page_planner():
    ss = st.session_state
    st.title("📅 Study Planner")
    if not ss.topics:
        st.info("Create a topic first, then I'll help you plan it.")
        return
    topic = topic_picker("Plan for")
    t = ss.topics[topic]
    plan = t["plan"]
    today = dt.date.today()

    with st.expander("⚙️ Set my availability & build a plan", expanded=not plan):
        st.caption("Pick your dates, the days you're free, and how long you can study. "
                   "You can change this any time and rebuild the plan.")
        with st.form("plan_form"):
            c1, c2 = st.columns(2)
            start = c1.date_input("Start date", today)
            end = c2.date_input("Exam / target date", today + dt.timedelta(days=7))
            days = st.multiselect("Days I'm available", WEEKDAYS, default=WEEKDAYS)
            mins = st.slider("Minutes per study session", 15, 180, 45, step=5)
            build = st.form_submit_button("Create / rebuild plan")
        if build:
            if end < start:
                st.error("The target date must be after the start date.")
            else:
                rows = build_plan(topic, start, end, [WEEKDAYS.index(d) for d in days], mins)
                if rows:
                    t["plan"] = rows
                    st.rerun()
                else:
                    st.error("No available days in that range, so I couldn't build a plan.")

    if not plan:
        st.info("Build a plan above ⬆️ and it will appear here.")
        tutor(topic)
        return

    work = work_rows(t)
    done = sum(bool(r["Done"]) for r in work)
    st.progress(done / len(work) if work else 0, text=f"{done} of {len(work)} study sessions done")

    late = [r for r in work if r["Date"] < today.isoformat() and not r["Done"]]
    if late:
        st.warning(f"⏰ You have {len(late)} missed session(s).")
        st.button("📆 Move missed sessions to today", on_click=move_missed, args=(topic,))

    today_rows = [(i, r) for i, r in enumerate(plan) if r["Date"] == today.isoformat() and r["Minutes"] > 0]
    st.subheader("Today")
    if today_rows:
        for i, r in today_rows:
            key = f"today_{topic}_{i}_{abs(hash(r['Task']))}_{int(r['Done'])}"
            st.checkbox(f"{r['Task']}  ·  {r['Minutes']} min", value=r["Done"], key=key,
                        on_change=toggle_task, args=(topic, i, key))
    else:
        st.caption("Nothing scheduled today. Enjoy the break! 🌿")

    st.subheader("Full schedule")
    st.caption("Edit any cell, tick **Done**, change dates, or add/remove rows with the table controls. "
               "Press **Save changes** when you're finished.")
    df = pd.DataFrame(plan)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    with st.form("plan_editor"):
        edited = st.data_editor(
            df, num_rows="dynamic", hide_index=True, key=f"editor_{topic}",
            column_config={
                "Date": st.column_config.DateColumn("Date", format="DD MMM YYYY", required=True),
                "Day": st.column_config.TextColumn("Day", disabled=True),
                "Task": st.column_config.TextColumn("What to study", width="large"),
                "Minutes": st.column_config.NumberColumn("Minutes", min_value=0, max_value=600, step=5),
                "Done": st.column_config.CheckboxColumn("Done"),
            },
        )
        saved = st.form_submit_button("💾 Save changes")
    if saved:
        rows = []
        for _, r in edited.dropna(subset=["Date"]).iterrows():
            d = pd.to_datetime(r["Date"]).date()
            rows.append({"Date": d.isoformat(), "Day": WEEKDAYS[d.weekday()],
                         "Task": str(r["Task"]) if pd.notna(r["Task"]) else "",
                         "Minutes": int(r["Minutes"]) if pd.notna(r["Minutes"]) else 0,
                         "Done": bool(r["Done"]) if pd.notna(r["Done"]) else False})
        t["plan"] = sorted(rows, key=lambda x: x["Date"])
        st.toast("Plan saved!", icon="📅")
        st.rerun()

    tutor(topic)


# ══════════════════════════════════════════════════════════════════════════
# Sidebar + main
# ══════════════════════════════════════════════════════════════════════════
def sidebar():
    with st.sidebar:
        st.title("🌸 Study Buddy")
        if THEME_CREATED:
            st.info("Pink & cream theme installed. Stop the app and run it once more to see it.")
        st.radio("Go to", PAGES, key="page", label_visibility="collapsed")
        st.divider()
        if env_key():
            st.caption("🔑 Gemini API key loaded")
        else:
            st.text_input("🔑 Gemini API key", type="password", key="api_key_input",
                          help="Get a free key at aistudio.google.com")
        st.selectbox("🔔 Remind me every", [0, 10, 20, 30, 45, 60], index=2, key="remind_every",
                     format_func=lambda m: "Off" if m == 0 else f"{m} minutes")
        reminder_ticker()
        st.divider()
        st.text_area("📝 Quick notes", key="quick_notes", height=200,
                     placeholder="Jot down thoughts while you study…")
        st.download_button("⬇️ Save quick notes", st.session_state.quick_notes or " ",
                           file_name="quick_notes.txt", key="dl_quick")


def main():
    load_state()
    if "pending_page" in st.session_state:
        st.session_state.page = st.session_state.pop("pending_page")
    sidebar()
    if not st.session_state.get("welcomed"):
        st.session_state.welcomed = True
        for m in get_reminders()[:2]:
            st.toast(m, icon="🔔")
    try:
        {HOME: page_home, NOTES: page_notes, CARDS: page_flashcards,
         QUIZ: page_quiz, PLANNER: page_planner}[st.session_state.page]()
    finally:
        save_state()


main()