"""Streamlit front end for the agentic face access-control system.

Run locally:   streamlit run streamlit_app.py
Deploy:        Streamlit Community Cloud, main file = streamlit_app.py
"""
import os

import streamlit as st

# Streamlit Cloud keeps API keys in "Secrets". Copy them into environment variables
# before the agent code reads them.
try:
    for _key in ("XAI_API_KEY", "XAI_MODEL", "GROQ_API_KEY", "GROQ_MODEL",
                 "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LLM_MODEL"):
        if _key in st.secrets and not os.getenv(_key):
            os.environ[_key] = str(st.secrets[_key])
except Exception:
    pass  # no secrets file locally; environment variables still work

import pandas as pd  # noqa: E402

import config  # noqa: E402
from src.agentic.pipeline import AccessPipeline  # noqa: E402
from src.agentic.vision import decode_image  # noqa: E402
from src.agentic import crew as agent_crew  # noqa: E402

st.set_page_config(page_title="Face Access Control", page_icon="🔐", layout="wide")

st.markdown("""
<style>
.verdict {border-radius: 14px; padding: 18px 22px; margin: 6px 0 14px; color: #fff;}
.verdict h2 {margin: 0; font-size: 1.9rem; letter-spacing: .02em; color: #fff;}
.verdict p {margin: 6px 0 0; font-size: 1.02rem; opacity: .95;}
.v-GRANTED {background: #1B7F4C;}
.v-DENIED {background: #B42318;}
.v-CHALLENGE {background: #9A6700;}
.v-RETRY {background: #475467;}
.who {display: inline-block; margin-top: 10px; padding: 3px 10px; border-radius: 999px;
      background: rgba(255,255,255,.18); font-size: .85rem;}
.step {border-left: 3px solid #1F4E79; padding: 4px 0 4px 12px; margin: 8px 0;}
.step b {color: #1F4E79;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading face models...")
def get_pipeline() -> AccessPipeline:
    return AccessPipeline()


pipe = get_pipeline()
ss = st.session_state
ss.setdefault("door_key", 0)
ss.setdefault("reg_key", 0)
ss.setdefault("pending_case", None)
ss.setdefault("outcome", None)
ss.setdefault("samples", [])

MODES = {
    "Only unclear cases (recommended)": "unclear",
    "Every photo (best for demos)": "always",
    "Off: original fixed checklist": "off",
}

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    mode_label = st.radio("When should the AI agent decide?", list(MODES), index=0)
    mode = MODES[mode_label]
    st.divider()
    ok, missing = pipe.vision.ready
    st.markdown(f"**Face models:** {'ready' if ok else 'missing ' + ', '.join(missing)}")
    model = agent_crew.resolve_model()
    if model:
        st.markdown(f"**Agent:** on, using {model}")
    else:
        st.warning("Agent is off: add XAI_API_KEY (Grok) or GROQ_API_KEY in Secrets. "
                   "The original checklist will decide until then.")
    st.caption("Clear cases are decided instantly. Unclear ones go to a CrewAI agent that "
               "inspects the evidence and can ask the person to turn their head.")

st.title("🔐 Face Access Control")
door_tab, reg_tab, admin_tab = st.tabs(["Door", "Register a person", "Admin"])


def show_outcome(o):
    who = f'<span class="who">Decided by: {o.decided_by}</span>'
    title = {"GRANTED": "Access granted", "DENIED": "Access denied",
             "CHALLENGE": "One more step", "RETRY": "Try again"}[o.status]
    if o.status != "CHALLENGE":  # the challenge banner is already shown above the camera
        st.markdown(f'<div class="verdict v-{o.status}"><h2>{title}</h2><p>{o.message}</p>{who}</div>',
                    unsafe_allow_html=True)
    for note in o.notes:
        st.info(note)
    left, right = st.columns([3, 2])
    with left:
        if o.image is not None:
            st.image(o.image, width="stretch")
    with right:
        if o.reasoning:
            st.markdown("**Why**")
            st.write(o.reasoning)
        ev = o.evidence or {}
        if ev.get("faces"):
            st.markdown("**What the models saw**")
            st.markdown(
                f"- Real-face probability: **{ev.get('live_probability', 0):.0%}**\n"
                f"- Best match: **{ev.get('identity') or 'nobody'}** (score {ev.get('match_score', 0):.2f}, "
                f"needs {ev.get('match_threshold')})\n"
                f"- Head turn: {ev.get('head_turn')} (0.5 = facing camera)")
    if o.trace:
        with st.expander(f"How the agent investigated ({len(o.trace)} tool calls)", expanded=o.status == "CHALLENGE"):
            for i, step in enumerate(o.trace, 1):
                st.markdown(f'<div class="step"><b>{i}. {step["tool"]}</b> {step["input"]}</div>',
                            unsafe_allow_html=True)
                st.json(step["output"], expanded=False)


# ---------------------------------------------------------------- door
with door_tab:
    if not ok:
        st.error("Face model files are missing: " + ", ".join(missing) + ". Run `python download_models.py`.")
    pending = ss.pending_case
    if pending:
        st.markdown(f'<div class="verdict v-CHALLENGE"><h2>Please do this</h2><p>{pending["instruction"]}</p></div>',
                    unsafe_allow_html=True)
        label = "Take the photo while doing that"
    else:
        label = "Look at the camera and take a photo"

    shot = st.camera_input(label, key=f"door_{ss.door_key}")
    c1, c2 = st.columns([1, 4])
    if pending and c1.button("Start over"):
        ss.pending_case, ss.outcome = None, None
        ss.door_key += 1
        st.rerun()

    if shot is not None:
        try:
            frame = decode_image(shot.getvalue())
        except ValueError as exc:
            st.error(str(exc))
        else:
            spinner = "The agent is checking..." if (mode != "off" and model) else "Checking..."
            with st.spinner(spinner):
                if pending:
                    outcome = pipe.answer_challenge(pending["case_id"], frame)
                else:
                    outcome = pipe.start(frame, mode)
            ss.outcome = outcome
            ss.pending_case = ({"case_id": outcome.case_id, "instruction": outcome.instruction}
                               if outcome.status == "CHALLENGE" else None)
            ss.door_key += 1  # reset the camera for the next photo
            st.rerun()

    if ss.outcome is not None:
        show_outcome(ss.outcome)

# ---------------------------------------------------------------- register
with reg_tab:
    st.write(f"Take {config.ENROLLMENT_MIN_SAMPLES} to {config.ENROLLMENT_MAX_SAMPLES} photos of the person, "
             "looking at the camera, in even light. Only a face template is stored, never the photos.")
    name = st.text_input("Full name").strip()
    replace = False
    if name and pipe.vision.users.name_exists(name):
        replace = st.checkbox(f"{name} is already registered. Replace their face with these photos.")

    snap = st.camera_input(f"Photo {len(ss.samples) + 1}", key=f"reg_{ss.reg_key}")
    if snap is not None:
        ss.samples.append(snap.getvalue())
        ss.samples = ss.samples[:config.ENROLLMENT_MAX_SAMPLES]
        ss.reg_key += 1
        st.rerun()

    if ss.samples:
        st.caption(f"{len(ss.samples)} photo(s) taken")
        st.image([decode_image(b)[:, :, ::-1] for b in ss.samples], width=110)

    b1, b2 = st.columns([1, 1])
    can_register = bool(name) and len(ss.samples) >= config.ENROLLMENT_MIN_SAMPLES
    if b1.button("Register", type="primary", disabled=not can_register):
        if pipe.vision.users.name_exists(name) and not replace:
            st.error("That name is already registered. Tick the box above to replace their face, or use another name.")
        else:
            good, reasons = [], []
            with st.spinner("Checking each photo..."):
                for i, data in enumerate(ss.samples, 1):
                    emb, why = pipe.vision.enrollment_embedding(decode_image(data))
                    (good.append(emb) if emb is not None else reasons.append(f"Photo {i}: {why}"))
            if len(good) >= config.ENROLLMENT_MIN_SAMPLES:
                pipe.vision.users.add_user(name, good)
                st.success(f"Registered {name} with {len(good)} face templates.")
                ss.samples = []
            else:
                st.error(f"Only {len(good)} of {len(ss.samples)} photos passed; "
                         f"{config.ENROLLMENT_MIN_SAMPLES} are needed.")
                for r in reasons:
                    st.write("- " + r)
    if b2.button("Clear photos", disabled=not ss.samples):
        ss.samples = []
        st.rerun()

# ---------------------------------------------------------------- admin
with admin_tab:
    st.subheader("People")
    users = pipe.vision.users.list_users()
    if not users:
        st.write("Nobody is registered yet. Use the Register tab.")
    for u in users:
        a, b, c = st.columns([3, 2, 1])
        a.write(f"**{u['name']}**")
        b.write("Active" if u["status"] == "active" else "Blocked")
        if u["status"] == "active":
            if c.button("Block", key=f"blk_{u['id']}"):
                pipe.vision.users.set_status(u["id"], "blocked")
                st.rerun()
        elif c.button("Unblock", key=f"ublk_{u['id']}"):
            pipe.vision.users.set_status(u["id"], "active")
            st.rerun()

    st.subheader("Decisions")
    st.caption("Every step: who decided (fast path, agent, safety rule, fixed rules) and why.")
    decisions = pipe.audit.recent(100)
    if decisions:
        df = pd.DataFrame(decisions)[["timestamp", "decided_by", "decision", "person", "reasoning"]]
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(df, hide_index=True)
    else:
        st.write("No decisions yet.")

    st.subheader("Access log")
    logs = pipe.logs.get_recent(100)
    if logs:
        df = pd.DataFrame(logs).drop(columns=["id"])
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(df, hide_index=True)
    else:
        st.write("No access attempts yet.")
