import streamlit as st
import io
from career_coach import (
    extract_text_from_pdf, parse_resume, build_profile_summary,
    suggest_careers_with_steps, SimpleRetriever, format_advice_block
)

st.set_page_config(page_title="Resume Career Coach", page_icon="🧭", layout="wide")

st.title("🧭 Resume Career Coach – PDF → Insights → Advice")

# --- Sidebar ---
with st.sidebar:
    st.header("1) Upload Resume PDF")
    uploaded = st.file_uploader("Upload your resume PDF", type=["pdf"])
    st.caption("Tip: Export your latest resume to PDF for best results.")

    st.header("2) Chat Settings")
    top_k = st.slider("Knowledge snippets to use", 1, 10, 3)
    temperature = st.slider("Response creativity", 0.0, 1.0, 0.2, 0.1)

    st.markdown("---")
    st.caption("This app runs locally with rule-based + retrieval logic (no external API).")

# --- Session State Defaults ---
for key, default in {
    "retriever": None,
    "resume_text": "",
    "parsed": None,
    "advice": [],
    "chat_history": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- Layout ---
col1, col2 = st.columns(2)

# --- Left: Resume Overview ---
with col1:
    st.subheader("📄 Extracted Resume Overview")
    if uploaded:
        try:
            resume_bytes = uploaded.read()
            text = extract_text_from_pdf(io.BytesIO(resume_bytes)) or ""
            if not text.strip():
                st.error("Could not extract text from PDF. Try another export.")
            else:
                st.session_state.resume_text = text
                parsed = parse_resume(text)
                st.session_state.parsed = parsed

                profile = build_profile_summary(parsed)
                st.write(profile)

                retriever = SimpleRetriever.from_texts([
                    ("resume_full_text", text),
                    ("skills_section", "\n".join(parsed.get("skills_list", []))),
                    ("education_section", parsed.get("education_text", "")),
                    ("experience_section", parsed.get("experience_text", "")),
                    ("projects_section", parsed.get("projects_text", "")),
                    ("achievements_section", parsed.get("achievements_text", "")),
                ])
                st.session_state.retriever = retriever
                st.success("Resume processed and indexed for chat.")
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
    else:
        st.info("Upload a PDF in the sidebar to begin.")

# --- Right: Career Suggestions ---
with col2:
    st.subheader("💡 Career Path Suggestions + Next Steps")
    if st.session_state.parsed:
        st.session_state.advice = suggest_careers_with_steps(st.session_state.parsed)
        for blk in st.session_state.advice:
            st.markdown(format_advice_block(blk), unsafe_allow_html=True)
    else:
        st.caption("Suggestions will appear here after you upload a resume.")

# --- Chatbot ---
st.markdown("---")
st.subheader("💬 Career Coaching Chatbot")

if not st.session_state.retriever:
    st.warning("Upload your resume first to enable chat.")
else:
    for m in st.session_state.chat_history:
        st.chat_message(m["role"]).write(m["content"])

    user_msg = st.chat_input("Ask for career guidance...")
    if user_msg:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})

        hits = st.session_state.retriever.search(user_msg, top_k=top_k)
        parsed = st.session_state.parsed or {}
        advice_blocks = suggest_careers_with_steps(parsed)

        response_parts = ["Here's guidance tailored to your resume and question:"]
        if hits:
            response_parts.append("**Relevant resume highlights:**")
            for i, (doc_id, score, text) in enumerate(hits, 1):
                excerpt = (text[:400] + "…") if len(text) > 400 else text
                response_parts.append(f"{i}. _{doc_id}_ (score {score:.2f}): {excerpt}")

        q = user_msg.lower()
        # Role / job guidance
        if any(k in q for k in ["role", "job", "position", "career", "path"]):
            response_parts.append("**Suggested roles:**")
            for blk in advice_blocks:
                response_parts.append(f"- **{blk['career']}** (skills: {', '.join(blk['matched_skills']) or 'general'})")

        # Project guidance
        if any(k in q for k in ["project", "portfolio", "github"]):
            response_parts += [
                "**Projects you could add:**",
                "- End-to-end project with problem, data, model, evaluation, deployment.",
                "- Include README with metrics, screenshots, and brief impact write-up."
            ]

        # Resume improvement
        if any(k in q for k in ["resume", "cv", "ats", "tailor", "improve"]):
            response_parts += [
                "**Resume improvements (ATS-friendly):**",
                "- Use action verbs and quantify impact.",
                "- Group skills into categories.",
                "- Keep to 1 page (<5 yrs exp), else 2 pages."
            ]

        if not any(k in q for k in ["role", "job", "position", "career", "project", "portfolio", "github", "resume", "cv", "ats", "tailor", "improve"]):
            response_parts.append("**General guidance:** Quantify results, highlight impact, match target roles.")

        if advice_blocks:
            response_parts.append("**Next Steps (personalized):**")
            for blk in advice_blocks[:2]:
                if blk["next_steps"]:
                    response_parts.append(f"- {blk['career']}: {blk['next_steps'][0]}")
        else:
            response_parts.append("**Next Steps:** Add a clear Skills section.")

        reply = "\n\n".join(response_parts)
        st.chat_message("assistant").write(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
