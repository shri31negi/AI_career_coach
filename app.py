
import streamlit as st
import io
from career_coach import (
    extract_text_from_pdf, parse_resume, build_profile_summary,
    suggest_careers_with_steps, SimpleRetriever, format_advice_block
)

st.set_page_config(page_title="Resume Career Coach", page_icon="🧭", layout="wide")

st.title("🧭 Resume Career Coach – PDF → Insights → Advice")

with st.sidebar:
    st.header("1) Upload Resume PDF")
    uploaded = st.file_uploader("Upload your resume PDF", type=["pdf"])
    st.caption("Tip: Export your latest resume to PDF for best results.")
    st.header("2) Chat Settings")
    top_k = st.slider("Knowledge snippets to use", 1, 10, 3)
    temperature = st.slider("Response creativity (0 = factual, 1 = creative)", 0.0, 1.0, 0.2, 0.1)
    st.markdown("---")
    st.caption("This app runs fully locally with rule-based + retrieval logic (no external API).")

if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "parsed" not in st.session_state:
    st.session_state.parsed = None
if "advice" not in st.session_state:
    st.session_state.advice = []

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📄 Extracted Resume Overview")
    if uploaded is not None:
        resume_bytes = uploaded.read()
        text = extract_text_from_pdf(io.BytesIO(resume_bytes))
        st.session_state.resume_text = text

        parsed = parse_resume(text)
        st.session_state.parsed = parsed

        profile = build_profile_summary(parsed)
        st.write(profile)

        retriever = SimpleRetriever.from_texts(
            [
                ("resume_full_text", text),
                ("skills_section", "\n".join(parsed.get("skills_list", []))),
                ("education_section", parsed.get("education_text", "")),
                ("experience_section", parsed.get("experience_text", "")),
                ("projects_section", parsed.get("projects_text", "")),
                ("achievements_section", parsed.get("achievements_text", "")),
            ]
        )
        st.session_state.retriever = retriever

        st.success("Resume processed and indexed for chat.")
    else:
        st.info("Upload a PDF in the sidebar to begin.")

with col2:
    st.subheader("💡 Career Path Suggestions + Next Steps")
    if st.session_state.parsed:
        st.session_state.advice = suggest_careers_with_steps(st.session_state.parsed.get("skills_list", []))
        for blk in st.session_state.advice:
            st.markdown(format_advice_block(blk), unsafe_allow_html=True)
    else:
        st.caption("Suggestions will appear here after you upload a resume.")

st.markdown("---")
st.subheader("💬 Career Coaching Chatbot")

if st.session_state.retriever is None:
    st.warning("Upload your resume first to enable chat.")
else:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  

    for m in st.session_state.chat_history:
        if m["role"] == "user":
            st.chat_message("user").write(m["content"])
        else:
            st.chat_message("assistant").write(m["content"])

    user_msg = st.chat_input("Ask for career guidance (e.g., suitable roles, projects to add, how to tailor resume)")
    if user_msg:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})

        hits = st.session_state.retriever.search(user_msg, top_k=top_k)

        parsed = st.session_state.parsed or {}
        skills = parsed.get("skills_list", [])
        adverbs = ["clearly", "concisely", "actionably"]
        response_parts = []
        response_parts.append("Here's guidance tailored to your resume and question:")

        if hits:
            response_parts.append("**Relevant resume highlights:**")
            for i, (doc_id, score, text) in enumerate(hits, start=1):
                excerpt = (text[:400] + "…") if len(text) > 400 else text
                response_parts.append(f"{i}. _{doc_id}_ (match {score:.3f}): {excerpt}")

        
        q = user_msg.lower()
        if any(k in q for k in ["role", "job", "position", "career", "path"]):
            response_parts.append("**Suggested roles** based on your skills:")
            for blk in suggest_careers_with_steps(skills):
                response_parts.append(f"- **{blk['career']}** — top skills matched: {', '.join(blk['matched_skills']) or 'core skills'}")

        if any(k in q for k in ["project", "portfolio", "github"]):
            response_parts.append("**Projects you could add:**")
            response_parts += [
                "- End-to-end project showcasing problem, data, modeling/architecture, evaluation, and deployment.",
                "- Include README with metrics, demo screenshots, and brief write-up of impact."
            ]

        if any(k in q for k in ["resume", "cv", "ats", "tailor", "improve"]):
            response_parts.append("**Resume improvements (ATS-friendly):**")
            response_parts += [
                "- Use action verbs and quantify impact (e.g., 'improved accuracy by 7%').",
                "- Group skills into categories (Languages, Libraries, Cloud, Tools).",
                "- Keep to 1 page for <5 years experience; 2 pages otherwise.",
            ]

        if not any(k in q for k in ["role", "job", "position", "career", "project", "portfolio", "github", "resume", "cv", "ats", "tailor", "improve"]):

            response_parts.append("**General guidance:** Focus your resume on impact, quantify results, and ensure skills match target roles.")

        
        if skills:
            response_parts.append("**Next Steps (based on your skills):**")
            for blk in suggest_careers_with_steps(skills):
                response_parts.append(f"- For **{blk['career']}**: {blk['next_steps'][0]}")
        else:
            response_parts.append("**Next Steps:** Add a clear 'Skills' section with your strongest tools/technologies.")

        reply = "\n\n".join(response_parts)
        st.chat_message("assistant").write(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
