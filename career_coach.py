
import pdfplumber, re, math
from typing import List, Dict, Tuple
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


import pdfplumber

def extract_text_from_pdf(file) -> str:
    """Extract raw text from a PDF file-like object using pdfplumber."""
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text.strip()




SECTION_HEADERS = [
    r"education", r"experience", r"work experience", r"skills",
    r"projects", r"achievements", r"certifications", r"publications",
]

SKILL_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\+\#\.\-]{1,}", re.IGNORECASE)

TECH_SKILL_HINTS = {
    "python","java","javascript","typescript","c","c++","c#","go","rust","sql","nosql",
    "html","css","react","angular","vue","next.js","node","node.js","spring","springboot",
    "docker","kubernetes","aws","azure","gcp","linux","git","github","gitlab",
    "tensorflow","pytorch","sklearn","scikit-learn","nlp","opencv","pandas","numpy",
    "spark","hadoop","airflow","dbt","kafka","redis","mongodb","postgres","mysql",
    "powerbi","tableau","excel","matplotlib","seaborn","plotly",
    "flutter","android","ios","swift","kotlin","dart",
    "fastapi","flask","django","rest","graphql",
    "ml","dl","machine","deep","transformers","llm","genai","rag",
    "devops","ci/cd","microservices","terraform","ansible",
}

def _extract_section(text: str, header: str) -> str:
    pattern = re.compile(rf"(?is){header}\s*[:\-]?\s*(.*?)(?=(" + "|".join(SECTION_HEADERS) + r")\s*[:\-]|\Z)")
    m = pattern.search(text)
    return m.group(1).strip() if m else ""

def parse_resume(text: str) -> Dict:
    lower = text.lower()
    parsed = {}

    parsed["education_text"] = _extract_section(text, r"education")
    parsed["experience_text"] = _extract_section(text, r"(work\s+)?experience")
    parsed["skills_text"] = _extract_section(text, r"skills")
    parsed["projects_text"] = _extract_section(text, r"projects")
    parsed["achievements_text"] = _extract_section(text, r"achievements|awards|honors")

   
    tokens = [t.lower() for t in SKILL_TOKEN.findall(parsed.get("skills_text",""))]
    skills_list = []
    for t in tokens:
        if t in TECH_SKILL_HINTS or t.replace(".","") in TECH_SKILL_HINTS:
            skills_list.append(t)
    
    seen = set()
    skills_list = [s for s in skills_list if not (s in seen or seen.add(s))]

    parsed["skills_list"] = skills_list

    return parsed


def build_profile_summary(parsed: Dict) -> str:
    skills = parsed.get("skills_list", [])
    edu = parsed.get("education_text","").strip().splitlines()[:3]
    exp = parsed.get("experience_text","").strip().splitlines()[:5]

    lines = []
    lines.append("**Detected Skills:** " + (", ".join(skills) if skills else "Not clearly listed."))
    if edu and any(e.strip() for e in edu):
        lines.append("**Education (snippet):** " + " | ".join([e.strip() for e in edu if e.strip()]))
    if exp and any(e.strip() for e in exp):
        lines.append("**Experience (snippet):** " + " | ".join([e.strip() for e in exp if e.strip()]))
    if parsed.get("projects_text","").strip():
        proj_line = parsed["projects_text"].strip().splitlines()[0]
        lines.append("**Projects (snippet):** " + proj_line[:180] + ("…" if len(proj_line) > 180 else ""))
    return "\n\n".join(lines)


CAREER_RULES = [
    {
        "career": "Data Scientist / ML Engineer",
        "match": {"any": {"python","pandas","numpy","sklearn","tensorflow","pytorch","ml","dl","nlp","transformers"}},
        "next_steps": [
            "Ship an end-to-end ML project with EDA → model → evaluation → deployment.",
            "Publish a notebook + README with metrics and lessons learned.",
            "Add MLOps basics: experiment tracking and reproducible pipelines.",
        ],
    },
    {
        "career": "Backend Engineer",
        "match": {"any": {"java","spring","springboot","node","node.js","fastapi","flask","django","graphql","rest","microservices"}},
        "next_steps": [
            "Build a production-grade REST API with tests and CI.",
            "Add auth, rate limiting, and observability (logging/metrics).",
            "Containerize and deploy to a cloud provider.",
        ],
    },
    {
        "career": "Frontend / Full‑Stack Engineer",
        "match": {"any": {"react","angular","vue","next.js","typescript","javascript","html","css"}},
        "next_steps": [
            "Create a multi-page app with routing, state management, and forms.",
            "Optimize bundle size and Core Web Vitals; add accessibility checks.",
            "Integrate a backend or third‑party API and deploy.",
        ],
    },
    {
        "career": "Cloud / DevOps Engineer",
        "match": {"any": {"aws","azure","gcp","docker","kubernetes","terraform","ansible","linux","ci/cd"}},
        "next_steps": [
            "Containerize a service and orchestrate with Kubernetes.",
            "Write IaC (Terraform) to provision cloud resources.",
            "Set up CI/CD with automated tests and security scans.",
        ],
    },
    {
        "career": "Data Engineer / Analytics Engineer",
        "match": {"any": {"spark","airflow","dbt","kafka","postgres","mysql","mongodb","sql","nosql","powerbi","tableau"}},
        "next_steps": [
            "Build a batch or streaming pipeline and document the lineage.",
            "Model data into clean marts and build a BI dashboard.",
            "Automate with orchestration and add tests/SLAs.",
        ],
    },
]

def _matches(rule_skills: set, candidate_skills: set) -> bool:
    return len(rule_skills.intersection(candidate_skills)) > 0

def suggest_careers_with_steps(skills_list: List[str]) -> List[Dict]:
    candidate = set([s.lower() for s in skills_list])
    results = []
    for rule in CAREER_RULES:
        needed = set([s.lower() for s in rule["match"]["any"]])
        if _matches(needed, candidate):
            results.append({
                "career": rule["career"],
                "matched_skills": sorted(list(needed.intersection(candidate))),
                "next_steps": rule["next_steps"],
            })
    if not results:
        results.append({
            "career": "General Tech Path",
            "matched_skills": [],
            "next_steps": [
                "Clarify your 'Skills' section with specific tools and versions.",
                "Add 2–3 quantified achievements per role or project.",
                "Target a role and mirror its keywords in your resume.",
            ],
        })
    return results

def format_advice_block(block: Dict) -> str:
    ms = ", ".join(block["matched_skills"]) if block["matched_skills"] else "core skills"
    steps = "".join([f"<li>{s}</li>" for s in block["next_steps"]])
    return f"""
<div style="padding:12px;border:1px solid #ddd;border-radius:8px;margin-bottom:10px;">
  <b>{block['career']}</b><br/>
  <i>Matched:</i> {ms}
  <ul>{steps}</ul>
</div>
"""


class SimpleRetriever:
    """
    TF‑IDF + cosine similarity retriever over provided documents.
    Acts as lightweight 'training' on the uploaded resume for grounding chat answers.
    """
    def __init__(self, ids: List[str], texts: List[str]):
        self.ids = ids
        self.texts = texts
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(texts)

    @classmethod
    def from_texts(cls, docs: List[Tuple[str, str]]):
        ids = [d[0] for d in docs]
        texts = [d[1] for d in docs]
        return cls(ids, texts)

    def search(self, query: str, top_k: int = 3):
        qvec = self.vectorizer.transform([query])
        sims = cosine_similarity(qvec, self.matrix).ravel()
     
        idxs = sims.argsort()[::-1][:top_k]
        results = []
        for i in idxs:
            if self.texts[i].strip():
                results.append((self.ids[i], float(sims[i]), self.texts[i]))
        return results
