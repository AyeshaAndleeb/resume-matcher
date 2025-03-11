import streamlit as st
import spacy
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2  # For PDF file support
from io import StringIO

# Function to download and load SpaCy model
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.warning("Downloading SpaCy model 'en_core_web_sm'... This may take a few minutes.")
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp

# Load SpaCy model
nlp = load_spacy_model()

# Function to extract text from a PDF file
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from a TXT file
def extract_text_from_txt(uploaded_file):
    return uploaded_file.read().decode("utf-8")

# Function to parse resume text
def parse_resume(resume_text):
    doc = nlp(resume_text)
    skills = []
    education = []
    experience = []

    for ent in doc.ents:
        if ent.label_ == "ORG":  # Example: Extract organizations as skills
            skills.append(ent.text)
        elif "education" in ent.text.lower():  # Example: Extract education
            education.append(ent.text)
        elif "experience" in ent.text.lower():  # Example: Extract experience
            experience.append(ent.text)

    return {
        "skills": skills,
        "education": education,
        "experience": experience
    }

# Function to compute similarity between resume and job descriptions
def compute_similarity(resume_text, job_description):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0]

# Function to rank job matches
def rank_matches(resume_text, job_descriptions):
    matches = []
    for job_desc in job_descriptions:
        similarity = compute_similarity(resume_text, job_desc)
        matches.append((job_desc, similarity))
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches

# Streamlit App
st.title("AI-Powered Resume Parser and Job Matcher")

# Upload resume
uploaded_file = st.file_uploader("Upload your resume (TXT, PDF)", type=["txt", "pdf"])
if uploaded_file is not None:
    # Extract text based on file type
    if uploaded_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "text/plain":
        resume_text = extract_text_from_txt(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a TXT or PDF file.")
        st.stop()

    st.success("Resume Uploaded and Parsed Successfully!")

    # Parse resume
    parsed_resume = parse_resume(resume_text)
    st.subheader("Extracted Information:")
    st.write("**Skills:**", ", ".join(parsed_resume["skills"]))
    st.write("**Education:**", ", ".join(parsed_resume["education"]))
    st.write("**Experience:**", ", ".join(parsed_resume["experience"]))

    # Input job descriptions
    st.subheader("Job Matching")
    job_descriptions = st.text_area("Paste job descriptions (one per line)")
    if job_descriptions:
        job_descriptions = job_descriptions.split("\n")
        matches = rank_matches(resume_text, job_descriptions)

        st.subheader("Top Job Matches:")
        for idx, (job_desc, similarity) in enumerate(matches, 1):
            st.write(f"**Match {idx}**")
            st.write(f"**Job Description:** {job_desc}")
            st.write(f"**Similarity Score:** {similarity:.2f}")
            st.write("---")
else:
    st.info("Please upload a resume to get started.")