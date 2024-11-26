import PyPDF2
from transformers import pipeline
import streamlit as st

# Using smaller, efficient models
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6",framework="pt")  # Efficient BART
ner = pipeline("ner", model="dslim/bert-base-NER")  # Efficient BERT for NER

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Streamlit interface for file upload and query input
st.title("Automated Legal Workflow")
uploaded_file = st.file_uploader("Upload Legal Document (PDF)", type="pdf")

if uploaded_file:
    # Extract text from the PDF
    document_text = extract_text_from_pdf(uploaded_file)

    # Summarize the document
    st.write("Summarizing the document...")
    summary = summarizer(document_text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
    st.write("Summary:")
    st.write(summary)

    # Perform Named Entity Recognition (NER) on the document
    if st.button("Extract"):
        st.write("Extracting entities...")
        entities = ner(document_text)
        st.write("Named Entities:")
        for entity in entities:
            st.write(f"{entity['word']}: {entity['entity']}")
            
def generate_legal_draft(prompt):
    generator = pipeline('text-generation', model="facebook/opt-350m")
    draft = generator(prompt, max_length=100)[0]['generated_text']
    return draft

prompt = "Prepare a non-disclosure agreement for a company."
draft = generate_legal_draft(prompt)
st.write("Generated Legal Draft:")
st.write(draft)
