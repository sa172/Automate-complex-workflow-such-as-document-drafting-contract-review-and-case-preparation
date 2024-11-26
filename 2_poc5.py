import PyPDF2
from transformers import pipeline
import streamlit as st

# Load the Hugging Face models for summarization and NER
# Replace with free-tier LLaMA or other models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def summarize_in_chunks(text, max_chunk_size=500):
    # Split the text into chunks of max_chunk_size words
    chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    return ' '.join(summaries)

# Streamlit interface for file upload and query input
st.title("Automated Legal Workflow")
uploaded_file = st.file_uploader("Upload Legal Document (PDF)", type="pdf")


if uploaded_file:
    # Extract text from the PDF
    document_text = extract_text_from_pdf(uploaded_file)
    document_text =document_text[:1024]
    # Summarize the document
    st.write("Summarizing the document...")
    summary = summarize_in_chunks(document_text)
    st.write("Summary:")
    # st.write(summary)
    st.text_area("Summarized Text" , document_text,height=100)
    if st.button("Extract"):
    # Perform Named Entity Recognition (NER) on the document
        st.write("Extracting entities...")
        entities = ner(document_text)
        st.write("Named Entities:")
        for entity in entities:
            st.write(f"{entity['word']}: {entity['entity']}")
# Function to generate legal draft based on user prompt
def generate_legal_draft(prompt):
    generator = pipeline('text-generation', model="facebook/opt-350m")  # Free LLaMA-like model
    draft = generator(prompt, max_length=100)[0]['generated_text']
    return draft

# Example usage
prompt = st.text_input("Enter a input for the draft generation")
if st.button("Generate"):
    draft = generate_legal_draft(prompt)
    st.write("Generated Legal Draft:")
    st.write(draft)
