import PyPDF2
from transformers import pipeline
# from celery import Celery
import streamlit as st
import tempfile
import os

# Initializing Celery
# app = Celery('legal_workflow', broker='redis://localhost:6379/0', backend = 'redis://localhost:6379/0')

# without specifying the model by default it will take the bert model "facebook/bart-large-cnn" for summarization.
summarizer = pipeline("summarization",model ="facebook/bart-large-cnn" )
# for this the default model will be used is "dbmdz/bert-large-cased-finetuned-conll03-english"  for name entity recognition.
ner = pipeline("ner",model = "dbmdz/bert-large-cased-finetuned-conll03-english")

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def process_document(text):
    # Summarize document
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
    
    # Extract entities
    entities = ner(text[:10000])  # Limit to first 10000 characters for speed
    
    return {
        "summary": summary,
        "entities": entities
    }

def generate_draft(prompt): 
    # In a real-world scenario, you'd use a more sophisticated model
    # This is a placeholder for demonstration
    return f"Draft based on prompt: {prompt}\n\n[Insert detailed legal language here...]"

def main():
    st.title("Legal Document Processor")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        text = extract_text_from_pdf(uploaded_file)
        
        st.subheader("Document Text")
        st.text_area("Extracted Text", text[:1000] + "...", height=200)
        
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                result = process_document(text)
            
            st.subheader("Document Summary")
            st.write(result["summary"])
            
            st.subheader("Extracted Entities")
            entities_text = "\n".join([f"{e['word']}: {e['entity']}" for e in result["entities"]])
            st.text_area("Entities", entities_text, height=200)
    
    st.subheader("Generate Draft")
    prompt = st.text_input("Enter a prompt for draft generation")
    if st.button("Generate Draft"):
        with st.spinner("Generating draft..."):
            draft = generate_draft(prompt)
        st.text_area("Generated Draft", draft, height=300)

if __name__ == "__main__":
    main()

# def process_document(text, max_length=1000000):
#     if not text:
#         return {"summary": "No text to process. The PDF might be empty or unreadable.", "entities": []}

#     result = {"summary": "", "entities": []}

#     # Summarize document
#     try:
#         chunks = [text[i:i+1000] for i in range(0, min(len(text), max_length), 1000)]
#         summaries = summarizer(chunks, max_length=30, min_length=10, do_sample=False)
#         result["summary"] = " ".join([s['summary_text'] for s in summaries])
#     except Exception as e:
#         result["summary"] = f"Summarization failed: {str(e)}"

#     # Extract entities
#     try:
#         result["entities"] = ner(text[:min(len(text), max_length)])
#     except Exception as e:
#         result["entities"] = []

#     return result

# def save_uploadedfile(uploadedfile):
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
#         tmp_file.write(uploadedfile.getvalue())
#         return tmp_file.name
    
# def generate_draft(prompt):
#     # In a real-world scenario, you'd use a more sophisticated model
#     # This is a placeholder for demonstration
#     return f"Draft based on prompt: {prompt}\n\n[Insert detailed legal language here...]"


# def main():
#     st.title("Batch Legal Document Processor")

#     uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
#     if uploaded_files:
#         st.write(f"Number of files uploaded: {len(uploaded_files)}")
        
#         if st.button("Process Documents"):
#             progress_bar = st.progress(0)
#             results = []

#             for i, uploaded_file in enumerate(uploaded_files):
#                 st.write(f"Processing: {uploaded_file.name}")
                
#                 # Save the uploaded file temporarily
#                 temp_file_path = save_uploadedfile(uploaded_file)
                
#                 # Process the document
#                 with open(temp_file_path, "rb") as file:
#                     text = extract_text_from_pdf(file)
#                     if text:
#                         result = process_document(text)
#                         results.append({"filename": uploaded_file.name, "result": result})
#                     else:
#                         results.append({"filename": uploaded_file.name, "result": {"summary": "Failed to extract text", "entities": []}})
                
#                 # Remove the temporary file
#                 os.unlink(temp_file_path)
                
#                 # Update progress
#                 progress_bar.progress((i + 1) / len(uploaded_files))

#             # Display results
#             for result in results:
#                 st.subheader(f"Results for {result['filename']}")
#                 st.write("Summary:")
#                 st.write(result['result']['summary'])
#                 st.write("Entities:")
#                 entities_text = "\n".join([f"{e['word']}: {e['entity']}" for e in result['result']['entities']])
#                 st.text_area("", entities_text, height=100)


    
# # def main():
# #     st.title("Automated Legal Workflow with AI")
# #     uploaded_file = st.file_uploader("Upload a Legal Document (PDF)", type="pdf")
# #     # user_prompt = st.text_input("Enter a prompt for drafting (e.g., 'Prepare an NDA')")
    
# #     if uploaded_file is not None:
# #         text = extract_text_from_pdf(uploaded_file)
        
# #         st.subheader("Document Text")
# #         st.text_area("Extracted Text ",text[:1000] + '...',height = 200)
# #         if st.button('Process Document'):
# #             with st.spinner("Processing Document..."):
# #                 result = process_document(text)
# #             st.subheader("Document Summary")
# #             st.write(result['summary'])
# #             st.subheader("Extracted Entities")
# #             entities_text  ='\n'.join([f"{e['word']}:{e['entity']}" for e in result['entities']])
# #             st.text_area("Entities", entities_text , height = 200)
            
#     st.subheader("Generating Draft")
#     prompt = st.text_input("Enter a prompt for draft generation")
#     if st.button("Generate Draft"):
#         with st.spinner("Generating Draft..."):
#             draft = generate_draft(prompt)
#         st.text_area("Generated Draft", draft, height=300)
           

# if __name__ == "__main__":
#     main()


        