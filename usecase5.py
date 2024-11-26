
import requests
from dotenv import load_dotenv
import os
import google.generativeai as genai
import faiss
import numpy as np
from transformers import pipeline
import PyPDF2
import streamlit as st

genai.configure(api_key = os.getenv('GOOGLE_API_KEY'))

# for m in genai.list_models():
#   if 'embedContent' in m.supported_generation_methods:
#     print(m.name)


def extract_text_from_pdf(uploaded_file):
  reader = PyPDF2.PdfReader(uploaded_file)
  text = ""
  for page in reader.pages:
      text += page.extract_text().encode('utf-8').decode('utf-8') + "\n"
  return text

def chunk_document(document_text, chunk_size=300, overlap=50):

  words = document_text.split()  # Split text into words
  chunks = []
  
  for i in range(0, len(words), chunk_size - overlap):
      chunk = " ".join(words[i:i + chunk_size])
      chunks.append(chunk)
  
  return chunks

def generate_embedding(text):
  model = 'models/embedding-001'
  embedding = genai.embed_content(
      model=model,
      content=text,
      task_type="retrieval_document"
  )
  return embedding

# Step 3: Build FAISS index
def build_faiss_index(document_text):
  chunks = chunk_document(document_text)
  embeddings = []
  for chunk in chunks:
    embedding_response = generate_embedding(chunk)  # it output is in dict format
    # print(type(embedding_response))
    embedding = embedding_response.get('embedding') 
    if embedding is not None:
        embeddings.append(embedding)
    else:
        st.warning(f"Embedding not found for chunk: {chunk}")

  # Ensure the embeddings are in the correct format for FAISS
  embedding_matrix = np.array(embeddings).astype('float32')
  index = faiss.IndexFlatL2(embedding_matrix.shape[1])  # L2 distance
  index.add(embedding_matrix)  # Add embeddings to the index
  return index, chunks

# Step 4: Function to search for similar documents
def search_similar_documents(query, index , chunks):
  query_embedding = generate_embedding(query)
  query_embedding = np.array(query_embedding.get('embedding')).astype('float32').reshape(1, -1)
  distances, indices = index.search(query_embedding, 5)  # Search for top 5 nearest chunks
  return [chunks[i] for i in indices[0]]

# Step 5: Function to extract key clauses from a document
def extract_key_clauses(document_text):
  
  # model = 'models/embedding-001'
  
  # # Use the Gemini API to extract key clauses
  # extracted_clauses = genai.extract_key_phrases(  # Placeholder for actual function
  #     model=model,
  #     content=document_text,
  #     task_type="key_clause_extraction"
  # )
  
    # Assume the API returns a list of key clauses or phrases
  # return extracted_clauses
  key_clauses = []  # Placeholder for extracted clauses
  sentences = document_text.split('. ')
  key_clauses = sentences[:3]  # Example logic: take first 3 sentences
  return key_clauses

# Step 6: Summarizing legal documents
summarizer = pipeline("summarization")

def summarize_document(document_text):
  summary = summarizer(document_text, max_length=150, min_length=30, do_sample=False)
  return summary[0]['summary_text']

# Step 7: Drafting initial responses
def draft_response(query):
  model = genai.GenerativeModel("gemini-1.5-flash")
  response = model.generate_content(f"draft for:{query}")
  return response.text
  #   # Implement logic to draft a response based on the query
  # response = f"Draft response for: {query}"  # Placeholder for drafting logic
  # return response

def main():
  st.title("Legal Document Workflow")

  # File upload for the single legal document
  uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

  if uploaded_file:
    # Extract the text from the uploaded PDF file
    document_text = extract_text_from_pdf(uploaded_file)
    
    # Build FAISS index from sections of the document
    if 'faiss_index' not in st.session_state:
      index, sections = build_faiss_index(document_text)
      st.session_state['faiss_index'] = index
      st.session_state['sections'] = sections
    else:
      index = st.session_state['faiss_index']
      sections = st.session_state['sections']

    st.success("Document processed and indexed successfully!")

    # Search for the most relevant section based on the query
    query = st.text_input("Enter a query to search within the document:")
    if query:
      relevant_sections = search_similar_documents(query, index, sections)
      print(type(relevant_sections))
      # st.write("Relevant Sections:", relevant_sections)
      st.text_area('Similar Search',relevant_sections[:2000],height=100)

      # Extract key clauses from the most relevant section
      key_clauses = extract_key_clauses(relevant_sections[0])
      # st.write("Extracted Key Clauses:", key_clauses)
      st.text_area('Extracted Key Clause',key_clauses[:1000],height=100)

      # Summarize the most relevant section
      summary = summarize_document(relevant_sections[0])
      # st.write("Section Summary:", summary)
      st.text_area('Summarized Text',summary[:2000],height=100)

      # Draft a response based on the query
      response = draft_response(query)
      st.write("Draft Response:", response)
      
if __name__ == "__main__":
  main()
