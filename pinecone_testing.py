import os
import io
import streamlit as st
import fitz  # PyMuPDF
import PyPDF2
import re
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import openai
import numpy as np
from sklearn.mixture import GaussianMixture
from pinecone import Pinecone, ServerlessSpec
import json
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

# Load environment variables
load_dotenv(find_dotenv())

# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = 'us-east-1'  # Correct Pinecone environment
MODEL_NAME = os.getenv("LLM_MODEL")
TEMPERATURE = 0.5

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "new-index"

# Connect to the existing Pinecone index or create if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Assuming the dimension for embeddings
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region=PINECONE_ENV
        )
    )

index = pc.Index(index_name)

# Streamlit configuration
st.set_page_config(page_title="PDF Content Analyzer")
st.header("PDF Content Analyzer")
st.sidebar.title("Options")

# Clear conversation
clear_button = st.sidebar.button("Clear Conversation", key="clear")
if clear_button or "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in content extraction and question answering."}
    ]

# Function to extract text from PDF by page and ensure patient name is stored in metadata
def extract_text_by_page_with_metadata(pdf_file):
    page_texts = []
    patient_name = None
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        for page_number in range(len(reader.pages)):
            page = reader.pages[page_number]
            text = page.extract_text()

            # Try to detect patient name on the page
            if not patient_name:
                patient_name = extract_patient_name(text)

            # Store the extracted text and associated metadata
            if text.strip():
                page_texts.append((page_number + 1, text, patient_name)) 
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
    return page_texts

# Function to extract patient name from text
def extract_patient_name(text):
    name_match = re.search(r"Name:\s+([A-Za-z\s]+)", text)
    if name_match:
        return name_match.group(1).strip()
    return None

# Function to chunk text by page and include patient name in metadata
def chunk_text_by_page(page_texts, chunk_size=100):
    chunks = []
    chunk_page_numbers = []
    patient_names = []
    for page_number, text, patient_name in page_texts:
        sentences = sent_tokenize(text)
        current_chunk = ''
        for sentence in sentences:
            if len(current_chunk.split()) + len(sentence.split()) <= chunk_size:
                current_chunk += sentence + ' '
            else:
                chunks.append(current_chunk.strip())
                chunk_page_numbers.append(page_number)
                patient_names.append(patient_name)
                current_chunk = sentence + ' '
        if current_chunk:
            chunks.append(current_chunk.strip())
            chunk_page_numbers.append(page_number)
            patient_names.append(patient_name)
    return chunks, chunk_page_numbers, patient_names

# Function to embed chunks
def embed_chunks(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, convert_to_tensor=True).cpu().numpy()
    return embeddings

# Function to summarize text using GPT
def summarize_text_gpt(text):
    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
            ],
            max_tokens=150,
            temperature=0.5
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return "Error occurred."

# Recursive clustering and summarization
def recursive_clustering(embeddings, chunks, depth=2, current_depth=1):
    if current_depth > depth or len(chunks) < 2:
        return {f'level_{current_depth}': chunks}
    
    gmm = GaussianMixture(n_components=2, covariance_type='tied')
    gmm.fit(embeddings)
    cluster_labels = gmm.predict(embeddings)
    clusters = {}
    for idx, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(chunks[idx])

    summaries = {}
    for cluster_id, cluster_chunks in clusters.items():
        combined_text = ' '.join(cluster_chunks)
        summary = summarize_text_gpt(combined_text)
        summary_embedding = embed_chunks([summary])
        summaries[cluster_id] = recursive_clustering(summary_embedding, [summary], depth, current_depth + 1)

    return summaries

# Function to clear the Pinecone index
def clear_pinecone_index():
    index.delete(delete_all=True)

# Function to read filenames from a JSON file
def read_uploaded_files(json_file_path):
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as file:
            return json.load(file)
    return []

# Function to write filenames to a JSON file
def write_uploaded_files(json_file_path, file_list):
    with open(json_file_path, 'w') as file:
        json.dump(file_list, file, indent=4)

# Function to add a new filename to the JSON file
def add_uploaded_file(json_file_path, filename):
    file_list = read_uploaded_files(json_file_path)
    if filename not in file_list:
        file_list.append(filename)
    write_uploaded_files(json_file_path, file_list)

def clear_uploaded_files(json_file_path):
    try:
        write_uploaded_files(json_file_path, [])
        st.sidebar.success("File history cleared from JSON file.")
    except Exception as e:
        st.error(f"Error clearing JSON file: {e}")
    clear_pinecone_index()

# Sidebar: Display uploaded files
json_file_path = 'uploaded_files.json'  # Define the JSON file path for storing uploaded filenames

clear_DB_button = st.sidebar.button("Clear File History", key="ClearFileHistory")
if clear_DB_button:
    clear_uploaded_files(json_file_path)
    st.experimental_rerun()

uploaded_files = read_uploaded_files(json_file_path)
st.sidebar.title("Uploaded Files")
if uploaded_files:
    for file in uploaded_files:
        st.sidebar.write(file)
else:
    st.sidebar.write("No files uploaded.")

# Upload PDF files
pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Process uploaded files
if pdf_files:
    try:
        all_embeddings = []
        page_data = []

        for pdf_file in pdf_files:
            pdf_data = io.BytesIO(pdf_file.getvalue())
            page_texts = extract_text_by_page_with_metadata(pdf_data)
            pdf_name = pdf_file.name
            page_data.append((pdf_name, page_texts))
            add_uploaded_file(json_file_path, pdf_name)

            for pdf_name, page_texts in page_data:
                chunks, chunk_page_numbers, patient_names = chunk_text_by_page(page_texts)
                embeddings = embed_chunks(chunks)
                hierarchical_structure = recursive_clustering(embeddings, chunks, 2)
                all_embeddings.append((embeddings, chunks, chunk_page_numbers, pdf_name, hierarchical_structure))

            st.sidebar.header("Extracted Text from PDFs")
            for pdf_name, page_texts in page_data:
                st.sidebar.subheader(pdf_name)

            for embeddings, chunks, chunk_page_numbers, pdf_name, hierarchical_structure in all_embeddings:
                metadata = [{"pdf_name": pdf_name, "chunk": chunk, "page_number": chunk_page_numbers[i], "patient_name": patient_names[i], "chunk_index": i, "hierarchical_level": 1} for i, chunk in enumerate(chunks)]

                ids = [f"{pdf_name}-{i}" for i in range(len(metadata))]
                vectors = [{"id": id, "values": embedding.tolist(), "metadata": meta} for id, embedding, meta in zip(ids, embeddings, metadata)]
                
                index.upsert(vectors)

                for level, summaries in hierarchical_structure.items():
                    if isinstance(level, str) and "level_" in level:
                        level_num = int(level.split('_')[1])
                        for summary in summaries:
                            summary_embedding = embed_chunks([summary])[0]
                            ids.append(f"{pdf_name}-summary-{level_num}")
                            vectors.append({"id": f"{pdf_name}-summary-{level_num}", "values": summary_embedding.tolist(), "metadata": {"pdf_name": pdf_name, "chunk": summary, "page_number": None, "patient_name": patient_names, "chunk_index": -1, "hierarchical_level": level_num}})
                
                index.upsert(vectors)
                
            st.sidebar.success("Data inserted into Pinecone successfully!")
    except Exception as e:
        st.error(f"Failed to process PDFs: {e}")

# Define retrieval functions using Pinecone
def retrieve_documents(query_embedding, top_k=10):
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_values=True,
        include_metadata=True
    )
    return results

def create_context_from_metadata(results):
    context_chunks = []
    context_metadata = []

    for result in results['matches']:
        metadata = result['metadata']
        chunk = metadata.get("chunk", "")
        context_chunks.append(chunk)
        context_metadata.append(metadata)

    context = " ".join(context_chunks)
    return context, context_metadata

def normalize_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*-\s*", "-", text)
    return text.strip()

def highlight_pdf_text(pdf_path, highlight_text, page_numbers):
    try:
        document = fitz.open(pdf_path)
        for page_num in page_numbers:
            page = document.load_page(page_num - 1)
            text_instances = page.search_for(highlight_text)
            if text_instances:
                for inst in text_instances:
                    highlight = page.add_highlight_annot(inst)
                    highlight.update()

        # Save the modified PDF to the Downloads directory
        downloads_path = Path.home() / "Downloads" / f"highlighted_{Path(pdf_path).name}"
        document.save(downloads_path)
        document.close()

        return downloads_path

    except Exception as e:
        print(f"Error in highlight_pdf_text: {e}")
        return None

def get_next_file_path(base_dir, base_filename, file_extension="pdf"):
    """
    This function returns the next available file path with an incrementing number.
    """
    base_path = Path(base_dir) / base_filename
    count = 1
    while (base_path.with_stem(f"{base_filename}{count}").with_suffix(f".{file_extension}")).exists():
        count += 1
    return base_path.with_stem(f"{base_filename}{count}").with_suffix(f".{file_extension}")
base_directory = Path.home() / "Downloads"  # Set your desired directory
base_filename = "highlighted_output"

def find_answer_gpt(question, context_metadata):
    try:
        context_info = json.dumps(context_metadata, indent=2)

        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant who reads reports."},
                {"role": "user",
                 "content": f"""Please answer the following question by returning only a JSON response. The JSON response should include the following fields: 'highlight', 'filename', 'page_number', and 'ans'. Use this format:

{{
  "highlight": "[Exact text to be highlighted]",
  "filename": "[Name of the PDF file]",
  "page_number": [Page number(s)],
  "ans": "[Direct answer to the query]"
}}
You are given 10 metadata containing fields like chunk and patient name. Answer the question by correctly matching the name in query and patient_name in metadata only.
Ensure the JSON response is valid and all fields are correctly formatted. Now, please answer the question: {question}

Context metadata: {context_info}"""}
            ],
            max_tokens=150,
            temperature=TEMPERATURE
        )
        
        raw_response = response.choices[0].message['content'].strip()
        return raw_response

    except Exception as e:
        print(f"Error occurred in find_answer_gpt: {e}")
        return "Error occurred."
    
def test_highlight_pdf_text(pdf_path, highlight_text1):
    try:
        # Open the PDF file
        document = fitz.open(pdf_path)

        # Iterate through all pages
        for page_num in range(len(document)):
            page = document.load_page(page_num)  # PyMuPDF uses 0-based indexing
            text_instances = page.search_for(highlight_text1)

            if text_instances:
                for inst in text_instances:
                    highlight = page.add_highlight_annot(inst)
                    highlight.update()
                print(f"Highlighted '{highlight_text1}' on page {page_num + 1}")
            else:
                print(f"No matching text found on page {page_num + 1} for '{highlight_text1}'")

        # Save the modified PDF to a temporary in-memory buffer
        temp_file = io.BytesIO()
        document.save(temp_file)
        document.close()
        temp_file.seek(0)  # Ensure the stream is at the start

        return temp_file, None  # No error

    except Exception as e:
        print(f"Error in test_highlight_pdf_text: {e}")
        return None, e  # Return the error
    
def parse_gpt_response(response):
    try:
        data = json.loads(response)
        return data
    except json.JSONDecodeError as je:
        print(f"JSONDecodeError: {je}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Streamlit interface for querying
query = st.text_input("Enter your query:", key="query_input")

if query:
    query_embedding = embed_chunks([query])[0]
    results = retrieve_documents(query_embedding)
    if results:
        with st.spinner("Assistant is typing..."):
            context, context_metadata = create_context_from_metadata(results)
            st.write("Context Metadata:")
            st.write(context_metadata)

            if context_metadata:
                answer = find_answer_gpt(query, context_metadata)
                st.write(f"Answer: {answer}")

                data = parse_gpt_response(answer)

                if data:
                    try:
                        highlight_text = data.get("highlight", "")
                        filename = data.get("filename", "")
                        page_numbers = data.get("page_number", [])
                        ans = data.get("ans", "")

                        # Display the direct answer
                        st.write(f"Answer: {ans}")
                        directory = "/Users/sarthakgarg/Documents/Sarthak/"
                        pdf_path = os.path.join(directory, filename)

                        highlight_text1 = highlight_text
                        output_path = get_next_file_path(base_directory, base_filename)

                        # Run the test function
                        highlighted_pdf, error = test_highlight_pdf_text(pdf_path, highlight_text1)
                        if highlighted_pdf:
                            with open(output_path, "wb") as f:
                                f.write(highlighted_pdf.getbuffer())
                            st.success(f"PDF Highlight successful. File saved as {output_path.name}")
                        else:
                            st.error(f"Highlighting failed: {error}")

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        print(f"Exception: {e}")
                else:
                    st.error("Failed to parse the GPT response.")
