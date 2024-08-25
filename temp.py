from flask import Blueprint, request, jsonify
import json
import os
import openai
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, utility, MilvusException
from datetime import datetime

query_bp = Blueprint('query_bp', __name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("LLM_MODEL")
TEMPERATURE = 0.5

# Connect to Milvus
def connect_to_milvus():
    connections.connect("default", host="0.0.0.0", port="19530")
    if utility.has_collection("exmpcollection1"):
        return Collection("exmpcollection1")
    return None

collection = connect_to_milvus()

if collection is None:
    raise Exception("Failed to connect to Milvus. Exiting...", 400)

# Query the collection and retrieve either summaries or detailed context (child nodes)
@query_bp.route("/query", methods=["POST"])
def query_pdf():
    try:
        data = request.get_json()
        query = data.get("query", "")
        filenames = data.get("filenames", None)  # Get the filenames from the request data

        query_embedding = embed_chunks([query])[0]
        
        # Determine whether to retrieve a high-level summary or child nodes for detailed context
        use_children = should_use_children(query)

        # Retrieve the documents from Milvus based on query type (summary or child nodes)
        results = retrieve_documents(query_embedding, filenames=filenames,top_k=10,use_children=True)
        #print(results)

        if not results:
            return jsonify({"error": "No results found"}), 404

        print("AG Testing context metadata resulsts before context building")
        print (results)
        print("AG Testing done")
         
        context, context_metadata = create_context_from_metadata(results)
        
        if not context_metadata:
            return jsonify({"error": "No context metadata found"}), 404

        # Generate answer using GPT with the selected context
        answer = find_answer_gpt(query, context_metadata)
        return answer

    except MilvusException as e:
        return jsonify(f"Milvus Error: {e}")
    except KeyError as e:
        return jsonify({"error": f"Missing key: {e}"}), 400  # Bad Request
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Embedding chunks for query
def embed_chunks(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, convert_to_tensor=True).cpu().numpy()
    return embeddings

# Retrieve documents from Milvus
def retrieve_documents(query_embedding, filenames=None, top_k=10, use_children=False):
    if not collection:
        return []

    search_params = {"metric_type": "COSINE", "params": {"ef": 200}}

    if filenames:
        filenames_expr = f"pdf_name in [{', '.join([f'\"{filename}\"' for filename in filenames])}]"
        expr = filenames_expr
    else:
        expr = None

    # Perform the search in Milvus
    results = collection.search(
        data=[query_embedding],
        anns_field="embeddings",
        param=search_params,
        limit=top_k,
        expr=expr,
        output_fields=["metadata"]
    )

    all_metadata = []
    print("Printing Metadata")
    print(results)
    print("Metadata printing over")

    # Process the search results
    for result in results:
        for hit in result:
            # Debugging: Log the structure of hit
            # # print ("1")
            # print("Processing hit:", hit)

            # Access metadata directly from hit
            metadata_str = hit.entity.get("metadata")
            # print ("3")
            # print(metadata_str)
            # print ("2")

            if not metadata_str:
                print("No metadata found in hit.")  # Debugging
                continue

            # Fix the escaping issue by loading the string as JSON
            if isinstance(metadata_str, str):
                try:
                    # Convert escaped string back to dictionary
                    metadata_str = metadata_str.replace("\\'", "'")  # Handle escaped single quotes
                    metadata = json.loads(metadata_str)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")  # Debugging
                    metadata = {}
            else:
                metadata = metadata_str  # If already a dict, no need to decode

            all_metadata.append(metadata)
            print ("Appended metadata")

            # If detailed context (child nodes) is needed and 'subtree' exists, retrieve child nodes
            if use_children and 'subtree' in metadata:
                child_metadata = fetch_child_nodes(metadata)
                all_metadata.extend(child_metadata)

    return all_metadata  # Return all the metadata collected





# Fetch child nodes if needed for detailed context
def fetch_child_nodes(parent_metadata):
    child_nodes = []
    subtree = parent_metadata.get('subtree', {})
    for level, clusters in subtree.items():
        for cluster_id, data in clusters.items():
            # Add all the chunks from the child nodes
            child_nodes.extend(data['chunks'])
    return child_nodes

# Decide whether to use child nodes (detailed context) or high-level summaries
def should_use_children(query):
    detailed_keywords = ["detail", "specific", "examples", "explain", "how", "why", "use", "show"]
    high_level_keywords = ["overview", "summary", "high-level", "general"]

    # Check for detailed keywords in the query
    for keyword in detailed_keywords:
        if keyword in query.lower():
            return True  # Use detailed context (child nodes)
    
    # Check for high-level keywords
    for keyword in high_level_keywords:
        if keyword in query.lower():
            return False  # Use high-level summary (node itself)

    # Use query length to infer intent (longer queries tend to be detailed)
    query_length = len(query.split())
    if query_length > 5:
        return True  # Longer queries are often asking for details
    
    return False  # Default to using high-level summaries

# Create the context from metadata (including summaries or child nodes)
def create_context_from_metadata(results):
    context_chunks = []
    context_metadata = []

    for hit in results:  # Each result is directly a hit, no need for nested looping
        # metadata_str = hit.get("metadata")  # Directly access metadata
        metadata_str = hit  # Directly access metadata

        # Ensure metadata exists
        if not metadata_str:
            print("No metadata found in hit.")  # Debugging
            continue

        # Convert metadata to dictionary if it is in string (JSON) format
        metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
        chunk = metadata.get("chunk", "")
        hierarchical_path = metadata.get("cluster_path", "")  # Get the hierarchical path
        summary = metadata.get("summary", "")  # Get the summary if it exists

        # If using the summary, add it; otherwise, append the chunk itself
        if summary:
            context_chunks.append(f"Summary: {summary}\nChunk: {chunk}")
        else:
            context_chunks.append(chunk)

        context_metadata.append(metadata)

    context = " ".join(context_chunks)
    return context, context_metadata


# GPT model to generate answer
def find_answer_gpt(question, context_metadata):
    context_info = json.dumps(context_metadata, indent=2)

    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant who reads reports."},
            {"role": "user", "content": f"""Please answer the following question by returning only a JSON response. 
            The JSON response should include the following fields: 'highlight', 'filename', 'page_number', and 'ans'. 
            Use this format:

{{
  "highlight": "[Exact text to be highlighted]",
  "filename": "[Name of the PDF file]",
  "page_number": [Page number(s)],
  "ans": "[Direct answer to the query]"
}}
Now, please answer the question: {question}

Context metadata: {context_info}"""}
        ],
        max_tokens=200,
        temperature=TEMPERATURE
    )
    
    raw_response = response.choices[0].message['content'].strip()

    save_interaction_to_json(question, raw_response, context_metadata)

    return raw_response

# Save interaction to JSON for logging
def save_interaction_to_json(question, answer, context_metadata):
    interaction_data = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "context_metadata": context_metadata
    }
    output_dir = "interactions"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a unique filename based on the current timestamp
    filename = f"interaction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Write the data to a JSON file
    with open(filepath, 'w') as json_file:
        json.dump(interaction_data, json_file, indent=2)
####_----------------------
from flask import Blueprint, request, jsonify
import io
import json
import openai
import re
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility, MilvusException
import PyPDF2
import numpy as np
from sklearn.mixture import GaussianMixture

upload_bp = Blueprint('upload_bp', __name__)

def connect_to_milvus():
    try:
        connections.connect("default", host="0.0.0.0", port="19530")
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=384),
            FieldSchema(name="pdf_name", dtype=DataType.VARCHAR, max_length=255),  # Scalar field for the filename
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        schema = CollectionSchema(fields, "index1")

        if not utility.has_collection("exmpcollection1"):
            collection = Collection("exmpcollection1", schema)
            index_params = {
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024},
                "metric_type": "COSINE"
            }
            collection.create_index(field_name="embeddings", index_params=index_params)
            collection.load()
            print("Collection 'exmpcollection1' created successfully.")
        else:
            collection = Collection("exmpcollection1")
            print("Collection 'exmpcollection1' loaded successfully.")
        
        return collection
    except MilvusException as e:
        print(f"Failed to connect to Milvus or create the collection: {e}")
        return None

collection = connect_to_milvus()

@upload_bp.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file:
        try:
            filename = file.filename.lower()  # Convert filename to lowercase
            
            # Ensure the collection is initialized
            if collection is None:
                return jsonify({"error": "Failed to initialize collection."}), 500

            pdf_data = io.BytesIO(file.read())
            page_texts, error = extract_text_by_page_with_metadata(pdf_data)
            if error:
                return jsonify({"error": error}), 500
            
            # Chunk the text and create embeddings
            chunks, chunk_page_numbers, patient_names = chunk_text_by_page(page_texts)
            embeddings = embed_chunks(chunks)
            
            # Perform bottom-up clustering and summarization (RAPTOR Indexing)
            hierarchical_structure = raptor_indexing(embeddings, chunks, depth=2)

            # Prepare metadata for each chunk, including the hierarchical path and summary
            metadata = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    "pdf_name": filename,
                    "chunk": chunk,
                    "page_number": chunk_page_numbers[i],
                    "patient_name": patient_names[i],
                    "chunk_index": i,
                    "hierarchical_level": 1,  # Dynamic updates for each level
                    "cluster_path": hierarchical_structure.get(f'level_1', {}).get(f'cluster_{i}', {}).get('path', None),  # Path
                    "summary": hierarchical_structure.get(f'level_1', {}).get(f'cluster_{i}', {}).get('summary', None)  # Summary
                }
                metadata.append(chunk_metadata)
            
            # Prepare data to insert into Milvus
            metadata_json = [json.dumps(m) for m in metadata]
            pdf_names = [filename] * len(embeddings)  # Ensure this matches the length of embeddings

            # Insert data into Milvus
            collection.insert([embeddings, pdf_names, metadata_json])
            collection.flush()

            return jsonify({"message": "File uploaded and processed successfully"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

# Extract text from PDF pages and detect metadata
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
        return str(e), None
    return page_texts, None

def extract_patient_name(text):
    name_match = re.search(r"Name:\s+([A-Za-z\s]+)", text)
    if name_match:
        return name_match.group(1).strip()
    return None

# Chunk text by page, keeping each chunk around 100 tokens
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

# Embed each chunk into vector representations
def embed_chunks(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, convert_to_tensor=True).cpu().numpy()
    return embeddings

# RAPTOR indexing: Perform bottom-up clustering and summarization
def raptor_indexing(embeddings, chunks, depth=2):
    return bottom_up_clustering(embeddings, chunks, depth)

def bottom_up_clustering(embeddings, chunks, depth=2, current_depth=1, cluster_path='cluster_0'):
    """
    Perform bottom-up clustering and summarization recursively, ensuring every chunk has a path and summary.
    """
    # Base case: if only one chunk or max depth is reached, return as a single cluster
    if current_depth > depth or len(chunks) < 2:
        # If there is only one chunk, it still gets a path and a default summary
        summary = summarize_text_gpt(' '.join(chunks)) if len(chunks) == 1 else None
        return {
            f'level_{current_depth}': {
                "chunks": chunks,
                "path": cluster_path,
                "summary": summary
            }
        }

    # Perform clustering using GMM
    gmm = GaussianMixture(n_components=2, covariance_type='tied')
    gmm.fit(embeddings)
    cluster_labels = gmm.predict(embeddings)

    clusters = {}
    for idx, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append((chunks[idx], idx))

    hierarchical_structure = {f'level_{current_depth}': {}}

    for cluster_id, cluster_data in clusters.items():
        cluster_chunks = [chunk for chunk, _ in cluster_data]
        combined_text = ' '.join([chunk for chunk, _ in cluster_data])
        
        # Summarize the cluster
        summary = summarize_text_gpt(combined_text)
        
        # Ensure every cluster has a unique path and summary
        new_cluster_path = f"{cluster_path}_sub_{cluster_id}"

        # Re-embed the summary and recursively apply clustering for deeper levels
        summary_embedding = embed_chunks([summary])[0]

        # Recursively cluster or finalize the path and summary for the current level
        subtree = bottom_up_clustering(
            [summary_embedding], 
            cluster_chunks,  # Pass the original chunks for further clustering
            depth, 
            current_depth + 1,
            cluster_path=new_cluster_path
        )
        
        # Add the cluster's summary, path, and subtree structure to the hierarchy
        hierarchical_structure[f'level_{current_depth}'][f'cluster_{cluster_id}'] = {
            'summary': summary,
            'chunks': cluster_chunks,  # Keep the original chunks
            'path': new_cluster_path,
            'subtree': subtree  # Link to the subtree, showing the path to inner chunks
        }

    return hierarchical_structure


# Use GPT for summarization of clusters
def summarize_text_gpt(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
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