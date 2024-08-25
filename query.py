# from datetime import datetime
# from pymilvus import connections, Collection, utility, MilvusException
# from flask import Blueprint, request, jsonify
# import json
# import os
# import openai
# from sentence_transformers import SentenceTransformer

# query_bp = Blueprint('query_bp', __name__)

# openai.api_key = os.getenv("OPENAI_API_KEY")
# MODEL_NAME = os.getenv("LLM_MODEL")
# TEMPERATURE = 0.5

# def connect_to_milvus():
#     connections.connect("default", host="0.0.0.0", port="19530")
#     if utility.has_collection("exmpcollection1"):
#         return Collection("exmpcollection1")
#     return None

# collection = connect_to_milvus()

# if collection is None:
#      raise Exception("Failed to connect to Milvus. Exiting...", 400)

# @query_bp.route("/query", methods=["POST"])
# def query_pdf():
#     try:
#         data = request.get_json()
#         query = data.get("query", "")
#         filenames = data.get("filenames", None)  # Get the filename from the request data

#         query_embedding = embed_chunks([query])[0]
#         results = retrieve_documents(query_embedding, filenames=filenames)  # Pass filename to retrieval function
#         if not results:
#             return jsonify({"error": "No results found"}), 404

#         context, context_metadata = create_context_from_metadata(results)
#         if not context_metadata:
#             return jsonify({"error": "No context metadata found"}), 404
#         print(context_metadata)
#         answer = find_answer_gpt(query, context_metadata)
#         return answer
    
#     except MilvusException as e:
#         return jsonify(f"Milvus Error: {e}")
#     except KeyError as e:
#         return jsonify({"error": f"Missing key: {e}"}), 400  # Bad Request
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# def embed_chunks(chunks):
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     embeddings = model.encode(chunks, convert_to_tensor=True).cpu().numpy()
#     return embeddings

# def retrieve_documents(query_embedding, filenames=None, top_k=10):
#     if not collection:
#         return []

#     search_params = {"metric_type": "COSINE", "params": {"ef": 200}}
    
#     # Build the expression to filter by filename if provided
#     # expr = f"pdf_name == '{filename}'" if filename else None
#    # Build the expression to filter by multiple filenames if provided
#     if filenames:
#         filenames_expr = f"pdf_name in [{', '.join([f'\"{filename}\"' for filename in filenames])}]"
#         expr = filenames_expr
#     else:
#         expr = None


#     results = collection.search(
#         data=[query_embedding],
#         anns_field="embeddings",
#         param=search_params,
#         limit=top_k,
#         expr=expr,  # Use the expression to filter by filename
#         output_fields=["metadata"]
#     )
#     return results

# def create_context_from_metadata(results):
#     context_chunks = []
#     context_metadata = []

#     for result in results:
#         for hit in result:
#             metadata_str = hit.entity.get("metadata")
#             metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
#             chunk = metadata.get("chunk", "")
#             context_chunks.append(chunk)
#             context_metadata.append(metadata)

#     context = " ".join(context_chunks)
#     return context, context_metadata

# def find_answer_gpt(question, context_metadata):
#     context_info = json.dumps(context_metadata, indent=2)

#     response = openai.ChatCompletion.create(
#         model=MODEL_NAME,
#         messages=[
#             {"role": "system", "content": "You are a helpful medical assistant who reads reports."},
#             {"role": "user",
#                 "content": f"""Please answer the following question by returning only a JSON response. The JSON response should include the following fields: 'highlight', 'filename', 'page_number', and 'ans'. Use this format:

# {{
# "highlight": "[Exact text to be highlighted]",
# "filename": "[Name of the PDF file]",
# "page_number": [Page number(s)],
# "ans": "[Direct answer to the query]"
# }}
# You are given 10 metadata containing fields like chunk and patient name. Answer the question by correctly matching the name in query in metadata only.
# Ensure the JSON response is valid and all fields are correctly formatted. Now, please answer the question: {question}

# Context metadata: {context_info}"""}
#         ],
#         max_tokens=200,
#         temperature=TEMPERATURE
#     )
    
#     raw_response = response.choices[0].message['content'].strip()

#     # Save the input, output, and context metadata to a JSON file
#     save_interaction_to_json(question, raw_response, context_metadata)

#     return raw_response

# def save_interaction_to_json(question, answer, context_metadata):
#     interaction_data = {
#         "timestamp": datetime.now().isoformat(),
#         "question": question,
#         "answer": answer,
#         "context_metadata": context_metadata
#     }
#     output_dir = "interactions"
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Generate a unique filename based on the current timestamp
#     filename = f"interaction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
#     filepath = os.path.join(output_dir, filename)
    
#     # Write the data to a JSON file
#     with open(filepath, 'w') as json_file:
#         json.dump(interaction_data, json_file, indent=2)

from datetime import datetime
from pymilvus import connections, Collection, utility, MilvusException
from flask import Blueprint, request, jsonify
import json
import os
import openai
from sentence_transformers import SentenceTransformer, CrossEncoder
import nltk
from nltk.tokenize import word_tokenize

# Ensure you have downloaded necessary NLTK data
nltk.download('punkt')

query_bp = Blueprint('query_bp', __name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("LLM_MODEL")
TEMPERATURE = 0.5

# Initialize the Cross-Encoder model for re-ranking
cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def connect_to_milvus():
    connections.connect("default", host="0.0.0.0", port="19530")
    if utility.has_collection("exmpcollection1"):
        return Collection("exmpcollection1")
    return None

collection = connect_to_milvus()

if collection is None:
    raise Exception("Failed to connect to Milvus. Exiting...", 400)

@query_bp.route("/query", methods=["POST"])
def query_pdf():
    try:
        data = request.get_json()
        query = data.get("query", "")
        filenames = data.get("filenames", None)  # Get the filename from the request data
        
        if not query:
            return jsonify({"error": "Query parameter is missing"}), 400  # Bad Request

        query_embedding = embed_chunks([query])[0]

        # Initial retrieval from Milvus (BM25 or vector-based search)
        results = retrieve_documents(query_embedding, query, filenames=filenames)
        if not results:
            return jsonify({"error": "No results found"}), 404

        # Re-rank using Cross-Encoder
        reranked_results = cross_encoder_rerank(query, results)

        context, context_metadata = create_context_from_metadata(reranked_results)
        if not context_metadata:
            return jsonify({"error": "No context metadata found"}), 404

        answer = find_answer_gpt(query, context_metadata)
        return answer

    except MilvusException as e:
        return jsonify(f"Milvus Error: {e}")
    except KeyError as e:
        return jsonify({"error": f"Missing key: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def embed_chunks(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, convert_to_tensor=True).cpu().numpy()
    return embeddings

def retrieve_documents(query_embedding, query, filenames=None, top_k=15):
    if not collection:
        return []

    search_params = {"metric_type": "COSINE", "params": {"ef": 200}}

    if filenames:
        filenames_expr = f"pdf_name in [{', '.join([f'\"{filename}\"' for filename in filenames])}]"
        expr = filenames_expr
    else:
        expr = None

    results = collection.search(
        data=[query_embedding],
        anns_field="embeddings",
        param=search_params,
        limit=top_k,
        expr=expr,
        output_fields=["metadata"]
    )

    # Get the chunks and their metadata from the results
    retrieved_chunks = []
    for result in results:
        for hit in result:
            metadata_str = hit.entity.get("metadata")
            metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
            chunk = metadata.get("chunk", "")
            retrieved_chunks.append((chunk, metadata))  # Store both chunk and metadata

    return retrieved_chunks

def cross_encoder_rerank(query, retrieved_chunks):
    """
    Re-ranks the retrieved documents using a Cross-Encoder model.
    :param query: The query string.
    :param retrieved_chunks: List of (chunk, metadata) tuples.
    :return: List of re-ranked (chunk, metadata) tuples.
    """
    print(retrieved_chunks)
    # Prepare inputs for Cross-Encoder
    cross_encoder_input = [(query, chunk) for chunk, _ in retrieved_chunks]

    # Get relevance scores from the Cross-Encoder model
    scores = cross_encoder_model.predict(cross_encoder_input)

    # Combine chunks, metadata, and scores
    chunks_with_scores = [(chunk, metadata, score) for (chunk, metadata), score in zip(retrieved_chunks, scores)]

    # Sort by score in descending order
    ranked_chunks = sorted(chunks_with_scores, key=lambda x: x[2], reverse=True)

    # Return the top 5 ranked chunks with their metadata
    return [(chunk, metadata) for chunk, metadata, _ in ranked_chunks[:5]]

def create_context_from_metadata(reranked_results):
    context_chunks = []
    context_metadata = []

    for chunk, metadata in reranked_results:
        context_chunks.append(chunk)
        context_metadata.append(metadata)

    context = " ".join(context_chunks)
    return context, context_metadata

# def find_answer_gpt(question, context_metadata):
#     context_info = json.dumps(context_metadata, indent=2)

#     response = openai.ChatCompletion.create(
#         model=MODEL_NAME,
#         messages=[
#             {"role": "system", "content": "You are a helpful medical assistant who reads reports."},
#             {"role": "user",
#                 "content": f"""Please answer the following question by returning only a JSON response. The JSON response should include the following fields: 'highlight', 'filename', 'page_number', and 'ans'. Use this format:

# {{
# "highlight": "[Need exact text to be highlighted from original_text field in context_metadata. Do not use summary_text field of context_metadata.
# If there are multiple items in context_metadata from where the ans is created then return all of them in different arrays]",
# "filename": "[Name of the PDF file.If query is from multiple pdfs return all the names of pdfs the answer is being generated]",
# "page_number": [Page number(s).Make sure to mention all the page numbers from where the ans in being made from each file if using multiple files],
# "ans": "[Give a natural lanuage responce which should be good to read]"
# }}
# Answer the question: {question}

# Context metadata: {context_info}"""}
#         ],
#         max_tokens=1000,
#         temperature=TEMPERATURE
#     )

#     raw_response = response.choices[0].message['content'].strip()

#     save_interaction_to_json(question, raw_response, context_metadata)

#     return raw_response

def find_answer_gpt(question, context_metadata):
    # Convert context metadata to JSON format
    context_info = json.dumps(context_metadata, indent=2)

    # Call the OpenAI API for LLM assistance
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant who reads reports."},
            {"role": "user",
                "content": f"""Please answer the following question by returning only a JSON response. The JSON response should include the following fields: 'highlight', 'filename', 'page_number', and 'ans'. Use this format:

{{
"highlight": [
    {{
      "text": "[Need exact text to be highlighted from original_text field in context_metadata. Do not use summary_text field of context_metadata. Each highlight must correspond to the part of the text from the original_text.]",
      "page_number": [Page number of the highlighted text],
      "filename": "[PDF file name where this highlighted text is located]"
    }},
    [Repeat for each highlight from different files and page numbers]
],
"ans": "[Give a natural language response summarizing the highlighted sections]"
}}
Answer the question: {question}

Context metadata: {context_info}"""}
        ],
        max_tokens=1000,
        temperature=TEMPERATURE
    )

    # Extract and clean the raw response
    raw_response = response.choices[0].message['content'].strip()

    # Save the interaction for future reference
    save_interaction_to_json(question, raw_response, context_metadata)

    # Return the processed raw response
    return raw_response

def save_interaction_to_json(question, answer, context_metadata):
    interaction_data = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "context_metadata": context_metadata
    }
    output_dir = "interactions"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"interaction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as json_file:
        json.dump(interaction_data, json_file, indent=2)
