from datetime import datetime
from pymilvus import connections, Collection, utility, MilvusException
from flask import Blueprint, request, jsonify
import json
import os
import openai
from sentence_transformers import SentenceTransformer

query_bp = Blueprint('query_bp', __name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("LLM_MODEL")
TEMPERATURE = 0.5

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

        query_embedding = embed_chunks([query])[0]
        results = retrieve_documents(query_embedding, filenames=filenames)  # Pass filename to retrieval function
        if not results:
            return jsonify({"error": "No results found"}), 404

        context, context_metadata = create_context_from_metadata(results)
        if not context_metadata:
            return jsonify({"error": "No context metadata found"}), 404

        answer = find_answer_gpt(query, context_metadata)
        return answer
    
    except MilvusException as e:
        return jsonify(f"Milvus Error: {e}")
    except KeyError as e:
        return jsonify({"error": f"Missing key: {e}"}), 400  # Bad Request
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def embed_chunks(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, convert_to_tensor=True).cpu().numpy()
    return embeddings

def retrieve_documents(query_embedding, filenames=None, top_k=10):
    if not collection:
        return []

    search_params = {"metric_type": "COSINE", "params": {"ef": 200}}
    
    # Build the expression to filter by filename if provided
    # expr = f"pdf_name == '{filename}'" if filename else None
   # Build the expression to filter by multiple filenames if provided
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
        expr=expr,  # Use the expression to filter by filename
        output_fields=["metadata"]
    )
    return results

def create_context_from_metadata(results):
    context_chunks = []
    context_metadata = []

    for result in results:
        for hit in result:
            metadata_str = hit.entity.get("metadata")
            metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
            chunk = metadata.get("chunk", "")
            context_chunks.append(chunk)
            context_metadata.append(metadata)

    context = " ".join(context_chunks)
    return context, context_metadata

def find_answer_gpt(question, context_metadata):
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
You are given 10 metadata containing fields like chunk and patient name. Answer the question by correctly matching the name in query in metadata only.
Ensure the JSON response is valid and all fields are correctly formatted. Now, please answer the question: {question}

Context metadata: {context_info}"""}
        ],
        max_tokens=200,
        temperature=TEMPERATURE
    )
    
    raw_response = response.choices[0].message['content'].strip()

    # Save the input, output, and context metadata to a JSON file
    save_interaction_to_json(question, raw_response, context_metadata)

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
    
    # Generate a unique filename based on the current timestamp
    filename = f"interaction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Write the data to a JSON file
    with open(filepath, 'w') as json_file:
        json.dump(interaction_data, json_file, indent=2)
