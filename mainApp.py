
# @app.route("/upload", methods=["POST"])
# def upload_pdf():


# @app.route("/query", methods=["POST"])
# def query_pdf():

from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
import os
from upload import upload_pdf
from query import query_pdf

app = Flask(__name__)

# Ensure the data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

# API 1: File Operations
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        file_path = upload_pdf(file) #Calling our main function from upload.py
        return jsonify({"file_path": file_path, "message": "File uploaded successfully"})
    except Exception as e:
        return jsonify({"error": f"File upload failed: {str(e)}"}), 500

@app.route('/files/<file_id>', methods=['GET'])
def retrieve_file(file_id):
    file_path = get_file(file_id)
    if file_path:
        return send_file(file_path)
    else:
        return jsonify({"error": "File not found"}), 404

# API 2: Query Handling
@app.route('/query', methods=['POST'])
def query_handler():
    input_data = request.json
    if not input_data:
        return jsonify({"error": "Invalid input"}), 400

    try:
        response = process_query(input_data)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": f"Query processing failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)