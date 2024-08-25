import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone
api_key = os.getenv("PINECONE_API_KEY")
if api_key is None:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

pc = Pinecone(api_key=api_key)

# Define index name and dimension
index_name = 'testingggg'
dimension = 384

# Delete the existing index if it exists
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)

# Create a new index with the correct dimension
pc.create_index(
    name=index_name,
    dimension=dimension,
    metric='cosine',
    spec=ServerlessSpec(
        cloud='aws',
        region='us-east-1'
    )
)

# Connect to the new index
index = pc.Index(index_name)

# Load the model and generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = ["Example text 1", "Example text 2"]
embeddings = model.encode(texts).tolist()  # Convert to list for Pinecone

# Insert vectors into the index
index.upsert(
    vectors=[
        {"id": f"vec{i}", "values": embedding}
        for i, embedding in enumerate(embeddings)
    ]
)

print("Vectors inserted successfully!")
