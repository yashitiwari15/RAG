from pymilvus import utility, connections

def connect_to_milvus():
    try:
        # Connect to Milvus
        connections.connect("default", host="0.0.0.0", port="19530")
        print("Connected to Milvus successfully.")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        return None

def drop_collection(collection_name):
    try:
        # Check if the collection exists in Milvus
        if utility.has_collection(collection_name):
            # Drop the collection
            utility.drop_collection(collection_name)
            print(f"Collection '{collection_name}' dropped successfully.")
        else:
            print(f"Collection '{collection_name}' does not exist.")
    except Exception as e:
        print(f"Failed to drop collection '{collection_name}': {e}")

if __name__ == "__main__":
    # Connect to Milvus
    connect_to_milvus()

    # Specify the collection name you want to drop
    collection_name = "exmpcollection1"

    # Drop the collection
    drop_collection(collection_name)
