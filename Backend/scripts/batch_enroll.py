# batch_enroll.py
import os
import uuid
import cv2
import numpy as np
from PIL import Image
import pinecone
from face_embedder import FaceEmbedder
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_ENVIRONMENT

IMAGES_FOLDER = "images"


def init_pinecone():
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=512,  # FaceNet embedding dimension
            metric="cosine"
        )
    index = pc.Index(PINECONE_INDEX_NAME)
    return index


def get_existing_ids(index):
    """Get all existing IDs from Pinecone to avoid duplicates"""
    try:
        # Fetch all IDs by querying with a dummy vector and high top_k
        dummy_vector = [0.0] * 512  # FaceNet produces 512-dim vectors
        result = index.query(
            vector=dummy_vector,
            top_k=10000,  # Get as many as possible
            include_metadata=False
        )
        if hasattr(result, 'matches') and result.matches:
            return [match.id for match in result.matches]
    except Exception as e:
        print(f"Warning: Could not fetch existing IDs: {e}")
    return []


def generate_random_id():
    """Generate a unique random ID"""
    return str(uuid.uuid4())


def extract_name_from_filename(filename):
    """Extract display name from filename (remove extension)"""
    return os.path.splitext(filename)[0]


def process_single_image(image_path, embedder, existing_ids):
    """Process a single image and return embedding data if face detected"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image: {image_path}")
            return None
        
        # Use face embedder to detect face and get embedding
        embedding, box = embedder.frame_to_embedding(img)
        if embedding is None:
            print(f"No face detected in: {image_path}")
            return None
        
        # Generate ID and name
        person_id = generate_random_id()
        display_name = extract_name_from_filename(os.path.basename(image_path))
        
        return {
            "id": person_id,
            "values": embedding.tolist(),
            "metadata": {"name": display_name},
        }
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def batch_enroll():
    """Process all images in the images folder and enroll new ones"""
    print("Starting batch enrollment...")
    
    # Initialize
    index = init_pinecone()
    embedder = FaceEmbedder()
    
    # Get existing IDs to avoid duplicates
    print("Checking existing embeddings...")
    existing_ids = get_existing_ids(index)
    print(f"Found {len(existing_ids)} existing embeddings")
    
    # Check if images folder exists
    if not os.path.exists(IMAGES_FOLDER):
        print(f"Creating {IMAGES_FOLDER} folder...")
        os.makedirs(IMAGES_FOLDER)
        print(f"Please place images in the {IMAGES_FOLDER} folder and run again.")
        return
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    for filename in os.listdir(IMAGES_FOLDER):
        if os.path.splitext(filename)[1].lower() in image_extensions:
            image_files.append(filename)
    
    if not image_files:
        print(f"No images found in {IMAGES_FOLDER} folder.")
        return
    
    print(f"Found {len(image_files)} images to process...")
    
    # Process each image
    new_embeddings = []
    processed_count = 0
    skipped_count = 0
    
    for filename in image_files:
        image_path = os.path.join(IMAGES_FOLDER, filename)
        
        # Check if already processed (simple check by filename in metadata)
        # This is a basic approach - you could also maintain a separate tracking file
        already_processed = False
        try:
            result = index.query(
                vector=[0.0] * 512,
                top_k=1000,
                filter={"name": extract_name_from_filename(filename)},
                include_metadata=True
            )
            if hasattr(result, 'matches') and result.matches:
                already_processed = True
                print(f"Skipping {filename} - already processed")
                skipped_count += 1
        except:
            pass  # If filtering fails, assume not processed
        
        if not already_processed:
            embedding_data = process_single_image(image_path, embedder, existing_ids)
            if embedding_data:
                new_embeddings.append(embedding_data)
                processed_count += 1
                print(f"Processed: {filename} -> ID: {embedding_data['id']}")
    
    # Batch upsert to Pinecone
    if new_embeddings:
        print(f"\nUpserting {len(new_embeddings)} new embeddings...")
        try:
            index.upsert(vectors=new_embeddings)
            print("Batch enrollment completed successfully!")
            print(f"Processed: {processed_count} new images")
            print(f"Skipped: {skipped_count} already processed images")
        except Exception as e:
            print(f"Error upserting to Pinecone: {e}")
    else:
        print("No new images to process.")


if __name__ == "__main__":
    batch_enroll()