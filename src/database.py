
import pickle
from pathlib import Path
import torch
from PIL import Image
from src.detector import get_aligned_face
import sys
sys.path.append("../")

def get_face_embedding_from_image(image_path, mtcnn, model):
    img = Image.open(image_path)
    face_tensor = get_aligned_face(img, mtcnn)
    if face_tensor is None:
        print(f"Warning: no face detected in {image_path}")
        return None
    from src.embedder import get_embedding
    return get_embedding(face_tensor, model)

def register_person(person_name, folder_path, mtcnn, model):
    embeddings = []
    for photo_path in Path(folder_path).glob("*"):
        emb = get_face_embedding_from_image(photo_path, mtcnn, model)
        if emb is not None:
            embeddings.append(emb)
    if not embeddings:
        print(f"Warning: no valid faces found for {person_name}")
        return None
    return torch.stack(embeddings).mean(dim=0)

def build_database(known_faces_dir, mtcnn, model):
    database = {}
    for person_folder in Path(known_faces_dir).iterdir():
        if person_folder.is_dir():
            emb = register_person(person_folder.name, person_folder, mtcnn, model)
            if emb is not None:
                database[person_folder.name] = emb
    return database

def save_database(database, path):
    with open(path, "wb") as f:
        pickle.dump(database, f)

def load_database(path):
    with open(path, "rb") as f:
        return pickle.load(f)