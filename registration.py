from src.detector import load_mtcnn, get_aligned_face
from src.embedder import load_facenet_model, get_embedding
from src.database import load_database
from src.recognizer import recognize_face
from PIL import Image

mtcnn = load_mtcnn()
model = load_facenet_model()
database = load_database("data/known_faces/db.pkl")

test_img = Image.open("data/test_images/dhoni2.jpg")
face_tensor = get_aligned_face(test_img, mtcnn)
embedding = get_embedding(face_tensor, model)

name, distance = recognize_face(embedding, database)
print(f"Recognized: {name} (distance: {distance:.3f})")