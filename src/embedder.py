from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from pathlib import Path

def load_facenet_model():
    return InceptionResnetV1(pretrained="vggface2").eval()

def get_embedding(face_tensor, model):
    if face_tensor.dim() == 3:
        face_tensor = face_tensor.unsqueeze(0)
    return model(face_tensor)


