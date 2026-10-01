from facenet_pytorch import MTCNN

def load_mtcnn(image_size=160, margin=0, keep_all=False):
    return MTCNN(image_size=image_size, margin=margin, keep_all=keep_all)

def detect_faces(img, mtcnn):
    """Returns boxes, probs — for visualization/sanity-checking (Day 1 style)."""
    boxes, probs = mtcnn.detect(img)
    return boxes, probs

def get_aligned_face(img, mtcnn):
    """Returns a cropped, aligned 160x160 face tensor ready for FaceNet (Day 2+ style)."""
    return mtcnn(img)
