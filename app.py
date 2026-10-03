import cv2
from PIL import Image
from src.detector import load_mtcnn, get_aligned_face
from src.embedder import load_facenet_model, get_embedding
from src.database import load_database
from src.recognizer import recognize_face

mtcnn = load_mtcnn(keep_all=True)   # detect multiple faces per frame
model = load_facenet_model()
database = load_database("data/known_faces/db.pkl")

cap = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # OpenCV uses BGR, MTCNN/PIL expect RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame)

    detection = mtcnn.detect(img)
    boxes = detection[0]

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            face_crop = img.crop((x1, y1, x2, y2))

            face_tensor = get_aligned_face(face_crop, mtcnn)
            if face_tensor is not None:
                embedding = get_embedding(face_tensor, model)
                name, distance = recognize_face(embedding, database)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({distance:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Face Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()