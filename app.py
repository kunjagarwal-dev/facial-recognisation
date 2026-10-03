"""
Face Attendance System — Streamlit app.
Live webcam recognition + attendance logging + new-person registration.
"""

from pathlib import Path

import cv2
import streamlit as st
from PIL import Image

from src.attendance import already_logged_today, log_attendance
from src.database import build_database, load_database, save_database
from src.detector import get_aligned_face, load_mtcnn
from src.embedder import get_embedding, load_facenet_model
from src.recognizer import recognize_face

st.set_page_config(page_title="Face Attendance", layout="wide")
st.title("Live Face Attendance")

PROJECT_DIR = Path(__file__).resolve().parent
KNOWN_FACES_DIR = PROJECT_DIR / "data" / "known_faces"
DB_PATH = PROJECT_DIR / "data" / "known_faces" / "db.pkl"


@st.cache_resource
def load_everything():
    mtcnn = load_mtcnn(keep_all=True)
    model = load_facenet_model()
    if DB_PATH.exists():
        database = load_database(DB_PATH)
    elif KNOWN_FACES_DIR.exists():
        database = build_database(KNOWN_FACES_DIR, mtcnn, model)
        save_database(database, DB_PATH)
    else:
        database = {}
    return mtcnn, model, database


mtcnn, model, database = load_everything()

if "checked_this_session" not in st.session_state:
    st.session_state.checked_this_session = set()

# ---------- sidebar: registration ----------
st.sidebar.header("Register New Person")
new_name = st.sidebar.text_input("Name")
register_clicked = st.sidebar.button("Capture & Register")
st.sidebar.caption("Click multiple times from different angles for a more robust registration.")

if register_clicked:
    if not new_name.strip():
        st.sidebar.error("Enter a name first.")
    else:
        reg_cap = cv2.VideoCapture(0)
        ret, frame = reg_cap.read()
        reg_cap.release()

        if not ret:
            st.sidebar.error("Could not capture from webcam.")
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            face_tensor = get_aligned_face(img, mtcnn)

            if face_tensor is None:
                st.sidebar.error("No face detected — try again.")
            else:
                save_dir = KNOWN_FACES_DIR / new_name.strip()
                save_dir.mkdir(parents=True, exist_ok=True)
                existing_count = len(list(save_dir.glob("*.jpg")))
                img.save(save_dir / f"{new_name.strip()}_{existing_count + 1}.jpg")

                st.sidebar.success(f"Captured photo {existing_count + 1} for {new_name.strip()}")

                new_database = build_database(KNOWN_FACES_DIR, mtcnn, model)
                save_database(new_database, DB_PATH)

                st.cache_resource.clear()
                st.sidebar.success("Database updated — reloading...")
                st.rerun()

st.sidebar.divider()
st.sidebar.write(f"**Registered people:** {len(database)}")
if not database:
    st.sidebar.warning("No face profiles are loaded. Register a person before starting recognition.")
if database:
    st.sidebar.write(", ".join(database.keys()))

# ---------- main: live recognition ----------
run = st.checkbox("Start camera")
frame_placeholder = st.empty()
status_placeholder = st.empty()

if run:
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Webcam not available.")
            break

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

                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{name} ({distance:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    if name != "Unknown" and name not in st.session_state.checked_this_session:
                        if not already_logged_today(name):
                            log_attendance(name)
                            status_placeholder.success(f"Logged: {name}")
                        st.session_state.checked_this_session.add(name)

        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(display_frame, channels="RGB")

    cap.release()
else:
    frame_placeholder.info("Check the box above to start the camera.")