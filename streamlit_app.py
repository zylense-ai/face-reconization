import streamlit as st
import cv2
import numpy as np
import pandas as pd
import face_recognition
import os
import csv
from PIL import Image
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="ZyLense.ai Face Recognition",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #A23B72;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .recognition-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .recognized-face {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        color: white;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .unknown-face {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        color: white;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .stats-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 30px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Configuration
CSV_PATH = "attendance/registered_guardians.csv"
UNKNOWN_LABEL = "Unknown"
ACCURACY_THRESHOLD = 0.6

# Initialize session state
if 'face_database_loaded' not in st.session_state:
    st.session_state.face_database_loaded = False
if 'known_encodings' not in st.session_state:
    st.session_state.known_encodings = []
if 'known_names' not in st.session_state:
    st.session_state.known_names = []
if 'recognition_active' not in st.session_state:
    st.session_state.recognition_active = False
if 'recognition_results' not in st.session_state:
    st.session_state.recognition_results = []
if 'session_faces' not in st.session_state:
    st.session_state.session_faces = set()

def load_face_database():
    """
    Load face encodings and names from CSV file
    Returns tuple (known_encodings, known_names)
    """
    known_encodings = []
    known_names = []
    
    if not os.path.exists(CSV_PATH):
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        with open(CSV_PATH, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=['Name', 'ImagePath'])
            writer.writeheader()
        return known_encodings, known_names
    
    try:
        df = pd.read_csv(CSV_PATH)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (_, row) in enumerate(df.iterrows()):
            progress = (idx + 1) / len(df)
            progress_bar.progress(progress)
            status_text.text(f"Loading face database... {idx + 1}/{len(df)}")
            
            if not os.path.exists(row['ImagePath']):
                st.warning(f"Image not found: {row['ImagePath']}")
                continue
                
            image = face_recognition.load_image_file(row['ImagePath'])
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(row['Name'])
            else:
                st.warning(f"No face detected in {row['ImagePath']}")
        
        progress_bar.empty()
        status_text.empty()
        st.success(f"Database loaded: {len(known_names)} known faces")
        return known_encodings, known_names
        
    except Exception as e:
        st.error(f"Error loading database: {str(e)}")
        return [], []

def recognize_faces_from_frame(frame, known_encodings, known_names):
    """Process a single frame for face recognition"""
    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings_current_frame = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=1)
    
    face_names = []
    
    for current_face_encoding in face_encodings_current_frame:
        name = UNKNOWN_LABEL
        
        # Compare with known faces
        if known_encodings:
            matches = face_recognition.compare_faces(
                known_encodings, 
                current_face_encoding, 
                tolerance=ACCURACY_THRESHOLD
            )
            
            if True in matches:
                # Calculate face distances
                face_distances = face_recognition.face_distance(
                    known_encodings, 
                    current_face_encoding
                )
                best_match_idx = np.argmin(face_distances)
                
                if matches[best_match_idx]:
                    name = known_names[best_match_idx]
        
        face_names.append(name)
    
    # Scale locations back to original frame size
    scaled_face_locations = []
    for (top, right, bottom, left) in face_locations:
        scaled_face_locations.append((
            top * 4, 
            right * 4, 
            bottom * 4, 
            left * 4
        ))
    
    return scaled_face_locations, face_names

def draw_face_boxes(frame, face_locations, face_names):
    """Draw bounding boxes and labels on the frame"""
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Choose color based on recognition
        color = (0, 255, 0) if name != UNKNOWN_LABEL else (0, 0, 255)
        
        # Draw rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
    
    return frame

# Main App Layout
st.markdown('<h1 class="main-header">Guardian Face Recognition</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced AI-Powered Face Recognition System</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Control Panel")
    
    # Database Management
    st.markdown("#### Database Management")
    
    if st.button("Load Face Database", key="load_db"):
        with st.spinner("Loading face database..."):
            st.session_state.known_encodings, st.session_state.known_names = load_face_database()
            st.session_state.face_database_loaded = True
    
    # Database Stats
    if st.session_state.face_database_loaded:
        st.markdown(f"**Database Status:** Loaded")
        st.markdown(f"**Known Faces:** {len(st.session_state.known_names)}")
        
        if st.session_state.known_names:
            st.markdown("**Registered Names:**")
            for name in st.session_state.known_names:
                st.markdown(f"• {name}")
    else:
        st.markdown("**Database Status:** Not Loaded")
    
    # Recognition Settings
    st.markdown("#### Recognition Settings")
    accuracy_threshold = st.slider(
        "Recognition Accuracy Threshold",
        min_value=0.3,
        max_value=1.0,
        value=0.6,
        step=0.1,
        help="Lower values = more lenient, Higher values = stricter"
    )
    
    # Update global threshold
    ACCURACY_THRESHOLD = accuracy_threshold

# Main Content Area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Live Camera Feed")
    
    # Camera controls
    camera_col1, camera_col2 = st.columns(2)
    
    with camera_col1:
        start_camera = st.button("Start Camera", key="start_cam")
    
    with camera_col2:
        stop_camera = st.button("Stop Camera", key="stop_cam")
    
    # Video feed placeholder
    video_placeholder = st.empty()
    
    # Handle camera controls
    if start_camera and st.session_state.face_database_loaded:
        st.session_state.recognition_active = True
        st.session_state.recognition_results = []  # Reset results
        st.session_state.session_faces = set()  # Reset session faces
    elif start_camera and not st.session_state.face_database_loaded:
        st.error("Please load the face database first!")
    
    if stop_camera:
        st.session_state.recognition_active = False

with col2:
    st.markdown("### Recognition Results")
    
    # Show results only after camera is stopped
    if not st.session_state.recognition_active and st.session_state.recognition_results:
        st.markdown('<div class="recognition-box">', unsafe_allow_html=True)
        st.markdown("**Session Summary:**")
        
        # Show unique faces detected in session
        if st.session_state.session_faces:
            recognized_faces = [face for face in st.session_state.session_faces if face != UNKNOWN_LABEL]
            unknown_count = len([face for face in st.session_state.session_faces if face == UNKNOWN_LABEL])
            
            if recognized_faces:
                st.markdown("**Recognized Faces:**")
                for face in recognized_faces:
                    st.markdown(f'<div class="recognized-face">{face}</div>', unsafe_allow_html=True)
            
            if unknown_count > 0:
                st.markdown(f'<div class="unknown-face">Unknown faces detected: {unknown_count}</div>', unsafe_allow_html=True)
        else:
            st.markdown("No faces detected in this session")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Session statistics
        st.markdown('<div class="stats-container">', unsafe_allow_html=True)
        st.markdown("**Session Statistics:**")
        st.markdown(f"**Total Detections:** {len(st.session_state.recognition_results)}")
        st.markdown(f"**Unique Faces:** {len(st.session_state.session_faces)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif st.session_state.recognition_active:
        st.info("Camera is active. Results will be shown after stopping the camera.")
    
    else:
        st.info("Start camera to begin face recognition.")

# Camera processing
if st.session_state.recognition_active:
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Error: Could not open camera")
        st.session_state.recognition_active = False
    else:
        # Process frames
        frame_placeholder = video_placeholder.empty()
        
        while st.session_state.recognition_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Error reading frame")
                break
            
            # Perform face recognition
            face_locations, face_names = recognize_faces_from_frame(
                frame, 
                st.session_state.known_encodings, 
                st.session_state.known_names
            )
            
            # Draw boxes on frame
            frame_with_boxes = draw_face_boxes(frame, face_locations, face_names)
            
            # Convert frame to RGB for display
            frame_rgb = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
            
            # Display frame
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Store recognition results
            if face_names:
                st.session_state.recognition_results.extend(face_names)
                for name in face_names:
                    st.session_state.session_faces.add(name)
            
            # Small delay to prevent excessive processing
            time.sleep(0.1)
        
        # Release camera
        cap.release()
        cv2.destroyAllWindows()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Powered by <strong>ZyLense.ai</strong></p>
    </div>
    """,
    unsafe_allow_html=True
)