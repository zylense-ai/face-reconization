import cv2
import numpy as np
import pandas as pd
import face_recognition
import os
import csv

# Configuration
CSV_PATH = "attendance/registered_guardians.csv"
UNKNOWN_LABEL = "Unknown"
ACCURACY_THRESHOLD = 0.6  # Higher value = stricter recognition

def load_face_database():
    """
    Load face encodings and names from CSV file
    Returns tuple (known_encodings, known_names)
    """
    known_encodings = []
    known_names = []
    
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=['Name', 'ImagePath'])
            writer.writeheader()
        return known_encodings, known_names
    
    try:
        df = pd.read_csv(CSV_PATH)
        for _, row in df.iterrows():
            if not os.path.exists(row['ImagePath']):
                print(f"Image not found: {row['ImagePath']}")
                continue
                
            image = face_recognition.load_image_file(row['ImagePath'])
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(row['Name'])
            else:
                print(f"No face detected in {row['ImagePath']}")
                
        print(f"Database loaded: {len(known_names)} known faces")
        return known_encodings, known_names
        
    except Exception as e:
        print(f"Error loading database: {str(e)}")
        return [], []

def recognize_faces():
    # Initialize video capture
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open camera")
        return

    # Load known faces
    known_encodings, known_names = load_face_database()
    
    # Initialize variables
    process_frame = True
    
    print("Starting face recognition. Press 'q' to exit...")
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error reading frame")
            continue
            
        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Option A: safest
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        
        # Initialize these lists outside the `if process_frame` block
        # so they are available for display even if processing is skipped
        face_locations = []
        face_names = []
        scaled_face_locations = []
        
        # Process every other frame
        if process_frame:
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_frame)
            
            # Ensure face_locations is a list of tuples for face_encodings
            # This is the crucial fix!
            face_encodings_current_frame = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=1)
            
            for i, current_face_encoding in enumerate(face_encodings_current_frame):
                name = UNKNOWN_LABEL
                
                # Compare with known faces
                # Ensure known_encodings is not empty before comparing
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
            for (top, right, bottom, left) in face_locations:
                scaled_face_locations.append((
                    top * 4, 
                    right * 4, 
                    bottom * 4, 
                    left * 4
                ))
        
        process_frame = not process_frame  # Toggle frame processing
        
        # Display results
        for (top, right, bottom, left), name in zip(scaled_face_locations, face_names):
            # Draw rectangle and label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        
        cv2.imshow('Face Recognition', frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    video_capture.release()
    cv2.destroyAllWindows()
    print("System stopped")

if __name__ == "__main__":
    recognize_faces()