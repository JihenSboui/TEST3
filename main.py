import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os
import joblib
import pandas as pd
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# 🔐 Charger la clé d'identification
cred = credentials.Certificate("firebase_key.json")  # chemin vers le fichier téléchargé
firebase_admin.initialize_app(cred)

# 🔌 Connexion à Firestore
db = firestore.client()

#  Charger le modèle de risque
model = joblib.load('modele_risque_rula_compat.pkl')

#Configuration des risques
RISK_COLORS = {
    1: (0, 255, 0),   # Vert
    3: (0, 255, 255), # Jaune 
    5: (0, 0, 255)    # Rouge
}

RISK_LABELS = {
    1: "Faible",
    3: "Modere",
    5: "Eleve"
}

ANGLE_COLS = [
    "Right_Shoulder_Angle", "Left_Shoulder_Angle",
    "Right_Elbow_Angle", "Left_Elbow_Angle",
    "Right_Wrist_Rotation", "Left_Wrist_Rotation",
    "Torso_Angle", "Neck_Angle"
]
def prepare_input_data(angles):
    """Prépare les données pour le modèle ML"""
    return pd.DataFrame([[
        angles['Right_Shoulder_Angle'],
        angles['Left_Shoulder_Angle'],
        angles['Right_Elbow_Angle'],
        angles['Left_Elbow_Angle'],
        angles['Right_Wrist_Rotation'],
        angles['Left_Wrist_Rotation'],
        angles['Torso_Angle'],
        angles['Neck_Angle']
    ]], columns=ANGLE_COLS)

def display_risk_assessment(image, predictions):
    """Affiche les résultats de risque"""
    y_position = 40
    parts = [
        "Epaule D", "Epaule G",
        "Coude D", "Coude G",
        "Poignet D", "Poignet G",
        "Torse", "Cou"
    ]
    
    for part, prediction in zip(parts, predictions):
        color = RISK_COLORS.get(prediction, (255, 255, 255))
        text = f"{part}: {RISK_LABELS.get(prediction, 'N/A')}"
        cv2.putText(image, text, (image.shape[1] - 250, y_position),  # A droite
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_position += 30



#  Initialisation de MediaPipe BlazePose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def calculate_neck_angle(landmarks, img_w, img_h):
    # Get required landmarks
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    # Create midpoints
    shoulder_mid = np.array([(left_shoulder.x + right_shoulder.x)/2,
                            (left_shoulder.y + right_shoulder.y)/2,
                            (left_shoulder.z + right_shoulder.z)/2])
   
    hip_mid = np.array([(left_hip.x + right_hip.x)/2,
                       (left_hip.y + right_hip.y)/2,
                       (left_hip.z + right_hip.z)/2])
   
    # Create spine vector (from hip to shoulder)
    spine_vector = np.array([shoulder_mid[0] - hip_mid[0],
                            shoulder_mid[1] - hip_mid[1],
                            shoulder_mid[2] - hip_mid[2]])
   
    # Create neck vector (from shoulder to nose)
    neck_vector = np.array([nose.x - shoulder_mid[0],
                           nose.y - shoulder_mid[1],
                           nose.z - shoulder_mid[2]])
   
    # Create vertical reference vector (pointing up)
    # In MediaPipe, y increases downward, so our vertical vector is negative in y
    vertical_vector = np.array([0, -1, 0])
   
    # Calculate angle between spine and vertical (to account for torso inclination)
    spine_vertical_dot = np.dot(spine_vector, vertical_vector)
    spine_norm = np.linalg.norm(spine_vector)
    vertical_norm = np.linalg.norm(vertical_vector)
    spine_angle_rad = np.arccos(np.clip(spine_vertical_dot / (spine_norm * vertical_norm), -1.0, 1.0))
   
    # Calculate angle between neck and vertical
    neck_vertical_dot = np.dot(neck_vector, vertical_vector)
    neck_norm = np.linalg.norm(neck_vector)
    neck_angle_rad = np.arccos(np.clip(neck_vertical_dot / (neck_norm * vertical_norm), -1.0, 1.0))
   
    # Convert to degrees
    spine_angle = np.degrees(spine_angle_rad)
    neck_angle = np.degrees(neck_angle_rad)
   
    # Calculate neck flexion relative to torso
    # If neck is more forward than spine, we have flexion
    neck_flexion = max(0, neck_angle - spine_angle)
   
    # Get pixel coordinates for visualization
    nose_px = (int(nose.x * img_w), int(nose.y * img_h))
    shoulder_mid_px = (int(shoulder_mid[0] * img_w), int(shoulder_mid[1] * img_h))
    hip_mid_px = (int(hip_mid[0] * img_w), int(hip_mid[1] * img_h))
   
    return neck_flexion, nose_px, shoulder_mid_px, hip_mid_px

def calculate_3d_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    # Vecteurs BA et BC
    ba = a - b
    bc = c - b
    # Produit scalaire pour trouver l'angle
    dot_product = np.dot(ba, bc)
    norm_product = np.linalg.norm(ba) * np.linalg.norm(bc)
    angle_rad = np.arccos(dot_product / norm_product)
    return np.degrees(angle_rad)

def calculate_wrist_rotation(landmarks, side="RIGHT"):
    if side == "RIGHT":
        wrist_id = mp_pose.PoseLandmark.RIGHT_WRIST
        elbow_id = mp_pose.PoseLandmark.RIGHT_ELBOW
        shoulder_id = mp_pose.PoseLandmark.RIGHT_SHOULDER
        hip_id = mp_pose.PoseLandmark.RIGHT_HIP
        index_id = mp_pose.PoseLandmark.RIGHT_INDEX
        pinky_id = mp_pose.PoseLandmark.RIGHT_PINKY
    else:
        wrist_id = mp_pose.PoseLandmark.LEFT_WRIST
        elbow_id = mp_pose.PoseLandmark.LEFT_ELBOW
        shoulder_id = mp_pose.PoseLandmark.LEFT_SHOULDER
        hip_id = mp_pose.PoseLandmark.LEFT_HIP
        index_id = mp_pose.PoseLandmark.LEFT_INDEX
        pinky_id = mp_pose.PoseLandmark.LEFT_PINKY

    # Extract 3D coordinates
    wrist = np.array([landmarks[wrist_id].x, landmarks[wrist_id].y, landmarks[wrist_id].z])
    elbow = np.array([landmarks[elbow_id].x, landmarks[elbow_id].y, landmarks[elbow_id].z])
    shoulder = np.array([landmarks[shoulder_id].x, landmarks[shoulder_id].y, landmarks[shoulder_id].z])
    hip = np.array([landmarks[hip_id].x, landmarks[hip_id].y, landmarks[hip_id].z])
   
    # Use hand landmarks if available for better accuracy
    try:
        index = np.array([landmarks[index_id].x, landmarks[index_id].y, landmarks[index_id].z])
        pinky = np.array([landmarks[pinky_id].x, landmarks[pinky_id].y, landmarks[pinky_id].z])
        has_hand_landmarks = True
    except (IndexError, AttributeError):
        has_hand_landmarks = False
   
    # Step 1: Create anatomical reference frame
    # Forearm axis (primary axis for rotation)
    forearm_vector = wrist - elbow
    forearm_length = np.linalg.norm(forearm_vector)
    forearm_unit = forearm_vector / forearm_length if forearm_length > 0 else np.array([0, 0, 0])
   
    # Upper arm axis (secondary reference)
    upper_arm_vector = shoulder - elbow
    upper_arm_length = np.linalg.norm(upper_arm_vector)
    upper_arm_unit = upper_arm_vector / upper_arm_length if upper_arm_length > 0 else np.array([0, 0, 0])
   
    # Create vertical reference (based on gravity direction)
    vertical_vector = np.array([0, -1, 0])  # Y-axis points down in MediaPipe
   
    # Step 2: Create rotation reference plane
    # This plane is perpendicular to the forearm and aligned with gravity when possible
   
    # First attempt to use the elbow-shoulder-hip plane as reference
    torso_vector = hip - shoulder
    torso_length = np.linalg.norm(torso_vector)
    if torso_length > 0:
        torso_unit = torso_vector / torso_length
    else:
        torso_unit = np.array([0, 1, 0])  # Fallback
   
    # Create normal vector to the rotation plane
    # We want a vector perpendicular to both forearm and a reference plane
    reference_normal = np.cross(forearm_unit, vertical_vector)
    reference_normal_length = np.linalg.norm(reference_normal)
   
    # If the cross product is too small (forearm aligned with vertical),
    # use the upper arm as additional reference
    if reference_normal_length < 0.1:
        reference_normal = np.cross(forearm_unit, upper_arm_unit)
        reference_normal_length = np.linalg.norm(reference_normal)
   
    # Normalize the normal vector
    if reference_normal_length > 0:
        reference_normal = reference_normal / reference_normal_length
    else:
        # Fallback if all vectors are collinear
        if side == "RIGHT":
            reference_normal = np.array([0, 0, 1])  # Right-hand coordinate system
        else:
            reference_normal = np.array([0, 0, -1])
   
    # Step 3: Analyze hand orientation if hand landmarks are available
    if has_hand_landmarks:
        # Vector across the hand (index to pinky)
        hand_vector = index - pinky
        hand_length = np.linalg.norm(hand_vector)
       
        if hand_length > 0:
            hand_unit = hand_vector / hand_length
           
            # Project hand vector onto the rotation plane
            # Calculate how much the hand vector aligns with the rotation plane normal
            dot_product = np.dot(hand_unit, reference_normal)
           
            # Project the hand vector onto the plane
            hand_projected = hand_unit - dot_product * reference_normal
            hand_projected_length = np.linalg.norm(hand_projected)
           
            if hand_projected_length > 0:
                hand_projected = hand_projected / hand_projected_length
               
                # Create a reference vector on the plane
                in_plane_reference = np.cross(forearm_unit, reference_normal)
                in_plane_reference_length = np.linalg.norm(in_plane_reference)
               
                if in_plane_reference_length > 0:
                    in_plane_reference = in_plane_reference / in_plane_reference_length
                   
                    # Calculate angle between reference and hand projection
                    cos_angle = np.clip(np.dot(hand_projected, in_plane_reference), -1.0, 1.0)
                    rotation_angle = np.degrees(np.arccos(cos_angle))
                   
                    # Determine rotation direction
                    direction = np.dot(np.cross(in_plane_reference, hand_projected), forearm_unit)
                    if direction < 0:
                        rotation_angle = 360 - rotation_angle
                   
                    # Map to 0-180 range (0 = supination, 180 = pronation)
                    # Adjust based on which side we're analyzing
                    if side == "RIGHT":
                        if rotation_angle > 180:
                            rotation_angle = 360 - rotation_angle
                    else:  # LEFT
                        if rotation_angle <= 180:
                            rotation_angle = 180 - rotation_angle
                        else:
                            rotation_angle = 540 - rotation_angle
                   
                    return rotation_angle
   
    # Step 4: Fallback method if hand landmarks aren't available or calculation failed
    # Use the relationship between forearm and upper arm
   
    # Project upper arm onto the plane perpendicular to forearm
    dot_product = np.dot(upper_arm_unit, forearm_unit)
    upper_arm_perpendicular = upper_arm_unit - dot_product * forearm_unit
    upper_arm_perp_length = np.linalg.norm(upper_arm_perpendicular)
   
    if upper_arm_perp_length > 0:
        upper_arm_perpendicular = upper_arm_perpendicular / upper_arm_perp_length
       
        # Create a reference vector on the plane
        horizontal_reference = np.array([1, 0, 0])  # Use world-x as default
       
        # Project horizontal reference onto rotation plane
        dot_product = np.dot(horizontal_reference, forearm_unit)
        horizontal_projected = horizontal_reference - dot_product * forearm_unit
        horizontal_length = np.linalg.norm(horizontal_projected)
       
        if horizontal_length > 0:
            horizontal_projected = horizontal_projected / horizontal_length
           
            # Calculate angle
            cos_angle = np.clip(np.dot(upper_arm_perpendicular, horizontal_projected), -1.0, 1.0)
            rotation_angle = np.degrees(np.arccos(cos_angle))
           
            # Adjust based on anatomical constraints
            # This is a rough approximation without hand landmarks
            rotation_angle = min(rotation_angle, 180)
           
            # Side-specific adjustments
            if side == "LEFT":
                rotation_angle = 180 - rotation_angle
           
            return rotation_angle
   
    # Final fallback - use wrist Z-coordinate compared to elbow
    # This is the least accurate method but provides something
    wrist_depth_diff = wrist[2] - elbow[2]
    if side == "RIGHT":
        # For right hand, positive z-diff tends to mean supination (lower angle)
        rotation_angle = 90 - (wrist_depth_diff * 300)  # Scale factor needs calibration
    else:
        # For left hand, positive z-diff tends to mean pronation (higher angle)
        rotation_angle = 90 + (wrist_depth_diff * 300)  # Scale factor needs calibration
   
    # Clamp to valid range
    rotation_angle = np.clip(rotation_angle, 0, 180)
   
    return rotation_angle

def calculate_shoulder_rula(shoulder_angle, landmarks, side):
    # Base score basé sur l'angle
    if shoulder_angle < 20:
        base_score = 1
    elif 20 <= shoulder_angle < 45:
        base_score = 2
    elif 45 <= shoulder_angle < 90:
        base_score = 3
    else:
        base_score = 4

    # Vérification de l'élévation de l'épaule
    if side == "RIGHT":
        shoulder_landmark = mp_pose.PoseLandmark.RIGHT_SHOULDER
        hip_landmark = mp_pose.PoseLandmark.RIGHT_HIP
    else:
        shoulder_landmark = mp_pose.PoseLandmark.LEFT_SHOULDER
        hip_landmark = mp_pose.PoseLandmark.LEFT_HIP

    shoulder_y = landmarks[shoulder_landmark].y
    hip_y = landmarks[hip_landmark].y
    elevated = (shoulder_y < hip_y - 0.05)  # Seuil d'élévation

    # Vérification de l'abduction de l'épaule
    left_hip_x = landmarks[mp_pose.PoseLandmark.LEFT_HIP].x
    right_hip_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x
    midline_x = (left_hip_x + right_hip_x) / 2
    shoulder_x = landmarks[shoulder_landmark].x
    distance_from_midline = abs(shoulder_x - midline_x)
    abducted = (distance_from_midline > 0.1)  # Seuil d'abduction

    # Points supplémentaires
    additional = 0
    if elevated:
        additional += 1
    if abducted:
        additional += 1

    return base_score + additional


def draw_angle(image, point1, point2, point3, angle, color=(0, 255, 0), radius=40):
    center = tuple(point2)
    v1 = np.array(point1) - np.array(center)
    v2 = np.array(point3) - np.array(center)
    angle1 = math.degrees(math.atan2(v1[1], v1[0]))
    angle2 = math.degrees(math.atan2(v2[1], v2[0]))
    if angle1 > angle2:
        angle1, angle2 = angle2, angle1

    cv2.ellipse(image, center, (radius, radius), 0, angle1, angle2, color, 2)
    cv2.putText(image, f"{angle:.1f}°", (center[0] + radius, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def calculate_torso_angle_3d(landmarks, img_w, img_h):
    left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z])
    right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                               landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z])
    left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP].y,
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP].z])
    right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y,
                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z])
    shoulder_mid = (left_shoulder + right_shoulder) / 2
    hip_mid = (left_hip + right_hip) / 2
    torso_vector = shoulder_mid - hip_mid
    vertical_vector = np.array([0, -1, 0])
    dot_product = np.dot(torso_vector, vertical_vector)
    norm_product = np.linalg.norm(torso_vector) * np.linalg.norm(vertical_vector)
    angle_rad = np.arccos(dot_product / norm_product)
    torso_angle = np.degrees(angle_rad)
    shoulder_mid_px = (int(shoulder_mid[0] * img_w), int(shoulder_mid[1] * img_h))
    hip_mid_px = (int(hip_mid[0] * img_w), int(hip_mid[1] * img_h))
    return torso_angle, shoulder_mid_px, hip_mid_px

def process_frame(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    img_h, img_w = image.shape[:2]
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
       
        img_h, img_w = image.shape[:2]
       
        def get_coords(landmark):
            return (int(landmarks[landmark].x * img_w), int(landmarks[landmark].y * img_h))
       
        right_shoulder = get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)
        right_elbow = get_coords(mp_pose.PoseLandmark.RIGHT_ELBOW)
        right_wrist = get_coords(mp_pose.PoseLandmark.RIGHT_WRIST)
       
        left_shoulder = get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER)
        left_elbow = get_coords(mp_pose.PoseLandmark.LEFT_ELBOW)
        left_wrist = get_coords(mp_pose.PoseLandmark.LEFT_WRIST)
       
        right_hip = get_coords(mp_pose.PoseLandmark.RIGHT_HIP)
        left_hip = get_coords(mp_pose.PoseLandmark.LEFT_HIP)
        neck_angle, nose_px, shoulder_mid_px, hip_mid_px = calculate_neck_angle(landmarks, img_w, img_h)
       
        # Calcul des angles
        right_elbow_angle = calculate_3d_angle(right_shoulder, right_elbow, right_wrist)
        left_elbow_angle = calculate_3d_angle(left_shoulder, left_elbow, left_wrist)
       
        right_shoulder_angle = calculate_3d_angle(get_coords(mp_pose.PoseLandmark.RIGHT_ELBOW),
                                                  get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER),
                                                  right_hip)

        left_shoulder_angle = calculate_3d_angle(get_coords(mp_pose.PoseLandmark.LEFT_ELBOW),
                                                 get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER),
                                                 left_hip)

        right_wrist_rotation = calculate_wrist_rotation(landmarks, side="RIGHT")
        left_wrist_rotation = calculate_wrist_rotation(landmarks, side="LEFT")
       
        torso_angle, shoulder_mid_px, hip_mid_px = calculate_torso_angle_3d(landmarks, img_w, img_h)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        # 🔹 Préparer les données pour le modèle
        angles = {
            'Right_Shoulder_Angle': right_shoulder_angle,
            'Left_Shoulder_Angle': left_shoulder_angle,
            'Right_Elbow_Angle': right_elbow_angle,
            'Left_Elbow_Angle': left_elbow_angle,
            'Right_Wrist_Rotation': right_wrist_rotation,
            'Left_Wrist_Rotation': left_wrist_rotation,
            'Torso_Angle': torso_angle,
            'Neck_Angle': neck_angle
        }
        
        # 🔹 Prédiction des risques
        input_data = prepare_input_data(angles)
        predictions = model.predict(input_data)[0]
        #firebase
        upload_to_firestore(angles, predictions)

        
        # 🔹 Afficher les résultats
        display_risk_assessment(image, predictions)
       
        cv2.line(image, shoulder_mid_px, hip_mid_px, (255, 255, 0), 3)
        cv2.putText(image, f"Trunk: {torso_angle:.1f}°", (shoulder_mid_px[0] - 50, shoulder_mid_px[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        

        draw_angle(image, right_shoulder, right_elbow, right_wrist, right_elbow_angle, (255, 0, 0))
        draw_angle(image, left_shoulder, left_elbow, left_wrist, left_elbow_angle, (255, 0, 0))
        draw_angle(image, right_elbow, right_shoulder, right_hip, right_shoulder_angle, (0, 255, 255))
        draw_angle(image, left_elbow, left_shoulder, left_hip, left_shoulder_angle, (0, 255, 255))

        cv2.putText(image, f"{right_elbow_angle:.1f}°", (right_elbow[0] - 30, right_elbow[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(image, f"{left_elbow_angle:.1f}°", (left_elbow[0] - 30, left_elbow[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.putText(image, f"{right_wrist_rotation:.1f}°", (right_wrist[0] - 30, right_wrist[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, f"{left_wrist_rotation:.1f}°", (left_wrist[0] - 30, left_wrist[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(image, f"{right_shoulder_angle:.1f}°", (right_shoulder[0] - 40, right_shoulder[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(image, f"{left_shoulder_angle:.1f}°", (left_shoulder[0] - 40, left_shoulder[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
       
        #cv2.line(image, nose_px, shoulder_mid_px, (255, 0, 255), 2)
        cv2.line(image, shoulder_mid_px, hip_mid_px, (255, 0, 255), 2)
        #draw_angle(image, nose_px, shoulder_mid_px, hip_mid_px, neck_angle, (255, 0, 255), 30)
        #cv2.putText(image, f"Neck: {neck_angle:.1f}°",
                   #(shoulder_mid_px[0] + 10, shoulder_mid_px[1] + 20),
                  # cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        # Réorganiser l'affichage du cou
        # Et pour le cou :
        cv2.line(image, nose_px, shoulder_mid_px, (255, 0, 255), 2)
        draw_angle(image, nose_px, shoulder_mid_px, hip_mid_px, neck_angle, (255, 0, 255), 25)
        cv2.putText(image, f"Neck: {neck_angle:.1f}°",
               (nose_px[0] + 10, nose_px[1] - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
    # Afficher l'indication de sauvegarde Excel
    status_text = "Press 'q' to quit and save data"
    cv2.putText(image, status_text, (10, img_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
   
    return image
def upload_to_firestore(angles, predictions):
    data = {
        "timestamp": datetime.utcnow().isoformat(),  # format standard ISO
        "angles": angles,
        "risks": predictions.tolist() if hasattr(predictions, 'tolist') else predictions
    }
    db.collection("posture_analysis").add(data)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = process_frame(frame)
            cv2.imshow("Posture Analysis with Risk Assessment", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
