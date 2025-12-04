import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
face_mesh = mp_face_mesh.FaceMesh()

# Version 1: 左臉+右臉
final_mesh_map_v1 = [
    # 左臉
    [212, 186, 57], [216, 212, 186], [216, 214, 212], [207, 214, 216], [207, 214, 192], 
    [192, 187, 207], [187, 147, 192], [123, 147, 187], [123, 205, 187], [205, 187, 216],
    [205, 206, 216], [123, 35, 229], [123, 229, 205], [126, 205, 229], [232, 229, 126], 
    [36, 203, 206], [205, 206, 36], [36, 203, 129], [143, 123, 35], [126, 129, 36],
    
    # 右臉
    [452, 449, 358], [358, 449, 425], [449, 425, 352], [449, 352, 265], [425, 352, 411], 
    [411, 352, 376], [425, 436, 411], [376, 433, 416], [376, 416, 411], [411, 416, 427], 
    [427, 434, 436], [427, 434, 416], [436, 434, 432], [436, 432, 410], [410, 287, 432], 
    [436, 426, 425], [425, 426, 266], [423, 266, 426], [423, 358, 266], [265, 372, 352], 
    [266, 355, 358],
]

# Version 2: 左臉+右臉+額頭
final_mesh_map_v2 = [
    # 左臉
    [212, 186, 57], [216, 212, 186], [216, 214, 212], [207, 214, 216], [207, 214, 192], 
    [192, 187, 207], [187, 147, 192], [123, 147, 187], [123, 205, 187], [205, 187, 216],
    [205, 206, 216], [123, 35, 229], [123, 229, 205], [126, 205, 229], [232, 229, 126], 
    [36, 203, 206], [205, 206, 36], [36, 203, 129], [143, 123, 35], [126, 129, 36],
    
    # 右臉
    [452, 449, 358], [358, 449, 425], [449, 425, 352], [449, 352, 265], [425, 352, 411], 
    [411, 352, 376], [425, 436, 411], [376, 433, 416], [376, 416, 411], [411, 416, 427], 
    [427, 434, 436], [427, 434, 416], [436, 434, 432], [436, 432, 410], [410, 287, 432], 
    [436, 426, 425], [425, 426, 266], [423, 266, 426], [423, 358, 266], [265, 372, 352], 
    [266, 355, 358],
    
    # 額頭 version 2
    [6, 168, 122], [122, 168, 193], [122, 193, 245], [245, 189, 193], [245, 244, 189], 
    [244, 189, 190], [6, 168, 351], [351, 168, 417], [417, 351, 465], [465, 417, 413], 
    [413, 465, 464], [464, 413, 414], [414, 464, 463], [193, 8, 168], [8, 55, 193], 
    [193, 55, 221], [193, 189, 221], [168, 8, 417], [8, 417, 285], [441, 285, 417], 
    [441, 413, 417], [9, 8, 55], [9, 55, 107], [9, 8, 285], [9, 285, 336],
    [9, 151, 107], [151, 107, 108], [9, 151, 336], [336, 151, 337],
    # 額頭 improved
    [67, 69, 109], [69, 109, 108], [109, 10, 108], [108, 151, 10], [10, 151, 337], 
    [10, 337, 338], [337, 338, 299], [338, 299, 297],
]

def vector_cross(a, b):
    x = a[1]*b[2] - a[2]*b[1]
    y = a[2]*b[0] - a[0]*b[2]
    z = a[0]*b[1] - a[1]*b[0]
    return np.asarray([x, y, z])

def calculate_angle(n_v):
    camera_n_v = np.array([0, 0, -1])
    angle = (180 * np.arccos(n_v.dot(camera_n_v) / (np.linalg.norm(n_v) * np.linalg.norm(camera_n_v))) / np.pi) % 360
    if angle > 90:
        angle = angle - 180
        angle = abs(angle)
    return angle

def calculate_mesh_angle(point_of_468, mesh_map):
    mesh_angle = []
    for mesh in mesh_map:
        vector1 = point_of_468[mesh[1]] - point_of_468[mesh[0]]
        vector2 = point_of_468[mesh[2]] - point_of_468[mesh[0]]
        result = vector_cross(vector1, vector2)
        angle = calculate_angle(result)
        mesh_angle.append(int(angle < 60))
    return mesh_angle

def map_2D_angle_map(mesh_angle, image, point_of_468_2D, mesh_map):
    angle_map = np.zeros((image.shape[0], image.shape[1]))
    for id, flag in enumerate(mesh_angle):
        if flag:
            point1 = point_of_468_2D[mesh_map[id][0]]
            point2 = point_of_468_2D[mesh_map[id][1]]
            point3 = point_of_468_2D[mesh_map[id][2]]
            myROI = [point1, point2, point3]
            cv2.fillPoly(angle_map, [np.array(myROI)], 1)
    return angle_map

def process_roi_extraction(folder_path, subject_id, version):
    """
    Process ROI extraction using FaceMesh
    version: 1 for 左臉+右臉, 2 for 左臉+右臉+額頭
    """
    # Select mesh map based on version
    if version == 1:
        mesh_map = final_mesh_map_v1
        version_text = "V1: Face L+R"
    else:
        mesh_map = final_mesh_map_v2
        version_text = "V2: Face L+R + Forehead"
    
    # Results storage
    results = []
    saved_visualization = False
    
    # Ensure subject_id is unique
    print(f"Processing: {subject_id} with {version_text}", file=sys.stderr)
    
    # Process frames 300-600
    for frame_num in range(300, 601):
        # Construct image path - try multiple formats
        possible_paths = [
            os.path.join(folder_path, f'{frame_num}.bmp'),
            os.path.join(folder_path, f'{frame_num:03d}.bmp'),
            os.path.join(folder_path, f'{frame_num:05d}.bmp'),
            os.path.join(folder_path, f'00{frame_num}.bmp'),
        ]
        
        image_path = None
        for path in possible_paths:
            if os.path.exists(path):
                image_path = path
                break
        
        if image_path is None:
            # Image not found
            results.append({
                'frame': frame_num,
                'R_avg': 0,
                'G_avg': 0,
                'B_avg': 0,
                'success': 0
            })
            continue
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            results.append({
                'frame': frame_num,
                'R_avg': 0,
                'G_avg': 0,
                'B_avg': 0,
                'success': 0
            })
            continue
        
        # Split channels
        B_channel, G_channel, R_channel = cv2.split(image)
        
        # Process with MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(image_rgb)
        
        if face_results.multi_face_landmarks:
            # Get face landmarks
            face_landmarks = face_results.multi_face_landmarks[0]
            
            # Extract 3D and 2D points
            point_of_468_3D = []
            point_of_468_2D = []
            
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                z = landmark.z * 300
                
                point_of_468_2D.append((x, y))
                point_of_468_3D.append(np.asarray([landmark.x, landmark.y, landmark.z]))
            
            # Calculate mesh angles
            mesh_angle = calculate_mesh_angle(point_of_468_3D, mesh_map)
            
            # Create angle mask
            angle_mask = map_2D_angle_map(mesh_angle, image, point_of_468_2D, mesh_map)
            mask = angle_mask.astype(np.uint8)
            
            # Apply mask to channels
            masked_R = R_channel * mask
            masked_G = G_channel * mask
            masked_B = B_channel * mask
            
            # Calculate averages
            R_avg = masked_R[masked_R != 0].mean() if np.any(masked_R != 0) else 0
            G_avg = masked_G[masked_G != 0].mean() if np.any(masked_G != 0) else 0
            B_avg = masked_B[masked_B != 0].mean() if np.any(masked_B != 0) else 0
            
            results.append({
                'frame': frame_num,
                'R_avg': R_avg,
                'G_avg': G_avg,
                'B_avg': B_avg,
                'success': 1
            })
            
            # Save mask visualization for frame 300 (as sample)
            if frame_num == 300 and not saved_visualization:
                # Create visualization
                skin_pic = cv2.merge([masked_B, masked_G, masked_R])
                
                # Create a copy for drawing
                vis_image = image.copy()
                
                # Draw mesh triangles on the image
                for idx, mesh in enumerate(mesh_map):
                    if idx < len(mesh_angle) and mesh_angle[idx]:
                        pts = np.array([point_of_468_2D[mesh[0]], 
                                       point_of_468_2D[mesh[1]], 
                                       point_of_468_2D[mesh[2]]], np.int32)
                        cv2.polylines(vis_image, [pts], True, (0, 255, 0), 1)
                
                # Add text labels with subject ID
                cv2.putText(vis_image, f"{subject_id}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(vis_image, version_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(skin_pic, "Masked Region", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Save visualization - combine original and masked
                combined = np.hstack([vis_image, skin_pic])
                
                # Create output directory if needed
                output_dir = os.path.join(os.path.dirname(folder_path), 'FaceMesh_Output')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # Save with unique filename
                vis_filename = f'{subject_id}_v{version}_visualization.jpg'
                vis_path = os.path.join(output_dir, vis_filename)
                cv2.imwrite(vis_path, combined)
                
                # Also save in folder for backward compatibility
                vis_path2 = os.path.join(folder_path, vis_filename)
                cv2.imwrite(vis_path2, combined)
                
                saved_visualization = True
                
                # Print path for debugging
                print(f"VISUALIZATION_SAVED:{vis_path}", file=sys.stderr)
        else:
            # No face detected
            results.append({
                'frame': frame_num,
                'R_avg': 0,
                'G_avg': 0,
                'B_avg': 0,
                'success': 0
            })
    
    # Output results to stdout for MATLAB to capture
    for result in results:
        print(f"{result['frame']},{result['R_avg']},{result['G_avg']},{result['B_avg']},{result['success']}")
    
    return results

def main():
    if len(sys.argv) != 4:
        print("Usage: python facemesh_roi_extraction.py <folder_path> <subject_id> <version>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    subject_id = sys.argv[2]
    version = int(sys.argv[3])
    
    # Process ROI extraction
    process_roi_extraction(folder_path, subject_id, version)

if __name__ == "__main__":
    main()