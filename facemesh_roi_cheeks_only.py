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

# Only cheeks: Left + Right
final_mesh_map_cheeks = [
    # 左臉頰
    [212, 186, 57], [216, 212, 186], [216, 214, 212], [207, 214, 216], [207, 214, 192], 
    [192, 187, 207], [187, 147, 192], [123, 147, 187], [123, 205, 187], [205, 187, 216],
    [205, 206, 216], [123, 35, 229], [123, 229, 205], [126, 205, 229], [232, 229, 126], 
    [36, 203, 206], [205, 206, 36], [36, 203, 129], [143, 123, 35], [126, 129, 36],
    
    # 右臉頰
    [452, 449, 358], [358, 449, 425], [449, 425, 352], [449, 352, 265], [425, 352, 411], 
    [411, 352, 376], [425, 436, 411], [376, 433, 416], [376, 416, 411], [411, 416, 427], 
    [427, 434, 436], [427, 434, 416], [436, 434, 432], [436, 432, 410], [410, 287, 432], 
    [436, 426, 425], [425, 426, 266], [423, 266, 426], [423, 358, 266], [265, 372, 352], 
    [266, 355, 358],
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

def process_roi_extraction(folder_path, subject_id, output_dir=None):
    """
    Process ROI extraction using FaceMesh - Cheeks only (Left + Right)
    
    Args:
        folder_path: path to folder containing video frames
        subject_id: unique identifier for the subject
        output_dir: local directory to save outputs (default: current directory)
    """
    mesh_map = final_mesh_map_cheeks
    version_text = "Cheeks Only (L+R)"
    
    # Set output directory to local path (avoid NAS permission issues)
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'FaceMesh_Output')
    
    # Create output directory
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create output directory {output_dir}: {e}", file=sys.stderr)
        output_dir = os.getcwd()
    
    # Results storage
    results = []
    saved_visualization = False
    
    print(f"Processing: {subject_id} with {version_text}", file=sys.stderr)
    print(f"Output directory: {output_dir}", file=sys.stderr)
    
    # 掃描資料夾中的所有 BMP 圖片
    import glob
    bmp_files = glob.glob(os.path.join(folder_path, '*.bmp'))
    
    if not bmp_files:
        print(f"Warning: No BMP files found in {folder_path}", file=sys.stderr)
        return results
    
    # 排序檔案（按數字順序）
    def extract_number(filename):
        """從檔名中提取數字"""
        import re
        numbers = re.findall(r'\d+', os.path.basename(filename))
        return int(numbers[0]) if numbers else 0
    
    bmp_files = sorted(bmp_files, key=extract_number)
    
    print(f"Found {len(bmp_files)} BMP files", file=sys.stderr)
    print(f"First file: {os.path.basename(bmp_files[0])}", file=sys.stderr)
    print(f"Last file: {os.path.basename(bmp_files[-1])}", file=sys.stderr)
    
    # 處理所有圖片
    for frame_idx, image_path in enumerate(bmp_files):
        frame_num = frame_idx + 1  # 從 1 開始編號
        
        # 每 100 幀顯示一次進度
        if frame_num % 100 == 0:
            print(f"Processing frame {frame_num}/{len(bmp_files)}...", file=sys.stderr)
        
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
            
            # Save mask visualization for middle frame (as sample)
            middle_frame = len(bmp_files) // 2
            if frame_num == middle_frame and not saved_visualization:
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
                
                # Save with unique filename to LOCAL output directory
                vis_filename = f'{subject_id}_cheeks_only_visualization.jpg'
                vis_path = os.path.join(output_dir, vis_filename)
                
                try:
                    cv2.imwrite(vis_path, combined)
                    saved_visualization = True
                    print(f"VISUALIZATION_SAVED:{vis_path}", file=sys.stderr)
                except Exception as e:
                    print(f"Warning: Could not save visualization: {e}", file=sys.stderr)
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
    if len(sys.argv) < 3:
        print("Usage: python facemesh_roi_cheeks_only.py <folder_path> <subject_id> [output_dir]")
        print("  folder_path: path to folder containing video frames (can be on NAS)")
        print("  subject_id: unique identifier for the subject")
        print("  output_dir: (optional) local directory to save outputs (default: ./FaceMesh_Output)")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    subject_id = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Process ROI extraction
    process_roi_extraction(folder_path, subject_id, output_dir)

if __name__ == "__main__":
    main()