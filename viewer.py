import open3d as o3d
import numpy as np
import os
import time

# Configuration
DATA_PATH = "./data/custom_av"
FRAME_LIST_FILE = os.path.join(DATA_PATH, "ImageSets", "train.txt")
POINTS_FOLDER = os.path.join(DATA_PATH, "points")
LABELS_FOLDER = os.path.join(DATA_PATH, "labels")
VIEW_FILE = os.path.join(DATA_PATH, "view.json")

# Load frame IDs
with open(FRAME_LIST_FILE, 'r') as f:
    frame_ids = [line.strip() for line in f.readlines()]

def load_npy_pointcloud(file_path):
    points = np.load(file_path)
    xyz = points[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.paint_uniform_color([1.0, 1.0, 1.0])  # 모든 점을 흰색으로
    return pcd

def create_bbox(center, size, yaw, color):
    R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw])
    box = o3d.geometry.OrientedBoundingBox(center, R, size)
    box.color = color
    return box

def create_heading_arrow(center, yaw, length=2.0):
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.05,
        cone_radius=0.1,
        cylinder_height=length * 0.8,
        cone_height=length * 0.2
    )
    arrow.paint_uniform_color([1, 0, 0])  # Red

    # Step 1: 기본 방향 +Z → +X 로 눕히기
    R_to_x = o3d.geometry.get_rotation_matrix_from_xyz([-np.pi / 2, 0, 0])
    arrow.rotate(R_to_x, center=(0, 0, 0))

    # Step 2: yaw 회전 적용 (Z축 기준)
    R_yaw = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw - np.pi/2])
    arrow.rotate(R_yaw, center=(0, 0, 0))

    # Step 3: 중심 위치로 이동
    arrow.translate(center)
    return arrow




def load_labels(label_path):
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            x, y, z = map(float, parts[0:3])
            dx, dy, dz = map(float, parts[3:6])
            yaw = float(parts[6])
            color = [1, 0, 0]  # Red
            labels.append((x, y, z, dx, dy, dz, yaw, color))
    return labels

def save_viewpoint(vis):
    params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(VIEW_FILE, params)
    print(f"Viewpoint saved to {VIEW_FILE}")

def deferred_load_viewpoint(vis):
    if os.path.exists(VIEW_FILE):
        params = o3d.io.read_pinhole_camera_parameters(VIEW_FILE)
        vis.get_view_control().convert_from_pinhole_camera_parameters(params)
        print(f"[Deferred] Viewpoint loaded from {VIEW_FILE}")
    else:
        print("[Warning] No saved viewpoint found.")

def visualize_frames():
    vis = o3d.visualization.VisualizerWithKeyCallback()
    if not vis.create_window(window_name='LiDAR Viewer'):
        print("[ERROR] Failed to create Open3D window.")
        return

    frame_idx = 0
    deferred_view_loaded = False
    last_key_time = 0
    key_delay = 0.1  # seconds

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    opt = vis.get_render_option()
    opt.point_size = 1.0
    opt.background_color = np.array([0.0, 0.0, 0.0])  # 검정
    def update_scene():
        nonlocal deferred_view_loaded
        vis.clear_geometries()

        frame_id = frame_ids[frame_idx]
        pc_path = os.path.join(POINTS_FOLDER, f"{frame_id}.npy")
        label_path = os.path.join(LABELS_FOLDER, f"{frame_id}.txt")

        pcd = load_npy_pointcloud(pc_path)
        vis.add_geometry(pcd)

        labels = load_labels(label_path)
        for x, y, z, dx, dy, dz, yaw, color in labels:
            center = [x, y, z]
            size = [dx, dy, dz]
            box = create_bbox(center, size, yaw, color)
            vis.add_geometry(box)

            arrow = create_heading_arrow(center, yaw)
            vis.add_geometry(arrow)

        vis.add_geometry(coordinate_frame)

        vis.poll_events()
        vis.update_renderer()

        if os.path.exists(VIEW_FILE):
            params = o3d.io.read_pinhole_camera_parameters(VIEW_FILE)
            vis.get_view_control().convert_from_pinhole_camera_parameters(params)

        if not deferred_view_loaded:
            deferred_view_loaded = True

    def debounce():
        nonlocal last_key_time
        now = time.time()
        if now - last_key_time >= key_delay:
            last_key_time = now
            return True
        return False

    def next_frame(vis):
        nonlocal frame_idx
        if debounce():
            frame_idx = (frame_idx + 1) % len(frame_ids)
            update_scene()
        return False

    def prev_frame(vis):
        nonlocal frame_idx
        if debounce():
            frame_idx = (frame_idx - 1 + len(frame_ids)) % len(frame_ids)
            update_scene()
        return False

    def save_view(vis):
        save_viewpoint(vis)
        return False

    def quit_viewer(vis):
        print("Quitting viewer.")
        vis.close()
        return False

    vis.register_key_callback(ord("D"), next_frame)
    vis.register_key_callback(ord("A"), prev_frame)
    vis.register_key_callback(ord("F"), save_view)
    vis.register_key_callback(ord("Q"), quit_viewer)

    update_scene()
    vis.run()

if __name__ == "__main__":
    visualize_frames()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import open3d as o3d
# import numpy as np
# import os
# import time
# import pickle
# import argparse

# try:
#     import yaml
#     _HAS_YAML = True
# except Exception:
#     _HAS_YAML = False

# DATA_PATH       = "./data/custom_av"
# FRAME_LIST_FILE = os.path.join(DATA_PATH, "ImageSets", "test.txt")
# POINTS_FOLDER   = os.path.join(DATA_PATH, "points")
# RESULT_PKL      = "./result_s.pkl"     # 예측 결과(pkl)
# SCORE_THR       = 0.10                           # 점수 임계값 
# NMS_IOU_THR     = 0.10                           
# VIEW_FILE       = os.path.join(DATA_PATH, "view.json")

# CLASS_COLOR = {
#     "Vehicle":    [1.0, 0.0, 0.0],
#     "Pedestrian": [0.0, 1.0, 0.0],
#     "Cyclist":    [0.0, 0.0, 1.0],
# }
# LABEL_COLOR = {  
#     1: [1.0, 0.0, 0.0],
#     2: [0.0, 1.0, 0.0],
#     3: [0.0, 0.0, 1.0],
# }

# def read_thresholds_from_cfg(cfg_path):
#     """
#     cfg YAML에서 (score_thresh, nms_iou) 추출.
#     - MODEL.POST_PROCESSING.SCORE_THRESH
#     - MODEL.POST_PROCESSING.NMS_CONFIG.NMS_THRESH
#     클래스별 리스트일 경우 평균값 사용.
#     """
#     score = None
#     nms_iou = None
#     if not (_HAS_YAML and cfg_path and os.path.exists(cfg_path)):
#         return score, nms_iou
#     try:
#         with open(cfg_path, 'r') as f:
#             y = yaml.safe_load(f) or {}
#         pp = ((y.get('MODEL') or {}).get('POST_PROCESSING') or {})
#         s = pp.get('SCORE_THRESH', None) or pp.get('SCORE_THRESH_LIST', None)
#         if isinstance(s, (list, tuple)):
#             score = float(np.mean(s))
#         elif s is not None:
#             score = float(s)
#         nc = pp.get('NMS_CONFIG', {}) or {}
#         nt = nc.get('NMS_THRESH', None) or nc.get('NMS_THRESH_TEST', None) or nc.get('NMS_THRESH_LIST', None)
#         if isinstance(nt, (list, tuple)):
#             nms_iou = float(np.mean(nt))
#         elif nt is not None:
#             nms_iou = float(nt)
#     except Exception as e:
#         print(f"[WARN] cfg read failed: {e}")
#     return score, nms_iou

# def parse_args_and_apply_cfg():
#     global SCORE_THR, NMS_IOU_THR, RESULT_PKL, DATA_PATH, FRAME_LIST_FILE, POINTS_FOLDER, VIEW_FILE
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--cfg_file", type=str, default=None, help="OpenPCDet cfg yaml")
#     ap.add_argument("--score_thr", type=float, default=None, help="override score threshold")
#     ap.add_argument("--nms_iou", type=float, default=None, help="override NMS IoU threshold (BEV AABB)")
#     ap.add_argument("--result_pkl", type=str, default=RESULT_PKL)
#     ap.add_argument("--data_path", type=str, default=DATA_PATH)
#     ap.add_argument("--frame_list", type=str, default=None, help="path to test.txt (override)")
#     args, _ = ap.parse_known_args()

#     s, n = read_thresholds_from_cfg(args.cfg_file)
#     if s is not None and args.score_thr is None:
#         SCORE_THR = s
#     if n is not None and args.nms_iou is None:
#         NMS_IOU_THR = n

#     if args.score_thr is not None:
#         SCORE_THR = args.score_thr
#     if args.nms_iou is not None:
#         NMS_IOU_THR = args.nms_iou

#     RESULT_PKL = args.result_pkl
#     DATA_PATH = args.data_path
#     POINTS_FOLDER = os.path.join(DATA_PATH, "points")
#     VIEW_FILE = os.path.join(DATA_PATH, "view.json")

#     global frame_ids
#     frame_list = args.frame_list or os.path.join(DATA_PATH, "ImageSets", "test.txt")
#     with open(frame_list, 'r') as f:
#         frame_ids = [line.strip() for line in f.readlines()]

#     print(f"[INFO] SCORE_THR={SCORE_THR:.3f}, NMS_IOU_THR={NMS_IOU_THR:.3f}")
#     print(f"[INFO] RESULT_PKL={RESULT_PKL}")
#     print(f"[INFO] DATA_PATH={DATA_PATH}, POINTS_FOLDER={POINTS_FOLDER}")
#     return args

# #포인트 로더 
# def load_npy_pointcloud(file_path):
#     points = np.load(file_path)  # (N, 3/4)
#     xyz = points[:, :3]
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(xyz)
#     pcd.paint_uniform_color([1.0, 1.0, 1.0])  # 흰색
#     return pcd, xyz.shape[0]

# def create_bbox(center, size, yaw, color):
#     R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw])
#     box = o3d.geometry.OrientedBoundingBox(center, R, size)
#     box.color = color
#     return box

# def create_heading_arrow(center, yaw, length=2.0):
#     arrow = o3d.geometry.TriangleMesh.create_arrow(
#         cylinder_radius=0.05, cone_radius=0.1,
#         cylinder_height=length * 0.8, cone_height=length * 0.2
#     )
#     arrow.paint_uniform_color([1, 0, 0])  # 빨강
#     R_to_x = o3d.geometry.get_rotation_matrix_from_xyz([-np.pi / 2, 0, 0])  # +Z→+X
#     arrow.rotate(R_to_x, center=(0, 0, 0))
#     R_yaw = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw - np.pi/2])
#     arrow.rotate(R_yaw, center=(0, 0, 0))
#     arrow.translate(center)
#     return arrow

# def load_predictions(pkl_path):
#     with open(pkl_path, 'rb') as f:
#         det_annos = pickle.load(f)  # list of dict
#     return {str(anno['frame_id']): anno for anno in det_annos}

# def boxes_lidar_to_bev_aabb_xyxy(boxes_lidar):
#     """
#     boxes_lidar: (N,7) [x,y,z,dx,dy,dz,yaw]
#     → (N,4) [x1,y1,x2,y2]  (회전 박스의 AABB)
#     """
#     out = np.zeros((len(boxes_lidar), 4), dtype=np.float32)
#     for i, (x, y, z, dx, dy, dz, yaw) in enumerate(boxes_lidar):
#         hx, hy = dx / 2.0, dy / 2.0
#         # corners local
#         corners = np.array([[ hx,  hy],
#                             [ hx, -hy],
#                             [-hx,  hy],
#                             [-hx, -hy]], dtype=np.float32)
#         c, s = np.cos(yaw), np.sin(yaw)
#         R = np.array([[c, -s],
#                       [s,  c]], dtype=np.float32)
#         pts = (corners @ R.T) + np.array([x, y], dtype=np.float32)
#         x1, y1 = pts[:, 0].min(), pts[:, 1].min()
#         x2, y2 = pts[:, 0].max(), pts[:, 1].max()
#         out[i] = [x1, y1, x2, y2]
#     return out

# def iou_xyxy(a, b):
#     """
#     a: (4,), b: (M,4)  → IoU: (M,)
#     """
#     xx1 = np.maximum(a[0], b[:, 0])
#     yy1 = np.maximum(a[1], b[:, 1])
#     xx2 = np.minimum(a[2], b[:, 2])
#     yy2 = np.minimum(a[3], b[:, 3])
#     w = np.maximum(0.0, xx2 - xx1)
#     h = np.maximum(0.0, yy2 - yy1)
#     inter = w * h
#     area_a = (a[2] - a[0]) * (a[3] - a[1]) + 1e-6
#     area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) + 1e-6
#     return inter / (area_a + area_b - inter + 1e-6)

# def nms_bev_aabb(boxes_xyxy, scores, iou_thr):
#     if len(boxes_xyxy) == 0:
#         return []
#     order = np.argsort(scores)[::-1]
#     keep = []
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#         if order.size == 1:
#             break
#         ious = iou_xyxy(boxes_xyxy[i], boxes_xyxy[order[1:]])
#         remain = np.where(ious <= iou_thr)[0]
#         order = order[1:][remain]
#     return keep

# def visualize_frames():
#     args = parse_args_and_apply_cfg()

#     vis = o3d.visualization.VisualizerWithKeyCallback()
#     if not vis.create_window(window_name='LiDAR Viewer'):
#         print("[ERROR] Failed to create Open3D window.")
#         return

#     # 렌더 옵션 포인트 사이즈 및 컬러
#     opt = vis.get_render_option()
#     opt.point_size = 1.0
#     opt.background_color = np.array([0.0, 0.0, 0.0])

#     # 예측 결과 로드
#     try:
#         preds_by_frame = load_predictions(RESULT_PKL)
#     except Exception as e:
#         print(f"[ERROR] Failed to read {RESULT_PKL}: {e}")
#         return

#     frame_idx = 0
#     last_key_time = 0
#     key_delay = 0.1  # seconds
#     coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

#     def debounce():
#         nonlocal last_key_time
#         now = time.time()
#         if now - last_key_time >= key_delay:
#             last_key_time = now
#             return True
#         return False

#     def update_scene():
#         vis.clear_geometries()

#         frame_id = frame_ids[frame_idx]
#         pc_path = os.path.join(POINTS_FOLDER, f"{frame_id}.npy")

#         # 포인트
#         if not os.path.exists(pc_path):
#             print(f"[WARN] point file not found: {pc_path}")
#             num_pts = 0
#         else:
#             pcd, num_pts = load_npy_pointcloud(pc_path)
#             vis.add_geometry(pcd)

#         # 예측
#         anno = preds_by_frame.get(str(frame_id))
#         if anno is None:
#             alt = str(frame_id).lstrip('0') or "0"
#             anno = preds_by_frame.get(alt)

#         num_boxes_raw = 0
#         num_boxes_after_score = 0
#         num_boxes_after_nms = 0

#         if anno is None:
#             print(f"[WARN] No prediction found for frame_id={frame_id}")
#         else:
#             boxes  = np.asarray(anno.get('boxes_lidar', []), dtype=np.float32)  # (N,7)
#             names  = np.asarray(anno.get('name', []))
#             scores = np.asarray(anno.get('score', []), dtype=np.float32)
#             labels = np.asarray(anno.get('pred_labels', []), dtype=np.int32)

#             num_boxes_raw = len(boxes)

#             # 1) score threshold
#             keep_score = np.arange(num_boxes_raw)
#             if scores.size:
#                 keep_score = np.where(scores >= float(SCORE_THR))[0]
#             boxes_f  = boxes[keep_score]
#             scores_f = scores[keep_score] if scores.size else np.ones((len(keep_score),), dtype=np.float32)
#             names_f  = names[keep_score]  if names.size  else []
#             labels_f = labels[keep_score] if labels.size else []
#             num_boxes_after_score = len(boxes_f)

        
#             keep_nms = np.arange(len(boxes_f))
#             if len(boxes_f) and NMS_IOU_THR is not None and NMS_IOU_THR > 0:
#                 aabbs = boxes_lidar_to_bev_aabb_xyxy(boxes_f)
#                 keep_nms = nms_bev_aabb(aabbs, scores_f, float(NMS_IOU_THR))
#             boxes_v  = boxes_f[keep_nms]
#             scores_v = scores_f[keep_nms]
#             names_v  = names_f[keep_nms]  if len(names_f)  else []
#             labels_v = labels_f[keep_nms] if len(labels_f) else []
#             num_boxes_after_nms = len(boxes_v)

#             #시각화
#             for i in range(len(boxes_v)):
#                 x, y, z, dx, dy, dz, yaw = boxes_v[i].tolist()
#                 if len(names_v):
#                     color = CLASS_COLOR.get(str(names_v[i]), [1, 0, 0])
#                 elif len(labels_v):
#                     color = LABEL_COLOR.get(int(labels_v[i]), [1, 0, 0])
#                 else:
#                     color = [1, 0, 0]
#                 vis.add_geometry(create_bbox([x, y, z], [dx, dy, dz], yaw, color))
#                 vis.add_geometry(create_heading_arrow([x, y, z], yaw))

#         vis.add_geometry(coord)
#         vis.poll_events()
#         vis.update_renderer()

#         if os.path.exists(VIEW_FILE):
#             try:
#                 params = o3d.io.read_pinhole_camera_parameters(VIEW_FILE)
#                 vis.get_view_control().convert_from_pinhole_camera_parameters(params)
#             except Exception:
#                 pass

#         # --- 콘솔 출력: 프레임/포인트/박스 통계 ---
#         print(f"[Frame {frame_idx+1}/{len(frame_ids)}] id={frame_id} | points={num_pts} "
#               f"| boxes raw={num_boxes_raw}, after_score={num_boxes_after_score}, after_nms={num_boxes_after_nms}")

#     def next_frame(_):
#         nonlocal frame_idx
#         if debounce():
#             frame_idx = (frame_idx + 1) % len(frame_ids)
#             update_scene()
#         return False

#     def prev_frame(_):
#         nonlocal frame_idx
#         if debounce():
#             frame_idx = (frame_idx - 1 + len(frame_ids)) % len(frame_ids)
#             update_scene()
#         return False

#     def save_view(_):
#         params = vis.get_view_control().convert_to_pinhole_camera_parameters()
#         o3d.io.write_pinhole_camera_parameters(VIEW_FILE, params)
#         print(f"Viewpoint saved to {VIEW_FILE}")
#         return False

#     def quit_viewer(_):
#         print("Quitting viewer.")
#         vis.close()
#         return False

#     vis.register_key_callback(ord("D"), next_frame)
#     vis.register_key_callback(ord("A"), prev_frame)
#     vis.register_key_callback(ord("F"), save_view)
#     vis.register_key_callback(ord("Q"), quit_viewer)

#     update_scene()
#     vis.run()

# if __name__ == "__main__":
#     visualize_frames()

