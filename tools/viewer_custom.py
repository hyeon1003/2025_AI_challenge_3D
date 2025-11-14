import open3d as o3d
import numpy as np
import os
import time
import pickle
import argparse

try:
    import yaml
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

DATA_PATH       = "../data/custom_av"
FRAME_LIST_FILE = os.path.join(DATA_PATH, "ImageSets", "test.txt")
POINTS_FOLDER   = os.path.join(DATA_PATH, "points")
RESULT_PKL      = "./result.pkl"     # 예측 결과(pkl)
SCORE_THR       = 0.10                           # 점수 임계값 
NMS_IOU_THR     = 0.10                           
VIEW_FILE       = os.path.join(DATA_PATH, "view.json")

CLASS_COLOR = {
    "Vehicle":    [1.0, 0.0, 0.0],
    "Pedestrian": [0.0, 1.0, 0.0],
    "Cyclist":    [0.0, 0.0, 1.0],
}
LABEL_COLOR = {  
    1: [1.0, 0.0, 0.0],
    2: [0.0, 1.0, 0.0],
    3: [0.0, 0.0, 1.0],
}

def read_thresholds_from_cfg(cfg_path):
    """
    cfg YAML에서 (score_thresh, nms_iou) 추출.
    - MODEL.POST_PROCESSING.SCORE_THRESH
    - MODEL.POST_PROCESSING.NMS_CONFIG.NMS_THRESH
    클래스별 리스트일 경우 평균값 사용.
    """
    score = None
    nms_iou = None
    if not (_HAS_YAML and cfg_path and os.path.exists(cfg_path)):
        return score, nms_iou
    try:
        with open(cfg_path, 'r') as f:
            y = yaml.safe_load(f) or {}
        pp = ((y.get('MODEL') or {}).get('POST_PROCESSING') or {})
        s = pp.get('SCORE_THRESH', None) or pp.get('SCORE_THRESH_LIST', None)
        if isinstance(s, (list, tuple)):
            score = float(np.mean(s))
        elif s is not None:
            score = float(s)
        nc = pp.get('NMS_CONFIG', {}) or {}
        nt = nc.get('NMS_THRESH', None) or nc.get('NMS_THRESH_TEST', None) or nc.get('NMS_THRESH_LIST', None)
        if isinstance(nt, (list, tuple)):
            nms_iou = float(np.mean(nt))
        elif nt is not None:
            nms_iou = float(nt)
    except Exception as e:
        print(f"[WARN] cfg read failed: {e}")
    return score, nms_iou

def parse_args_and_apply_cfg():
    global SCORE_THR, NMS_IOU_THR, RESULT_PKL, DATA_PATH, FRAME_LIST_FILE, POINTS_FOLDER, VIEW_FILE
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_file", type=str, default=None, help="OpenPCDet cfg yaml")
    ap.add_argument("--score_thr", type=float, default=None, help="override score threshold")
    ap.add_argument("--nms_iou", type=float, default=None, help="override NMS IoU threshold (BEV AABB)")
    ap.add_argument("--result_pkl", type=str, default=RESULT_PKL)
    ap.add_argument("--data_path", type=str, default=DATA_PATH)
    ap.add_argument("--frame_list", type=str, default=None, help="path to test.txt (override)")
    args, _ = ap.parse_known_args()

    s, n = read_thresholds_from_cfg(args.cfg_file)
    if s is not None and args.score_thr is None:
        SCORE_THR = s
    if n is not None and args.nms_iou is None:
        NMS_IOU_THR = n

    if args.score_thr is not None:
        SCORE_THR = args.score_thr
    if args.nms_iou is not None:
        NMS_IOU_THR = args.nms_iou

    RESULT_PKL = args.result_pkl
    DATA_PATH = args.data_path
    POINTS_FOLDER = os.path.join(DATA_PATH, "points")
    VIEW_FILE = os.path.join(DATA_PATH, "view.json")

    global frame_ids
    frame_list = args.frame_list or os.path.join(DATA_PATH, "ImageSets", "test.txt")
    with open(frame_list, 'r') as f:
        frame_ids = [line.strip() for line in f.readlines()]

    print(f"[INFO] SCORE_THR={SCORE_THR:.3f}, NMS_IOU_THR={NMS_IOU_THR:.3f}")
    print(f"[INFO] RESULT_PKL={RESULT_PKL}")
    print(f"[INFO] DATA_PATH={DATA_PATH}, POINTS_FOLDER={POINTS_FOLDER}")
    return args

#포인트 로더 
def load_npy_pointcloud(file_path):
    points = np.load(file_path)  # (N, 3/4)
    xyz = points[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.paint_uniform_color([1.0, 1.0, 1.0])  # 흰색
    return pcd, xyz.shape[0]

def create_bbox(center, size, yaw, color):
    R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw])
    box = o3d.geometry.OrientedBoundingBox(center, R, size)
    box.color = color
    return box

def create_heading_arrow(center, yaw, length=2.0):
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.05, cone_radius=0.1,
        cylinder_height=length * 0.8, cone_height=length * 0.2
    )
    arrow.paint_uniform_color([1, 0, 0])  # 빨강
    R_to_x = o3d.geometry.get_rotation_matrix_from_xyz([-np.pi / 2, 0, 0])  # +Z→+X
    arrow.rotate(R_to_x, center=(0, 0, 0))
    R_yaw = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw - np.pi/2])
    arrow.rotate(R_yaw, center=(0, 0, 0))
    arrow.translate(center)
    return arrow

def load_predictions(pkl_path):
    with open(pkl_path, 'rb') as f:
        det_annos = pickle.load(f)  # list of dict
    return {str(anno['frame_id']): anno for anno in det_annos}

def boxes_lidar_to_bev_aabb_xyxy(boxes_lidar):
    """
    boxes_lidar: (N,7) [x,y,z,dx,dy,dz,yaw]
    → (N,4) [x1,y1,x2,y2]  (회전 박스의 AABB)
    """
    out = np.zeros((len(boxes_lidar), 4), dtype=np.float32)
    for i, (x, y, z, dx, dy, dz, yaw) in enumerate(boxes_lidar):
        hx, hy = dx / 2.0, dy / 2.0
        # corners local
        corners = np.array([[ hx,  hy],
                            [ hx, -hy],
                            [-hx,  hy],
                            [-hx, -hy]], dtype=np.float32)
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s],
                      [s,  c]], dtype=np.float32)
        pts = (corners @ R.T) + np.array([x, y], dtype=np.float32)
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        out[i] = [x1, y1, x2, y2]
    return out

def iou_xyxy(a, b):
    """
    a: (4,), b: (M,4)  → IoU: (M,)
    """
    xx1 = np.maximum(a[0], b[:, 0])
    yy1 = np.maximum(a[1], b[:, 1])
    xx2 = np.minimum(a[2], b[:, 2])
    yy2 = np.minimum(a[3], b[:, 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    area_a = (a[2] - a[0]) * (a[3] - a[1]) + 1e-6
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) + 1e-6
    return inter / (area_a + area_b - inter + 1e-6)

def nms_bev_aabb(boxes_xyxy, scores, iou_thr):
    if len(boxes_xyxy) == 0:
        return []
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_xyxy(boxes_xyxy[i], boxes_xyxy[order[1:]])
        remain = np.where(ious <= iou_thr)[0]
        order = order[1:][remain]
    return keep

def visualize_frames():
    """
    기존 단일 뷰어 대신 두 개의 Open3D 창을 생성합니다.
    왼쪽 창에는 원시 포인트 클라우드만 표시하고, 오른쪽 창에는 예측
    결과(박스 및 화살표 포함)를 함께 표시합니다. 다른 기능은 기존
    구현을 유지합니다.
    """
    args = parse_args_and_apply_cfg()

    # --- Raw point cloud viewer (no key callback needed) ---
    vis_raw = o3d.visualization.Visualizer()
    if not vis_raw.create_window(window_name='LiDAR Viewer - Raw'):
        print("[ERROR] Failed to create Open3D window for raw view.")
        return

    # --- Prediction viewer with key callbacks ---
    vis_pred = o3d.visualization.VisualizerWithKeyCallback()
    if not vis_pred.create_window(window_name='LiDAR Viewer - Prediction'):
        print("[ERROR] Failed to create Open3D window for prediction view.")
        return

    # Render options for both views
    opt_raw = vis_raw.get_render_option()
    opt_raw.point_size = 1.0
    opt_raw.background_color = np.array([0.0, 0.0, 0.0])

    opt_pred = vis_pred.get_render_option()
    opt_pred.point_size = 1.0
    opt_pred.background_color = np.array([0.0, 0.0, 0.0])

    # 예측 결과 로드
    try:
        preds_by_frame = load_predictions(RESULT_PKL)
    except Exception as e:
        print(f"[ERROR] Failed to read {RESULT_PKL}: {e}")
        return

    frame_idx = 0
    last_key_time = 0
    key_delay = 0.1  # seconds
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    def debounce():
        nonlocal last_key_time
        now = time.time()
        if now - last_key_time >= key_delay:
            last_key_time = now
            return True
        return False

    def update_scene():
        # Clear old geometries
        vis_raw.clear_geometries()
        vis_pred.clear_geometries()

        # Determine current frame and point path
        frame_id = frame_ids[frame_idx]
        pc_path = os.path.join(POINTS_FOLDER, f"{frame_id}.npy")

        # Load point cloud if available
        if not os.path.exists(pc_path):
            print(f"[WARN] point file not found: {pc_path}")
            num_pts = 0
            pcd = None
        else:
            pcd, num_pts = load_npy_pointcloud(pc_path)
            # Add point cloud to both viewers
            vis_raw.add_geometry(pcd)
            vis_pred.add_geometry(pcd)

        # Prediction annotation retrieval with flexible key
        anno = preds_by_frame.get(str(frame_id))
        if anno is None:
            # Some datasets may drop leading zeros
            alt = str(frame_id).lstrip('0') or "0"
            anno = preds_by_frame.get(alt)

        num_boxes_raw = 0
        num_boxes_after_score = 0
        num_boxes_after_nms = 0

        if anno is None:
            print(f"[WARN] No prediction found for frame_id={frame_id}")
        else:
            boxes  = np.asarray(anno.get('boxes_lidar', []), dtype=np.float32)  # (N,7)
            names  = np.asarray(anno.get('name', []))
            scores = np.asarray(anno.get('score', []), dtype=np.float32)
            labels = np.asarray(anno.get('pred_labels', []), dtype=np.int32)

            num_boxes_raw = len(boxes)

            # 1) score threshold
            keep_score = np.arange(num_boxes_raw)
            if scores.size:
                keep_score = np.where(scores >= float(SCORE_THR))[0]
            boxes_f  = boxes[keep_score]
            scores_f = scores[keep_score] if scores.size else np.ones((len(keep_score),), dtype=np.float32)
            names_f  = names[keep_score]  if names.size  else []
            labels_f = labels[keep_score] if labels.size else []
            num_boxes_after_score = len(boxes_f)

            # 2) NMS filtering
            keep_nms = np.arange(len(boxes_f))
            if len(boxes_f) and NMS_IOU_THR is not None and NMS_IOU_THR > 0:
                aabbs = boxes_lidar_to_bev_aabb_xyxy(boxes_f)
                keep_nms = nms_bev_aabb(aabbs, scores_f, float(NMS_IOU_THR))
            boxes_v  = boxes_f[keep_nms]
            scores_v = scores_f[keep_nms]
            names_v  = names_f[keep_nms]  if len(names_f)  else []
            labels_v = labels_f[keep_nms] if len(labels_f) else []
            num_boxes_after_nms = len(boxes_v)

            # Add predicted boxes and heading arrows to prediction viewer only
            for i in range(len(boxes_v)):
                x, y, z, dx, dy, dz, yaw = boxes_v[i].tolist()
                if len(names_v):
                    color = CLASS_COLOR.get(str(names_v[i]), [1, 0, 0])
                elif len(labels_v):
                    color = LABEL_COLOR.get(int(labels_v[i]), [1, 0, 0])
                else:
                    color = [1, 0, 0]
                vis_pred.add_geometry(create_bbox([x, y, z], [dx, dy, dz], yaw, color))
                vis_pred.add_geometry(create_heading_arrow([x, y, z], yaw))

        # Add coordinate axes to both viewers
        vis_raw.add_geometry(coord)
        vis_pred.add_geometry(coord)

        # Apply saved viewpoint to the prediction viewer
        if os.path.exists(VIEW_FILE):
            try:
                params = o3d.io.read_pinhole_camera_parameters(VIEW_FILE)
                vis_pred.get_view_control().convert_from_pinhole_camera_parameters(params)
            except Exception:
                pass

        # Update both viewers
        vis_raw.poll_events()
        vis_raw.update_renderer()
        vis_pred.poll_events()
        vis_pred.update_renderer()

        # --- 콘솔 출력: 프레임/포인트/박스 통계 ---
        print(f"[Frame {frame_idx+1}/{len(frame_ids)}] id={frame_id} | points={num_pts} "
              f"| boxes raw={num_boxes_raw}, after_score={num_boxes_after_score}, after_nms={num_boxes_after_nms}")

    def next_frame(_):
        nonlocal frame_idx
        if debounce():
            frame_idx = (frame_idx + 1) % len(frame_ids)
            update_scene()
        return False

    def prev_frame(_):
        nonlocal frame_idx
        if debounce():
            frame_idx = (frame_idx - 1 + len(frame_ids)) % len(frame_ids)
            update_scene()
        return False

    def save_view(_):
        params = vis_pred.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(VIEW_FILE, params)
        print(f"Viewpoint saved to {VIEW_FILE}")
        return False

    def quit_viewer(_):
        print("Quitting viewer.")
        # Close both windows
        vis_raw.close()
        vis_pred.close()
        return False

    # Register key callbacks on prediction viewer
    vis_pred.register_key_callback(ord("D"), next_frame)
    vis_pred.register_key_callback(ord("A"), prev_frame)
    vis_pred.register_key_callback(ord("F"), save_view)
    vis_pred.register_key_callback(ord("Q"), quit_viewer)

    # Initialize first scene
    update_scene()

    # Run event loop (only prediction viewer needs run, raw window updates via update_scene())
    vis_pred.run()

if __name__ == "__main__":
    visualize_frames()