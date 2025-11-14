# # point_range_robust.py



# import pickle, numpy as np, os

# INFO_T = '/home/linux/OpenPCDet/data/custom_av/custom_av_infos_train.pkl'
# INFO_V = '/home/linux/OpenPCDet/data/custom_av/custom_av_infos_val.pkl'
# CLASSES = {'Vehicle', 'Pedestrian', 'Cyclist'}   
# P_LO, P_HI = 0.5, 99.5                           # 퍼센타일 컷
# Z_CLAMP = (-3.5, 4.0)                            # 권장 Z 범위(필요시 살짝 조절)
# XY_MARGIN = 1.0                                   # XY 여유

# def collect_boxes(pkl):
#     if not os.path.exists(pkl): return np.empty((0,7)), []
#     infos = pickle.load(open(pkl, 'rb'))
#     all_boxes, all_names = [], []
#     for it in infos:
#         ann = it.get('annos', {})
#         boxes = ann.get('gt_boxes_lidar', None)      #[x,y,z,dx,dy,dz,yaw]
#         names = ann.get('name', None)                 
#         if boxes is None or names is None: 
#             continue
#         # 우리가 쓰는 클래스만 필터
#         mask = np.array([n in CLASSES for n in names], dtype=bool)
#         if mask.any():
#             all_boxes.append(boxes[mask])
#             all_names.extend([n for n in names[mask]])
#     if not all_boxes:
#         return np.empty((0,7)), []
#     return np.concatenate(all_boxes, axis=0), all_names

# def robust_range():
#     b1, _ = collect_boxes(INFO_T)
#     b2, _ = collect_boxes(INFO_V)
#     boxes = np.concatenate([b1,b2], axis=0) if b2.size else b1
#     if boxes.size == 0:
#         raise RuntimeError("No GT boxes found for selected classes.")

#     # 박스 중심과 크기 min/max 근사
#     centers = boxes[:, :3]
#     dims    = boxes[:, 3:6]
#     half    = dims / 2
#     mins    = centers - half
#     maxs    = centers + half

#     # 이상치 억제
#     lo = np.percentile(mins, P_LO, axis=0)   # (x_min, y_min, z_min) 쪽
#     hi = np.percentile(maxs, P_HI, axis=0)   # (x_max, y_max, z_max) 쪽

#     # XY는 대칭으로 통일 
#     x_abs = max(abs(lo[0]), abs(hi[0])) + XY_MARGIN
#     y_abs = max(abs(lo[1]), abs(hi[1])) + XY_MARGIN

#     z_min = max(lo[2] - 0.5, Z_CLAMP[0])    
#     z_max = min(hi[2] + 0.5, Z_CLAMP[1])

#     rec = np.array([-x_abs, -y_abs, z_min, x_abs, y_abs, z_max], dtype=float)
#     return rec

# if __name__ == "__main__":
#     rec = robust_range()
#     print("권장 POINT_CLOUD_RANGE:", np.round(rec, 2))


# quick_range_coverage_check.py
import pickle, numpy as np

info_pkl = '/home/linux/OpenPCDet/data/custom_av/custom_av_infos_train.pkl'
# 새 범위 입력
rng = np.array([-70.80, -70.0, -4.0, 70.80, 70.80, 4.0])  # 예시(0.08*8 스냅)
lo, hi = rng[:3], rng[3:]

tot = {'Vehicle':0,'Pedestrian':0,'Cyclist':0}
miss = {'Vehicle':0,'Pedestrian':0,'Cyclist':0}

infos = pickle.load(open(info_pkl,'rb'))
for info in infos:
    annos = info['annos']
    boxes = annos['gt_boxes_lidar']      # [x,y,z,dx,dy,dz,heading]
    names = annos['name']                # 클래스 문자열 리스트
    c = boxes[:, :3]; half = boxes[:,3:6]/2
    bmin = c - half; bmax = c + half
    inside = np.all((bmin>=lo)&(bmax<=hi), axis=1)
    for cls, ok in zip(names, inside):
        if cls in tot:
            tot[cls]+=1
            miss[cls]+= (not ok)

print("총/범위밖/비율(%) =", {k:(tot[k], miss[k], round(100*miss[k]/max(1,tot[k]),2)) for k in tot})
