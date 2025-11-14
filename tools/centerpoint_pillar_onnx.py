# # export_centerpoint_pillar_onnx.py
# import os
# from pathlib import Path
# import re
# import numpy as np
# import torch
# import torch.nn as nn
# from types import SimpleNamespace

# from pcdet.config import cfg, cfg_from_yaml_file
# from pcdet.models import build_network
# from pcdet.utils import common_utils

# # === 사용자 경로 설정 ===
# CFG_PATH = 'cfgs/custom_av/centerpoint_pillar_1x_long_epoch.yaml' # 설정파일 경로

# CKPT_PATH_HINT = '../output/custom_av/centerpoint_pillar_1x_long_epoch/default/ckpt/checkpoint_epoch_80.pth' # 모델파일 경로 현재는 centerPointPillar 모델만 가능 .pth-> .onnx PFE,RPN 두가지가 나옴

# ONNX_DIR = Path('onnx_out'); ONNX_DIR.mkdir(exist_ok=True, parents=True)# 변환된 onnx 파일들이 저장되는 위치

# # ---------------------- 유틸 ----------------------
# def _find_in_dict(d, key):
#     if isinstance(d, dict):
#         if key in d: return d[key]
#         for v in d.values():
#             r = _find_in_dict(v, key)
#             if r is not None: return r
#     elif isinstance(d, (list, tuple)):
#         for v in d:
#             r = _find_in_dict(v, key)
#             if r is not None: return r
#     return None

# def _extract_voxel_and_range_from_cfg(cfg_dict):
#     voxel_size = _find_in_dict(cfg_dict, 'VOXEL_SIZE')
#     pc_range   = _find_in_dict(cfg_dict, 'POINT_CLOUD_RANGE')
#     if voxel_size is None or pc_range is None:
#         raise ValueError('yaml에서 VOXEL_SIZE / POINT_CLOUD_RANGE 를 찾지 못했습니다.')
#     return np.array(voxel_size, np.float32), np.array(pc_range, np.float32)

# def _get_pillar_limits(cfg_dict):
#     max_points = _find_in_dict(cfg_dict, 'MAX_POINTS_PER_VOXEL')
#     max_voxels = _find_in_dict(cfg_dict, 'MAX_NUMBER_OF_VOXELS')
#     num_point_features = _find_in_dict(cfg_dict, 'NUM_POINT_FEATURES')

#     if isinstance(max_voxels, dict):
#         max_voxels = int(max(max_voxels.values()))
#     elif isinstance(max_voxels, (list, tuple)):
#         max_voxels = int(max(max_voxels))
#     elif max_voxels is None:
#         max_voxels = 40000
#     else:
#         max_voxels = int(max_voxels)

#     max_points = int(max_points) if max_points is not None else 32
#     num_point_features = int(num_point_features) if num_point_features is not None else 5
#     return max_points, max_voxels, num_point_features

# def _get_num_bev_channels_from_cfg_or_model(cfg_dict, model):
#     n_yaml = _find_in_dict(cfg_dict, 'NUM_BEV_FEATURES')
#     if n_yaml is not None:
#         try: return int(n_yaml)
#         except: pass
#     mtb = getattr(model, 'map_to_bev', None)
#     if mtb is not None and hasattr(mtb, 'num_bev_features'):
#         return int(mtb.num_bev_features)
#     return 64

# def make_dummy_dataset_from_cfg(cfg):
#     voxel_size, pc_range = _extract_voxel_and_range_from_cfg(cfg)
#     vx, vy, vz = voxel_size
#     x_min, y_min, z_min, x_max, y_max, z_max = pc_range

#     nx = int(np.floor((x_max - x_min) / vx))
#     ny = int(np.floor((y_max - y_min) / vy))
#     nz = int(np.floor((z_max - z_min) / vz))
#     grid_size_xyz = np.array([nx, ny, nz], dtype=np.int64)

#     npf = _find_in_dict(cfg, 'NUM_POINT_FEATURES') or 5

#     # OpenPCDet 템플릿이 참조하는 필드를 모두 채움
#     dummy = SimpleNamespace(
#         class_names=list(cfg.CLASS_NAMES),
#         point_cloud_range=np.array(pc_range, np.float32),
#         voxel_size=np.array(voxel_size, np.float32),
#         grid_size=grid_size_xyz,
#         num_point_features=int(npf),
#         point_feature_encoder=SimpleNamespace(num_point_features=int(npf)),
#         depth_downsample_factor=1,   # 공통 템플릿에서 요구
#         mode='test'
#     )
#     return dummy

# def resolve_ckpt(path_hint: str) -> str:
#     """지정 경로가 없으면 output/**/ckpt/에서 'best' 우선, 그다음 epoch 숫자 큰 순으로 자동 선택"""
#     p = Path(path_hint)
#     if p.is_file():
#         print('[INFO] CKPT fixed:', p)
#         return str(p)

#     print('[WARN] 지정한 CKPT가 없음. 자동 탐색을 수행합니다...')
#     search_roots = [Path('output')]
#     cands = []
#     for root in search_roots:
#         if root.is_dir():
#             cands += list(root.rglob('ckpt/*.pth'))

#     if not cands:
#         raise FileNotFoundError(
#             f'체크포인트를 찾지 못했습니다. 힌트: {path_hint}\n'
#             f"다음 경로들에 .pth가 있는지 확인하세요: output/**/ckpt/"
#         )

#     # best 먼저, 없으면 epoch 숫자 큰 순
#     def score(pp: Path):
#         name = pp.name.lower()
#         m = re.search(r'epoch[_-]?(\d+)', name)
#         epoch = int(m.group(1)) if m else -1
#         is_best = int('best' in name)
#         return (is_best, epoch)  # best가 우선, 그 다음 epoch
#     cands.sort(key=score, reverse=True)
#     best = cands[0]
#     print('[INFO] CKPT resolved:', best)
#     return str(best)

# # ---------------------- 래퍼 ----------------------
# class PFEWrapper(nn.Module):
#     def __init__(self, vfe_module):
#         super().__init__()
#         self.vfe = vfe_module
#     @torch.no_grad()
#     def forward(self, voxels, num_points, coords):
#         batch_dict = {
#             'voxels': voxels,
#             'voxel_num_points': num_points,
#             'voxel_coords': coords
#         }
#         batch_dict = self.vfe(batch_dict)
#         pf = batch_dict.get('pillar_features', batch_dict.get('voxel_features'))
#         return pf

# # [PATCH] RPNWrapper: pred_dicts를 batch.forward_ret_dict에서 우선 수집 + post_processing 우회
# class RPNWrapper(nn.Module):
#     """BEV -> 2D backbone -> CenterHead 원시 예측 텐서만 뽑아내기 (NMS/후처리 우회)"""
#     def __init__(self, backbone_2d, dense_head):
#         super().__init__()
#         self.backbone_2d = backbone_2d
#         self.dense_head = dense_head
#         self.last_out_keys = []

#     # --- 유틸: 중첩 구조에서 (경로,Tensor) 재귀 수집 ---
#     def _walk_tensors(self, obj, path="", out=None):
#         if out is None: out = []
#         if isinstance(obj, torch.Tensor):
#             out.append((path, obj))
#         elif isinstance(obj, dict):
#             for k, v in obj.items():
#                 self._walk_tensors(v, f"{path}.{k}" if path else k, out)
#         elif isinstance(obj, (list, tuple)):
#             for i, v in enumerate(obj):
#                 self._walk_tensors(v, f"{path}[{i}]" if path else f"[{i}]", out)
#         return out

#     # --- 유틸: 경로 문자열에 alias가 포함된 첫 텐서 선택 ---
#     def _pick_first(self, tensor_pool, aliases):
#         aliases = [a.lower() for a in aliases]
#         for path, ten in tensor_pool:
#             p = path.lower()
#             if any(a in p for a in aliases):
#                 return ten, path
#         return None, None

#     @torch.no_grad()
#     def forward(self, bev):  # bev: [B, C, H, W]
#         # 1) 필수 키
#         batch_dict = {
#             'spatial_features': bev,
#             'batch_size': int(bev.shape[0]),
#             'mode': 'test',
#         }

#         # 2) 2D 백본
#         batch_dict = self.backbone_2d(batch_dict)
#         batch_dict['batch_size'] = int(bev.shape[0])  # 안전 재설정

#         # 3) post_processing 우회
#         orig_pp = getattr(self.dense_head, 'post_processing', None)
#         if orig_pp is not None:
#             self.dense_head.post_processing = lambda x: ([], {})

#         # 4) 헤드 실행 (반환값은 무시: in-place / forward_ret_dict에 저장)
#         _ = self.dense_head(batch_dict)

#         # 5) post_processing 복구
#         if orig_pp is not None:
#             self.dense_head.post_processing = orig_pp

#         # 6) pred_dicts 획득 (task별 딕셔너리 리스트)
#         pred_dicts = None
#         candidates = [
#             batch_dict.get('pred_dicts', None),
#             (batch_dict.get('forward_ret_dict', {}) or {}).get('pred_dicts', None)
#                 if isinstance(batch_dict.get('forward_ret_dict', None), dict) else None,
#             (getattr(self.dense_head, 'forward_ret_dict', {}) or {}).get('pred_dicts', None)
#                 if isinstance(getattr(self.dense_head, 'forward_ret_dict', None), dict) else None
#         ]
#         for cand in candidates:
#             if isinstance(cand, (list, tuple)) and len(cand) > 0:
#                 pred_dicts = cand
#                 break

#         # 7) 원시 예측 수집: task별 텐서를 채널(dim=1)로 concat
#         wanted = [
#             ('heatmap', ['heatmap', 'hm']),
#             ('reg',     ['reg']),
#             ('height',  ['height', 'z']),
#             ('dim',     ['dim', 'dims', 'whd']),
#             ('rot',     ['rot', 'rotcos', 'rotsin', 'rotation']),
#             ('vel',     ['vel', 'velocity']),
#             ('iou',     ['iou'])
#         ]

#         outs, names = [], []
#         if pred_dicts is not None:
#             for std_name, aliases in wanted:
#                 per_task = []
#                 for tdict in pred_dicts:
#                     if not isinstance(tdict, dict): 
#                         continue
#                     found = None
#                     # 정확/부분 일치 모두 시도
#                     for a in aliases:
#                         if a in tdict and isinstance(tdict[a], torch.Tensor):
#                             found = tdict[a]; break
#                         # _head_0 같은 변형 키 대응
#                         for k, v in tdict.items():
#                             if isinstance(v, torch.Tensor) and a in k:
#                                 found = v; break
#                         if found is not None: break
#                     if found is not None:
#                         per_task.append(found)
#                 if per_task:
#                     try:
#                         # 과제별 채널 concat (B, C_sum, H, W)
#                         cat = torch.cat(per_task, dim=1)
#                     except Exception:
#                         # 모양 불일치 시 첫 텐서만 사용(최소한 동작)
#                         cat = per_task[0]
#                     outs.append(cat); names.append(std_name)

#         # 8) 보강: 그래도 비었다면 전체 dict 재귀 스캔으로 보완
#         if not outs:
#             pool = []
#             pool += self._walk_tensors(batch_dict)
#             frd = getattr(self.dense_head, 'forward_ret_dict', None)
#             if isinstance(frd, dict):
#                 pool += self._walk_tensors(frd)
#             for std_name, aliases in wanted:
#                 ten, _ = self._pick_first(pool, aliases)
#                 if ten is not None:
#                     outs.append(ten); names.append(std_name)

#         # 9) 실패 시 가용 경로를 보여주며 에러
#         if not outs:
#             pool = []
#             pool += self._walk_tensors(batch_dict)
#             frd = getattr(self.dense_head, 'forward_ret_dict', None)
#             if isinstance(frd, dict):
#                 pool += self._walk_tensors(frd)
#             avail = [p for p, _ in pool]
#             raise RuntimeError(f"원시 예측 텐서를 찾지 못했습니다. 수집 경로 예시: {avail[:60]} ...")

#         self.last_out_keys = names
#         return tuple(outs) if len(outs) > 1 else outs[0]


#         # 9) 보강: forward_ret_dict / batch_dict에서도 느슨하게 탐색
#         def pick_from(src: dict, key: str):
#             if not isinstance(src, dict):
#                 return None
#             if key in src and isinstance(src[key], torch.Tensor):
#                 return src[key]
#             if f'{key}_head_0' in src and isinstance(src[f'{key}_head_0'], torch.Tensor):
#                 return src[f'{key}_head_0']
#             for kk, vv in src.items():
#                 if isinstance(vv, torch.Tensor) and key in kk:
#                     return vv
#             return None

#         for k in wanted_order:
#             if k not in found:
#                 # batch.forward_ret_dict → dense_head.forward_ret_dict → batch_dict 순으로 보강 탐색
#                 for _, src in srcs:
#                     t = pick_from(src, k)
#                     if t is not None:
#                         found[k] = t
#                         break

#         # 10) 수집 결과 정리
#         outs, names = [], []
#         for k in wanted_order:
#             if k in found:
#                 outs.append(found[k])
#                 names.append(k)

#         if not outs:
#             # 디버깅: 모든 가능한 키를 펼쳐서 보여주기
#             avail = []
#             # pred_dicts 내부 키들
#             if isinstance(pred_dicts, (list, tuple)):
#                 for i, d in enumerate(pred_dicts):
#                     if isinstance(d, dict):
#                         avail += [f'pred_dicts[{i}].{kk}' for kk in d.keys()]
#             # 각 소스 dict 키
#             for tag, src in srcs:
#                 if isinstance(src, dict):
#                     avail += [f'{tag}.{kk}' for kk in src.keys()]
#             raise RuntimeError(
#                 "원시 예측 텐서(heatmap/reg/height/dim/rot/vel/iou)를 찾지 못했습니다. "
#                 f"존재 키: {sorted(set(avail))}"
#             )

#         self.last_out_keys = names
#         return tuple(outs) if len(outs) > 1 else outs[0]





# # ---------------------- 메인 ----------------------
# def main():
#     # 스크립트 파일명을 onnx.py 로 하지 마세요. (표준 onnx 패키지와 충돌합니다)
#     torch.set_grad_enabled(False)
#     logger = common_utils.create_logger()
#     cfg_from_yaml_file(CFG_PATH, cfg)

#     dummy_ds = make_dummy_dataset_from_cfg(cfg)
#     model = build_network(
#         model_cfg=cfg.MODEL,
#         num_class=len(cfg.CLASS_NAMES),
#         dataset=dummy_ds
#     )

#     ckpt_path = resolve_ckpt(CKPT_PATH_HINT)
#     model.load_params_from_file(filename=ckpt_path, logger=logger, to_cpu=True)
#     model.cuda().eval()

#     voxel_size, pc_range = _extract_voxel_and_range_from_cfg(cfg)
#     vx, vy, vz = voxel_size
#     x_min, y_min, z_min, x_max, y_max, z_max = pc_range
#     W = int(np.floor((x_max - x_min)/vx))
#     H = int(np.floor((y_max - y_min)/vy))

#     max_points, max_voxels, num_point_feat = _get_pillar_limits(cfg)
#     num_bev_ch = _get_num_bev_channels_from_cfg_or_model(cfg, model)

#     print('[INFO] class_names   =', dummy_ds.class_names)
#     print('[INFO] pc_range      =', pc_range.tolist())
#     print('[INFO] voxel_size    =', voxel_size.tolist())
#     print('[INFO] grid(H,W)     =', H, W)
#     print('[INFO] num_bev_ch    =', num_bev_ch)
#     print('[INFO] max_voxels    =', max_voxels)
#     print('[INFO] max_points    =', max_points)
#     print('[INFO] num_pt_feat   =', num_point_feat)

#     # --- 1) PFE ONNX ---
#     pfe = PFEWrapper(model.vfe).cuda().eval()
#     voxels = torch.zeros((max_voxels, max_points, num_point_feat), dtype=torch.float32, device='cuda')
#     num_points = torch.ones((max_voxels,), dtype=torch.int32, device='cuda')
#     coords = torch.zeros((max_voxels, 4), dtype=torch.int32, device='cuda')  # (b,z,y,x)
#     ONNX_DIR.mkdir(exist_ok=True, parents=True)
#     pfe_path = str(ONNX_DIR/'pfe.onnx')
#     torch.onnx.export(
#         pfe, (voxels, num_points, coords), pfe_path,
#         input_names=['voxels','num_points','coords'],
#         output_names=['pillar_features'],
#         opset_version=17, dynamic_axes=None
#     )
#     print('[OK] export:', pfe_path)

#     # --- 2) RPN ONNX 내보내기 ---
#     rpn = RPNWrapper(model.backbone_2d, model.dense_head).cuda().eval()
#     bev = torch.zeros((1, int(num_bev_ch), H, W), dtype=torch.float32, device='cuda')
#     dummy_out = rpn(bev)

#     # 출력 이름: 래퍼가 표준 키 순서(heatmap, reg, height, dim, rot, vel, iou)로 설정
#     if isinstance(dummy_out, torch.Tensor):
#         out_names = [rpn.last_out_keys[0] if rpn.last_out_keys else 'out0']
#     else:
#         out_names = rpn.last_out_keys if rpn.last_out_keys else [f'out{i}' for i in range(len(dummy_out))]

#     rpn_path = str(ONNX_DIR/'rpn.onnx')
#     torch.onnx.export(
#         rpn, (bev,), rpn_path,
#         input_names=['bev'],
#         output_names=out_names,
#         opset_version=17, dynamic_axes=None
#     )
#     print('[OK] export:', rpn_path)

# if __name__ == '__main__':
#     main()




# export_centerpoint_pillar_onnx.py
import os
from pathlib import Path
import re
import numpy as np
import torch
import torch.nn as nn
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.utils import common_utils

# === 사용자 경로 설정 ===
CFG_PATH = 'cfgs/custom_av/centerpoint_pillar_1x_long_epoch.yaml'  # 설정 파일 경로
CKPT_PATH_HINT = '../output/custom_av/centerpoint_pillar_1x_long_epoch/default/ckpt/checkpoint_epoch_80.pth'  # ckpt 경로 힌트
ONNX_DIR = Path('onnx_out'); ONNX_DIR.mkdir(exist_ok=True, parents=True)  # ONNX 저장 폴더

# ---------------------- 유틸 ----------------------
def _find_in_dict(d, key):
    if isinstance(d, dict):
        if key in d: return d[key]
        for v in d.values():
            r = _find_in_dict(v, key)
            if r is not None: return r
    elif isinstance(d, (list, tuple)):
        for v in d:
            r = _find_in_dict(v, key)
            if r is not None: return r
    return None

def _extract_voxel_and_range_from_cfg(cfg_dict):
    voxel_size = _find_in_dict(cfg_dict, 'VOXEL_SIZE')
    pc_range   = _find_in_dict(cfg_dict, 'POINT_CLOUD_RANGE')
    if voxel_size is None or pc_range is None:
        raise ValueError('yaml에서 VOXEL_SIZE / POINT_CLOUD_RANGE 를 찾지 못했습니다.')
    return np.array(voxel_size, np.float32), np.array(pc_range, np.float32)

def _get_pillar_limits(cfg_dict):
    max_points = _find_in_dict(cfg_dict, 'MAX_POINTS_PER_VOXEL')
    max_voxels = _find_in_dict(cfg_dict, 'MAX_NUMBER_OF_VOXELS')
    num_point_features = _find_in_dict(cfg_dict, 'NUM_POINT_FEATURES')

    if isinstance(max_voxels, dict):
        max_voxels = int(max(max_voxels.values()))
    elif isinstance(max_voxels, (list, tuple)):
        max_voxels = int(max(max_voxels))
    elif max_voxels is None:
        max_voxels = 40000
    else:
        max_voxels = int(max_voxels)

    max_points = int(max_points) if max_points is not None else 32
    num_point_features = int(num_point_features) if num_point_features is not None else 5
    return max_points, max_voxels, num_point_features

def _get_num_bev_channels_from_cfg_or_model(cfg_dict, model):
    n_yaml = _find_in_dict(cfg_dict, 'NUM_BEV_FEATURES')
    if n_yaml is not None:
        try: return int(n_yaml)
        except: pass
    mtb = getattr(model, 'map_to_bev', None)
    if mtb is not None and hasattr(mtb, 'num_bev_features'):
        return int(mtb.num_bev_features)
    return 64

def make_dummy_dataset_from_cfg(cfg_obj):
    voxel_size, pc_range = _extract_voxel_and_range_from_cfg(cfg_obj)
    vx, vy, vz = voxel_size
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range

    nx = int(np.floor((x_max - x_min) / vx))
    ny = int(np.floor((y_max - y_min) / vy))
    nz = int(np.floor((z_max - z_min) / vz))
    grid_size_xyz = np.array([nx, ny, nz], dtype=np.int64)

    npf = _find_in_dict(cfg_obj, 'NUM_POINT_FEATURES') or 5

    # OpenPCDet 템플릿이 참조하는 필드 채우기
    dummy = SimpleNamespace(
        class_names=list(cfg_obj.CLASS_NAMES),
        point_cloud_range=np.array(pc_range, np.float32),
        voxel_size=np.array(voxel_size, np.float32),
        grid_size=grid_size_xyz,
        num_point_features=int(npf),
        point_feature_encoder=SimpleNamespace(num_point_features=int(npf)),
        depth_downsample_factor=1,
        mode='test'
    )
    return dummy

def resolve_ckpt(path_hint: str) -> str:
    """지정 경로가 없으면 output/**/ckpt/에서 'best' 우선, 그다음 epoch 큰 순으로 자동 선택"""
    p = Path(path_hint)
    if p.is_file():
        print('[INFO] CKPT fixed:', p)
        return str(p)

    print('[WARN] 지정한 CKPT가 없음. 자동 탐색을 수행합니다...')
    search_roots = [Path('output'), Path('../output')]
    cands = []
    for root in search_roots:
        if root.is_dir():
            cands += list(root.rglob('ckpt/*.pth'))

    if not cands:
        raise FileNotFoundError(
            f'체크포인트를 찾지 못했습니다. 힌트: {path_hint}\n'
            f"다음 경로들에 .pth가 있는지 확인하세요: output/**/ckpt/ 또는 ../output/**/ckpt/"
        )

    def score(pp: Path):
        name = pp.name.lower()
        m = re.search(r'epoch[_-]?(\d+)', name)
        epoch = int(m.group(1)) if m else -1
        is_best = int('best' in name)
        return (is_best, epoch)
    cands.sort(key=score, reverse=True)
    best = cands[0]
    print('[INFO] CKPT resolved:', best)
    return str(best)

# ---------------------- 래퍼 ----------------------
class PFEWrapper(nn.Module):
    def __init__(self, vfe_module):
        super().__init__()
        self.vfe = vfe_module
    @torch.no_grad()
    def forward(self, voxels, num_points, coords):
        batch_dict = {
            'voxels': voxels,
            'voxel_num_points': num_points,
            'voxel_coords': coords
        }
        batch_dict = self.vfe(batch_dict)
        pf = batch_dict.get('pillar_features', batch_dict.get('voxel_features'))
        return pf

class RPNWrapper(nn.Module):
    """
    BEV 특징(spatial_features)을 받아 2D 백본 + CenterHead를 거쳐
    원시 헤드 맵(heatmap/reg/height/dim/rot[/vel][/iou])을 ONNX로 내보내기 위한 래퍼.
    - post_processing 비활성화(디코딩/후처리 우회)
    - separate_head_preds / forward_ret_dict / pred_dicts(list) 등 모든 경로 탐색
    - 멀티태스크(head가 클래스 그룹별로 나뉘는 경우) 채널(dim=1) concat
    """
    def __init__(self, backbone_2d: nn.Module, dense_head: nn.Module):
        super().__init__()
        self.backbone_2d = backbone_2d
        self.dense_head = dense_head
        # 표준 키와 별칭
        self.WANT = {
            "heatmap": ["heatmap", "hm"],
            "reg":     ["reg", "offset", "ofs", "center", "centers"],
            "height":  ["height", "z"],
            "dim":     ["dim", "dims", "whd", "size"],
            "rot":     ["rot", "rotation", "rotcos", "rotsin"],
            "vel":     ["vel", "velocity"],
            "iou":     ["iou"],
        }
        self.REQUIRED = ["heatmap", "reg", "height", "dim", "rot"]
        self.last_out_keys: List[str] = []

    # ---------- 유틸 ----------
    @staticmethod
    def _to_list(x):
        return x if isinstance(x, (list, tuple)) else [x]

    @staticmethod
    def _is_tensor(x):
        return isinstance(x, torch.Tensor)

    def _get_by_alias(self, d: Dict[str, Any], aliases: List[str]):
        # 정확 키 우선
        for a in aliases:
            v = d.get(a, None)
            if self._is_tensor(v):
                return v
        # 부분 문자열 일치
        for k, v in d.items():
            if self._is_tensor(v) and any(a in k for a in aliases):
                return v
        return None

    def _cat_over_tasks(self, items: List[torch.Tensor]) -> torch.Tensor:
        if not items:
            return None
        if len(items) == 1:
            return items[0]
        try:
            return torch.cat(items, dim=1)  # (B, sumC, H, W)
        except Exception:
            # 예외적으로 (B,H,W,C) 등의 형태면 채널축 유추
            t0 = items[0]
            ch_dim = int(torch.argmax(torch.tensor(list(t0.shape))).item())
            return torch.cat(items, dim=ch_dim)

    def _gather_from_separate_head_preds(self, shp: Dict[str, List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
        outs = {}
        for name, aliases in self.WANT.items():
            if isinstance(shp, dict):
                candidates = []
                if name in shp and isinstance(shp[name], (list, tuple)):
                    candidates = [t for t in shp[name] if self._is_tensor(t)]
                else:
                    for k, v in shp.items():
                        if any(a in k for a in aliases) and isinstance(v, (list, tuple)):
                            candidates.extend([t for t in v if self._is_tensor(t)])
                if candidates:
                    outs[name] = self._cat_over_tasks(candidates)
        return outs

    def _gather_from_pred_dicts_list(self, pred_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        outs = {}
        for name, aliases in self.WANT.items():
            per_task = []
            for tdict in self._to_list(pred_dicts):
                if isinstance(tdict, dict):
                    ten = self._get_by_alias(tdict, aliases)
                    if self._is_tensor(ten):
                        per_task.append(ten)
            if per_task:
                outs[name] = self._cat_over_tasks(per_task)
        return outs

    def _gather_from_forward_ret_dict(self, frd: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        outs = {}
        for name, aliases in self.WANT.items():
            key_hits = [k for k in frd.keys() if (k == name) or any(a in k for a in aliases)]
            per_task = []
            for k in key_hits:
                v = frd[k]
                if self._is_tensor(v):
                    per_task.append(v)
                elif isinstance(v, (list, tuple)):
                    per_task.extend([t for t in v if self._is_tensor(t)])
            if per_task:
                outs[name] = self._cat_over_tasks(per_task)
        return outs

    def _merge_preferring(self, *outs_list: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        merged = {}
        for outs in outs_list:
            for k, v in (outs or {}).items():
                if k not in merged and self._is_tensor(v):
                    merged[k] = v
        return merged

    # ---------- 메인 ----------
    @torch.no_grad()
    def forward(self, bev: torch.Tensor):
        """
        입력: bev (B, C, H, W)  -> 'spatial_features'
        출력: (heatmap, reg, height, dim, rot[, vel][, iou]) (존재하는 것만 순서대로)
        """
        assert bev.dim() == 4, f"bev shape must be (B,C,H,W), got {tuple(bev.shape)}"
        B = int(bev.shape[0])

        # 1) batch_dict 준비
        batch_dict = {
            'spatial_features': bev,
            'batch_size': B,
            'mode': 'test',
        }

        # 2) 2D 백본 수행 -> batch_dict 갱신
        self.backbone_2d.eval()
        batch_dict = self.backbone_2d(batch_dict)
        batch_dict['batch_size'] = B

        # 3) post_processing 임시 우회
        orig_pp = getattr(self.dense_head, 'post_processing', None)
        if orig_pp is not None:
            self.dense_head.post_processing = lambda x: ([], {})

        # 4) 헤드 실행 (in-place 저장)
        self.dense_head.eval()
        _ = self.dense_head(batch_dict)

        # 5) 복구
        if orig_pp is not None:
            self.dense_head.post_processing = orig_pp

        # 6) 여러 경로에서 원시 예측 수집
        outs_from_separate = {}
        shp = getattr(self.dense_head, 'separate_head_preds', None)
        if isinstance(shp, dict):
            outs_from_separate = self._gather_from_separate_head_preds(shp)

        outs_from_frd = {}
        frd_head = getattr(self.dense_head, 'forward_ret_dict', None)
        if isinstance(frd_head, dict):
            outs_from_frd = self._gather_from_forward_ret_dict(frd_head)

        outs_from_batch_frd = {}
        frd_batch = batch_dict.get('forward_ret_dict', None)
        if isinstance(frd_batch, dict):
            outs_from_batch_frd = self._gather_from_forward_ret_dict(frd_batch)

        outs_from_pred_list = {}
        pred_list = None
        if isinstance(frd_head, dict):
            pred_list = frd_head.get('pred_dicts', None)
        if pred_list is None and isinstance(frd_batch, dict):
            pred_list = frd_batch.get('pred_dicts', None)
        if pred_list is None:
            pred_list = batch_dict.get('pred_dicts', None)
        if isinstance(pred_list, (list, tuple)) and len(pred_list) > 0:
            outs_from_pred_list = self._gather_from_pred_dicts_list(pred_list)

        outs = self._merge_preferring(
            outs_from_separate, outs_from_frd, outs_from_batch_frd, outs_from_pred_list
        )

        # 7) 필수 키 확인
        missing = [k for k in self.REQUIRED if k not in outs]
        if missing:
            have = list(outs.keys())
            raise RuntimeError(
                f"[RPNWrapper] 필수 출력 누락: {missing}. "
                f"현재 수집된 키: {have}. "
                f"별칭 매핑: {self.WANT}"
            )

        # 8) 표준 순서로 출력 구성
        order = ["heatmap", "reg", "height", "dim", "rot", "vel", "iou"]
        ret_keys = [k for k in order if k in outs]
        ret = [outs[k].contiguous() for k in ret_keys]
        self.last_out_keys = ret_keys
        return tuple(ret)

# ---------------------- 메인 ----------------------
def main():
    # 파일명을 onnx.py 로 하지 마세요(onnx 패키지와 충돌)
    torch.set_grad_enabled(False)
    logger = common_utils.create_logger()
    cfg_from_yaml_file(CFG_PATH, cfg)

    dummy_ds = make_dummy_dataset_from_cfg(cfg)
    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dummy_ds
    )

    ckpt_path = resolve_ckpt(CKPT_PATH_HINT)
    model.load_params_from_file(filename=ckpt_path, logger=logger, to_cpu=True)
    model.cuda().eval()

    voxel_size, pc_range = _extract_voxel_and_range_from_cfg(cfg)
    vx, vy, vz = voxel_size
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    W = int(np.floor((x_max - x_min)/vx))
    H = int(np.floor((y_max - y_min)/vy))

    max_points, max_voxels, num_point_feat = _get_pillar_limits(cfg)
    num_bev_ch = _get_num_bev_channels_from_cfg_or_model(cfg, model)

    print('[INFO] class_names   =', dummy_ds.class_names)
    print('[INFO] pc_range      =', pc_range.tolist())
    print('[INFO] voxel_size    =', voxel_size.tolist())
    print('[INFO] grid(H,W)     =', H, W)
    print('[INFO] num_bev_ch    =', num_bev_ch)
    print('[INFO] max_voxels    =', max_voxels)
    print('[INFO] max_points    =', max_points)
    print('[INFO] num_pt_feat   =', num_point_feat)

    # --- 1) PFE ONNX ---
    pfe = PFEWrapper(model.vfe).cuda().eval()
    voxels = torch.zeros((max_voxels, max_points, num_point_feat), dtype=torch.float32, device='cuda')
    num_points = torch.ones((max_voxels,), dtype=torch.int32, device='cuda')
    coords = torch.zeros((max_voxels, 4), dtype=torch.int32, device='cuda')  # (b,z,y,x)
    pfe_path = str(ONNX_DIR/'pfe.onnx')
    torch.onnx.export(
        pfe, (voxels, num_points, coords), pfe_path,
        input_names=['voxels','num_points','coords'],
        output_names=['pillar_features'],
        opset_version=17, dynamic_axes=None
    )
    print('[OK] export:', pfe_path)

    # --- 2) RPN ONNX ---
    rpn = RPNWrapper(model.backbone_2d, model.dense_head).cuda().eval()
    bev = torch.zeros((1, int(num_bev_ch), H, W), dtype=torch.float32, device='cuda')

    with torch.no_grad():
        dummy_out = rpn(bev)
        out_keys = rpn.last_out_keys

    # 표준 순서 고정(존재하는 것만)
    std_order = ['heatmap','reg','height','dim','rot','vel','iou']
    out_names = [k for k in std_order if k in out_keys]
    if ('reg' not in out_names):
        print('[WARN] RPN 원시 출력에 reg 가 포함되지 않았습니다. 별칭/키를 확인하세요.', out_keys)

    rpn_path = str(ONNX_DIR/'rpn.onnx')
    torch.onnx.export(
        rpn, (bev,), rpn_path,
        input_names=['bev'],
        output_names=out_names,
        opset_version=17, dynamic_axes=None
    )
    print('[OK] export:', rpn_path, '-> outputs:', out_names)

if __name__ == '__main__':
    main()
