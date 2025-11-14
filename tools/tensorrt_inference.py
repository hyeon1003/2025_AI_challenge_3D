# # run_trt_centerpoint_pillar.py
# import os, math, numpy as np, tensorrt as trt
# import pycuda.driver as cuda; import pycuda.autoinit  # noqa: F401
# import torch

# # ==== 고정 파라미터 (당신의 YAML과 ONNX가정에 맞춤) ====
# VOXEL_SIZE = np.array([0.25, 0.25, 8.0], np.float32)
# PC_RANGE   = np.array([-70.0, -70.0, -4.0,  70.0,  70.0,  4.0], np.float32)
# MAX_VOXELS, MAX_POINTS_PER_VOXEL, NUM_POINT_FEATURES = 150000, 20, 4
# NUM_BEV_CH, H, W = 64, 560, 560

# PFE_ENGINE = "engines/pfe_fp16.engine"
# RPN_ENGINE = "engines/rpn_fp16.engine"

# # ==== 간단한 numpy 하드-보xelizer (B=1) ====
# def voxelize(points: np.ndarray):
#     # points: [N,4] in (x,y,z,intensity)
#     x_min, y_min, z_min, x_max, y_max, z_max = PC_RANGE
#     vx, vy, vz = VOXEL_SIZE
#     mask = (
#         (points[:,0] >= x_min) & (points[:,0] < x_max) &
#         (points[:,1] >= y_min) & (points[:,1] < y_max) &
#         (points[:,2] >= z_min) & (points[:,2] < z_max)
#     )
#     pts = points[mask]
#     if pts.shape[0] == 0:
#         raise RuntimeError("Input point cloud empty after range filter.")

#     ix = np.floor((pts[:,0] - x_min) / vx).astype(np.int32)
#     iy = np.floor((pts[:,1] - y_min) / vy).astype(np.int32)
#     # pillar 네트는 z축을 1셀로 가정
#     iz = np.zeros_like(ix, dtype=np.int32)

#     # (y,x) 키로 버킷
#     keys = iy * W + ix
#     order = np.argsort(keys)
#     keys = keys[order]; pts = pts[order]

#     voxels = np.zeros((MAX_VOXELS, MAX_POINTS_PER_VOXEL, NUM_POINT_FEATURES), np.float32)
#     num_points = np.zeros((MAX_VOXELS,), np.int32)
#     coords = np.zeros((MAX_VOXELS, 4), np.int32)  # (b,z,y,x)

#     cur_key, vcount, pcount = -1, -1, 0
#     for k, p in zip(keys, pts):
#         if k != cur_key:
#             vcount += 1
#             if vcount >= MAX_VOXELS: break
#             cur_key = k; pcount = 0
#             y = k // W; x = k % W
#             coords[vcount] = np.array([0, 0, y, x], np.int32)  # b=0, z=0
#         if pcount < MAX_POINTS_PER_VOXEL:
#             voxels[vcount, pcount, :] = p.astype(np.float32)
#             pcount += 1; num_points[vcount] = pcount

#     used = max(0, vcount + 1)
#     if used == 0:
#         raise RuntimeError("No voxels created (all filtered or MAX_VOXELS=0).")

#     return voxels[:used], num_points[:used], coords[:used]

# # ==== TensorRT 간단 래퍼 ====
# class TRTModule:
#     def __init__(self, engine_path):
#         logger = trt.Logger(trt.Logger.ERROR)
#         with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
#             self.engine = runtime.deserialize_cuda_engine(f.read())
#         self.context = self.engine.create_execution_context()
#         self.bindings = [None] * self.engine.num_bindings
#         self.device_mem = {}
#         self.host_mem = {}
#         self.stream = cuda.Stream()

#     def _alloc_like(self, name, shape, dtype=np.float32):
#         nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
#         if name not in self.device_mem:
#             dptr = cuda.mem_alloc(nbytes)
#             self.device_mem[name] = dptr
#         # 호스트 버퍼는 I/O 때마다 새로 만들어도 무방
#         self.host_mem[name] = np.empty(shape, dtype=dtype)
#         bidx = self.engine.get_binding_index(name)
#         self.bindings[bidx] = int(self.device_mem[name])
#         return self.host_mem[name]

#     def infer(self, feeds: dict, fetches: list):
#         # 입력 복사
#         for name, arr in feeds.items():
#             arr = np.ascontiguousarray(arr)
#             h = self._alloc_like(name, arr.shape, arr.dtype)
#             np.copyto(h, arr)
#             cuda.memcpy_htod_async(self.device_mem[name], h, self.stream)

#         # 출력 준비(엔진 바인딩 shape 고정 가정)
#         out_host = {}
#         for name in fetches:
#             bidx = self.engine.get_binding_index(name)
#             shape = tuple(self.engine.get_binding_shape(bidx))
#             dtype = np.float32  # FP16 엔진이라도 바인딩 dtype은 FP32로 오는 경우가 많음
#             h = self._alloc_like(name, shape, dtype)
#             out_host[name] = h

#         # 실행
#         self.context.execute_async_v2(self.bindings, self.stream.handle)
#         # 출력 복사
#         for name in fetches:
#             cuda.memcpy_dtoh_async(out_host[name], self.device_mem[name], self.stream)
#         self.stream.synchronize()
#         return out_host

# # ==== BEV scatter (GPU, 벡터화) ====
# def scatter_pillars_to_bev(pillar_feats: np.ndarray, coords: np.ndarray):
#     # pillar_feats: [Nv, 64] (float32, host) → CUDA 텐서로 올려서 scatter
#     feats = torch.from_numpy(pillar_feats).cuda()          # [Nv, C]
#     crd   = torch.from_numpy(coords).cuda()                # [Nv, 4] (b,z,y,x)
#     y, x = crd[:,2].long(), crd[:,3].long()
#     lin = (y * W + x).long()                               # [Nv]

#     bev_flat = torch.zeros((NUM_BEV_CH, H*W), device='cuda', dtype=feats.dtype)
#     idx = lin.unsqueeze(0).expand(NUM_BEV_CH, lin.numel()) # [C,Nv]
#     bev_flat.scatter_add_(1, idx, feats.t())               # (C,Nv) -> (C, HW)
#     bev = bev_flat.view(NUM_BEV_CH, H, W).unsqueeze(0)     # [1,C,H,W]
#     return bev

# def main():
#     import argparse
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--points", type=str, required=False, help="N×4 .npy point cloud")
#     args = ap.parse_args()

#     # 1) 입력 포인트 로드(+voxelize)
#     if args.points and os.path.isfile(args.points):
#         pts = np.load(args.points).astype(np.float32)  # [N,4]
#     else:
#         # 데모용 랜덤 포인트(실전에선 꼭 실제 .npy 사용)
#         N = 200000
#         rng = np.random.default_rng(0)
#         xs = rng.uniform(PC_RANGE[0], PC_RANGE[3], size=N)
#         ys = rng.uniform(PC_RANGE[1], PC_RANGE[4], size=N)
#         zs = rng.uniform(PC_RANGE[2], PC_RANGE[5], size=N)
#         is_ = rng.uniform(0.0, 1.0, size=N)
#         pts = np.stack([xs, ys, zs, is_], axis=1).astype(np.float32)

#     voxels, num_points, coords = voxelize(pts)
#     # pad to MAX for 엔진 바인딩 고정 대응
#     used = voxels.shape[0]
#     vox_pad = np.zeros((MAX_VOXELS, MAX_POINTS_PER_VOXEL, NUM_POINT_FEATURES), np.float32)
#     np_pts  = np.zeros((MAX_VOXELS,), np.int32)
#     crd_pad = np.zeros((MAX_VOXELS, 4), np.int32)
#     vox_pad[:used] = voxels; np_pts[:used] = num_points; crd_pad[:used] = coords

#     # 2) PFE TRT 실행
#     pfe = TRTModule(PFE_ENGINE)
#     pfe_out = pfe.infer(
#         {
#             "voxels": vox_pad,
#             "num_points": np_pts,
#             "coords": crd_pad,
#         },
#         fetches=["pillar_features"],
#     )
#     pillar_features = pfe_out["pillar_features"][:used, :]  # [Nv,64]

#     # 3) BEV scatter (CUDA / PyTorch)
#     bev = scatter_pillars_to_bev(pillar_features, coords)    # [1,64,560,560]

#     # 4) RPN TRT 실행
#     rpn = TRTModule(RPN_ENGINE)
#     rpn_out = rpn.infer(
#         {"bev": bev.detach().cpu().numpy()},
#         fetches=["heatmap","reg","height","dim","rot"]
#     )
#     for k,v in rpn_out.items():
#         print(f"{k}: {v.shape}")

#     # (옵션) 상위 픽 몇 개만 확인
#     heat = torch.from_numpy(rpn_out["heatmap"]).sigmoid()     # [1,3,H,W]
#     topk = torch.topk(heat.view(1, 3, -1), k=5, dim=-1).values.cpu().numpy()
#     print("top-heat logits(sigmoid):", topk)

# if __name__ == "__main__":
#     main()


# tensorrt_inference.py
import os
import time
import numpy as np
import tensorrt as trt
import torch

# ==== 모델/데이터 파라미터 ====
VOXEL_SIZE = np.array([0.25, 0.25, 8.0], np.float32)
PC_RANGE   = np.array([-70.0, -70.0, -4.0, 70.0, 70.0, 4.0], np.float32)
MAX_VOXELS, MAX_POINTS_PER_VOXEL, NUM_POINT_FEATURES = 150000, 20, 4
NUM_BEV_CH, H, W = 64, 560, 560

PFE_ENGINE = "engines/pfe_fp16.engine"
RPN_ENGINE = "engines/rpn_fp16.engine"

# ----- 간단한 voxelizer (B=1) -----
def voxelize(points: np.ndarray):
    x_min, y_min, z_min, x_max, y_max, z_max = PC_RANGE
    vx, vy, vz = VOXEL_SIZE
    m = (
        (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] < y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] < z_max)
    )
    pts = points[m].astype(np.float32)
    if pts.shape[0] == 0:
        raise RuntimeError("No points in range.")

    ix = np.floor((pts[:, 0] - x_min) / vx).astype(np.int32)
    iy = np.floor((pts[:, 1] - y_min) / vy).astype(np.int32)
    keys = iy * W + ix
    order = np.argsort(keys)
    keys, pts = keys[order], pts[order]

    voxels = np.zeros((MAX_VOXELS, MAX_POINTS_PER_VOXEL, NUM_POINT_FEATURES), np.float32)
    num_points = np.zeros((MAX_VOXELS,), np.int32)
    coords = np.zeros((MAX_VOXELS, 4), np.int32)  # (b,z,y,x)

    cur_key, vcount, pcount = -1, -1, 0
    for k, p in zip(keys, pts):
        if k != cur_key:
            vcount += 1
            if vcount >= MAX_VOXELS:
                break
            cur_key = k
            pcount = 0
            y = k // W
            x = k % W
            coords[vcount] = (0, 0, y, x)
        if pcount < MAX_POINTS_PER_VOXEL:
            voxels[vcount, pcount, :] = p
            pcount += 1
            num_points[vcount] = pcount

    used = max(0, vcount + 1)
    if used == 0:
        raise RuntimeError("No voxels created.")
    return voxels[:used], num_points[:used], coords[:used]

# ----- dtype 매핑 -----
def trt_dtype_to_torch(dt: trt.DataType):
    if dt == trt.DataType.FLOAT: return torch.float32
    if dt == trt.DataType.HALF:  return torch.float16
    if dt == trt.DataType.INT32: return torch.int32
    raise ValueError(f"Unsupported TRT dtype: {dt}")

# ==============================================
# TensorRT 8/9/10 호환 래퍼 (Torch 텐서 포인터 사용)
# ==============================================
class TRTModuleCompat:
    def __init__(self, engine_path: str):
        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # TRT10 여부 감지(신 API)
        self.is_trt10 = hasattr(self.engine, "num_io_tensors") and hasattr(self.context, "set_tensor_address")
        # TRT8/9 여부 감지(구 API)
        self.is_legacy = hasattr(self.engine, "num_bindings") and hasattr(self.context, "execute_v2")

        if not (self.is_trt10 or self.is_legacy):
            raise RuntimeError("Unsupported TensorRT Python API version.")

        if self.is_trt10:
            # IO 텐서 메타 수집
            self.io_desc = []
            for i in range(self.engine.num_io_tensors):
                name  = self.engine.get_tensor_name(i)
                mode  = self.engine.get_tensor_mode(name)  # INPUT / OUTPUT
                dtype = self.engine.get_tensor_dtype(name)
                shape = tuple(self.engine.get_tensor_shape(name))
                self.io_desc.append((name, mode, dtype, shape))
            self.dev_tensors = {}  # name -> torch.cuda.Tensor
        else:
            # 레거시: 바인딩 인덱스 기반
            self.num_bindings = self.engine.num_bindings
            self.bindings_ptrs = [0] * self.num_bindings
            self.name_of = [self.engine.get_binding_name(i) for i in range(self.num_bindings)]
            self.is_input = [self.engine.binding_is_input(i) for i in range(self.num_bindings)]
            self.dtype_of = [self.engine.get_binding_dtype(i) for i in range(self.num_bindings)]
            self.dev_tensors = {}  # name -> torch.cuda.Tensor

    def _gpu_time_ms(self, fn):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)  # ms

    def _current_stream_handle(self) -> int:
        """PyTorch CUDA stream 핸들을 int로 추출."""
        s = torch.cuda.current_stream()
        try:
            return int(s.cuda_stream)  # PyTorch 2.x
        except Exception:
            # 예비 경로
            return int(torch.cuda.current_stream().cuda_stream)

    def infer(self, feeds: dict, fetches: list, to_numpy=True, measure=True):
        wall_t0 = time.perf_counter()

        if self.is_trt10:
            # ----- 입력 설정 -----
            for name, mode, dtype, _ in self.io_desc:
                if mode == trt.TensorIOMode.INPUT:
                    if name not in feeds:
                        raise KeyError(f"Missing input '{name}'")
                    arr = feeds[name]
                    tdtype = trt_dtype_to_torch(dtype)
                    xt = torch.as_tensor(arr, device="cuda", dtype=tdtype).contiguous()
                    self.dev_tensors[name] = xt
                    self.context.set_input_shape(name, tuple(xt.shape))
                    self.context.set_tensor_address(name, xt.data_ptr())

            # ----- 출력 버퍼 준비 -----
            for name, mode, dtype, _ in self.io_desc:
                if mode == trt.TensorIOMode.OUTPUT:
                    # 동적일 수 있으므로 context에서 shape 조회
                    real_shape = tuple(self.context.get_tensor_shape(name))
                    tdtype = trt_dtype_to_torch(dtype)
                    yt = torch.empty(real_shape, device="cuda", dtype=tdtype)
                    self.dev_tensors[name] = yt
                    self.context.set_tensor_address(name, yt.data_ptr())

            # ----- 실행 (TRT10: execute_v3 또는 execute_async_v3 폴백) -----
            if hasattr(self.context, "execute_v3"):
                run_fn = lambda: self.context.execute_v3()
                if measure:
                    gpu_ms = self._gpu_time_ms(run_fn)
                else:
                    ok = run_fn()
                    if not ok: raise RuntimeError("TRT execute_v3 failed.")
                    gpu_ms = None
            elif hasattr(self.context, "execute_async_v3"):
                stream_handle = self._current_stream_handle()
                run_fn = lambda: self.context.execute_async_v3(stream_handle=stream_handle)
                if measure:
                    gpu_ms = self._gpu_time_ms(run_fn)
                else:
                    ok = run_fn()
                    if not ok: raise RuntimeError("TRT execute_async_v3 failed.")
                    gpu_ms = None
            else:
                raise RuntimeError("Neither execute_v3 nor execute_async_v3 is available in this TRT build.")

        else:
            # ----- 입력 바인딩 설정 -----
            for i in range(self.num_bindings):
                if self.is_input[i]:
                    name = self.name_of[i]
                    if name not in feeds:
                        raise KeyError(f"Missing input '{name}'")
                    arr = feeds[name]
                    tdtype = trt_dtype_to_torch(self.dtype_of[i])
                    xt = torch.as_tensor(arr, device="cuda", dtype=tdtype).contiguous()
                    self.dev_tensors[name] = xt
                    try:
                        self.context.set_binding_shape(i, tuple(xt.shape))
                    except Exception:
                        pass
                    self.bindings_ptrs[i] = xt.data_ptr()

            # ----- 출력 shape 확인 후 버퍼 할당 -----
            for i in range(self.num_bindings):
                if not self.is_input[i]:
                    name = self.name_of[i]
                    dtype = self.dtype_of[i]
                    shape = tuple(self.context.get_binding_shape(i))
                    if any(s < 0 for s in shape):
                        shape = tuple(self.engine.get_binding_shape(i))
                    tdtype = trt_dtype_to_torch(dtype)
                    yt = torch.empty(shape, device="cuda", dtype=tdtype)
                    self.dev_tensors[name] = yt
                    self.bindings_ptrs[i] = yt.data_ptr()

            # ----- 실행 (TRT8/9) -----
            def _run():
                ok = self.context.execute_v2(self.bindings_ptrs)
                if not ok:
                    raise RuntimeError("TRT execute_v2 failed.")
            gpu_ms = self._gpu_time_ms(_run) if measure else (_run() or None)

        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - wall_t0) * 1000.0

        # ----- 결과 -----
        outs = {}
        for name in fetches:
            t = self.dev_tensors[name]
            outs[name] = t.detach().cpu().numpy() if to_numpy else t
        return outs, gpu_ms, wall_ms

# ----- BEV scatter -----
def scatter_pillars_to_bev(pillar_feats: torch.Tensor, coords: torch.Tensor):
    # pillar_feats: [Nv, C], coords: [Nv, 4] (b,z,y,x)
    y, x = coords[:, 2].long(), coords[:, 3].long()
    lin = (y * W + x).long()
    bev_flat = torch.zeros((NUM_BEV_CH, H * W), device='cuda', dtype=pillar_feats.dtype)
    idx = lin.unsqueeze(0).expand(NUM_BEV_CH, lin.numel())
    bev_flat.scatter_add_(1, idx, pillar_feats.t())
    return bev_flat.view(NUM_BEV_CH, H, W).unsqueeze(0)  # [1,64,560,560]

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=str, help="N×4 .npy point cloud")
    args = ap.parse_args()

    print("[INFO] TensorRT Python version:", trt.__version__)

    # 1) 입력 포인트
    if args.points and os.path.isfile(args.points):
        pts = np.load(args.points).astype(np.float32)
    else:
        # 데모용 랜덤 포인트
        N = 200000
        rng = np.random.default_rng(0)
        xs = rng.uniform(PC_RANGE[0], PC_RANGE[3], size=N)
        ys = rng.uniform(PC_RANGE[1], PC_RANGE[4], size=N)
        zs = rng.uniform(PC_RANGE[2], PC_RANGE[5], size=N)
        is_ = rng.uniform(0.0, 1.0, size=N)
        pts = np.stack([xs, ys, zs, is_], axis=1).astype(np.float32)

    voxels, num_points, coords = voxelize(pts)

    # pad → 엔진 입력 고정크기와 동일
    used = voxels.shape[0]
    vox_pad = np.zeros((MAX_VOXELS, MAX_POINTS_PER_VOXEL, NUM_POINT_FEATURES), np.float32)
    np_pts  = np.zeros((MAX_VOXELS,), np.int32)
    crd_pad = np.zeros((MAX_VOXELS, 4), np.int32)
    vox_pad[:used], np_pts[:used], crd_pad[:used] = voxels, num_points, coords

    # 2) PFE
    pfe = TRTModuleCompat(PFE_ENGINE)
    pfe_out, pfe_gpu_ms, pfe_wall_ms = pfe.infer(
        {"voxels": vox_pad, "num_points": np_pts, "coords": crd_pad},
        fetches=["pillar_features"],
        to_numpy=False, measure=True
    )
    pillar_all = pfe_out["pillar_features"]        # [MAX_VOXELS, 64] (cuda, fp16/32)
    pillar = pillar_all[:used, :]                  # [Nv, 64]
    coords_cuda = torch.as_tensor(coords, device='cuda', dtype=torch.int32)

    # 3) BEV scatter
    t0 = time.perf_counter()
    bev = scatter_pillars_to_bev(pillar, coords_cuda)  # [1,64,560,560]
    torch.cuda.synchronize()
    bev_ms = (time.perf_counter() - t0) * 1000.0

    # 4) RPN
    rpn = TRTModuleCompat(RPN_ENGINE)
    rpn_out, rpn_gpu_ms, rpn_wall_ms = rpn.infer(
        {"bev": bev},
        fetches=["heatmap", "reg", "height", "dim", "rot"],
        to_numpy=True, measure=True
    )

    # 결과 출력
    for k, v in rpn_out.items():
        print(f"{k}: {v.shape}")
    heat = torch.from_numpy(rpn_out["heatmap"]).sigmoid()
    top_vals = torch.topk(heat.view(1, 3, -1), k=5, dim=-1).values.cpu().numpy()
    print("top-heat(sigmoid) 5 vals:", top_vals)

    print(f"[TIME] PFE GPU {pfe_gpu_ms:.3f} ms | host {pfe_wall_ms:.3f} ms")
    print(f"[TIME] BEV scatter host {bev_ms:.3f} ms")
    print(f"[TIME] RPN GPU {rpn_gpu_ms:.3f} ms | host {rpn_wall_ms:.3f} ms")
    total_ms = pfe_wall_ms + bev_ms + rpn_wall_ms
    print(f"[TIME] TOTAL host ~{total_ms:.3f} ms  (≈ {1000.0/total_ms:.1f} FPS)")

if __name__ == "__main__":
    main()
