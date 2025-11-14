from functools import partial

import numpy as np
from PIL import Image

from ...utils import common_utils
from . import augmentor_utils, database_sampler


class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger

        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def disable_augmentation(self, augmentor_configs):
        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)
             
    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            gt_boxes, points, enable = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points, return_flip=True
            )
            data_dict['flip_%s'%cur_axis] = enable
            if 'roi_boxes' in data_dict.keys():
                num_frame, num_rois,dim = data_dict['roi_boxes'].shape
                roi_boxes, _, _ = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                data_dict['roi_boxes'].reshape(-1,dim), np.zeros([1,3]), return_flip=True, enable=enable
                )
                data_dict['roi_boxes'] = roi_boxes.reshape(num_frame, num_rois,dim)

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points, noise_rot = augmentor_utils.global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range, return_rot=True
        )
        if 'roi_boxes' in data_dict.keys():
            num_frame, num_rois,dim = data_dict['roi_boxes'].shape
            roi_boxes, _, _ = augmentor_utils.global_rotation(
            data_dict['roi_boxes'].reshape(-1, dim), np.zeros([1, 3]), rot_range=rot_range, return_rot=True, noise_rotation=noise_rot)
            data_dict['roi_boxes'] = roi_boxes.reshape(num_frame, num_rois,dim)

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['noise_rot'] = noise_rot
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        
        if 'roi_boxes' in data_dict.keys():
            gt_boxes, roi_boxes, points, noise_scale = augmentor_utils.global_scaling_with_roi_boxes(
                data_dict['gt_boxes'], data_dict['roi_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE'], return_scale=True
            )
            data_dict['roi_boxes'] = roi_boxes
        else:
            gt_boxes, points, noise_scale = augmentor_utils.global_scaling(
                data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE'], return_scale=True
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['noise_scale'] = noise_scale
        return data_dict

    def random_image_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_image_flip, config=config)
        images = data_dict["images"]
        depth_maps = data_dict["depth_maps"]
        gt_boxes = data_dict['gt_boxes']
        gt_boxes2d = data_dict["gt_boxes2d"]
        calib = data_dict["calib"]
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['horizontal']
            images, depth_maps, gt_boxes = getattr(augmentor_utils, 'random_image_flip_%s' % cur_axis)(
                images, depth_maps, gt_boxes, calib,
            )

        data_dict['images'] = images
        data_dict['depth_maps'] = depth_maps
        data_dict['gt_boxes'] = gt_boxes
        return data_dict

    def random_world_translation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_translation, config=config)
        noise_translate_std = config['NOISE_TRANSLATE_STD']
        assert len(noise_translate_std) == 3
        noise_translate = np.array([
            np.random.normal(0, noise_translate_std[0], 1),
            np.random.normal(0, noise_translate_std[1], 1),
            np.random.normal(0, noise_translate_std[2], 1),
        ], dtype=np.float32).T

        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        points[:, :3] += noise_translate
        gt_boxes[:, :3] += noise_translate
                
        if 'roi_boxes' in data_dict.keys():
            data_dict['roi_boxes'][:, :3] += noise_translate
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['noise_translate'] = noise_translate
        return data_dict

    def random_local_translation(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_translation, config=config)
        offset_range = config['LOCAL_TRANSLATION_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y', 'z']
            gt_boxes, points = getattr(augmentor_utils, 'random_local_translation_along_%s' % cur_axis)(
                gt_boxes, points, offset_range,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_local_rotation(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_rotation, config=config)
        rot_range = config['LOCAL_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points = augmentor_utils.local_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_local_scaling(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_scaling, config=config)
        gt_boxes, points = augmentor_utils.local_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['LOCAL_SCALE_RANGE']
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_frustum_dropout(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_world_frustum_dropout, config=config)

        intensity_range = config['INTENSITY_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for direction in config['DIRECTION']:
            assert direction in ['top', 'bottom', 'left', 'right']
            gt_boxes, points = getattr(augmentor_utils, 'global_frustum_dropout_%s' % direction)(
                gt_boxes, points, intensity_range,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_local_frustum_dropout(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_frustum_dropout, config=config)

        intensity_range = config['INTENSITY_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for direction in config['DIRECTION']:
            assert direction in ['top', 'bottom', 'left', 'right']
            gt_boxes, points = getattr(augmentor_utils, 'local_frustum_dropout_%s' % direction)(
                gt_boxes, points, intensity_range,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_local_pyramid_aug(self, data_dict=None, config=None):
        """
        Refer to the paper:
            SE-SSD: Self-Ensembling Single-Stage Object Detector From Point Cloud
        """
        if data_dict is None:
            return partial(self.random_local_pyramid_aug, config=config)

        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']

        gt_boxes, points, pyramids = augmentor_utils.local_pyramid_dropout(gt_boxes, points, config['DROP_PROB'])
        gt_boxes, points, pyramids = augmentor_utils.local_pyramid_sparsify(gt_boxes, points,
                                                                            config['SPARSIFY_PROB'],
                                                                            config['SPARSIFY_MAX_NUM'],
                                                                            pyramids)
        gt_boxes, points = augmentor_utils.local_pyramid_swap(gt_boxes, points,
                                                                 config['SWAP_PROB'],
                                                                 config['SWAP_MAX_NUM'],
                                                                 pyramids)
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    
    # 추가한 기법
    def random_points_dropout(self, data_dict=None, config=None):
        """
        프레임 전역 포인트를 확률적으로 줄이되,
        현재 포인트 수(N)가 ENSURE_MIN_KEEP보다 적을 때도 안전하게 동작하도록 보강.
        config:
        DROP_RATE: float in [0,1)
        ENSURE_MIN_KEEP: int
        PROB: float in [0,1]
        """
        if data_dict is None:
            return partial(self.random_points_dropout, config=config)

        prob = float(config.get('PROB', 1.0))
        if np.random.rand() > prob:
            return data_dict

        pts = data_dict.get('points', None)
        if pts is None:
            return data_dict
        N = pts.shape[0]
        if N == 0:
            return data_dict

        drop_rate = float(config.get('DROP_RATE', 0.3))
        min_keep  = int(config.get('ENSURE_MIN_KEEP', 0))
        if drop_rate <= 0:
            return data_dict

        # 남길 개수 = max(ENSURE_MIN_KEEP, (1 - drop_rate) * N)
        target_keep = max(min_keep, int(round((1.0 - drop_rate) * N)))
        # N보다 많이 남길 수는 없음
        target_keep = min(target_keep, N)

        # 남길 개수가 N과 같으면 실질적으로 드롭할 게 없음 → 원본 유지
        if target_keep >= N:
            return data_dict

        # target_keep개 만큼만 무작위로 "남길" 인덱스를 뽑는다
        keep_idx = np.random.choice(N, size=target_keep, replace=False)
        data_dict['points'] = pts[keep_idx]
        return data_dict

    def line_downsample(self, data_dict=None, config=None):
        """
        적응형 라인 다운샘플: 프레임의 '현재 수직 라인 수'를 추정해 TARGET_LINES에 맞춰 줄임.
        config:
        TARGET_LINES: int         # 목표 라인수 (예: 64)
        LINE_BINS: int            # 수직 각도 bin 수 (예: 128)
        TOLERANCE: int            # 허용 오차 (예: 4 -> 60~68 사이는 스킵)
        MIN_BIN_OCCUPANCY: int    # 한 bin을 '점유'로 볼 최소 포인트 수 (노이즈 방지)
        PROB: float in [0,1]      # 증강 적용 시도 확률 (1.0 권장, 내부에서 안전하게 스킵함)
        ELEVATION_FROM: 'auto' or [min_deg, max_deg]
        """
        if data_dict is None:
            return partial(self.line_downsample, config=config)

        prob = float(config.get('PROB', 1.0))
        if np.random.rand() > prob:
            return data_dict

        pts = data_dict.get('points', None)
        if pts is None or pts.shape[0] == 0:
            return data_dict

        target_lines = int(config.get('TARGET_LINES', 64))
        num_bins     = int(config.get('LINE_BINS', 128))
        tol          = int(config.get('TOLERANCE', 4))
        min_occ      = int(config.get('MIN_BIN_OCCUPANCY', 150))
        elev_from    = config.get('ELEVATION_FROM', 'auto')

        # 고도각 계산
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        r = np.sqrt(x**2 + y**2) + 1e-6
        elev = np.arctan2(z, r)

        # 각도 범위
        if isinstance(elev_from, (list, tuple)) and len(elev_from) == 2:
            mn = np.deg2rad(float(elev_from[0])); mx = np.deg2rad(float(elev_from[1]))
        else:
            mn, mx = elev.min(), elev.max()
            if (mx - mn) < np.deg2rad(5.0):
                pad = np.deg2rad(2.5); mn -= pad; mx += pad

        # bin 할당
        bins = ((elev - mn) / (mx - mn + 1e-9) * num_bins).astype(np.int32)
        bins = np.clip(bins, 0, num_bins - 1)

        # 현재 라인수(점유된 bin 수) 측정
        # 빈 bin(포인트 매우 적은 bin)은 제외
        counts = np.bincount(bins, minlength=num_bins)
        occupied = np.where(counts >= min_occ)[0]
        cur_lines = int(occupied.size)

        # 목표 근처면 스킵 (과소화 방지)
        if cur_lines <= target_lines + tol:
            return data_dict

        # 목표에 맞추기 위한 드롭 비율 계산
        # 남길 비율 = target / current → 드롭 비율 = 1 - 남길 비율
        keep_ratio = float(target_lines) / float(cur_lines)
        drop_ratio = max(0.0, min(1.0, 1.0 - keep_ratio))

        # '점유된 bin' 중에서만 드롭 대상 추출 (빈 bin은 어차피 영향 없음)
        num_drop = max(1, int(len(occupied) * drop_ratio))
        drop_bins = set(np.random.choice(occupied, size=num_drop, replace=False))

        keep_mask = np.array([b not in drop_bins for b in bins], dtype=bool)
        data_dict['points'] = pts[keep_mask]
        return data_dict
    
    def object_points_upsample(self, data_dict=None, config=None):
        """
        박스 내부 포인트가 적은 객체(클래스 지정)에 한해, 박스 내부 포인트를 복제+지터로 보강.
        config:
          UPSAMPLE_CLASSES: ['Pedestrian','Cyclist']  # 대상 클래스
          MIN_PTS: {'Pedestrian':15, 'Cyclist':20}    # 이보다 적으면 upsample 트리거
          TARGET_PTS: {'Pedestrian':30, 'Cyclist':40} # 목표 포인트 수
          JITTER_STD: 0.02                            # (m) 가우시안 지터 표준편차
          MAX_MULTIPLIER: 3                           # 과도 증가 방지
          PROB: float in [0,1]                        # (선택) 증강 적용 확률
        주의:
          - 클래스 이름은 dataset의 gt_names와 일치해야 함
          - 박스 내부 판별은 축회전 박스(heading)까지 고려
        """
        if data_dict is None:
            return partial(self.object_points_upsample, config=config)

        prob = float(config.get('PROB', 1.0))
        if np.random.rand() > prob:
            return data_dict

        if 'gt_boxes' not in data_dict or 'points' not in data_dict:
            return data_dict

        pts = data_dict['points']          # [N,C]
        boxes = data_dict['gt_boxes']      # [M,7]  [x,y,z,dx,dy,dz,heading]
        names = data_dict.get('gt_names', None)  # [M] list/ndarray of str
        if pts.shape[0] == 0 or boxes.shape[0] == 0 or names is None:
            return data_dict

        cls_set   = set(config.get('UPSAMPLE_CLASSES', []))
        min_pts_d = config.get('MIN_PTS', {})
        tgt_pts_d = config.get('TARGET_PTS', {})
        jitter    = float(config.get('JITTER_STD', 0.02))
        max_mul   = int(config.get('MAX_MULTIPLIER', 3))

        pts_xyz = pts[:, :3]
        out_list = [pts]

        # 박스 내부 판별 함수 (박스 좌표계로 회전/이동 후 절대값 비교)
        def in_box_mask(pts_xyz, box):
            cx, cy, cz, dx, dy, dz, yaw = box[:7]
            # 평면 회전(heading 반대 방향)
            c, s = np.cos(-yaw), np.sin(-yaw)
            rot = np.array([[c, -s], [s, c]], dtype=np.float32)
            xy = pts_xyz[:, :2] - np.array([cx, cy], dtype=np.float32)
            xy = xy @ rot.T
            z  = pts_xyz[:, 2] - cz
            return (np.abs(xy[:, 0]) <= dx / 2) & (np.abs(xy[:, 1]) <= dy / 2) & (np.abs(z) <= dz / 2)

        for bi, box in enumerate(boxes):
            cls_name = str(names[bi])
            if cls_name not in cls_set:
                continue

            mask = in_box_mask(pts_xyz, box)
            in_cnt = int(mask.sum())
            need_min = int(min_pts_d.get(cls_name, 0))
            target   = int(tgt_pts_d.get(cls_name, 0))

            # 박스 내부 포인트가 너무 적으면 보강
            if in_cnt > 0 and in_cnt < need_min and target > in_cnt:
                mul = min(max_mul, int(np.ceil((target - in_cnt) / max(in_cnt, 1))))
                base = pts[mask]
                dup  = np.repeat(base, mul, axis=0).copy()

                # xyz에 작은 가우시안 노이즈 추가(형태 왜곡 최소화)
                noise = np.random.normal(scale=jitter, size=dup[:, :3].shape).astype(dup.dtype)
                dup[:, :3] += noise
                out_list.append(dup)

        if len(out_list) > 1:
            data_dict['points'] = np.concatenate(out_list, axis=0)
        return data_dict
    
    # 여기까지가 추가한 증강기법들

    def imgaug(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.imgaug, config=config)
        imgs = data_dict["camera_imgs"]
        img_process_infos = data_dict['img_process_infos']
        new_imgs = []
        for img, img_process_info in zip(imgs, img_process_infos):
            flip = False
            if config.RAND_FLIP and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*config.ROT_LIM)
            # aug images
            if flip:
                img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
            img = img.rotate(rotate)
            img_process_info[2] = flip
            img_process_info[3] = rotate
            new_imgs.append(img)

        data_dict["camera_imgs"] = new_imgs
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )
        # if 'calib' in data_dict:
        #     data_dict.pop('calib')
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            if 'gt_boxes2d' in data_dict:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][gt_boxes_mask]

            data_dict.pop('gt_boxes_mask')
        return data_dict
