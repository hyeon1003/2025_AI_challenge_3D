import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans
import matplotlib as mpl
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.unicode_minus'] = False


DATA_ROOT = '../data/custom_av'
POINTS_DIR = os.path.join(DATA_ROOT, 'points')
LABELS_DIR = os.path.join(DATA_ROOT, 'labels')  


# 1) ImageSets/{train,val,test}.txt
# 2) splits/{train,val,test}.txt
# 3) points/{train,val,test}/ folder structure


def load_split_lists(data_root):
    split = {}
    for cand in ['ImageSets', 'splits', 'Split']:
        d = os.path.join(data_root, cand)
        if os.path.isdir(d):
            for name in ['train', 'val', 'test']:
                p = os.path.join(d, f'{name}.txt')
                if os.path.isfile(p):
                    with open(p) as f:
                        ids = [line.strip().split('.')[0] for line in f if line.strip()]
                    if ids:
                        split[name] = set(ids)
            if split:
                return split

    
    subdirs = {k: os.path.join(POINTS_DIR, k) for k in ['train','val','test']}
    if any(os.path.isdir(v) for v in subdirs.values()):
        for k, d in subdirs.items():
            if os.path.isdir(d):
                ids = [os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith('.npy')]
                if ids:
                    split[k] = set(ids)
        if split:
            return split

    all_ids = [os.path.splitext(f)[0] for f in os.listdir(POINTS_DIR) if f.endswith('.npy')]
    return {'all': set(all_ids)}

SPLITS = load_split_lists(DATA_ROOT)
print('발견된 splits:', {k: len(v) for k, v in SPLITS.items()})


def point_path(sample_id, split_name=None):
    cands = []
    if split_name:
        cands.append(os.path.join(POINTS_DIR, split_name, f'{sample_id}.npy'))
    cands.append(os.path.join(POINTS_DIR, f'{sample_id}.npy'))
    for p in cands:
        if os.path.isfile(p):
            return p
    return None

def label_path(sample_id, split_name=None):
    cands = []
    if split_name:
        cands.append(os.path.join(LABELS_DIR, split_name, f'{sample_id}.txt'))
    cands.append(os.path.join(LABELS_DIR, f'{sample_id}.txt'))
    for p in cands:
        if os.path.isfile(p):
            return p
    return None  


def estimate_layers(xyz, angle_resolution=0.4):
    """Estimate vertical layer count by quantizing elevation angle."""
    xy = np.linalg.norm(xyz[:, :2], axis=1)
    v_angles = np.degrees(np.arctan2(xyz[:, 2], xy))
    q = np.floor((v_angles - v_angles.min()) / angle_resolution)
    return int(np.unique(q).size)

def eda_for_split(split_name, ids):
    rows = []
    for sid in ids:
        p = point_path(sid, split_name if split_name!='all' else None)
        if p is None:
            continue
        pts = np.load(p)  
        xyz = pts[:, :3]
        rows.append({
            'id': sid,
            'point_count': xyz.shape[0],
            'layer_count': estimate_layers(xyz, angle_resolution=0.4),
            'split': split_name
        })
    df = pd.DataFrame(rows)
    return df

def label_stats_for_split(split_name, ids):
    obj_counter = Counter()
    frame_counter = Counter()
    total_labeled_frames = 0
    for sid in ids:
        lp = label_path(sid, split_name if split_name!='all' else None)
        if lp is None:
            continue
        total_labeled_frames += 1
        seen = set()
        with open(lp, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls = parts[-1]  
                obj_counter[cls] += 1
                seen.add(cls)
        for cls in seen:
            frame_counter[cls] += 1
    return obj_counter, frame_counter, total_labeled_frames

all_df = []
split_order = list(SPLITS.keys())
for split_name in split_order:
    print(f'\n[{split_name}] EDA 실행 중...')
    df = eda_for_split(split_name, SPLITS[split_name])
    print(df[['point_count','layer_count']].describe())
    all_df.append(df)

all_df = pd.concat(all_df, ignore_index=True)

def plot_hist_by_split(df, col, bins=40, title=''):
    plt.figure(figsize=(7,4))
    for s in df['split'].unique():
        sub = df[df['split']==s][col]
        plt.hist(sub, bins=bins, alpha=0.5, label=f'{s} (n={len(sub)})')
    plt.title(title if title else f'{col} distribution (by split)')
    plt.xlabel(col)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_hist_by_split(all_df, 'point_count', bins=40, title='Point Count Distribution (by split)')
plot_hist_by_split(all_df, 'layer_count', bins=40, title='Vertical Layer Count Distribution (by split)')

# K-means sensor clustering (64ch / 128ch)
# - Use both features but map clusters by MEAN POINT COUNT
#   (high points -> 128ch_like, low points -> 64ch_like)

features = all_df[['point_count','layer_count']].values
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(features)
all_df['cluster'] = kmeans.labels_


pc_means = all_df.groupby('cluster')['point_count'].mean()
mapping = {pc_means.idxmin():'64ch_like', pc_means.idxmax():'128ch_like'}
all_df['sensor_guess'] = all_df['cluster'].map(mapping)

print('\n[Overall] cluster stats (point_count-based mapping)')
print(all_df.groupby(['cluster'])[['point_count','layer_count']].agg(['mean','std','min','max']))

print('\n[By split] estimated sensor ratio (point_count-based)')
print(pd.crosstab(all_df['split'], all_df['sensor_guess'], normalize='index').round(3))

for split_name in split_order:
    ids = SPLITS[split_name]
    obj_cnt, frm_cnt, labeled_frames = label_stats_for_split(split_name, ids)
    print(f'\n[{split_name}] labeled frames: {labeled_frames}')
    if labeled_frames == 0:
        print('  No labels → skip object/frame distribution.')
        continue
    print('  Object counts:', dict(obj_cnt))
    print('  Frame counts :', dict(frm_cnt))

    if obj_cnt:
        classes = list(obj_cnt.keys())
        counts = [obj_cnt[c] for c in classes]
        plt.figure(figsize=(6,4))
        plt.bar(classes, counts)
        plt.title(f'[{split_name}] Object Count Distribution')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()


out_csv = os.path.join(DATA_ROOT, 'eda_point_layer_sensor.csv')
all_df.to_csv(out_csv, index=False)
print('\nSaved EDA summary CSV:', out_csv)