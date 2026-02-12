import cv2
import numpy as np
from pathlib import Path
import glob
import cv2
import os


# ---------- 你已有的工具（略有改动/复用） ----------
def load_img(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def sift_keypoints_and_descriptors(img_gray):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img_gray, None)
    return kp, des

def match_descriptors(des1, des2, ratio=0.8):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn = flann.knnMatch(des1, des2, k=2)
    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good

def estimate_homography(kp1, kp2, matches, ransac_thresh=4.0):
    if len(matches) < 4:
        return None, None, 0
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)
    inliers = int(mask.sum()) if (mask is not None) else 0
    return H, mask, inliers

# ---------- 关键：把多张图都配准到 reference(-90deg) ----------
def register_to_reference(img, ref_img, kp_ref, des_ref, ratio=0.8, ransac_thresh=4.0):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift_keypoints_and_descriptors(g)
    if des is None or len(kp) < 4:
        return None, [], 0
    matches = match_descriptors(des, des_ref, ratio=ratio)
    H, mask, inliers = estimate_homography(kp, kp_ref, matches, ransac_thresh=ransac_thresh)
    return H, matches, inliers

def compute_global_canvas(h_ref, w_ref, transforms):
    """
    transforms: [(img, H_ref_from_img), ...]，注意包含 (ref_img, I)
    计算所有图的四角经变换后的范围，得到全景画布和用于正坐标的平移 T。
    """
    corners_all = []
    for img, H in transforms:
        h, w = img.shape[:2]
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        warp = cv2.perspectiveTransform(corners, H)
        corners_all.append(warp)

    all_corners = np.vstack(corners_all)
    xmin, ymin = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    xmax, ymax = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

    tx, ty = -xmin, -ymin
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=np.float64)
    width, height = (xmax - xmin), (ymax - ymin)
    return T, width, height

def feather_blend(dst, src, mask):
    """
    简单羽化融合：按距离边界的权重线性融合。
    dst: 目标全景画布 (H,W,3)
    src: 同尺寸的待融合图 (H,W,3)
    mask: 同尺寸单通道(0/1)，src有效区域
    """
    if mask.max() == 0:
        return dst
    # 距离变换作为权重
    inv_mask = (1 - mask).astype(np.uint8)
    dist_dst = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 3)
    dist_src = cv2.distanceTransform((1 - inv_mask), cv2.DIST_L2, 3)  # == mask
    weight_src = dist_src / (dist_src + dist_dst + 1e-6)
    weight_src = weight_src[..., None]  # (H,W,1)

    dst_f = dst.astype(np.float32)
    src_f = src.astype(np.float32)
    out = (dst_f * (1 - weight_src) + src_f * weight_src)
    return out.astype(np.uint8)

def warp_into_canvas(img, H, T, size):
    """
    把 img 经 T@H 变换到画布 size=(W,H)，返回变换图和有效区域mask
    """
    W, Hh = size
    M = T @ H
    warped = cv2.warpPerspective(img, M, (W, Hh))
    mask = cv2.warpPerspective(np.ones(img.shape[:2], dtype=np.uint8), M, (W, Hh))
    mask = (mask > 0).astype(np.uint8)
    return warped, mask

def stitch_all_to_minus90(image_paths, ref_path):
    # 读取 reference
    ref_img = load_img(ref_path)
    g_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    kp_ref, des_ref = sift_keypoints_and_descriptors(g_ref)

    transforms = [(ref_img, np.eye(3, dtype=np.float64))]
    H_map = {str(ref_path): np.eye(3, dtype=np.float64)}  # 记录参考图自身为 I

    failed = []
    for p in image_paths:
        if str(p) == str(ref_path):
            continue
        img = load_img(p)
        H, matches, inliers = register_to_reference(
            img, ref_img, kp_ref, des_ref, ratio=0.8, ransac_thresh=4.0
        )
        if H is None:
            failed.append((str(p), "H=None / insufficient matches"))
            continue
        transforms.append((img, H))
        H_map[str(p)] = H  # 👈 保存：该图 -> 参考图 的单应矩阵

    # 计算统一画布
    T, W, Hh = compute_global_canvas(ref_img.shape[0], ref_img.shape[1], transforms)

    panorama = np.zeros((Hh, W, 3), dtype=np.uint8)
    for img, H in transforms:
        warped, mask = warp_into_canvas(img, H, T, (W, Hh))
        if panorama.sum() == 0:
            panorama = warped
        else:
            warped_bg = warped.copy()
            warped_bg[mask == 0] = 0
            panorama = feather_blend(panorama, warped_bg, mask)

    return panorama, transforms, failed, T, H_map

import re
from collections import defaultdict

def parse_angle_from_path(p):
    """
    从文件名中解析角度，形如 scene_-90deg.jpg -> -90
    """
    m = re.search(r'_(\-?\d+)deg\.jpg$', os.path.basename(str(p)))
    return int(m.group(1)) if m else None

def register_pairwise(img_src, img_tgt, kp_tgt, des_tgt, ratio=0.8, ransac_thresh=4.0):
    """
    把 img_src 配准到目标 img_tgt，返回 H_tgt_from_src（从 src 到 tgt 的单应）
    """
    g = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    kp_src, des_src = sift_keypoints_and_descriptors(g)
    if des_src is None or len(kp_src) < 4:
        return None, 0
    matches = match_descriptors(des_src, des_tgt, ratio=ratio)
    H, mask, inliers = estimate_homography(kp_src, kp_tgt, matches, ransac_thresh=ransac_thresh)
    return H, inliers

def pano_stitch_subset_and_map_all(
    image_paths, 
    ref_path, 
    anchor_range=(-110, -70),  # 只用这个角度区间的视角参与拼接
    ratio=0.8, 
    ransac_thresh=4.0
):
    # 1) 读取参考图（-90）
    ref_img = load_img(ref_path)
    g_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    kp_ref, des_ref = sift_keypoints_and_descriptors(g_ref)

    # 2) 按角度划分 “锚定视角” 与 “仅映射视角”
    anchors, map_only = [], []
    ref_angle = parse_angle_from_path(ref_path)
    for p in image_paths:
        if str(p) == str(ref_path):
            anchors.append(p)  # 参考一定在锚定集
            continue
        a = parse_angle_from_path(p)
        if a is not None and (anchor_range[0] <= a <= anchor_range[1]):
            anchors.append(p)
        else:
            map_only.append(p)

    # 3) 先把锚定视角都配准到参考（用于画布与融合）
    transforms = [(ref_img, np.eye(3, dtype=np.float64))]
    H_map = {str(ref_path): np.eye(3, dtype=np.float64)}
    failed_anchor = []

    # 也缓存锚定视角的 SIFT，以便非锚定视角走“先到锚定”的后备路线
    anchor_cache = {}  # path -> dict(img, kp, des, H_ref_from_anchor)
    anchor_cache[str(ref_path)] = {
        "img": ref_img, "kp": kp_ref, "des": des_ref, "H_ref": np.eye(3, dtype=np.float64),
        "angle": ref_angle
    }

    for p in anchors:
        if str(p) == str(ref_path):
            continue
        img = load_img(p)
        H, matches, inliers = register_to_reference(
            img, ref_img, kp_ref, des_ref, ratio=ratio, ransac_thresh=ransac_thresh
        )
        if H is None:
            failed_anchor.append((str(p), "anchor->ref failed"))
            continue
        transforms.append((img, H))
        H_map[str(p)] = H

        # 缓存锚定视角的特征
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift_keypoints_and_descriptors(g)
        anchor_cache[str(p)] = {
            "img": img, "kp": kp, "des": des, "H_ref": H,
            "angle": parse_angle_from_path(p)
        }

    # 4) 计算画布（只用锚定视角）
    T, W, Hh = compute_global_canvas(ref_img.shape[0], ref_img.shape[1], transforms)

    # 5) 只融合锚定视角
    panorama = np.zeros((Hh, W, 3), dtype=np.uint8)
    first = True
    for img, H in transforms:
        warped, mask = warp_into_canvas(img, H, T, (W, Hh))
        if first:
            panorama = warped
            first = False
        else:
            warped_bg = warped.copy()
            warped_bg[mask == 0] = 0
            panorama = feather_blend(panorama, warped_bg, mask)

    # 6) 为 map-only 视角求 H_ref_from_img（不参与融合）
    failed_maponly = []
    # 先准备一个按角度的锚定列表，便于找“最近锚定”
    anchor_list = [
        (info["angle"], path, info) for path, info in anchor_cache.items() if info["angle"] is not None
    ]
    for p in map_only:
        img = load_img(p)

        # 6.1 直连到参考
        H_direct, matches, inliers = register_to_reference(
            img, ref_img, kp_ref, des_ref, ratio=ratio, ransac_thresh=ransac_thresh
        )
        if H_direct is not None:
            H_map[str(p)] = H_direct
            continue

        # 6.2 失败则找最近角度的锚定视角，先配准到锚定，再链式到参考
        a = parse_angle_from_path(p)
        best = None
        if a is not None and len(anchor_list) > 0:
            best = min(anchor_list, key=lambda x: abs(a - x[0]))  # (angle, path, info)

        if best is not None:
            _, anchor_path, info = best
            H_anchor, inliers2 = register_pairwise(img, info["img"], info["kp"], info["des"],
                                                  ratio=ratio, ransac_thresh=ransac_thresh)
            if H_anchor is not None:
                # 链式：img -> anchor -> ref
                H_ref = info["H_ref"] @ H_anchor
                H_map[str(p)] = H_ref
                continue

        # 6.3 仍失败则记录
        failed_maponly.append((str(p), "map-only failed to register (direct & via anchor)"))

    # 保存画布平移
    H_map['T'] = T
    H_map['canvas_size'] = np.array([W, Hh], dtype=np.int32)

    return panorama, transforms, failed_anchor, failed_maponly, T, H_map

def project_points_to_pano(points_xy, H_ref_from_img, T):
    """
    把某视角上的点投到 pano 画布坐标系
    """
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1,1,2)
    M = T @ H_ref_from_img
    proj = cv2.perspectiveTransform(pts, M).reshape(-1,2)
    return proj

# ----------------- 使用示例 -----------------
# 以 -90deg 为参考，把同场景不同角度都拼进去
# H_maps = {}  # 保存每个图到 -90 的单应矩阵
# for scene in ["livingroom", "kidroom", "workshop", "studyroom"]:
#     ref_path = Path(f"scenes/{scene}/norm/{scene}_-90deg.jpg")
#     image_paths = [Path(i) for i in glob.glob(f"scenes/{scene}/norm/*.jpg") if os.path.exists(f"annotations/{os.path.basename(i).replace('.jpg', '.json')}")]

#     panorama, transforms, failed, T, H_map = stitch_all_to_minus90(image_paths, ref_path)
#     # H_maps[scene] = H_map  # 保存每个场景的 H_map
#     H_map['T'] = T
#     np.savez(f"metadata/{scene}_H.npz", **{k: H_map[k] for k in H_map})
#     cv2.imwrite(f"metadata/{scene}_panorama.jpg", panorama)
#     print("Failed:", failed)

# # 如果你有一条在 angleX 上的轨迹 points_X (N,2)，想投到 -90：
# def project_traj_to_ref(points_xy, H_ref_from_img):
#     pts = np.asarray(points_xy, dtype=np.float32).reshape(-1,1,2)
#     proj = cv2.perspectiveTransform(pts, H_ref_from_img).reshape(-1,2)
#     return proj

if __name__ == "__main__":
    for scene in ["livingroom", "kidroom", "workshop", "studyroom"]:
        ref_path = Path(f"scenes/{scene}/norm/{scene}_-90deg.jpg")
        # 这里 image_paths 可以给“全部视角”（含 -120、-130 等）
        image_paths = [Path(i) for i in glob.glob(f"scenes/{scene}/norm/*.jpg") 
                    if os.path.exists(f"annotations/{os.path.basename(i).replace('.jpg', '.json')}")]

        pano, transforms, fail_anchor, fail_maponly, T, H_map = pano_stitch_subset_and_map_all(
            image_paths, ref_path, anchor_range=(-110, -70)
        )

        # 存储
        H_map_npz = {k: H_map[k] for k in H_map}
        np.savez(f"metadata/{scene}_H.npz", **H_map_npz)
        cv2.imwrite(f"metadata/{scene}_panorama.jpg", pano)
        print("Failed anchors:", fail_anchor)
        print("Failed map-only:", fail_maponly)