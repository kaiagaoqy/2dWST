import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import os
from shapely.geometry import Polygon, Point
from pathlib import Path
import numpy as np
from matplotlib import cm

from torchcpd import RigidRegistration, DeformableRegistration
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.colors as mcolors

import torch
h_ang , v_ang = np.rad2deg(2*np.arctan(35.4/2/40)), np.rad2deg(2*np.arctan(19.9/2/40))
h_res, v_res = 1920, 1080


device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# --- CPD (Coherent Point Drift) based nearest neighbor matching ---
def multi_resample(Y, L, n=8):
    """
    Perform multi-phase resampling of input trajectory Y to length L.

    Args:
        Y (ndarray): Original trajectory of shape (M, 2).
        L (int): Desired length of each resampled trajectory.
        n (int): Number of resampled versions to generate.

    Returns:
        resampled (list of ndarray): List of n trajectories, each of shape (L, 2).
        indices (list of ndarray): Corresponding index lists from the original Y.
    """
    N = len(Y)
    resampled, indices, dropped_indices = [], [], []
    for phase in np.linspace(0, 1, n, endpoint=False):
        start = phase * (N / L) # Start point for resampling
        idx = (start + np.arange(L) * (N / L)).astype(int) % N # Ensure indices wrap around
        idx = list(set(idx))
        resampled.append(Y[idx])
        indices.append(idx)
        # add dropped indices
        dropped_indices.append(np.setdiff1d(np.arange(N), idx))
    return resampled, indices, dropped_indices

def icp_2d(A, B, max_iter=50, tolerance=1e-5):
    """
    2D ICP for rigid alignment.

    Args:
        A (np.ndarray): Source (Nx2)
        B (np.ndarray): Target (Nx2)

    Returns:
        aligned (Nx2), R (2x2), t (2,)
    """
    src = A.copy()
    tgt = B.copy()
    prev_error = float('inf')

    for _ in range(max_iter):
        distances = cdist(src, tgt)
        indices = np.argmin(distances, axis=1)
        tgt_matched = tgt[indices]

        centroid_src = np.mean(src, axis=0)
        centroid_tgt = np.mean(tgt_matched, axis=0)

        src_centered = src - centroid_src
        tgt_centered = tgt_matched - centroid_tgt

        H = src_centered.T @ tgt_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = centroid_tgt - R @ centroid_src
        src = (R @ src.T).T + t

        error = np.mean(np.linalg.norm(src - tgt_matched, axis=1))
        if abs(prev_error - error) < tolerance:
            break
        prev_error = error

    return src, R, t

def constrain_rigid(Y, X, max_rot=5, scale_bounds=(0.95, 1.05), icp=False):
    """
    Enforce rotation and scale limits on a rigid CPD transformation.

    Args:
        Y (ndarray): Hand Traced point set of shape (M, 2).
        X (ndarray): Ground Truth point set of shape (M, 2).
        max_rot (float): Maximum allowed rotation in degrees.
        scale_bounds (tuple): (min_scale, max_scale).

    Returns:
        TY_fixed (ndarray): Rigidly aligned X with constraints applied.
        theta (float): Clipped rotation angle in radians.
        scale_clipped (float): Clipped scale factor.
    """
    if icp:
        # Use ICP for initial alignment
        TY, R_orig, t_orig = icp_2d(Y, X)
    else:
        reg = RigidRegistration(X=X, Y=Y, max_iterations=50, device=device)
        TY, _ = reg.register()
        # --- Compute original rotation angle ---
        R_orig = reg.R.cpu().numpy()
        t_orig = reg.t.cpu().numpy()
    angle_rad = np.arccos(np.clip((np.trace(R_orig) - 1) / 2, -1, 1))
    angle_deg = np.degrees(angle_rad)
    angle_deg_clipped = np.clip(angle_deg, -max_rot, max_rot)
    theta = np.radians(angle_deg_clipped)

    # --- Limit rotation ---
    R_fixed = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    # --- Limit scaling ---
    scale_seg = np.mean(np.linalg.norm(X - X.mean(0), axis=1))
    scale_Yr = np.mean(np.linalg.norm(Y - Y.mean(0), axis=1))
    scale_ratio = scale_Yr / scale_seg
    scale_clipped = np.clip(scale_ratio, scale_bounds[0], scale_bounds[1])

    # --- Reapply the constrained rigid transformation ---
    centroid_Y = Y.mean(0)
    centroid_X = X.mean(0)
    TY_rigid_fixed = (scale_clipped * R_fixed @ (Y - centroid_Y).T).T + centroid_X
    return TY_rigid_fixed, theta, scale_clipped


def hybrid_match(best_aligned, best_seg, D=None, tau=None, k=30):
    """
    Perform hybrid matching: Relaxed BBS followed by Hungarian supplementation.

    Args:
        best_aligned (ndarray): Aligned moving set of shape (M, 2).
        best_seg (ndarray): Reference segment of shape (N, 2).
        tau (float): Distance threshold; defaults to median pairwise distance.
        k (int): Top-k parameter for relaxed BBS.

    Returns:
        final_pairs (list of tuples): List of matched index pairs (i, j).
        unmatched (tuple): Two lists of unmatched indices: (unmatched_gt, unmatched_aligned_ht).
    """
    if D is None:
        # Compute distance matrix if not provided
        D = cdist(best_seg, best_aligned)
    if tau is None:
        tau = np.median(D)
    # === Relaxed BBS ===
    # D[i, j] is the distance from segment[i] to aligned[j]
    # ---- top-k for rows ----
    m, n = D.shape
    topk_cols_per_row = np.argsort(D, axis=1)[:, :k]
    row_indices_rowwise = np.repeat(np.arange(m), k)
    col_indices_rowwise = topk_cols_per_row.flatten()
    row_topk_coords = list(zip(row_indices_rowwise, col_indices_rowwise))

    # ---- top-k for columns ----
    topk_rows_per_col = np.argsort(D, axis=0)[:k, :]
    col_indices_colwise = np.repeat(np.arange(n), k)
    row_indices_colwise = topk_rows_per_col.flatten()
    col_topk_coords = list(zip(row_indices_colwise, col_indices_colwise))

    all_coords = row_topk_coords + col_topk_coords
    all_coords = list(set(all_coords)) # Remove duplicates

    # ---- Sort by D[i, j] value ----
    sorted_coords = sorted(all_coords, key=lambda ij: D[ij[0], ij[1]])
    # matched_j: deform idx -> (ref idx, dist)
    # matched_i: ref idx -> (deform idx, dist)
    matched_i, matched_j = {}, {} 
    # flat_indices = np.argsort(D, axis=None)          
    # coords = np.unravel_index(flat_indices, D.shape) # turn to (i, j) coordinates
    # sorted_coords = list(zip(coords[0], coords[1]))  # list of (i, j)
    for ref_idx, deform_idx in sorted_coords:
        if D[ref_idx, deform_idx] <= tau:
            if ref_idx not in matched_i and deform_idx not in matched_j:
                matched_i[ref_idx] = (deform_idx, D[ref_idx, deform_idx])
                matched_j[deform_idx] = (ref_idx, D[ref_idx, deform_idx])
                

    valid = [(ref_idx, deform_idx) for ref_idx, (deform_idx, d) in matched_i.items() if d <= tau]
    # valid = [(deform_idx, ref_idx) for deform_idx, ref_idx in relaxed_bbs_pairs if d <= tau]

    # Supplement with Hungarian
    rem_i = [i for i in range(D.shape[0]) if i not in {i for i, _ in valid}]
    rem_j = [j for j in range(D.shape[1]) if j not in {j for _, j in valid}]
    subD = np.where(D[np.ix_(rem_i, rem_j)] > tau, 1e6, D[np.ix_(rem_i, rem_j)]) # Mask distances above tau
    r, c = linear_sum_assignment(subD)
    supp = [(rem_i[r[i]], rem_j[c[i]]) for i in range(len(r)) if subD[r[i], c[i]] <= tau]
    final = valid + supp
    unmatched = (
        list(set(range(D.shape[0])) - {i for i, _ in final}),
        list(set(range(D.shape[1])) - {j for _, j in final})
    )
    return valid, supp, unmatched

import numpy as np



def pixel_to_ray(u, v, K_inv):
    """Convert pixel coordinates to a ray in camera space."""
    p_img = np.array([u, v, 1])
    ray = K_inv @ p_img
    ray = ray / np.linalg.norm(ray)
    return ray

def get_va(u, v):
    """Convert pixel coordinates to viewing angle in radians."""
    # fx, fy, cx, cy are camera intrinsic parameters
    fx, fy = 8.64859157e+02, 1.11134930e+04
    cx, cy = 9.63671546e+02, 5.27296795e+02
    K = np.array([[fx, 0, cx],
                [0, fy, cy],
                [0,  0,  1]])

    K_inv = np.linalg.inv(K)
    return pixel_to_ray(u, v, K_inv)

def diff_va(g1,g2):
    """Calculate the difference in viewing angles between two rays."""
    ray1 = get_va(g1[0], g1[1])
    ray2 = get_va(g2[0], g2[1])

    # Calculate the angle between the two rays
    cos_theta = np.dot(ray1, ray2)
    theta_rad = np.arccos(np.clip(cos_theta, -1, 1))
    theta_deg = np.rad2deg(theta_rad)
    return theta_deg


def CPD(A, B, save_path, save_fig=True, suffix='',filter_flag=False):
    """ Find the nearest neighbor in B for each point in A using CPD (Coherent Point Drift) algorithm.

    Args:
        A (List or ndarray): Ground Truth Trajectory of shape (N,2)
        B (List or ndarray): Hand Traced Trajectory of shape (N,2) 
        save_path (str): Path to save the results
        save_fig (bool, optional): Whether to save the figure. Defaults to True.
        suffix (str, optional): Suffix for the saved figure filename. Defaults to ''.
        filter_flag (bool, optional): Whether to apply filtering. Defaults to False.
    """

    
    # === Hyperparameters ===
    max_rotation_deg = 5
    scale_bounds = (0.95, 1.05)
    best_err = np.inf
    best_dropped_A = best_dropped_B = best_i = best_seg = best_aligned = best_P = best_orig_Y = None

    A = A[:-1].copy()
    B = B[:-1].copy()
    # Generate sample data
    if isinstance(A, list):
        A = np.array(A)
    if isinstance(B, list):
        B = np.array(B)
    # Convert to angle offsets
    A[:,0] = A[:,0] / h_res * h_ang
    A[:,1] = A[:,1] / v_res * v_ang
    B[:,0] = B[:,0] / h_res * h_ang
    B[:,1] = B[:,1] / v_res * v_ang

    L = min(len(B), len(A))
    topk = max(30, int(L * 0.3))  # Ensure L is at least 30% of the smaller set
    icp_flag = False  # Use ICP for initial alignment

    # Sliding window + resampling to find best alignment
    resampled_list_A, idx_list_A, dropped_idx_list_A = multi_resample(A, L, n=3)
    resampled_list_B, idx_list_B, dropped_idx_list_B = multi_resample(B, L, n=3)
    for segment, _, drop_idx_A in zip(resampled_list_A, idx_list_A, dropped_idx_list_A):
        for (Y_r, idx, drop_idx_B) in zip(resampled_list_B, idx_list_B, dropped_idx_list_B):
            # Constrained rigid
            # TY is transformed Y
            # Ensure segment and Y_r have the same length
            if len(segment) != len(Y_r):
                print(f"Skipping segment due to length mismatch: {len(segment)} vs {len(Y_r)}")
                continue
            TY_rigid, theta, scale = constrain_rigid(Y_r, segment, scale_bounds=scale_bounds, max_rot=max_rotation_deg, icp=icp_flag)
            # Nonrigid refinement
            # TY_final, P, w = adaptive_nonrigid(segment, TY_rigid)
            reg_nonrigid = DeformableRegistration(X=segment, Y=TY_rigid, beta=3, lamb=120, w=0, max_iterations=50, device=device)
            TY_final, _ = reg_nonrigid.register()
            TY_final = TY_final.cpu().numpy()
            P = reg_nonrigid.P.cpu().numpy()

            # Error metric
            D = cdist(segment, TY_final) # GT -> HT
            err = 0.5 * (np.min(D, axis=1).mean() + np.min(D, axis=0).mean())
            if err < best_err:
                best_err = err
                # best_i = start
                best_seg = segment.copy() # Kept GT
                best_aligned = TY_final.copy() # Hand Traced Traj after rigid trans
                best_P = P.copy()
                best_rigid = TY_rigid.copy() # Hand Traced Traj after non-rigid trans
                best_orig_Y = B[idx] # Kept Original Hand Traced traj
                best_dropped_A = A[drop_idx_A] # Dropped GT
                best_dropped_B = B[drop_idx_B] # Dropped Hand Traced traj
                best_D = D
                best_rot = theta
                best_scale = scale


    # === Soft assignment mapping ===
    inlier_mask = best_P.max(axis=0) > (1.0 / len(best_seg))
    match_indices = best_P.argmax(axis=0)  # aligned[j] → segment[i]

    # === Hybrid matching ===
    dist_top_k = min(8,len(best_D))  # Ensure we don't exceed the number of points
    row_topk_idx = np.argpartition(best_D, dist_top_k-1, axis=1)[:, :dist_top_k]
    row_topk_val = np.take_along_axis(best_D, row_topk_idx, axis=1).flatten()
    col_topk_idx = np.argpartition(best_D, dist_top_k-1, axis=0)[:dist_top_k, :]
    col_topk_val = np.take_along_axis(best_D, col_topk_idx, axis=0).flatten()
    nearest_dist = np.hstack((row_topk_val, col_topk_val))
    # nearest_dist = np.hstack((best_D[np.argpartition(best_D, dist_top_k, axis=1)][:, :dist_top_k].flatten(), 
    #                           np.argpartition(best_D, dist_top_k, axis=0)[:dist_top_k, :].flatten()))  # Find the minimum distance for each point
    tau = np.percentile(nearest_dist, 50)  # distance threshold
    # valid_pairs refers to pairs from relaxed BBS
    # supp_pairs refers to pairs from Hungarian matching
    valid_pairs, supp_pairs, (unmatched_ref, unmatched_align) = hybrid_match(best_aligned, best_seg, D=best_D, tau=tau, k=topk)
    # Final matches are pairs of (original Y, segment)
    final_matches = [(best_seg[i].tolist(), best_orig_Y[j].tolist()) for i, j in valid_pairs + supp_pairs]
    rigid_matches = [(best_seg[i].tolist(), best_rigid[j].tolist()) for i, j in valid_pairs + supp_pairs]
    distances = cdist(best_seg, best_orig_Y)[[i for i, j in valid_pairs + supp_pairs], [j for i, j in valid_pairs + supp_pairs]]
    procrustes_distance = cdist(best_seg, best_rigid)[[i for i, j in valid_pairs + supp_pairs], [j for i, j in valid_pairs + supp_pairs]]
    procrustes_distance = np.mean(procrustes_distance)
    aligned_distance = cdist(best_seg, best_aligned)[[i for i, j in valid_pairs + supp_pairs], [j for i, j in valid_pairs + supp_pairs]]
    aligned_distance = np.mean(aligned_distance)


    if not save_fig:
        return distances, final_matches, rigid_matches, (procrustes_distance, aligned_distance)
    # Normalize distances for coloring
    norm = mcolors.Normalize(vmin=min(distances), vmax=max(distances))
    cmap = cm.get_cmap("RdYlGn_r")  # Red (high) to Green (low)

    # Create the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    # Offset for second set of points (to align them side by side)
    offset_x = 0  # Width of the first image
    
    # Plot BBS (relaxed BBS) pairs
    # valid from relaxed BBS
    bbs_i = [i for i,_ in valid_pairs]
    bbs_j = [j for _,j in valid_pairs]
    ax1.scatter(best_seg[bbs_i,0], best_seg[bbs_i,1], c='black', label='Ground-Truth (BBS Matched)', s=20)
    ax1.scatter(best_orig_Y[bbs_j,0], best_orig_Y[bbs_j,1], c='red', label='Hand-Trace (BBS Reference)', s=20)
    # Draw valid matches
    for i,j in valid_pairs:
        d = distances[bbs_i.index(i)]  # Get the distance for the current pair
        color = cmap(norm(d)) 
        ax1.plot([best_seg[i,0], best_orig_Y[j,0]], [best_seg[i,1], best_orig_Y[j,1]], color=color, linewidth=1, linestyle='-')
    # Plot Hungarian supplemental pairs
    sup_i = [i for i,_ in supp_pairs]
    sup_j = [j for _,j in supp_pairs]
    ax1.scatter(best_seg[sup_i,0], best_seg[sup_i,1], facecolors='none', edgecolors='black',linewidths=1, label='Ground-Truth (Hungarian Matched)', s=20)
    ax1.scatter(best_orig_Y[sup_j,0], best_orig_Y[sup_j,1], facecolors='none', edgecolors='red',linewidths=1, label='Hand-Trace (Hungarian Reference)', s=20)
    for i,j in supp_pairs:
        d = distances[len(bbs_i) + sup_i.index(i)]
        color = cmap(norm(d))
        ax1.plot([best_seg[i,0], best_orig_Y[j,0]], [best_seg[i,1], best_orig_Y[j,1]], color=color, linestyle='--', lw=0.6)
    # Draw unmatched points    
    # if len(unmatched_align) or len(best_dropped_A):
    ax1.scatter(best_seg[unmatched_ref,0], best_seg[unmatched_ref,1], c='black', marker='x', alpha=0.5, label='Unmatched Ground-Truth', s=20)

    # if len(unmatched_ref) or len(best_dropped_B):
    ax1.scatter(best_orig_Y[unmatched_align,0], best_orig_Y[unmatched_align,1], c='red', marker='x', alpha=0.5, label='Unmatched Hand-Trace', s=20)
    ax1.scatter(best_dropped_A[:,0], best_dropped_A[:,1], c='brown', marker='x', alpha=0.7, label='Dropped Ground-Truth', s=20)
    ax1.scatter(best_dropped_B[:,0], best_dropped_B[:,1], c='orange', marker='x', alpha=0.7, label='Dropped Hand-Trace', s=20)

    # all_x = np.concatenate([np.array(best_orig_Y)[:, 0], np.array(best_seg)[:, 0]])
    # all_y = np.concatenate([np.array(best_orig_Y)[:, 1], np.array(best_seg)[:, 1]])
    # Compute limits with padding
    # x_pad = (all_x.max() - all_x.min()) * 0.1
    # y_pad = (all_y.max() - all_y.min()) * 0.1

    # Hide axes
    # ax.axis("off")
    # Add legend for Set A and Set B
    # ax1.set_xlim(all_x.min() - x_pad, all_x.max() + x_pad, size=12)
    # ax1.set_ylim(all_y.min() - y_pad, all_y.max() + y_pad, size=12)
    # put legend outside the canvas below the title horizontally
    ax1.legend(loc="upper center", ncol=4, fontsize=10, bbox_to_anchor=(0.5, 1.15))
    ax1.set_xlabel('Horizontal (deg)', fontsize=14)
    ax1.set_ylabel('Vertical (deg)', fontsize=14)
    ax1.axis('equal'); ax1.invert_yaxis()


    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Empty array for ScalarMappable
    cbar = plt.colorbar(sm, ax=ax1, fraction=0.03, pad=0.04)
    cbar.set_label("Viewing Angle Offset (deg)", fontsize=12)
    cbar.set_ticks(np.linspace(min(distances), max(distances), num=5))  # Set meaningful tick values
    cbar.ax.set_yticklabels([f"{tick:.2f}" for tick in np.linspace(min(distances), max(distances), num=5)])  # Show actual distances

    # plot 2: Non-rigid vs Rigid
    ax2.plot(A[:, 0], A[:, 1], 'k-', label='Ground Truth')
    ax2.scatter(best_orig_Y[:, 0], best_orig_Y[:, 1], c='#7EACB5', label='Original Hand Trace', s=20)

    ax2.scatter(best_rigid[:, 0], best_rigid[:, 1], c='#FADFA1', label='Rigid Init Hand Trace', s=20)
    ax2.scatter(best_aligned[:, 0], best_aligned[:, 1],facecolors='none', edgecolors='#EF5A6F',linewidths=2, label='Non-Rigid Aligned Hand Trace', s=20)
    for j in range(len(best_orig_Y)):
        if inlier_mask[j]:
            ax2.plot([best_orig_Y[j,0], best_aligned[j,0]], [best_orig_Y[j,1], best_aligned[j,1]], 'b--', lw=0.5)
    ax2.legend(loc="upper center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, 1.15))
    ax2.axis('equal'); ax2.grid(False); ax2.invert_yaxis()
    ax2.set_xlabel('Horizontal (deg)', fontsize=14)
    ax2.set_ylabel('Vertical (deg)', fontsize=14)
    

    # Show the plot
    os.makedirs(str(save_path), exist_ok=True)
    save_path = str(save_path / f"nn{suffix}.jpg")
    
    mean_distance = np.mean(distances)
    cover_rate_gt = len(final_matches) / len(A)
    cover_rate_pred = len(final_matches) / len(B)
    
    # Convert final matches back to original scale
    # final_matches = [((gt_i * h_res / h_ang, gt_j * v_res / v_ang), (ht_i * h_res / h_ang, ht_j * v_res / v_ang)) for ((gt_i,gt_j), (ht_i,ht_j)) in final_matches]

    plt.suptitle(f"Mean Distance: {mean_distance:.3f} deg, Procrustes Distance: {procrustes_distance:.3f} deg, Aligned Distance: {aligned_distance:.3f} deg\n \
              Cover Rate GT: {cover_rate_gt:.3f}, Cover Rate Pred: {cover_rate_pred:.3f} \n \
                Rotation={best_rot:.3f} deg, Scale={best_scale*100:.3f}%", fontsize=14)#, y=1.1)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return distances, final_matches, rigid_matches, (procrustes_distance, aligned_distance)
