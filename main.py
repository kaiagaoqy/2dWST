from shapely.geometry import Polygon, Point, MultiPolygon
import numpy as np
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import os
from matplotlib import pyplot as plt

def plot_trajectory(file, trajectory, gt_traj, gt_label, save_path):
    SCENE_BASE = 'scenes'
    plt.figure(figsize=(10, 10))
    img = plt.imread(os.path.join(SCENE_BASE, file))
    plt.imshow(img)
    plt.scatter(gt_traj[:,0], gt_traj[:,1], c='green', s=1, label='Ground Truth')
    plt.scatter(trajectory[:, 0], trajectory[:, 1], c='red', s=1, label='Hand Drawn')
    # 在 Ground Truth 轨迹起点加文字
    plt.text(
        min(gt_traj[:,0]), min(gt_traj[:,1]) - 15, gt_label,
        fontsize=12,
        color="#FFFBE6",
        weight="bold",
        bbox=dict(facecolor="#00712D", alpha=0.9, edgecolor="none", pad=2)
    )
    plt.axis('off')
    plt.legend()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def extract_outermost_boundary(geometry):
    # 如果是 MultiPolygon，合并所有多边形
    if geometry.geom_type == 'MultiPolygon':
        unified = unary_union(geometry)
    elif geometry.geom_type == 'Polygon':
        unified = geometry
    else:
        raise ValueError("Unsupported geometry type")
        
    # 提取最外层的边界
    if unified.geom_type == 'Polygon':
            return np.array(unified.exterior.coords)
    elif unified.geom_type == 'MultiPolygon':
        # 如果仍然是 MultiPolygon，取外层最大的 Polygon
        largest_polygon = max(unified.geoms, key=lambda p: p.area)
        return np.array(largest_polygon.exterior.coords)
    

def sample_poly(polygon, interval=5):
    """
    Sample points along the edges of a polygon at regular intervals.

    Parameters:
    - polygon (shapely.geometry.Polygon): The polygon to sample points from.
    - interval (float): The distance between sampled points.

    Returns:
    - list: A polygon with sampled points.
    """
    if isinstance(polygon, Polygon):
        # Get the exterior boundary of the polygon (a LineString).
        boundary = polygon.exterior
    else:
        boundary = polygon
        
    # Determine the total length of the boundary.
    total_length = boundary.length
    if total_length == 0:
        return Polygon(polygon) if not isinstance(polygon, Polygon) else polygon

    # Compute the number of sample points.
    # We use np.arange to generate distances along the boundary.
    sample_distances = np.arange(0, total_length, interval)

    # Optionally, add the final point to cover the entire boundary.
    if sample_distances[-1] < total_length:
        sample_distances = np.append(sample_distances, total_length)

    # Interpolate points along the boundary at these distances.
    sampled_points = [boundary.interpolate(d) for d in sample_distances]
    return Polygon(sampled_points) if len(sampled_points) > 4 else (Polygon(polygon) if not isinstance(polygon, Polygon) else polygon)


def list_depth(lst):
    if isinstance(lst, list) and lst:   # 如果是非空 list
        return 1 + max(list_depth(item) for item in lst)
    else:
        return 0
    
def get_poly_from_traj(pred_traj):
    if not isinstance(pred_traj, np.ndarray):
        depth = list_depth(pred_traj)  # Check the depth of the trajectory list
        if depth > 2:
            traj_li = [np.array(item).reshape(-1, 2) for item in pred_traj]
            pred_traj = np.concatenate(traj_li, axis=0)
        elif depth == 2:
            pred_traj = np.array(pred_traj)
        else:
            raise ValueError("Invalid trajectory format: expected a list of lists or a 2D array.")
    traj_poly = Polygon(pred_traj)
    traj_poly = Polygon(extract_outermost_boundary(traj_poly))
    traj_poly = sample_poly(traj_poly, interval=5)
    traj = np.array(traj_poly.exterior.coords)

    if not traj_poly.is_valid:
        traj_poly = traj_poly.buffer(0)
        if isinstance(traj_poly, MultiPolygon):
            traj_poly = max([geom for geom in traj_poly.geoms if isinstance(geom, Polygon)], key=lambda p: p.area)
        elif isinstance(traj_poly, Polygon):
            pass
    assert traj_poly.is_valid, "Predicted trajectory polygon is invalid"
    return traj



def main():
    import json
    import os
    import glob
    import numpy as np
    from tqdm import tqdm
    from shapely.geometry import LinearRing
    ANNO_BASE = 'annotations'
    RESULTS_BASE = 'results'


    files = glob.glob('results/sub321/*.json')
    for sub_file in tqdm(files, desc='Processing subjects'):
        with open(sub_file, 'r') as f:
            data = json.load(f)
        data['match'] = []
        for ind in tqdm(range(len(data['file'])), desc='Processing trajectories'):
            try:
                file = data['file'][ind]
                anno = data['annotations'][ind]
                import itertools

                merged = list(itertools.chain.from_iterable(
                    itertools.chain.from_iterable(data['points'][ind])
                ))
                trajectory = np.array(merged).reshape(-1, 3)
                assert trajectory.shape[0] > 0, "Trajectory is empty"

                gt_file = os.path.join(ANNO_BASE, os.path.basename(file).replace('.jpg', '.json'))
                with open(gt_file, 'r') as f:
                    gt_data = json.load(f)
                imageHeight = gt_data['imageHeight']
                imageWidth = gt_data['imageWidth']

                trajectory[:, 1] = imageHeight - trajectory[:, 1]    # Flip y-axis for correct orientation
                gt_file = os.path.join(ANNO_BASE, os.path.basename(file).replace('.jpg', '.json'))
                gt_data = {shape['label']: get_poly_from_traj(shape['points']) for shape in gt_data['shapes']}
            except Exception as e:
                print(f"Error processing file {file} at index {ind}: {e}")
                data['match'].append({'matched_label': None, 'avg_distance': None, 'distances': [], 'final_matches': []})
                continue
            from utils.registor import CPD
            from pathlib import Path
            res_dict = {'ind': ind,}
            dist_dict = {}
            for gt_label, gt_traj in gt_data.items():
                try:
                    pred_traj = trajectory[:, :2]  # Use only x and y coordinates
                    if not np.array_equal(pred_traj[0], pred_traj[-1]):
                        pred_traj = np.vstack([pred_traj, pred_traj[0]])
                    pred_traj = sample_poly(LinearRing(pred_traj), interval=5)
                    pred_traj = np.array(pred_traj.exterior.coords)

                    json_path = '/'.join(sub_file.split('/')[1:3])
                    save_path = Path(RESULTS_BASE, json_path.replace('.json','').lower())
                    distances, final_matches, rigid_matches, (procrustes_distance, aligned_distance) = CPD(gt_traj, pred_traj, save_path=save_path, suffix='_{ind}'.format(ind=ind),save_fig=False)
                    avg_distance = np.mean(distances)
                    dist_dict[gt_label] = {
                        'avg_distance': avg_distance,
                        'distances': distances,
                        'final_matches': final_matches,
                        'draw_traj': pred_traj.tolist(),
                        'gt_rate': len(final_matches) / len(gt_traj) if len(gt_traj) > 0 else 0,
                        'pred_rate': len(final_matches) / len(pred_traj) if len(pred_traj) > 0 else 0
                    }
                except Exception as e:
                    # print(f"Error processing trajectory for {gt_label} in {file}: {e}")
                    continue
            
            if not dist_dict:
                print(f"No valid trajectories found for {file} in {sub_file}. Skipping.")
                data['match'].append({'matched_label': None, 'avg_distance': None, 'distances': [], 'final_matches': []})
                continue
            matched_gt = sorted(
                    dist_dict.items(),
                    key=lambda x: x[1]["avg_distance"]
                )[0]
            res_dict['matched_label'] = matched_gt[0]
            res_dict['avg_distance'] = matched_gt[1]['avg_distance']
            res_dict['distances'] = matched_gt[1]['distances'].tolist()
            res_dict['final_matches'] = matched_gt[1]['final_matches']
            res_dict['draw_traj'] = matched_gt[1]['draw_traj']
            res_dict['gt_rate'] = matched_gt[1]['gt_rate']
            res_dict['pred_rate'] = matched_gt[1]['pred_rate']
            data['match'].append(res_dict)

            CPD(gt_data[matched_gt[0]], res_dict['draw_traj'], save_path=save_path, suffix='_{ind}'.format(ind=ind),save_fig=True)
            plot_trajectory(file, trajectory, gt_data[matched_gt[0]], gt_label=matched_gt[0], save_path=str(save_path/'trajectory_{ind}.png'.format(ind=ind)))
        with open(os.path.join(RESULTS_BASE,json_path), 'w') as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()