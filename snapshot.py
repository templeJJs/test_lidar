"""Interactive 3D view of the first frame from a bag."""
import sqlite3
import numpy as np
import open3d as o3d
import sys, os

from visualize_bag import parse_pointcloud2_cdr, colorize_rails, find_rail_lines, apply_rail_lines, build_rail_meshes

bag_dir = sys.argv[1] if len(sys.argv) > 1 else 'for_hackathon/doubleT_platform'
db3 = [f for f in os.listdir(bag_dir) if f.endswith('.db3')][0]
db_path = os.path.join(bag_dir, db3)

conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute("SELECT data FROM messages ORDER BY timestamp ASC LIMIT 1")
blob = cur.fetchone()[0]
conn.close()

pts, intensity = parse_pointcloud2_cdr(blob)
rail_lines, ground_plane = find_rail_lines(pts, intensity)
rail_mask = apply_rail_lines(pts, rail_lines, ground_plane)
print(f"Points: {len(pts)}, Rail lines: {len(rail_lines)}, Rail pts: {rail_mask.sum()}")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
pcd.colors = o3d.utility.Vector3dVector(colorize_rails(pts, intensity, rail_mask))

y_min, y_max = pts[:, 1].min() + 1, pts[:, 1].max() - 1
rail_meshes = build_rail_meshes(rail_lines, ground_plane, y_range=(y_min, y_max))
print(f"Rail meshes: {len(rail_meshes)}")

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Frame 0 — 3D", width=1280, height=720)
vis.add_geometry(pcd)
for mesh in rail_meshes:
    vis.add_geometry(mesh)

opt = vis.get_render_option()
opt.point_size = 3.0
opt.background_color = np.array([1.0, 1.0, 1.0])

# Камера как в visualize_bag.py
ctr = vis.get_view_control()
ctr.set_lookat([0.0, -7.0, -1.0])    # центр основного облака (медиана Y=-7)
ctr.set_front([0.0, 0.95, 0.10])
ctr.set_up([0.0, 0.0, 1.0])
ctr.set_zoom(0.04)

vis.run()
vis.destroy_window()
