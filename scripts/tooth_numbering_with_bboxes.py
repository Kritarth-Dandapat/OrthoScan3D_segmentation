import torch
import numpy as np
import trimesh
from sklearn.cluster import DBSCAN
from models.dilated_tooth_seg_network import DilatedToothSegmentationNetwork
from dataset.preprocessing import PreTransform
from utils.mesh_io import read_mesh, save_mesh
from utils.teeth_numbering import _teeth_labels

# Parameters
PLY_PATH = '00OMSZGW_lower.ply'
CKPT_PATH = 'best_model_full_dataset_full_dataset_training_v1.pth'
OUT_PATH = '00OMSZGW_lower_with_bboxes.ply'
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'

# 1. Load mesh
mesh = trimesh.load(PLY_PATH)

# --- DOWNSAMPLE MESH ---
# Target number of faces (adjust as needed)
TARGET_FACES = 4096
if len(mesh.faces) > TARGET_FACES:
    # Use trimesh's simplification (requires open3d or pyfqmr, fallback to random sampling)
    try:
        mesh = mesh.simplify_quadratic_decimation(TARGET_FACES)
        print(f"Mesh downsampled to {len(mesh.faces)} faces using quadratic decimation.")
    except Exception as e:
        print(f"Quadratic decimation failed: {e}. Using random face sampling.")
        idx = np.random.choice(len(mesh.faces), TARGET_FACES, replace=False)
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[idx])
        print(f"Mesh downsampled to {len(mesh.faces)} faces by random sampling.")
else:
    print(f"Mesh has {len(mesh.faces)} faces, no downsampling needed.")

# 2. Extract features (as in PreTransform)
def extract_features(mesh):
    mesh_faces = torch.from_numpy(mesh.faces.copy()).float()
    mesh_triangles = torch.from_numpy(mesh.vertices[mesh.faces]).float()
    mesh_face_normals = torch.from_numpy(mesh.face_normals.copy()).float()
    mesh_vertices_normals = torch.from_numpy(mesh.vertex_normals[mesh.faces]).float()
    labels = torch.zeros(mesh_faces.shape[0], dtype=torch.long)  # dummy labels
    data = (mesh_faces, mesh_triangles, mesh_vertices_normals, mesh_face_normals, labels)
    pre = PreTransform(classes=17)
    pos, x, _ = pre(data)
    return pos, x

pos, x = extract_features(mesh)

# 3. Run segmentation model
model = DilatedToothSegmentationNetwork(num_classes=17, feature_dim=24).to(DEVICE)
# --- Handle DataParallel checkpoints ---
def load_ckpt_strip_module(model, ckpt_path):
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    return model

model = load_ckpt_strip_module(model, CKPT_PATH)
model.eval()

with torch.no_grad():
    x = x.unsqueeze(0).to(DEVICE)  # (1, N, 24)
    pos = pos.unsqueeze(0).to(DEVICE)  # (1, N, 3)
    logits = model(x, pos)  # (1, N, num_classes)
    pred_labels = torch.argmax(logits, dim=2).cpu().numpy().squeeze()  # (N,)

# 4. Cluster faces of each tooth class (excluding gum=0)
face_centers = x[0, :, 9:12].cpu().numpy()  # (N, 3)
clusters = []
bboxes = []
visual_colors = np.array(mesh.visual.face_colors)

for class_idx, class_name in _teeth_labels.items():
    if class_idx == 0 or class_idx not in pred_labels:
        continue  # skip gum or unused classes
    mask = pred_labels == class_idx
    if np.sum(mask) == 0:
        continue
    # DBSCAN clustering
    db = DBSCAN(eps=2.5, min_samples=10).fit(face_centers[mask])
    for cluster_id in np.unique(db.labels_):
        if cluster_id == -1:
            continue  # noise
        cluster_mask = (db.labels_ == cluster_id)
        indices = np.where(mask)[0][cluster_mask]
        clusters.append(indices)
        # Compute bounding box
        points = face_centers[indices]
        extents, transform = trimesh.bounds.oriented_bounds(points)
        box = trimesh.primitives.Box(extents=extents, transform=transform)
        bbox_corners = box.vertices
        bboxes.append((class_idx, class_name, bbox_corners))
        # Color faces in this cluster for visualization
        color = np.random.randint(0, 255, 3)
        visual_colors[indices, :3] = color
        visual_colors[indices, 3] = 255

# 5. Add bounding boxes as line entities to the mesh scene
scene = trimesh.Scene()
scene.add_geometry(mesh)
for class_idx, class_name, bbox in bboxes:
    # bbox is (8, 3) corners
    # Create lines for the bounding box
    lines = [
        [bbox[0], bbox[1]], [bbox[1], bbox[2]], [bbox[2], bbox[3]], [bbox[3], bbox[0]],  # bottom
        [bbox[4], bbox[5]], [bbox[5], bbox[6]], [bbox[6], bbox[7]], [bbox[7], bbox[4]],  # top
        [bbox[0], bbox[4]], [bbox[1], bbox[5]], [bbox[2], bbox[6]], [bbox[3], bbox[7]],  # sides
    ]
    for line in lines:
        tube = trimesh.load_path(np.array(line))
        scene.add_geometry(tube)

# 6. Save the mesh with colored faces and bounding boxes
mesh.visual.face_colors = visual_colors
scene.export(OUT_PATH)
print(f'Saved mesh with bounding boxes to {OUT_PATH}') 