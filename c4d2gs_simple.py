# -*- coding: utf-8 -*-
"""
C4D2GS Simple — Cinema 4D to Gaussian Splat
=================================================================

A simple Python script for Cinema 4D that generates synthetic COLMAP data.

Usage:
------
1. Open Cinema 4D and select the object you want to splat
2. Edit the SETTINGS section below to customize
3. Run this script: Plugins > Run Python Script !make sure the target object is selected!
4. Render the animation to produce the image sequence
5. Import the synthetic COLMAP data folder into your GS pipeline 

License: CC BY-NC 4.0
Author: Joel Tenenberg
"""

import c4d
import math
import json
import os
import random
import bisect

doc: c4d.documents.BaseDocument  # The currently active document.
op: c4d.BaseObject | None  # The primary selected object in `doc`. Can be `None`.

# ===========================================================================
# SETTINGS — Edit these values before running
# ===========================================================================

# Camera rig setup
CAMERA_COUNT = 120              # Number of viewpoints around the object
SPHERE_RADIUS = 300.0           # Distance from object center to each camera

# Render output
# Must be set before execution. Example: r"C:\renders\my_capture"
OUTPUT_PATH = r"" 
RESOLUTION_X = 1080
RESOLUTION_Y = 1080
FPS = 30
ENABLE_STRAIGHT_ALPHA = True       # If True, attempt to render with straight alpha channel.

# Exports
EXPORT_CAMERA_POSES_JSON = False       # Exports camera_poses.json (for nerfstudio)
EXPORT_COLMAP_DATA = True             # Exports synthetic COLMAP data files

# Sparse point cloud
SPARSE_POINT_COUNT = 20000    # Number of 3D points sampled from object surface
REPLACE_EXISTING_RIG = True     # Remove old GS_CameraRig before building new one

# ===========================================================================
# CORE FUNCTIONS
# ===========================================================================

def normalize_vec(v):
    """Normalize a vector."""
    length = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    if length <= 0.0:
        return c4d.Vector(0, 1, 0)
    return c4d.Vector(v.x / length, v.y / length, v.z / length)


def _normalize(v):
    return normalize_vec(v)


def dot(a, b):
    """Dot product of two vectors."""
    return a.x * b.x + a.y * b.y + a.z * b.z


def _dot(a, b):
    return dot(a, b)


def _clean_small(value, eps=1e-10):
    try:
        v = float(value)
    except Exception:
        return value
    if abs(v) < float(eps):
        return 0.0
    return v


def cross(a, b):
    """Cross product of two vectors."""
    return c4d.Vector(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )


def dot(a, b):
    """Dot product of two vectors."""
    return a.x * b.x + a.y * b.y + a.z * b.z


def look_at_matrix(camera_pos, target_pos, up_hint=None):
    """Build a camera-to-world matrix looking at target_pos."""
    if up_hint is None:
        up_hint = c4d.Vector(0, 1, 0)
    
    z_axis = normalize_vec(camera_pos - target_pos)
    if abs(dot(z_axis, up_hint)) > 0.999:
        up_hint = c4d.Vector(0, 0, 1)
    
    x_axis = normalize_vec(cross(up_hint, z_axis))
    y_axis = normalize_vec(cross(z_axis, x_axis))
    
    mg = c4d.Matrix()
    mg.off = camera_pos
    mg.v1 = x_axis
    mg.v2 = y_axis
    mg.v3 = z_axis
    return mg


def fibonacci_sphere_points(count):
    """Generate points on a sphere using golden-angle Fibonacci sphere."""
    if count <= 0:
        return []
    
    points = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    
    for i in range(count):
        y = 1.0 - (2.0 * i) / float(max(1, count - 1))
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * i
        points.append(c4d.Vector(math.sin(theta) * radius, y, math.cos(theta) * radius))
    
    return points


def get_object_center(obj):
    """Get world-space center of an object."""
    if obj is None:
        return c4d.Vector(0)
    # Convert local bounding-box center into world space.
    return obj.GetMp() * obj.GetMg()


def matrix_to_rows(mg):
    """Convert Cinema 4D matrix to 4x4 row-major format."""
    return [
        [mg.v1.x, mg.v2.x, mg.v3.x, mg.off.x],
        [mg.v1.y, mg.v2.y, mg.v3.y, mg.off.y],
        [mg.v1.z, mg.v2.z, mg.v3.z, mg.off.z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def rotation_matrix_to_quaternion(r):
    """Convert 3x3 rotation matrix to quaternion (qw, qx, qy, qz)."""
    trace = r[0][0] + r[1][1] + r[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return 0.25 * s, (r[2][1] - r[1][2]) / s, (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s
    elif (r[0][0] > r[1][1]) and (r[0][0] > r[2][2]):
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2.0
        return (r[2][1] - r[1][2]) / s, 0.25 * s, (r[0][1] + r[1][0]) / s, (r[0][2] + r[2][0]) / s
    elif r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2.0
        return (r[0][2] - r[2][0]) / s, (r[0][1] + r[1][0]) / s, 0.25 * s, (r[1][2] + r[2][1]) / s
    else:
        s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2.0
        return (r[1][0] - r[0][1]) / s, (r[0][2] + r[2][0]) / s, (r[1][2] + r[2][1]) / s, 0.25 * s


def c2w_to_colmap_extrinsics(mg):
    """Convert C4D camera-to-world matrix to COLMAP extrinsics (quat, translation, rotation)."""
    flip_y = c4d.Matrix()
    flip_y.v1 = c4d.Vector(1, 0, 0)
    flip_y.v2 = c4d.Vector(0, -1, 0)
    flip_y.v3 = c4d.Vector(0, 0, 1)
    flip_y.off = c4d.Vector(0, 0, 0)

    mg1 = mg * flip_y

    def _apply_flip_y(mat):
        out = c4d.Matrix()
        out.v1 = c4d.Vector(mat.v1.x, -mat.v1.y, mat.v1.z)
        out.v2 = c4d.Vector(mat.v2.x, -mat.v2.y, mat.v2.z)
        out.v3 = c4d.Vector(mat.v3.x, -mat.v3.y, mat.v3.z)
        out.off = c4d.Vector(mat.off.x, -mat.off.y, mat.off.z)
        return out

    mg2 = _apply_flip_y(mg1)

    c_pos = mg2.off
    r_w2c = [
        [mg2.v1.x, mg2.v1.y, mg2.v1.z],
        [mg2.v2.x, mg2.v2.y, mg2.v2.z],
        [mg2.v3.x, mg2.v3.y, mg2.v3.z],
    ]
    tx = -(r_w2c[0][0] * c_pos.x + r_w2c[0][1] * c_pos.y + r_w2c[0][2] * c_pos.z)
    ty = -(r_w2c[1][0] * c_pos.x + r_w2c[1][1] * c_pos.y + r_w2c[1][2] * c_pos.z)
    tz = -(r_w2c[2][0] * c_pos.x + r_w2c[2][1] * c_pos.y + r_w2c[2][2] * c_pos.z)
    qw, qx, qy, qz = rotation_matrix_to_quaternion(r_w2c)

    return (qw, qx, qy, qz), (tx, ty, tz), r_w2c


def project_world_to_image(mg, world_point, world_normal, fx, fy, cx, cy, require_front_facing=True):
    """Project world point to image coordinates with COLMAP conventions."""
    flip_y = c4d.Matrix(); flip_y.v1 = c4d.Vector(1,0,0); flip_y.v2 = c4d.Vector(0,-1,0); flip_y.v3 = c4d.Vector(0,0,1); flip_y.off = c4d.Vector(0,0,0)

    def _apply_flip_y_vec(v):
        return c4d.Vector(v.x, -v.y, v.z)

    def _apply_flip_y_mat(mat):
        out = c4d.Matrix()
        out.v1 = c4d.Vector(mat.v1.x, -mat.v1.y, mat.v1.z)
        out.v2 = c4d.Vector(mat.v2.x, -mat.v2.y, mat.v2.z)
        out.v3 = c4d.Vector(mat.v3.x, -mat.v3.y, mat.v3.z)
        out.off = c4d.Vector(mat.off.x, -mat.off.y, mat.off.z)
        return out

    mg1 = mg * flip_y
    mg2 = _apply_flip_y_mat(mg1)

    p_col = _apply_flip_y_vec(world_point)
    if world_normal is not None:
        n_col = _apply_flip_y_vec(world_normal)
        if require_front_facing:
            to_cam_col = _normalize(mg2.off - p_col)
            if _dot(to_cam_col, n_col) <= 0.0:
                return None

    local = (~mg2) * p_col
    x_cv = local.x
    y_cv = local.y
    z_cv = local.z
    if z_cv <= 1e-6:
        return None

    return (fx * (x_cv / z_cv)) + cx, (fy * (y_cv / z_cv)) + cy


def collect_triangles_from_object(obj):
    """Collect all triangles from an object (including caches/deformers)."""
    triangles = []
    
    def walk_hierarchy(node):
        if node is None:
            return
        if node.CheckType(c4d.Opolygon):
            mg = node.GetMg()
            points = node.GetAllPoints()
            if points:
                world_pts = [p * mg for p in points]
                for poly in node.GetAllPolygons():
                    a, b, c = world_pts[poly.a], world_pts[poly.b], world_pts[poly.c]
                    triangles.append((a, b, c))
                    if poly.c != poly.d:
                        d = world_pts[poly.d]
                        triangles.append((a, c, d))
        
        # Walk children
        child = node.GetDown()
        while child:
            walk_hierarchy(child)
            child = child.GetNext()
    
    # Check main object
    walk_hierarchy(obj)
    
    # Check deform/render caches
    for cache_source in [obj.GetDeformCache(), obj.GetCache()]:
        if cache_source:
            walk_hierarchy(cache_source)
    
    return triangles


def triangle_area(a, b, c):
    """Calculate area of a triangle."""
    return (cross(b - a, c - a).GetLength()) * 0.5


def sample_on_triangle(a, b, c):
    """Uniform random sample on triangle surface."""
    r1 = math.sqrt(random.random())
    r2 = random.random()
    return a * (1.0 - r1) + b * (r1 * (1.0 - r2)) + c * (r1 * r2)


def generate_sparse_points_from_surface(doc, obj, count=256):
    """Generate sparse points by area-weighted sampling from object surface (with normals)."""
    triangles = collect_triangles_from_object(obj)
    if not triangles:
        return None

    # Build area-weighted distribution
    areas = []
    cumulative = []
    running = 0.0

    for tri in triangles:
        ar = triangle_area(tri[0], tri[1], tri[2])
        if ar <= 1e-12:
            continue
        areas.append((tri, ar))
        running += ar
        cumulative.append(running)

    if running <= 0.0 or not areas:
        return None

    # Sample points and normals
    out = []
    for _ in range(count):
        r = random.random() * running
        idx = bisect.bisect_left(cumulative, r)
        if idx >= len(areas):
            idx = len(areas) - 1
        a, b, c = areas[idx][0]
        p = sample_on_triangle(a, b, c)
        n = normalize_vec(cross(b - a, c - a))
        out.append((p, n))

    return out


def generate_sparse_points_in_core_volume(target_obj, target_pos, count=256, radius_factor=0.35):
    """Fallback sparse points sampled inside a core sphere of the target bounds."""
    if target_obj is None:
        return None
    count = max(8, int(count))
    try:
        rad = target_obj.GetRad()
        diag = math.sqrt(rad.x ** 2 + rad.y ** 2 + rad.z ** 2)
        base_radius = max(diag * 2.5, 10.0)
    except Exception:
        base_radius = 300.0

    core_radius = max(1.0, base_radius * max(0.05, float(radius_factor)))
    out = []
    for _ in range(count):
        u = random.random()
        v = random.random()
        w = random.random()
        theta = 2.0 * math.pi * u
        phi = math.acos(max(-1.0, min(1.0, 2.0 * v - 1.0)))
        r = core_radius * (w ** (1.0 / 3.0))
        sx = r * math.sin(phi) * math.cos(theta)
        sy = r * math.cos(phi)
        sz = r * math.sin(phi) * math.sin(theta)
        out.append((target_pos + c4d.Vector(sx, sy, sz), None))
    return out


def get_output_extension():
    """Output extension is fixed to PNG."""
    return ".png"


def build_frame_image_path(frame_index):
    """Build output path for a single frame."""
    token = "{:04d}".format(frame_index)
    base = os.path.join(get_images_output_dir(), "gs_{}".format(token))
    ext = get_output_extension()
    return base if base.lower().endswith(ext.lower()) else base + ext


def get_render_output_dir():
    """Get export root output directory."""
    path = str(OUTPUT_PATH).strip()
    if not path:
        return ""
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def get_images_output_dir():
    """Get image sequence output directory under export root."""
    return os.path.join(get_render_output_dir(), "images")


def get_render_output_pattern():
    """Get C4D render output pattern for frame sequence."""
    return os.path.join(get_images_output_dir(), "gs_####")


def _copy_matrix(mg):
    out = c4d.Matrix()
    out.off = c4d.Vector(mg.off.x, mg.off.y, mg.off.z)
    out.v1 = c4d.Vector(mg.v1.x, mg.v1.y, mg.v1.z)
    out.v2 = c4d.Vector(mg.v2.x, mg.v2.y, mg.v2.z)
    out.v3 = c4d.Vector(mg.v3.x, mg.v3.y, mg.v3.z)
    return out


def _camera_matrices_for_export(doc, render_cam, frame_count, fps):
    if doc is None or render_cam is None or frame_count <= 0:
        return None

    current_time = doc.GetTime()
    out = []
    try:
        for frame in range(frame_count):
            doc.SetTime(c4d.BaseTime(frame, fps))
            try:
                doc.ExecutePasses(None, True, True, True, getattr(c4d, "BUILDFLAGS_NONE", 0))
            except Exception:
                pass
            out.append(_copy_matrix(render_cam.GetMg()))
    finally:
        doc.SetTime(current_time)
        try:
            doc.ExecutePasses(None, True, True, True, getattr(c4d, "BUILDFLAGS_NONE", 0))
        except Exception:
            pass

    if len(out) != frame_count:
        return None
    return out


def get_colmap_intrinsics(render_cam):
    """Get COLMAP intrinsics from render camera only."""
    focus_pid = getattr(c4d, "CAMERA_FOCUS", None)
    aperture_pid = getattr(c4d, "CAMERAOBJECT_APERTURE", None)

    if render_cam is None:
        raise ValueError("Render camera is required to read intrinsics.")
    if not focus_pid or not aperture_pid:
        raise ValueError("Cinema 4D camera intrinsics identifiers are unavailable.")

    focal_mm = float(render_cam[focus_pid])
    aperture_w_mm = float(render_cam[aperture_pid])
    if focal_mm <= 0 or aperture_w_mm <= 0:
        raise ValueError("Invalid camera intrinsics: focal length and aperture must be > 0.")
    if RESOLUTION_X <= 0 or RESOLUTION_Y <= 0:
        raise ValueError("Invalid resolution for intrinsics computation.")

    aperture_h_mm = aperture_w_mm * (float(RESOLUTION_Y) / float(RESOLUTION_X))
    fx = (focal_mm / aperture_w_mm) * float(RESOLUTION_X)
    fy = (focal_mm / aperture_h_mm) * float(RESOLUTION_Y)
    return {
        "fx": fx,
        "fy": fy,
        "cx": RESOLUTION_X * 0.5,
        "cy": RESOLUTION_Y * 0.5,
        "source": "render_camera",
    }


def write_colmap_files(world_points, target_pos, output_dir, render_cam, doc, target_obj, camera_matrices=None):
    """Export synthetic COLMAP data files (cameras.txt, images.txt, points3D.txt)."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    images_dir = os.path.join(output_dir, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    cameras_txt = os.path.join(output_dir, "cameras.txt")
    images_txt = os.path.join(output_dir, "images.txt")
    points3d_txt = os.path.join(output_dir, "points3D.txt")

    intrinsics = get_colmap_intrinsics(render_cam)
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    width = float(RESOLUTION_X)
    height = float(RESOLUTION_Y)

    if doc is None or target_obj is None:
        raise ValueError("A target object is required for synthetic COLMAP data export.")

    sparse_points_with_normals = generate_sparse_points_from_surface(doc, target_obj, SPARSE_POINT_COUNT)
    if not sparse_points_with_normals:
        raise ValueError("Could not sample sparse points from object surface. Ensure object has polygonal geometry.")

    if camera_matrices is None:
        camera_matrices = _camera_matrices_for_export(doc, render_cam, len(world_points), FPS)

    def _build_image_entries(use_camera_matrices):
        entries = []
        for i, world_pos in enumerate(world_points):
            if use_camera_matrices and camera_matrices and i < len(camera_matrices):
                mg = camera_matrices[i]
            else:
                mg = look_at_matrix(world_pos, target_pos)
            q, t, r_w2c = c2w_to_colmap_extrinsics(mg)
            entries.append({
                "image_id": i + 1,
                "name": os.path.basename(build_frame_image_path(i)),
                "q": q,
                "t": t,
                "r_w2c": r_w2c,
                "mg": mg,
                "obs": [],
            })
        return entries

    image_entries = _build_image_entries(use_camera_matrices=True)
    image_entries = sorted(image_entries, key=lambda e: e["name"])
    for idx, entry in enumerate(image_entries, start=1):
        entry["image_id"] = idx

    max_obs = max(2, int(12))

    def _build_tracks(require_front_facing):
        tracks = {}
        for entry in image_entries:
            entry["obs"] = []
        for pid, (p3d, nrm) in enumerate(sparse_points_with_normals, start=1):
            tracks[pid] = []
            candidates = []
            for entry in image_entries:
                projected = project_world_to_image(entry["mg"], p3d, nrm, fx, fy, cx, cy, require_front_facing=require_front_facing)
                if projected is None:
                    continue
                u, v = projected
                if 0.0 <= u < width and 0.0 <= v < height:
                    candidates.append((entry, u, v))
            for entry, u, v in candidates[:max_obs]:
                p2d_idx = len(entry["obs"])
                entry["obs"].append((u, v, pid))
                tracks[pid].append((entry["image_id"], p2d_idx))
        return tracks

    tracks_by_pid = _build_tracks(require_front_facing=True)
    valid_points = [(pid, p3d, tracks_by_pid[pid]) for pid, (p3d, _nrm) in enumerate(sparse_points_with_normals, start=1) if len(tracks_by_pid.get(pid, [])) >= 1]

    if not valid_points:
        tracks_by_pid = _build_tracks(require_front_facing=False)
        valid_points = [(pid, p3d, tracks_by_pid[pid]) for pid, (p3d, _nrm) in enumerate(sparse_points_with_normals, start=1) if len(tracks_by_pid.get(pid, [])) >= 1]

    if not valid_points:
        core_points = generate_sparse_points_in_core_volume(target_obj, target_pos, SPARSE_POINT_COUNT, 0.35)
        if core_points:
            sparse_points_with_normals = core_points
            tracks_by_pid = _build_tracks(require_front_facing=False)
            valid_points = [(pid, p3d, tracks_by_pid[pid]) for pid, (p3d, _nrm) in enumerate(sparse_points_with_normals, start=1) if len(tracks_by_pid.get(pid, [])) >= 1]

    if not valid_points:
        raise ValueError("No sparse point had >= 1 observation after visibility checks.")

    with open(cameras_txt, "w") as f:
        f.write("# Camera list\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        if abs(float(fx) - float(fy)) < 1e-6:
            f.write("1 SIMPLE_PINHOLE {} {} {} {} {}\n".format(int(RESOLUTION_X), int(RESOLUTION_Y), fx, cx, cy))
        else:
            f.write("1 PINHOLE {} {} {} {} {} {}\n".format(int(RESOLUTION_X), int(RESOLUTION_Y), fx, fy, cx, cy))

    with open(images_txt, "w") as f:
        f.write("# Image list\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write("# Number of images: {}\n".format(len(image_entries)))
        for entry in image_entries:
            qw, qx, qy, qz = entry["q"]
            tx, ty, tz = entry["t"]
            f.write("{} {} {} {} {} {} {} {} 1 {}\n".format(entry["image_id"], _clean_small(qw), _clean_small(qx), _clean_small(qy), _clean_small(qz), _clean_small(tx), _clean_small(ty), _clean_small(tz), entry["name"]))
            if entry["obs"]:
                f.write(" ".join("{} {} {}".format(o[0], o[1], o[2]) for o in entry["obs"]) + "\n")
            else:
                f.write("\n")

    with open(points3d_txt, "w") as f:
        f.write("# 3D point list\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write("# Number of points: {}\n".format(len(valid_points)))
        for pid, p3d, track in valid_points:
            track_flat = " ".join("{} {}".format(img_id, p2d_idx) for img_id, p2d_idx in track)
            colmap_x = p3d.x
            colmap_y = -p3d.y
            colmap_z = p3d.z
            f.write("{} {} {} {} 255 255 255 1.0 {}\n".format(pid, colmap_x, colmap_y, colmap_z, track_flat))

    return {
        "dir": output_dir,
        "intrinsics_source": intrinsics.get("source", "manual"),
        "points": len(valid_points),
    }


def write_pose_json(world_points, target_pos, output_path, render_cam):
    """Export camera poses as JSON (for nerfstudio/instant-ngp)."""
    intrinsics = get_colmap_intrinsics(render_cam)
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    
    frames = []
    for i, world_pos in enumerate(world_points):
        mg = look_at_matrix(world_pos, target_pos)
        hpb = c4d.utils.MatrixToHPB(mg)
        
        frames.append({
            "frame": i,
            "position": [world_pos.x, world_pos.y, world_pos.z],
            "rotation_hpb_rad": [hpb.x, hpb.y, hpb.z],
            "transform_matrix": matrix_to_rows(mg),
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        })
    
    payload = {
        "coordinate_system": "Cinema4D_Yup_RightHanded",
        "camera_model": "PINHOLE",
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": RESOLUTION_X,
        "height": RESOLUTION_Y,
        "frame_count": len(world_points),
        "frames": frames,
    }
    
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def find_object_by_name(root, name):
    """Find object by name in hierarchy."""
    obj = root
    while obj:
        if obj.GetName() == name:
            return obj
        child = obj.GetDown()
        if child:
            found = find_object_by_name(child, name)
            if found:
                return found
        obj = obj.GetNext()
    return None


def create_target_tag(cam, target):
    """Add target tag to camera pointing at target object."""
    tag = c4d.BaseTag(c4d.Ttargetexpression)
    if tag:
        tag[c4d.TARGETEXPRESSIONTAG_LINK] = target
        cam.InsertTag(tag)


def add_position_key(cam, time, pos, fps):
    """Add position keyframe to camera."""
    for axis_idx, axis_val in enumerate([pos.x, pos.y, pos.z]):
        if axis_idx == 0:
            did = c4d.DescID(
                c4d.DescLevel(c4d.ID_BASEOBJECT_REL_POSITION, c4d.DTYPE_VECTOR, 0),
                c4d.DescLevel(c4d.VECTOR_X, c4d.DTYPE_REAL, 0),
            )
        elif axis_idx == 1:
            did = c4d.DescID(
                c4d.DescLevel(c4d.ID_BASEOBJECT_REL_POSITION, c4d.DTYPE_VECTOR, 0),
                c4d.DescLevel(c4d.VECTOR_Y, c4d.DTYPE_REAL, 0),
            )
        else:
            did = c4d.DescID(
                c4d.DescLevel(c4d.ID_BASEOBJECT_REL_POSITION, c4d.DTYPE_VECTOR, 0),
                c4d.DescLevel(c4d.VECTOR_Z, c4d.DTYPE_REAL, 0),
            )
        
        track = cam.FindCTrack(did)
        if not track:
            track = c4d.CTrack(cam, did)
            cam.InsertTrackSorted(track)
        
        curve = track.GetCurve()
        key_data = curve.AddKey(time)
        if key_data:
            key = key_data["key"]
            key.SetValue(curve, float(axis_val))
            key.SetInterpolation(curve, c4d.CINTERPOLATION_STEP)


# ===========================================================================
# MAIN FUNCTION
# ===========================================================================

def main() -> None:
    if doc is None:
        c4d.gui.MessageDialog("No active Cinema 4D document.")
        return

    if not str(OUTPUT_PATH).strip():
        c4d.gui.MessageDialog(
            "Output Path is empty.\n\n"
            "Set OUTPUT_PATH in the script settings before execution.\n"
            "Example: r\"C:\\renders\\my_capture\""
        )
        return

    output_dir = get_render_output_dir()
    images_dir = get_images_output_dir()
    try:
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
    except Exception as e:
        c4d.gui.MessageDialog("Could not create export folders:\n{}".format(e))
        return

    # Prefer Script Manager's primary selection (`op`) when available.
    target_obj = op if op is not None else doc.GetActiveObject()
    if not target_obj:
        c4d.gui.MessageDialog("Select an object in the Object Manager.")
        return
    
    # Calculate target position
    target_pos = get_object_center(target_obj)
    
    # Generate camera positions
    unit_points = fibonacci_sphere_points(CAMERA_COUNT)
    world_points = [target_pos + p * SPHERE_RADIUS for p in unit_points]
    
    doc.StartUndo()
    try:
        # Remove old rig if requested
        if REPLACE_EXISTING_RIG:
            old_rig = find_object_by_name(doc.GetFirstObject(), "GS_CameraRig")
            if old_rig:
                doc.AddUndo(c4d.UNDOTYPE_DELETEOBJ, old_rig)
                old_rig.Remove()
        
        # Create rig null
        rig = c4d.BaseObject(c4d.Onull)
        rig.SetName("GS_CameraRig")
        rig.SetAbsPos(target_pos)
        doc.InsertObject(rig)
        doc.AddUndo(c4d.UNDOTYPE_NEWOBJ, rig)
        
        # Create target null
        target_null = c4d.BaseObject(c4d.Onull)
        target_null.SetName("GS_Target")
        target_null.InsertUnder(rig)
        target_null.SetAbsPos(target_pos)
        doc.AddUndo(c4d.UNDOTYPE_NEWOBJ, target_null)
        
        # Create static reference cameras
        for i, world_pos in enumerate(world_points):
            cam = c4d.BaseObject(c4d.Ocamera)
            cam.SetName("GS_Cam_{:04d}".format(i))
            cam.InsertUnder(rig)
            cam.SetAbsPos(world_pos)
            create_target_tag(cam, target_null)
            doc.AddUndo(c4d.UNDOTYPE_NEWOBJ, cam)
        
        # Create animated render camera
        render_cam = c4d.BaseObject(c4d.Ocamera)
        render_cam.SetName("GS_RenderCam")
        render_cam.InsertUnder(rig)
        render_cam.SetAbsPos(world_points[0])
        create_target_tag(render_cam, target_null)
        doc.AddUndo(c4d.UNDOTYPE_NEWOBJ, render_cam)
        
        # Keyframe position
        for frame, world_pos in enumerate(world_points):
            local_pos = world_pos - target_pos
            t = c4d.BaseTime(frame, FPS)
            add_position_key(render_cam, t, local_pos, FPS)
        
        # Configure render settings
        rd = doc.GetActiveRenderData()
        if rd:
            rd[c4d.RDATA_XRES] = RESOLUTION_X
            rd[c4d.RDATA_YRES] = RESOLUTION_Y
            rd[c4d.RDATA_FRAMERATE] = FPS
            rd[c4d.RDATA_SAVEIMAGE] = True
            rd[c4d.RDATA_PATH] = get_render_output_pattern()
            rd[c4d.RDATA_FORMAT] = c4d.FILTER_PNG
            rd[c4d.RDATA_FRAMESEQUENCE] = c4d.RDATA_FRAMESEQUENCE_ALLFRAMES
            rd[c4d.RDATA_FRAMEFROM] = c4d.BaseTime(0, FPS)
            rd[c4d.RDATA_FRAMETO] = c4d.BaseTime(len(world_points) - 1, FPS)

            if ENABLE_STRAIGHT_ALPHA:
                for attr_name in [
                    "RDATA_ALPHACHANNEL",
                    "RDATA_ALPHA",
                    "RDATA_STRAIGHT_ALPHA",
                    "RDATA_STRAIGHTALPHA",
                ]:
                    attr = getattr(c4d, attr_name, None)
                    if attr is not None:
                        rd[attr] = True

            if hasattr(c4d, "RDATA_CAMERA"):
                rd[c4d.RDATA_CAMERA] = render_cam

        # Select the created render camera and make it active for rendering.
        try:
            doc.SetActiveObject(render_cam, c4d.SELECTION_NEW)
        except Exception:
            pass

        bd = doc.GetActiveBaseDraw()
        if bd is not None and render_cam is not None:
            try:
                bd.SetSceneCamera(render_cam)
            except Exception:
                pass

        doc.SetTime(c4d.BaseTime(0, FPS))
        c4d.EventAdd()
        
        # Export files
        export_status = []
        
        if EXPORT_COLMAP_DATA:
            try:
                # Store COLMAP files at the export root
                colmap_dir = output_dir
                cam_matrices = _camera_matrices_for_export(doc, render_cam, len(world_points), FPS)
                result = write_colmap_files(world_points, target_pos, colmap_dir, render_cam, doc, target_obj, camera_matrices=cam_matrices)
                export_status.append("✓ Synthetic COLMAP data: {} points in {}".format(result["points"], result["dir"]))
            except Exception as e:
                export_status.append("✗ Synthetic COLMAP data export failed: {}".format(e))
        
        if EXPORT_CAMERA_POSES_JSON:
            try:
                pose_path = os.path.join(output_dir, "camera_poses.json")
                write_pose_json(world_points, target_pos, pose_path, render_cam)
                export_status.append("✓ Pose JSON: {}".format(pose_path))
            except Exception as e:
                export_status.append("✗ Pose JSON export failed: {}".format(e))
        
        msg = "Camera rig ready! {} cameras created.\n\n".format(len(world_points))
        msg += "\n".join(export_status)
        msg += "\n\nNext: Render animation to produce image sequence.\n"
        msg += "Then import the synthetic COLMAP data folder into your reconstruction pipeline."
        
        c4d.gui.MessageDialog(msg)
    
    finally:
        doc.EndUndo()


if __name__ == "__main__":
    main()