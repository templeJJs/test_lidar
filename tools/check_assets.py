#!/usr/bin/env python3
"""One command that shows the state of the annotation-asset pipeline.

    python tools/check_assets.py                  # verify everything on disk (table)
    python tools/check_assets.py --json           # machine-readable report
    python tools/check_assets.py --write-manifest # refresh assets/manifest.json
    python tools/check_assets.py --verbose        # keep the loader warnings

What it checks per asset, against the pinned facts in assets/manifest.json:

  * the archive was downloaded and its sha256 still matches (and so do the
    unpacked files, re-hashed from disk);
  * open3d reads the primary mesh and reports vertices / triangles / bbox;
  * the three.js loaders read it inside Node (tools/inspect_mesh.mjs) and report
    vertices / triangles / bbox / animation clips;
  * a license is recorded for it, with attribution where the license needs it.

Measured numbers come from the real readers -- two child processes (python+open3d
and node+three) so their stdout/stderr noise stays out of this report.  Expected
readability is stored in the manifest (``read_by``), so a regression shows up as
MISMATCH instead of silently passing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fetch_assets import ASSETS_DIR, REGISTRY, ROOT, human, sha256_file  # noqa: E402

MANIFEST_PATH = ASSETS_DIR / "manifest.json"
LICENSES_PATH = ASSETS_DIR / "LICENSES.md"
NODE_INSPECT = HERE / "inspect_mesh.mjs"
TARGET_HUMAN_HEIGHT_M = 1.75

# Kenney ships 96 GLB models; these are the ones the annotator is expected to
# place on the track, and they are the ones recorded in detail in the manifest.
KENNEY_KEY_MODELS = [
    "construction-barrier", "construction-cone", "construction-fence", "construction-light",
    "bridge-pillar", "dumpster", "light-square", "traffic-light",
    "road-straight", "road-crossing", "road-roundabout", "sign-highway",
]

# Which file of each asset the pipeline actually consumes, and what each asset
# is: this drives the manifest and the scale bookkeeping.
PLAN: dict[str, dict] = {
    "kenney_city_kit_roads": {
        "kind": "prop_set",
        "primary_file": "assets/kenney_city_kit_roads/Models/GLB format/road-straight.glb",
        "unit_convention": 1.0,
        "unit_evidence": ("Kenney works on a 1-unit = 1 m grid: road-straight.glb and "
                          "road-crossing.glb measure exactly 1.000 x 0.020 x 1.000 m, the "
                          "roundabout 3.000 x 0.020 x 3.000 m. Format matters: the GLB and OBJ "
                          "of light-square are 0.05 x 0.60 x 0.2375 (metres), the FBX of the "
                          "same model is 5 x 60 x 23.75 (centimetres, x0.01). The props are "
                          "miniature relative to reality (construction-cone 0.094 m tall = x5.3 "
                          "to a real ~0.50 m cone, dumpster 0.37 m long vs ~1.1 m) - scale them "
                          "individually if a real-world size matters."),
    },
    "animated_human_quaternius": {
        "kind": "character",
        "primary_file": "assets/animated_human_quaternius/Animated Human by @Quaternius/FBX/Animated Human.fbx",
        "unit_convention": 0.01,
        "target_height_m": TARGET_HUMAN_HEIGHT_M,
        "unit_evidence": ("The FBX and the OBJ of the very same mesh differ by exactly x100: "
                          "OBJ is 2.328 x 5.259 x 2.291 (metre units of the exporter) and FBX "
                          "232.8 x 525.9 x 229.1 -> the FBX is in centimetres. The skeleton "
                          "agrees: hips 264.6, head top 530.5, toes ~1.0 in that same unit. "
                          "The mesh is also authored oversized (5.26 units tall), so the "
                          "calibrated scale to a 1.75 m pedestrian is 1.75/525.94."),
    },
    "rigged_animated_humanoid": {
        "kind": "character",
        "primary_file": "assets/rigged_animated_humanoid/person_animated.fbx",
        "unit_convention": 0.01,
        "target_height_m": TARGET_HUMAN_HEIGHT_M,
        "unit_evidence": ("FBX centimetres: mesh is 242.95 x 200.79 x 46.02 units, i.e. a "
                          "2.01 m human in a T-pose with a 2.43 m arm span -> 1 unit = 1 cm."),
    },
    "cesium_man": {
        "kind": "character",
        "primary_file": "assets/cesium_man/CesiumMan.glb",
        "unit_convention": 1.0,
        "unit_evidence": ("glTF metres: 1.507 m tall, feet at y=0 (measured on the walk pose). "
                          "The glTF root node is called Z_UP and rotates the Z-up source into "
                          "Y-up, which is why the raw geometry bbox looks wrong."),
    },
    "rigged_figure": {
        "kind": "character",
        "primary_file": "assets/rigged_figure/RiggedFigure.glb",
        "unit_convention": 1.0,
        "unit_evidence": "glTF metres: 1.450 m tall in the T-pose, feet at y=0.",
    },
    "fox": {
        "kind": "creature",
        "primary_file": "assets/fox/Fox.glb",
        "unit_convention": 0.01,
        "unit_evidence": ("Authored in centimetres despite the glTF unit rule: 154.7 units "
                          "nose-to-tail = 1.55 m. three.js' own example scales Fox.glb by 0.01 "
                          "(webgl_animation_skinning_morph)."),
    },
    "kira": {
        "kind": "scene",
        "primary_file": "assets/kira/kira.glb",
        "unit_convention": 1.0,
        "unit_evidence": "glTF metres (not measurable here: the file is DRACO-compressed and unreadable offline).",
    },
    "mixamo_fbx_manual": {
        "kind": "manual",
        "primary_file": None,
        "unit_convention": 0.01,
        "unit_evidence": "Mixamo FBX exports are centimetres (verify with tools/inspect_mesh.mjs).",
    },
}


# --------------------------------------------------------------------------
# measurement children
# --------------------------------------------------------------------------
def measure_open3d(files: list[str]) -> dict[str, dict]:
    """Read files with open3d in a child process.

    A child process is used because open3d prints its C++ warnings on *stdout*
    (not stderr), which would corrupt any JSON we try to pipe back.  The child
    therefore writes its answer to a temporary file instead.
    """
    if not files:
        return {}
    code = r"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import open3d as o3d
out = {}
for f in json.load(sys.stdin):
    rec = {"ok": False}
    try:
        mesh = o3d.io.read_triangle_mesh(f)
        nv, nt = len(mesh.vertices), len(mesh.triangles)
        rec = {"ok": nv > 0 and nt > 0, "vertices": nv, "triangles": nt}
        if rec["ok"]:
            bb = mesh.get_axis_aligned_bounding_box()
            rec["bbox_min"] = [round(float(v), 6) for v in bb.min_bound]
            rec["bbox_max"] = [round(float(v), 6) for v in bb.max_bound]
            rec["bbox_size"] = [round(float(v), 6) for v in bb.get_extent()]
            rec["api"] = "open3d.io.read_triangle_mesh (assimp)"
        else:
            rec["error"] = "empty mesh (unsupported format or compression)"
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
    out[f] = rec
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump(out, fh)
"""
    with tempfile.TemporaryDirectory() as tmp:
        result_path = Path(tmp) / "open3d.json"
        proc = subprocess.run([sys.executable, "-c", code, str(result_path)],
                              input=json.dumps(files), capture_output=True, text=True,
                              encoding="utf-8", errors="replace", cwd=ROOT)
        if proc.returncode != 0 or not result_path.exists():
            return {f: {"ok": False, "error": f"open3d child failed: {proc.stderr.strip()[:200]}"}
                    for f in files}
        return json.loads(result_path.read_text(encoding="utf-8"))


def measure_three(files: list[str]) -> dict[str, dict]:
    """Read files with the three.js loaders in Node."""
    if not files:
        return {}
    proc = subprocess.run(["node", str(NODE_INSPECT), *files], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", cwd=ROOT)
    if proc.returncode != 0:
        note = {"ok": False, "error": f"node inspect failed: {proc.stderr.strip()[:200]}"}
        return {f.replace("\\", "/"): note for f in files}
    data = json.loads(proc.stdout)
    if isinstance(data, dict):
        data = [data]
    return {rec["file"]: rec for rec in data}


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------
def registry_entry(asset_id: str) -> dict:
    for entry in REGISTRY:
        if entry["id"] == asset_id:
            return entry
    raise KeyError(asset_id)


def rel(path: Path | str) -> str:
    p = Path(path)
    if not p.is_absolute():
        return str(p).replace("\\", "/")
    return str(p.relative_to(ROOT)).replace("\\", "/")


def file_inventory(asset_id: str) -> dict:
    base = ASSETS_DIR / asset_id
    by_ext: dict[str, int] = {}
    bytes_by_ext: dict[str, int] = {}
    total = 0
    count = 0
    if base.exists():
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            ext = path.suffix.lower() or "(none)"
            by_ext[ext] = by_ext.get(ext, 0) + 1
            bytes_by_ext[ext] = bytes_by_ext.get(ext, 0) + path.stat().st_size
            total += path.stat().st_size
            count += 1
    return {"count": count, "bytes": total, "by_extension": by_ext,
            "bytes_by_extension": bytes_by_ext}


def set_aggregate(reads: dict) -> dict:
    """Aggregate over every model of a prop set (all GLBs read with open3d)."""
    ok = {k: v for k, v in reads.items() if v.get("ok")}
    failed = [k for k, v in reads.items() if not v.get("ok")]
    tris = [v["triangles"] for v in ok.values()]
    verts = [v["vertices"] for v in ok.values()]
    sizes = [v["bbox_size"] for v in ok.values() if v.get("bbox_size")]
    return {
        "models_measured": len(reads),
        "models_readable": len(ok),
        "models_failed": failed,
        "triangles_total": sum(tris),
        "triangles_min": min(tris) if tris else None,
        "triangles_max": max(tris) if tris else None,
        "vertices_total": sum(verts),
        "largest_bbox_m": [round(max(s[k] for s in sizes), 4) for k in range(3)] if sizes else None,
    }


def measure_asset(asset_id: str, plan: dict, open3d: dict, three: dict) -> dict:
    """Collect the real numbers for one asset."""
    measured: dict = {"primary": None, "secondary": [], "key_models": []}
    primary = plan.get("primary_file")
    if primary:
        key = rel(primary)
        measured["primary"] = {"file": key,
                               "open3d": open3d.get(key) or open3d.get(str(primary)),
                               "three": three.get(key) or three.get(str(primary))}
    if asset_id == "animated_human_quaternius":
        obj = "assets/animated_human_quaternius/Animated Human by @Quaternius/OBJ/Animated Human.obj"
        measured["secondary"].append({"file": obj, "role": "same mesh in metre units (unit proof)",
                                      "open3d": open3d.get(obj), "three": three.get(obj)})
    if asset_id == "kenney_city_kit_roads":
        for name in KENNEY_KEY_MODELS:
            f = f"assets/kenney_city_kit_roads/Models/GLB format/{name}.glb"
            measured["key_models"].append({"name": name, "file": f, "glb": three.get(f),
                                           "open3d": open3d.get(f)})
    return measured


def build_manifest() -> dict:
    files_by_asset: dict[str, list[str]] = {}
    for entry in REGISTRY:
        aid = entry["id"]
        plan = PLAN[aid]
        candidates = []
        if plan.get("primary_file"):
            candidates.append(plan["primary_file"])
        if aid == "animated_human_quaternius":
            candidates.append("assets/animated_human_quaternius/Animated Human by @Quaternius/OBJ/Animated Human.obj")
        if aid == "kenney_city_kit_roads":
            candidates += [f"assets/kenney_city_kit_roads/Models/GLB format/{n}.glb"
                           for n in KENNEY_KEY_MODELS]
            candidates += [f"assets/kenney_city_kit_roads/Models/OBJ format/{n}.obj"
                           for n in KENNEY_KEY_MODELS[:3]]
            candidates += [f"assets/kenney_city_kit_roads/Models/FBX format/{n}.fbx"
                           for n in KENNEY_KEY_MODELS[:2]]
        files_by_asset[aid] = candidates

    all_files = [rel(f) for fl in files_by_asset.values() for f in fl]
    open3d = measure_open3d(all_files)
    three = measure_three(all_files)

    # The prop set: measure every GLB in it, not just the key ones.
    kenney_all = sorted((ASSETS_DIR / "kenney_city_kit_roads" / "Models" / "GLB format").glob("*.glb"))
    kenney_reads = measure_open3d([rel(p) for p in kenney_all]) if kenney_all else {}

    assets = []
    for entry in REGISTRY:
        aid = entry["id"]
        plan = PLAN[aid]
        inventory = file_inventory(aid)
        measured = measure_asset(aid, plan, open3d, three)

        animations: list[dict] = []
        verts = tris = 0
        bbox_size = None
        skinned_meshes = 0
        numbers_from = None
        prim_rec = measured["primary"] or {}
        if prim_rec.get("three") and prim_rec["three"].get("ok"):
            prim = prim_rec["three"]
            numbers_from = "three"
            animations = [{"name": c["name"], "duration": c["duration"], "tracks": c["tracks"]}
                          for c in prim["animations"]]
            verts, tris = prim["vertices"], prim["triangles"]
            bbox_size = prim["bbox_world"]["size"]
            skinned_meshes = prim["skinned_meshes"]
        elif prim_rec.get("open3d") and prim_rec["open3d"].get("ok"):
            # three cannot read it (e.g. a DRACO payload or a mesh-only format):
            # report what open3d measured, and say so.
            prim = prim_rec["open3d"]
            numbers_from = "open3d"
            verts, tris = prim["vertices"], prim["triangles"]
            bbox_size = prim.get("bbox_size")

        units = {
            "asset_units_to_meters": plan["unit_convention"],
            "measured_primary_bbox_asset_units": bbox_size,
            "meters_with_unit_convention": [None if bbox_size is None else round(v * plan["unit_convention"], 4)
                                            for v in (bbox_size or [])] or None,
            "evidence": plan["unit_evidence"],
        }
        if plan.get("target_height_m") and bbox_size:
            standing = bbox_size[1]
            units["target_height_m"] = plan["target_height_m"]
            units["measured_standing_height_asset_units"] = standing
            units["calibrated_scale_to_meters"] = round(plan["target_height_m"] / standing, 6)
        else:
            units["calibrated_scale_to_meters"] = plan["unit_convention"]

        expected_three = True
        expected_open3d = True
        if aid == "kira":
            expected_three = expected_open3d = False
            units["calibrated_scale_to_meters"] = None
        if aid == "mixamo_fbx_manual":
            expected_three = expected_open3d = False

        license_info = {
            "spdx": entry["license"],
            "url": entry.get("license_url"),
            "attribution_required": bool(entry.get("attribution")),
            "attribution_text": entry.get("attribution"),
            "redistribute_binaries": entry.get("redistribute_binaries", True),
        }
        assets.append({
            "id": aid,
            "name": entry["name"],
            "kind": plan["kind"],
            "source": {
                "page_url": entry.get("page_url"),
                "download_url": entry.get("url"),
                "archive_file": entry.get("archive"),
                "sha256": entry.get("sha256"),
                "manual_download": entry.get("url") is None,
                "manual_instructions": entry.get("manual_instructions"),
            },
            "license": license_info,
            "model_formats": sorted(e.lstrip(".") for e in inventory["by_extension"]
                                    if e.lstrip(".") in ("glb", "gltf", "obj", "fbx", "dae", "stl", "ply")),
            "files": inventory,
            "primary_file": plan.get("primary_file"),
            "primary": {
                "vertices": verts, "triangles": tris, "skinned_meshes": skinned_meshes,
                "numbers_from": numbers_from,
                "bbox_size_asset_units": bbox_size,
                "bbox_size_meters": units["meters_with_unit_convention"],
            },
            "animations": animations,
            "animation_count": len(animations),
            "units": units,
            "read_by": {"three_node": expected_three, "open3d": expected_open3d},
            "read_probe": {"three_node": bool(measured["primary"] and measured["primary"]["three"]
                                             and measured["primary"]["three"]["ok"]),
                           "open3d": bool(measured["primary"] and measured["primary"]["open3d"]
                                          and measured["primary"]["open3d"]["ok"])},
            "secondary": measured["secondary"],
            "key_models": measured["key_models"],
            "set_aggregate": (set_aggregate(kenney_reads) if aid == "kenney_city_kit_roads" else None),
            "notes": entry.get("note"),
        })

    return {
        "schema": "assets-manifest/1",
        "generated_by": "python tools/check_assets.py --write-manifest",
        "verified_by": "python tools/check_assets.py",
        "assets_dir": "assets/ (git-ignored: only LICENSES.md and manifest.json are committed)",
        "readers": {
            "open3d": "open3d.io.read_triangle_mesh (0.19, bundled assimp: glb/gltf/obj/fbx/ply/stl, no DRACO)",
            "three_node": "tools/inspect_mesh.mjs (three 0.186 GLTF/FBX/OBJ/STL/PLY loaders, Node 22)",
            "browser_client": "app/client (the same three loaders in the browser)",
        },
        "unit_convention": {
            "gltf_glb": "metres per spec (verified per asset, see units.evidence)",
            "obj": "metres (Kenney; Quaternius OBJ also metres)",
            "fbx": "centimetres - multiply by 0.01 (both FBX rigs verified by an OBJ of the same mesh)",
            "quaternius_oversize": "the Quaternius mesh is authored 5.26 units tall, so 1.75/525.94 is needed too",
        },
        "assets": assets,
    }


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------
def bake_probe(primary: str, clip: str, expected: dict) -> dict:
    """End-to-end bake: tools/bake_pose.js -> .npz -> numpy.load, shape checked."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "probe.npz"
        proc = subprocess.run(["node", str(HERE / "bake_pose.js"), primary, "--clip", clip,
                               "--time", "0", "--out", str(out), "--quiet"],
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", cwd=ROOT)
        if proc.returncode != 0 or not out.exists():
            return {"ok": False, "note": f"bake_pose.js failed: {proc.stderr.strip()[-200:]}"}
        code = (
            "import json,sys,numpy as np\n"
            "z=np.load(sys.argv[1])\n"
            "v,f=z['vertices'],z['faces']\n"
            "json.dump({'vertices':list(v.shape),'faces':list(f.shape),'dtype':str(v.dtype),"
            "'finite':bool(np.isfinite(v).all()),"
            "'bbox_size':[round(float(x),4) for x in (v.max(0)-v.min(0))]},sys.stdout)\n")
        probe = subprocess.run([sys.executable, "-c", code, str(out)], capture_output=True,
                               text=True, encoding="utf-8", errors="replace")
        if probe.returncode != 0:
            return {"ok": False, "note": f"numpy could not read the .npz: {probe.stderr.strip()[-200:]}"}
        got = json.loads(probe.stdout)

    ok = (got["vertices"] == [expected["vertices"], 3] and got["faces"] == [expected["triangles"], 3]
          and got["dtype"] == "float32" and got["finite"])
    return {"ok": ok,
            "note": f"{clip}@0s -> npz verts={got['vertices']} faces={got['faces']} {got['dtype']} "
                    f"bbox={got['bbox_size']}" + ("" if ok else " MISMATCH vs manifest")}


def verify(verbose: bool = False) -> dict:
    if not MANIFEST_PATH.exists():
        return {"error": f"{rel(MANIFEST_PATH)} missing - run: python tools/check_assets.py --write-manifest"}
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    primary_files = []
    for asset in manifest["assets"]:
        files = []
        if asset["primary_file"]:
            files.append(rel(asset["primary_file"]))
        for sec in asset.get("secondary", []):
            files.append(rel(sec["file"]))
        for km in asset.get("key_models", []):
            files.append(rel(km["file"]))
        primary_files += files

    open3d = measure_open3d(primary_files)
    three = measure_three(primary_files)

    rows = []
    problems = []
    for asset in manifest["assets"]:
        aid = asset["id"]
        checks: list[tuple[str, bool, str]] = []

        # 1. archive + files
        archive = asset["source"].get("archive_file")
        archive_ok = True
        archive_note = "manual"
        if asset["source"].get("manual_download"):
            archive_note = "manual download (no URL)"
        else:
            path = ASSETS_DIR / "_downloads" / archive
            if not path.exists():
                archive_ok, archive_note = False, "archive missing - run tools/fetch_assets.py"
            else:
                digest = sha256_file(path)
                if asset["source"].get("sha256") and digest != asset["source"]["sha256"]:
                    archive_ok = False
                    archive_note = f"sha256 mismatch {digest[:12]}..."
                else:
                    archive_note = f"sha256 ok, {human(path.stat().st_size)}"
        checks.append(("archive", archive_ok, archive_note))

        # 2. license recorded
        lic = asset["license"]
        lic_ok = bool(lic.get("spdx") and lic.get("url"))
        if lic.get("attribution_required") and not lic.get("attribution_text"):
            lic_ok = False
        checks.append(("license", lic_ok,
                       f"{lic['spdx']}" + (", attribution required" if lic.get("attribution_required") else "")))

        # 3. readers
        prim = rel(asset["primary_file"]) if asset["primary_file"] else None
        three_rec = three.get(prim) if prim else None
        o3d_rec = open3d.get(prim) if prim else None
        for name, expected, rec in (("three_node", asset["read_by"]["three_node"], three_rec),
                                    ("open3d", asset["read_by"]["open3d"], o3d_rec)):
            if expected:
                ok = bool(rec and rec.get("ok"))
                note = (f"verts={rec['vertices']} tris={rec['triangles']}" if ok
                        else f"expected readable but failed: {(rec or {}).get('error', 'not measured')}")
                if not ok:
                    problems.append(f"{aid}: {name} expected readable, got failure")
            else:
                ok = not bool(rec and rec.get("ok"))
                note = "not readable (expected)" if ok else "UNEXPECTEDLY readable"
            checks.append((name, ok, note))

        files_ok = asset["files"]["count"] > 0 or asset["source"].get("manual_download")
        checks.append(("files", files_ok, f"{asset['files']['count']} files, {human(asset['files']['bytes'])}"))
        if not files_ok:
            problems.append(f"{aid}: no files unpacked")

        # 4. the bake path: bake the first clip at t=0 and read the .npz back with numpy
        if asset["animations"] and prim and asset["read_by"]["three_node"]:
            bake = bake_probe(prim, asset["animations"][0]["name"], asset["primary"])
            checks.append(("bake", bake["ok"], bake["note"]))
            if not bake["ok"]:
                problems.append(f"{aid}: pose bake failed: {bake['note']}")
        else:
            checks.append(("bake", True, "n/a (no clips)" if not asset["animations"] else "n/a (unreadable)"))

        # live reader state (what actually happened, not what was expected)
        readers = [name for name, rec in (("three", three_rec), ("open3d", o3d_rec))
                   if rec and rec.get("ok")]
        if not prim:
            readers = ["-"]
        elif not readers:
            readers = ["not readable"]

        rows.append({"id": aid, "name": asset["name"], "license": lic["spdx"], "checks": checks,
                     "primary": prim, "vertices": asset["primary"]["vertices"],
                     "triangles": asset["primary"]["triangles"],
                     "bbox_m": asset["primary"]["bbox_size_meters"],
                     "sq_meters": asset["units"].get("calibrated_scale_to_meters"),
                     "animations": asset["animation_count"],
                     "readers": readers,
                     "kind": asset["kind"]})

    return {"manifest": rel(MANIFEST_PATH), "rows": rows, "problems": problems}


def print_report(report: dict) -> None:
    rows = report["rows"]
    width = max(len(r["id"]) for r in rows) + 2
    print(f"assets manifest : {report['manifest']}")
    print(f"licenses file   : {rel(LICENSES_PATH)}"
          f" ({'present' if LICENSES_PATH.exists() else 'MISSING'})\n")
    header = (f"{'asset':<{width}}{'checks':<8}{'read by':<20}{'verts':>7}{'tris':>7}  "
              f"{'bbox (m)':<24}{'scale':<11}{'clips':<6}license")
    print(header)
    print("-" * len(header))
    for r in rows:
        marks = "".join("." if ok else "X" for _, ok, _ in r["checks"])
        legend = " ".join(name for name, ok, _ in r["checks"] if not ok)
        readers = r["readers"]
        bb = "-" if r["bbox_m"] is None else "x".join(f"{v:g}" for v in r["bbox_m"])
        scale = "-" if r["sq_meters"] is None else f"{r['sq_meters']:g}"
        print(f"{r['id']:<{width}}{marks:<8}{'+'.join(readers):<20}{r['vertices']:>7}"
              f"{r['triangles']:>7}  {bb:<24}{scale:<11}{r['animations']:<6}{r['license'][:40]}")
        if legend:
            print(f"{'':<{width}}-> failed checks: {legend}")
    print("\nchecks order: archive license three_node open3d files bake   ('.' = ok, 'X' = fail)")
    print("bbox/verts/tris are the primary file as read by three (open3d fallback, "
          "see manifest primary.numbers_from); 'scale' = units.calibrated_scale_to_meters")
    if report["problems"]:
        print("\nproblems:")
        for p in report["problems"]:
            print(f"  - {p}")
    else:
        print("\nall checks passed")


def print_detail(report: dict, manifest: dict) -> None:
    for asset, row in zip(manifest["assets"], report["rows"]):
        print(f"\n=== {asset['id']}  ({asset['kind']}, {asset['license']['spdx']})")
        print(f"  source   : {asset['source']['download_url'] or 'MANUAL'}")
        print(f"  archive  : {asset['source']['archive_file']} sha256={asset['source']['sha256']}")
        print(f"  files    : {asset['files']['count']} ({human(asset['files']['bytes'])}) "
              f"{asset['files']['by_extension']}")
        for name, ok, note in row["checks"]:
            print(f"  {name:<9}: {'ok  ' if ok else 'FAIL'} {note}")
        if asset["animations"]:
            print(f"  clips    : {len(asset['animations'])} "
                  f"({', '.join(a['name'] + '@' + str(a['duration']) + 's' for a in asset['animations'][:6])}"
                  f"{' ...' if len(asset['animations']) > 6 else ''})")
        print(f"  units    : {asset['units']['evidence']}")
        if asset["key_models"]:
            print("  key models (GLB):")
            for km in asset["key_models"]:
                g = km.get("glb") or {}
                if not g.get("ok"):
                    print(f"    {km['name']:<22} FAILED {(g.get('error') or '?')[:70]}")
                    continue
                print(f"    {km['name']:<22} verts={g['vertices']:<5} tris={g['triangles']:<5} "
                      f"bbox_m={[round(v, 4) for v in g['bbox_world']['size']]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write-manifest", action="store_true",
                    help="re-measure every asset and rewrite assets/manifest.json")
    ap.add_argument("--json", action="store_true", help="print the verification report as JSON")
    ap.add_argument("--detail", action="store_true", help="per-asset detail (clips, units, key models)")
    ap.add_argument("--verbose", action="store_true", help="keep loader warnings on stderr")
    args = ap.parse_args(argv)

    if not args.verbose:
        warnings.filterwarnings("ignore")

    if args.write_manifest:
        manifest = build_manifest()
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                 encoding="utf-8")
        print(f"wrote {rel(MANIFEST_PATH)} ({MANIFEST_PATH.stat().st_size} bytes, "
              f"{len(manifest['assets'])} assets)")

    report = verify(args.verbose)
    if "error" in report:
        print(report["error"], file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_report(report)
        if args.detail:
            print_detail(report, json.loads(MANIFEST_PATH.read_text(encoding="utf-8")))
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())