#!/usr/bin/env python3
"""Download and unpack the 3D asset pack used for annotation.

Everything lands under ``assets/`` (which is git-ignored: the binaries must not
be committed, see assets/LICENSES.md).  Only this script, the license notes and
the manifest are versioned.

The run is idempotent: an archive that is already present and whose sha256
matches the pin in the registry (or the pin recorded in ``assets/manifest.json``)
is not downloaded again.  Pin a new asset by running once with ``--record`` and
copying the printed hashes into ``REGISTRY`` below.

Standard library only (urllib + zipfile).

    python tools/fetch_assets.py              # fetch everything that is missing
    python tools/fetch_assets.py --check      # verify what is already on disk
    python tools/fetch_assets.py --files      # also print a per-file hash table
    python tools/fetch_assets.py --only kenney_city_kit_roads
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT / "assets"
DOWNLOADS_DIR = ASSETS_DIR / "_downloads"
STATE_PATH = ASSETS_DIR / "_state.json"
REPORT_PATH = ASSETS_DIR / "_fetch_report.json"
MANIFEST_PATH = ASSETS_DIR / "manifest.json"

USER_AGENT = "test_lidar-asset-fetch/1.0 (local annotation pipeline; python-urllib)"

# --------------------------------------------------------------------------
# Registry.  ``sha256`` is the pin of the *downloaded archive*:
#   None  -> first run records the hash into assets/_state.json, fill the pin in
#   "..." -> every run verifies the hash, a mismatch is a hard error
# --------------------------------------------------------------------------
REGISTRY: list[dict] = [
    {
        "id": "kenney_city_kit_roads",
        "name": "Kenney City Kit Roads",
        "page_url": "https://kenney.nl/assets/city-kit-roads",
        "url": "https://kenney.nl/media/pages/assets/city-kit-roads/74288c9459-1787042796/kenney_city-kit-roads.zip",
        "license": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "redistribute_binaries": True,
        "archive": "kenney_city_kit_roads.zip",
        "sha256": "22058af3d68173a7cf9bda9f0e243a8cef6bd68168c302ebc76327063849674e",
        "keep": r"Models/.*\.(glb|obj|mtl|png|fbx)$|License\.txt$",
        "note": "Static street furniture: barrier, column, light, fences, containers. "
                "The zip ships GLB + OBJ + FBX (no STL folder, contrary to the store page wording).",
    },
    {
        "id": "animated_human_quaternius",
        "name": "Animated Human Low Poly (Quaternius)",
        "page_url": "https://opengameart.org/content/animated-human-low-poly",
        "url": "https://opengameart.org/sites/default/files/Animated%20Human%20by%20%40Quaternius_0.zip",
        "license": "CC0-1.0 (archive license text) - Quaternius also ships this pack under QAL v1.0 on quaternius.com",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "redistribute_binaries": False,
        "archive": "animated_human_quaternius.zip",
        "sha256": "dcd72162b5e59495efc629fb624d9e1deb12124b68291ad4fa3ce66b4fc24db3",
        "keep": r"\.(fbx|obj|mtl|dae|png)$|License\.txt$",
        "note": "QAL v1.0 forbids redistributing the assets *as assets* -> binaries stay out of git. FBX only (no GLB), the .blend is skipped: no Blender offline.",
    },
    {
        "id": "rigged_animated_humanoid",
        "name": "Rigged and Animated Humanoid",
        "page_url": "https://opengameart.org/content/rigged-and-animated-humanoid",
        "url": "https://opengameart.org/sites/default/files/animated%20humanoid.zip",
        "license": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "redistribute_binaries": True,
        "archive": "rigged_animated_humanoid.zip",
        "sha256": "8f90ddc45c250f0d47cf0cda5db6cbdc6e438689f8884118dc4c7e67153040f9",
        "keep": r".*\.(fbx|txt)$",
        "note": "Ships as FBX only (plus .blend) - three's FBXLoader reads it, open3d does too "
                "(assimp). Unity-oriented export: centimetres, 2.01 m tall, 18 clips.",
    },
    {
        "id": "cesium_man",
        "name": "CesiumMan",
        "page_url": "https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/CesiumMan",
        "url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/CesiumMan/glTF-Binary/CesiumMan.glb",
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "redistribute_binaries": True,
        "archive": "CesiumMan.glb",
        "sha256": "b7001eaeea8254bd44773bcd247e78696d94169388fbb2a1800fc69434e777d9",
        "keep": None,
        "single_file": True,
        "attribution": "CesiumMan by Cesium (Khronos glTF-Sample-Assets), CC-BY-4.0",
        "note": "Skinned walk cycle, the pose-baking reference asset.",
    },
    {
        "id": "rigged_figure",
        "name": "RiggedFigure",
        "page_url": "https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/RiggedFigure",
        "url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/RiggedFigure/glTF-Binary/RiggedFigure.glb",
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "redistribute_binaries": True,
        "archive": "RiggedFigure.glb",
        "sha256": "d6be85417d3e256861ee733eea6916093a7af7c79c16366181fd8abcaeb38cf5",
        "keep": None,
        "single_file": True,
        "attribution": "RiggedFigure by Cesium (Khronos glTF-Sample-Assets), CC-BY-4.0",
        "note": "Skinned but animation-free figure: negative control for the baker.",
    },
    {
        "id": "kira",
        "name": "kira (three.js example model)",
        "page_url": "https://github.com/mrdoob/three.js/tree/dev/examples/models/gltf",
        "url": "https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/models/gltf/kira.glb",
        "license": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "redistribute_binaries": True,
        "archive": "kira.glb",
        "sha256": "22bc87b9ee12aa7daa5b8c8c91bda245bae9f7e2c6a3ad8c4496ae7fa18910cf",
        "keep": None,
        "single_file": True,
        "note": "DRACO-compressed room scene, and its glTF carries no animations at all: "
                "readable in the browser client, not readable in Node/open3d (no DRACO decoder worker).",
    },
    {
        "id": "fox",
        "name": "Fox",
        "page_url": "https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Fox",
        "url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/Fox/glTF-Binary/Fox.glb",
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "redistribute_binaries": True,
        "archive": "Fox.glb",
        "sha256": "d97044e701822bac5a62696459b27d7b375aada5de8574ed4362edbba94771f7",
        "keep": None,
        "single_file": True,
        "attribution": "Fox by PixelMannen (opengameart.org/content/fox-and-shiba) and @tomkranis, "
                       "converted by @AsoboStudio with @scurest - CC-BY-4.0",
        "note": "Skinned quadruped with Survey/Walk/Run clips: second animated GLB, not DRACO-packed.",
    },
    {
        "id": "mixamo_fbx_manual",
        "name": "Mixamo character (any) - MANUAL, not fetched",
        "page_url": "https://www.mixamo.com/",
        "url": None,
        "license": "Adobe Mixamo terms (free use of the animations/characters, no redistribution as assets)",
        "license_url": "https://www.adobe.com/legal/terms.html",
        "redistribute_binaries": False,
        "manual": True,
        "manual_instructions": (
            "Mixamo needs an interactive session, there is no fetchable URL:\n"
            "  1. log in at https://www.mixamo.com/ (free Adobe account),\n"
            "  2. pick a character + an animation (walk/run/idle), set the frame range,\n"
            "  3. Download -> Format FBX Binary, Skin: With Skin, 'Frames per Second' 30,\n"
            "     Keyframe Reduction: none,\n"
            "  4. save it as assets/mixamo_fbx_manual/<name>.fbx and rerun\n"
            "     `python tools/fetch_assets.py --only mixamo_fbx_manual` (it only re-checks);\n"
            "  5. Mixamo FBX is centimetres: scale_to_meters = 0.01, verify with tools/inspect_mesh.mjs."
        ),
        "note": "Placeholder for licensed-for-use Mixamo content: never committed, always fetched by hand.",
    },
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n / 1.0:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GiB"


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def manifest_pin(asset_id: str) -> str | None:
    """sha256 declared for this asset in assets/manifest.json (if it exists)."""
    manifest = load_json(MANIFEST_PATH, None)
    if not isinstance(manifest, dict):
        return None
    for entry in manifest.get("assets", []):
        if entry.get("id") == asset_id:
            return (entry.get("archive") or {}).get("sha256")
    return None


def download(url: str, dest: Path) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    parts: list[bytes] = []
    with urllib.request.urlopen(req, timeout=180) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        got = 0
        last = 0.0
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            parts.append(chunk)
            got += len(chunk)
            now = time.time()
            if now - last > 1.0:
                last = now
                pct = f"{100.0 * got / total:5.1f}%" if total else "   ?  "
                print(f"      {pct}  {human(got)}", flush=True)
    data = b"".join(parts)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return data


def extract_zip(archive: Path, dest: Path, keep: str | None) -> list[Path]:
    pattern = re.compile(keep) if keep else None
    written: list[Path] = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename.replace("\\", "/")
            if pattern and not pattern.search(name):
                continue
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as out:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            written.append(target)
    return written


def asset_dir(entry: dict) -> Path:
    return ASSETS_DIR / entry["id"]


def list_files(entry: dict) -> list[Path]:
    base = asset_dir(entry)
    if not base.exists():
        return []
    return sorted(p for p in base.rglob("*") if p.is_file())


# --------------------------------------------------------------------------
# main flow
# --------------------------------------------------------------------------
def fetch_one(entry: dict, state: dict, force: bool, check_only: bool) -> dict:
    """Returns a result dict describing what happened for this asset."""
    asset_id = entry["id"]
    result = {"id": asset_id, "status": "", "archive": None, "files": [], "error": None}

    if entry.get("url") is None:  # manual download
        result["status"] = "manual"
        result["manual_instructions"] = entry.get("manual_instructions", "")
        return result

    archive_name = entry["archive"]
    stamp = state.setdefault("archives", {}).get(entry["url"]) or {}
    stored = DOWNLOADS_DIR / archive_name
    expected = entry.get("sha256") or stamp.get("sha256") or manifest_pin(asset_id)

    if stored.exists() and expected and sha256_file(stored) == expected:
        result["status"] = "cached"
        result["archive"] = {"file": str(stored.relative_to(ROOT)), "sha256": expected,
                             "bytes": stored.stat().st_size}
    elif check_only:
        result["status"] = "missing"
        result["error"] = "archive not on disk"
        return result
    else:
        print(f"  downloading {asset_id} <- {entry['url']}")
        try:
            data = download(entry["url"], stored)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            result["status"] = "error"
            result["error"] = f"download failed: {exc}"
            return result
        digest = sha256_bytes(data)
        if expected and digest != expected:
            result["status"] = "error"
            result["error"] = f"sha256 mismatch: expected {expected}, got {digest}"
            return result
        result["status"] = "downloaded"
        result["archive"] = {"file": str(stored.relative_to(ROOT)), "sha256": digest,
                             "bytes": len(data)}
        state.setdefault("archives", {})[entry["url"]] = {
            "file": archive_name, "sha256": digest, "bytes": len(data),
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    if not force and result["status"] == "cached":
        files = list_files(entry)
        if files:
            result["status"] = "ok"
            result["files"] = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size}
                               for p in files]
            return result

    # unpack
    dest = asset_dir(entry)
    if entry.get("single_file"):
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / archive_name
        target.write_bytes(stored.read_bytes())
        written = [target]
    else:
        written = extract_zip(stored, dest, entry.get("keep"))

    result["status"] = "ok" if result["status"] != "ok" else result["status"]
    result["files"] = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size}
                       for p in written]
    return result


def print_table(rows: list[tuple[str, int, str]]) -> None:
    w1 = max([len("file")] + [len(r[0]) for r in rows]) if rows else 4
    w2 = max([len("bytes")] + [len(human(r[1])) for r in rows]) if rows else 5
    print(f"  {'file'.ljust(w1)}  {'size'.rjust(w2)}  sha256")
    print(f"  {'-' * w1}  {'-' * w2}  {'-' * 16}")
    for name, size, digest in rows:
        print(f"  {name.ljust(w1)}  {human(size).rjust(w2)}  {digest}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="do not download: only verify/report what is already present")
    ap.add_argument("--force", action="store_true", help="re-download and re-extract everything")
    ap.add_argument("--files", action="store_true", help="print the per-file size/sha256 table")
    ap.add_argument("--only", action="append", default=None, help="restrict to an asset id")
    ap.add_argument("--json", action="store_true", help="print the machine-readable result")
    args = ap.parse_args(argv)

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    state = load_json(STATE_PATH, {"archives": {}})
    entries = [e for e in REGISTRY if not args.only or e["id"] in args.only]
    if not entries:
        print("no asset matched --only", file=sys.stderr)
        return 2

    print(f"assets dir: {ASSETS_DIR}")
    print(f"mode: {'check only' if args.check else ('force refetch' if args.force else 'incremental')}\n")

    results = []
    for entry in entries:
        print(f"[{entry['id']}] {entry['name']}")
        res = fetch_one(entry, state, args.force, args.check)
        if res["error"]:
            print(f"  !! {res['error']}")
        if res["status"] == "manual":
            print("  manual download required")
        results.append(res)
        print()

    save_json(STATE_PATH, state)

    # ---- report ---------------------------------------------------------
    rows: list[tuple[str, int, str]] = []
    report = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "assets": []}
    for entry, res in zip(entries, results):
        entry_report = {"id": entry["id"], "status": res["status"], "archive": res["archive"],
                        "error": res["error"], "files": []}
        if res["archive"]:
            rows.append((entry["id"], res["archive"]["bytes"], res["archive"]["sha256"]))
        for f in res.get("files", []):
            path = ROOT / f["path"]
            digest = sha256_file(path)
            entry_report["files"].append({"path": f["path"], "bytes": f["bytes"], "sha256": digest})
            if args.files:
                rows.append((f["path"].replace("\\", "/"), f["bytes"], digest))
        report["assets"].append(entry_report)

    print("downloaded archives")
    archived = {e["id"]: True for e in entries}
    print_table([r for r in rows if r[0] in archived] if not args.files else rows)

    print("\nsummary")
    for entry, res in zip(entries, results):
        n = len(res.get("files", []))
        total = sum(f["bytes"] for f in res.get("files", []))
        mark = {"ok": "ok", "cached": "ok (already on disk)", "missing": "MISSING",
                "error": "ERROR", "manual": "MANUAL"}.get(res["status"], res["status"])
        print(f"  {entry['id']:<28} {mark:<22} files={n:<4} {human(total)}")

    save_json(REPORT_PATH, report)
    print(f"\nreport: {REPORT_PATH.relative_to(ROOT)}")

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))

    failed = [r["id"] for r in results if r["status"] in ("error", "missing")]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())