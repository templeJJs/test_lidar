// Shared "three.js inside plain Node" helper for the asset pipeline.
//
// three 0.186 lives in app/client/node_modules (the client app owns it) and is
// NOT resolvable from the repo root, so every module here imports it by
// absolute path: app/client/node_modules/three/build/three.module.js and the
// matching examples/jsm loaders.  Both resolve the *same* module instance, so
// instanceof checks stay consistent.
//
// The loaders assume a browser.  We give them the smallest possible shims
// instead of pulling in jsdom:
//   self    -> globalThis          (GLTFLoader: self.URL)
//   window  -> {innerWidth,...}    (FBXLoader: camera aspect default)
//   textures -> a stub Texture     (no canvas/image decoder exists in Node;
//                                   geometry + skinning do not need pixels)
//
// Exports: loadThree(), loadAsset(), listClips(), bakePose(), meshStats().

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const REPO_ROOT = path.resolve(HERE, '..');

export const THREE_ROOT = process.env.THREE_ROOT
  ? path.resolve(process.env.THREE_ROOT)
  : path.join(REPO_ROOT, 'app', 'client', 'node_modules', 'three');

const loadersByExt = {
  '.glb': 'GLTFLoader',
  '.gltf': 'GLTFLoader',
  '.fbx': 'FBXLoader',
  '.obj': 'OBJLoader',
  '.stl': 'STLLoader',
  '.ply': 'PLYLoader',
};

let cached = null;

function installDomShims() {
  if (typeof globalThis.self === 'undefined') globalThis.self = globalThis;
  if (typeof globalThis.window === 'undefined') {
    globalThis.window = { innerWidth: 1280, innerHeight: 720, devicePixelRatio: 1 };
  }
}

/** Import three once: { THREE, loaders: {GLTFLoader, FBXLoader, OBJLoader, STLLoader, PLYLoader} }. */
export async function loadThree() {
  if (cached) return cached;
  installDomShims();

  const entry = path.join(THREE_ROOT, 'build', 'three.module.js');
  if (!fs.existsSync(entry)) {
    throw new Error(
      `three.js not found at ${entry}\n` +
      `Set THREE_ROOT to a directory containing build/three.module.js ` +
      `(expected: app/client/node_modules/three).`);
  }
  const imp = (rel) => import(pathToFileURL(path.join(THREE_ROOT, rel)).href);
  const THREE = await imp('build/three.module.js');

  // Textures: Node has no image decoder and no canvas.  Geometry, skinning and
  // animation clips are unaffected, so hand back an empty Texture.
  const stubLoad = function (url, onLoad) {
    const texture = new THREE.Texture();
    texture.name = typeof url === 'string' ? url : '';
    texture.userData.stub = 'three-dom-shim';
    if (onLoad) queueMicrotask(() => onLoad(texture));
    return texture;
  };
  THREE.TextureLoader.prototype.load = stubLoad;
  THREE.ImageLoader.prototype.load = stubLoad;
  if (THREE.ImageBitmapLoader) THREE.ImageBitmapLoader.prototype.load = stubLoad;

  const loaders = {};
  const wanted = ['GLTFLoader', 'FBXLoader', 'OBJLoader', 'STLLoader', 'PLYLoader', 'ColladaLoader'];
  for (const name of wanted) {
    try {
      const mod = await imp(`examples/jsm/loaders/${name}.js`);
      loaders[name] = mod[name];
    } catch (err) {
      loaders[name] = null; // ColladaLoader is optional (needs DOMParser)
    }
  }
  cached = { THREE, loaders, revision: THREE.REVISION };
  return cached;
}

function arrayBufferOf(buf) {
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

/**
 * Load an asset from disk into a three Object3D.
 * Returns { THREE, root, animations, format, loader, file, bytes }.
 */
export async function loadAsset(file, { three } = {}) {
  const { THREE, loaders } = three || (await loadThree());
  const ext = path.extname(file).toLowerCase();
  const loaderName = loadersByExt[ext];
  if (!loaderName) {
    throw new Error(`no three loader for ${ext} (supported: ${Object.keys(loadersByExt).join(', ')}` +
      ext === '.dae' ? '; .dae also needs DOMParser, unavailable in Node)' : ')');
  }
  const Loader = loaders[loaderName];
  if (!Loader) throw new Error(`${loaderName} is not available in ${THREE_ROOT}`);

  const buf = fs.readFileSync(file);
  const ab = arrayBufferOf(buf);
  let root, animations = [];
  const loader = new Loader();

  if (loaderName === 'GLTFLoader') {
    const gltf = await loader.parseAsync(ab, path.dirname(file) + path.sep);
    root = gltf.scene;
    animations = gltf.animations || [];
  } else if (loaderName === 'FBXLoader') {
    root = loader.parse(ab, path.dirname(file) + path.sep);
    animations = root.animations || [];
  } else if (loaderName === 'OBJLoader') {
    root = loader.parse(buf.toString('utf8'));
  } else {
    const geometry = loader.parse(ab);
    root = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
    root.name = path.basename(file);
  }

  root.updateMatrixWorld(true);
  return { THREE, root, animations, format: ext.slice(1), loader: loaderName,
           file, bytes: buf.length };
}

/** [{name, duration, tracks}] for every AnimationClip of a loaded asset. */
export function listClips(asset) {
  return asset.animations.map((clip) => ({
    name: clip.name,
    duration: Number(clip.duration.toFixed(6)),
    tracks: clip.tracks.length,
    track_names: clip.tracks.slice(0, 4).map((t) => t.name),
  }));
}

/** Every drawable mesh of a loaded root (skinned or not). */
export function collectMeshes(root) {
  const meshes = [];
  root.traverse((obj) => {
    if (obj.isMesh && obj.geometry && obj.geometry.attributes && obj.geometry.attributes.position) {
      meshes.push(obj);
    }
  });
  return meshes;
}

/** Triangles of a mesh geometry (indexed or not). */
export function triangleCount(mesh) {
  const g = mesh.geometry;
  return Math.floor((g.index ? g.index.count : g.attributes.position.count) / 3);
}

/** Per-mesh vertex/triangle counts without skinning (bind pose). */
export function meshStats(root) {
  const meshes = collectMeshes(root);
  const stats = [];
  for (const m of meshes) {
    stats.push({
      name: m.name || '(unnamed)',
      type: m.isSkinnedMesh ? 'SkinnedMesh' : 'Mesh',
      parent: m.parent ? m.parent.name : null,
      vertices: m.geometry.attributes.position.count,
      triangles: triangleCount(m),
      skinned: !!m.isSkinnedMesh,
      has_morph_targets: !!m.morphTargetInfluences && m.morphTargetInfluences.length > 0,
      material: m.material ? (m.material.name || m.material.type) : null,
    });
  }
  return {
    meshes: stats,
    vertices: stats.reduce((a, s) => a + s.vertices, 0),
    triangles: stats.reduce((a, s) => a + s.triangles, 0),
    skinned_meshes: stats.filter((s) => s.skinned).length,
    morph_meshes: stats.filter((s) => s.has_morph_targets).length,
  };
}

function bboxOfPoints(vertices) {
  if (!vertices || vertices.length === 0) return null;
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < vertices.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      const v = vertices[i + k];
      if (v < min[k]) min[k] = v;
      if (v > max[k]) max[k] = v;
    }
  }
  const size = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
  return {
    min: min.map(r6), max: max.map(r6), size: size.map(r6),
    diagonal: r6(Math.hypot(size[0], size[1], size[2])),
  };
}

const r6 = (v) => Math.round(v * 1e6) / 1e6;

/**
 * Bake a pose into world-space vertex + face arrays.
 *
 *   bakePose(asset, {clip: 'walk', time: 0.5})
 *
 * With `clip` the AnimationMixer is advanced to `time` seconds first; without a
 * clip the asset's rest (bind) pose is baked.  Vertices are world space of the
 * loaded glTF/FBX scene (Y-up, asset units), faces are indices into them, so a
 * consumer can offset/reindex several meshes without further bookkeeping.
 */
export function bakePose(asset, { clip, time = 0 } = {}) {
  const started = process.hrtime.bigint();
  const { THREE, root } = asset;

  // Snapshot the current local transforms so the bake is a pure function of
  // (asset, clip, time): an AnimationMixer writes straight into the bones and
  // whatever it leaves behind would otherwise leak into the next bake.
  const snapshot = [];
  root.traverse((obj) => {
    snapshot.push([obj, obj.position.clone(), obj.quaternion.clone(), obj.scale.clone()]);
  });
  const restore = () => {
    for (const [obj, position, quaternion, scale] of snapshot) {
      obj.position.copy(position);
      obj.quaternion.copy(quaternion);
      obj.scale.copy(scale);
    }
    root.updateMatrixWorld(true);
  };

  let usedClip = null;
  if (clip) {
    const found = asset.animations.find((c) => c.name === clip) || null;
    if (!found) {
      const names = asset.animations.map((c) => c.name).join(', ');
      throw new Error(`clip "${clip}" not found; available: [${names}]`);
    }
    usedClip = found;
    const mixer = new THREE.AnimationMixer(root);
    mixer.clipAction(found).play();
    mixer.setTime(time);
  } else if (time !== 0) {
    throw new Error('--time without --clip has no effect: the rest pose is time-independent');
  }
  root.updateMatrixWorld(true);

  try {
    return collectPose(asset, { THREE, root }, { usedClip, time, started });
  } finally {
    restore();
  }
}

function collectPose(asset, { THREE, root }, { usedClip, time, started }) {
  const meshes = collectMeshes(root);
  const vertices = [];
  const faces = [];
  const perMesh = [];
  const tmp = new THREE.Vector3();
  let vertexOffset = 0;
  let faceOffset = 0;

  for (const mesh of meshes) {
    const geometry = mesh.geometry;
    const count = geometry.attributes.position.count;
    for (let i = 0; i < count; i++) {
      // getVertexPosition()/applyBoneTransform() live in *mesh local* space
      // (three's shader applies matrixWorld after skinning), so multiply here:
      // this reproduces exactly where a viewer draws the vertex.
      mesh.getVertexPosition(i, tmp).applyMatrix4(mesh.matrixWorld);
      vertices.push(tmp.x, tmp.y, tmp.z);
    }
    const index = geometry.index ? geometry.index.array : null;
    const indices = index ? index.length : count;
    for (let i = 0; i < indices; i++) {
      faces.push((index ? index[i] : i) + vertexOffset);
    }
    const tris = Math.floor(indices / 3);
    perMesh.push({
      name: mesh.name || '(unnamed)',
      type: mesh.isSkinnedMesh ? 'SkinnedMesh' : 'Mesh',
      skinned: !!mesh.isSkinnedMesh,
      vertices: count, triangles: tris,
      vertex_offset: vertexOffset, face_offset: faceOffset,
    });
    vertexOffset += count;
    faceOffset += tris;
  }

  const positions = Float32Array.from(vertices);
  const indices32 = Int32Array.from(faces);
  const bakeMs = Number(process.hrtime.bigint() - started) / 1e6;

  return {
    vertices: positions,
    faces: indices32,
    perMesh,
    clip: usedClip ? usedClip.name : null,
    clip_duration: usedClip ? r6(usedClip.duration) : null,
    time: usedClip ? r6(time) : null,
    animations_available: asset.animations.map((c) => c.name),
    vertex_count: vertexOffset,
    triangle_count: faceOffset,
    bbox: bboxOfPoints(positions),
    bake_ms: Math.round(bakeMs * 1000) / 1000,
    space: 'gltf-world',
  };
}

export { bboxOfPoints };