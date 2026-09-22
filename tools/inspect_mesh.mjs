#!/usr/bin/env node
// Inspect a 3D asset with the three.js loaders inside plain Node.
//
//   node tools/inspect_mesh.mjs assets/cesium_man/CesiumMan.glb [more files...]
//   node tools/inspect_mesh.mjs --clips assets/cesium_man/CesiumMan.glb
//
// Prints one JSON object per file: read ok/failed, vertex/triangle counts,
// world-space bounding box (skinning applied at t=0), meshes, animation clips
// and load timing.  tools/check_assets.py shells out to this, but it is handy
// on its own when a new asset shows up.

import path from 'node:path';
import { loadThree, loadAsset, listClips, meshStats, bakePose } from './three_env.mjs';

const argv = process.argv.slice(2);
const showClips = argv.includes('--clips');
const files = argv.filter((a) => !a.startsWith('-'));

if (files.length === 0) {
  console.error('usage: node tools/inspect_mesh.mjs [--clips] <file.glb|.fbx|.obj|.stl|.ply> ...');
  process.exit(2);
}

const three = await loadThree();
const results = [];
for (const file of files) {
  const t0 = process.hrtime.bigint();
  const rec = { file: file.replace(/\\/g, '/'), ext: path.extname(file).toLowerCase() };
  try {
    const asset = await loadAsset(file, { three });
    const loadMs = Number(process.hrtime.bigint() - t0) / 1e6;
    const stats = meshStats(asset.root);
    // Bake the rest pose: for skinned meshes this is the real world-space
    // extent (bind pose + skeleton), not the raw geometry attribute bounds.
    const pose = bakePose(asset, {});
    rec.ok = true;
    rec.loader = asset.loader;
    rec.bytes = asset.bytes;
    rec.load_ms = Math.round(loadMs * 100) / 100;
    rec.vertices = stats.vertices;
    rec.triangles = stats.triangles;
    rec.skinned_meshes = stats.skinned_meshes;
    rec.morph_meshes = stats.morph_meshes;
    rec.meshes = stats.meshes;
    rec.bbox_world = pose.bbox;
    rec.bbox_units = 'asset';
    rec.animations = listClips(asset);
    if (showClips) rec.clips = pose.animations_available;
  } catch (err) {
    rec.ok = false;
    rec.error = `${err.name}: ${err.message}`;
  }
  results.push(rec);
}

console.log(JSON.stringify(results.length === 1 ? results[0] : results, null, 2));