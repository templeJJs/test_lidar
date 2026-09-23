#!/usr/bin/env node
// Bake an animation pose of a GLB/FBX into plain vertex + face arrays.
//
//   node tools/bake_pose.js <asset> --clip <name> --time 0.5 --out pose.npz
//   node tools/bake_pose.js <asset> --list
//   node tools/bake_pose.js <asset> --clip walk --times 0,0.5,1,1.5 --outdir assets/_poses
//
// Writes <out>.npz  (numpy.load-able: "vertices" (N,3) float32, "faces" (M,3) int32)
// and     <out>.json (clip, time, counts, bbox, per-mesh table, timings).
//
// Vertices come out in the *asset's* world space (Y-up, asset units) with
// skinning + morphs applied -- i.e. exactly what a viewer draws at that moment.
// Placing it in metres on the track is the Python side's job: see tools/README.md.
//
// `--loop once` matters at exactly `--time <clip duration>`: three's AnimationMixer
// wraps an action that reaches the clip duration, so a bake at t == duration under
// the default LoopRepeat silently returns the FIRST pose instead of the LAST one.
// Bake the final frame of a one-shot clip (death, jump, punch) with `--loop once`,
// otherwise the last moment is a duplicate of t=0.

'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');

const HERE = __dirname;

function usage(code = 2) {
  console.error(`usage:
  node tools/bake_pose.js <asset.glb|.fbx|.obj> --clip <name> --time <sec> --out <file.npz>
  node tools/bake_pose.js <asset> --list
  node tools/bake_pose.js <asset> --clip <name> --times 0,0.5,1 --outdir <dir>

options: --time t1,t2,...   several moments in one process (mixer state is restored between bakes)
         --scale S          multiply baked vertices by S and report it in the metadata
         --loop once|repeat action loop mode for the bake (default repeat; use once to get the
                            real final pose at t == clip duration, see the note at the top)
         --out FILE         output base name, .npz/.json are appended
         --outdir DIR       output directory for --times / default time list
         --quiet            only print the JSON metadata lines`);
  process.exit(code);
}

function parseArgs(argv) {
  const opts = { times: null, time: null, clip: null, out: null, outdir: null, list: false,
                 scale: 1, quiet: false, loop: 'repeat' };
  const files = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--clip') opts.clip = argv[++i];
    else if (a === '--time') opts.times = argv[++i];
    else if (a === '--times') opts.times = argv[++i];
    else if (a === '--out') opts.out = argv[++i];
    else if (a === '--outdir') opts.outdir = argv[++i];
    else if (a === '--scale') opts.scale = Number(argv[++i]);
    else if (a === '--loop') opts.loop = argv[++i];
    else if (a === '--list') opts.list = true;
    else if (a === '--quiet') opts.quiet = true;
    else if (a === '-h' || a === '--help') usage(0);
    else if (a.startsWith('-')) usage();
    else files.push(a);
  }
  if (opts.loop !== 'repeat' && opts.loop !== 'once') {
    console.error(`--loop must be "once" or "repeat", got ${JSON.stringify(opts.loop)}`);
    process.exit(2);
  }
  if (opts.loop === 'once' && !opts.clip) {
    console.error('--loop once needs --clip');
    process.exit(2);
  }
  if (files.length !== 1) usage();
  opts.file = files[0];
  if (!opts.times) opts.times = opts.time === null ? null : opts.time;
  opts.timeList = opts.times === null ? [] : String(opts.times).split(',').map((s) => Number(s.trim()));
  if (opts.timeList.some((t) => !Number.isFinite(t))) {
    console.error('--time must be a comma separated list of numbers');
    process.exit(2);
  }
  return opts;
}

function slug(text) {
  return String(text).replace(/[^A-Za-z0-9_.-]+/g, '_').replace(/^_+|_+$/g, '');
}

function fmtTime(t) {
  return `t${t.toFixed(3)}`.replace(/0+$/, '').replace(/\.$/, '');
}

(async () => {
  const opts = parseArgs(process.argv.slice(2));
  const { loadThree, loadAsset, listClips, bakePose } = await import(
    pathToFileURL(path.join(HERE, 'three_env.mjs')).href);
  const { writeNpz } = await import(pathToFileURL(path.join(HERE, 'npz.mjs')).href);
  const { createHash } = require('node:crypto');

  const ctx = await loadThree();
  const asset = await loadAsset(opts.file, ctx);
  const assetAbs = path.resolve(opts.file);
  const assetSha = createHash('sha256').update(fs.readFileSync(assetAbs)).digest('hex');
  const clips = listClips(asset);

  if (opts.list) {
    console.log(JSON.stringify({
      file: path.relative(process.cwd(), assetAbs).replace(/\\/g, '/'),
      loader: asset.loader, sha256: assetSha,
      clips: clips.map((c) => ({ name: c.name, duration: c.duration, tracks: c.tracks })),
    }, null, 2));
    return;
  }

  if (opts.timeList.length === 0) opts.timeList = [0];
  if (opts.timeList.length > 1 && !opts.outdir) {
    console.error('--time with several values needs --outdir');
    process.exit(2);
  }

  const baseName = slug(path.basename(opts.file, path.extname(opts.file)));
  const records = [];
  for (const time of opts.timeList) {
    const pose = bakePose(asset, { clip: opts.clip, time: opts.clip ? time : 0, loop: opts.loop });
    const vertices = opts.scale === 1
      ? pose.vertices
      : Float32Array.from(pose.vertices, (v) => v * opts.scale);
    const bbox = opts.scale === 1 ? pose.bbox : (() => {
      const min = [Infinity, Infinity, Infinity];
      const max = [-Infinity, -Infinity, -Infinity];
      for (let i = 0; i < vertices.length; i += 3) for (let k = 0; k < 3; k++) {
        if (vertices[i + k] < min[k]) min[k] = vertices[i + k];
        if (vertices[i + k] > max[k]) max[k] = vertices[i + k];
      }
      return { min, max, size: max.map((v, k) => v - min[k]),
               diagonal: Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) };
    })();

    const suffix = opts.clip ? `_${slug(opts.clip)}_${fmtTime(time)}` : '_restpose';
    let outBase;
    if (opts.out && opts.timeList.length === 1) outBase = opts.out.replace(/\.(npz|json)$/i, '');
    else if (opts.outdir) {
      fs.mkdirSync(opts.outdir, { recursive: true });
      outBase = path.join(opts.outdir, `${baseName}${suffix}`);
    } else {
      outBase = path.join(path.dirname(assetAbs), `${baseName}${suffix}`);
    }

    const zip = writeNpz(`${outBase}.npz`, {
      vertices: { data: vertices, shape: [vertices.length / 3, 3] },
      faces: { data: pose.faces, shape: [pose.faces.length / 3, 3] },
    });

    const meta = {
      format: 'bake_pose/1',
      asset: { path: path.relative(process.cwd(), assetAbs).replace(/\\/g, '/'),
               sha256: assetSha, bytes: asset.bytes, loader: asset.loader },
      clip: pose.clip, time_seconds: pose.time, clip_duration: pose.clip_duration,
      clip_loop_mode: pose.clip ? opts.loop : null,
      clips_available: pose.animations_available,
      space: `${pose.space} (Y-up, asset units${opts.scale !== 1 ? `, scaled by ${opts.scale}` : ''})`,
      scale_applied: opts.scale,
      vertices: pose.vertex_count, triangles: pose.triangle_count,
      bbox: {
        min: bbox.min.map((v) => Math.round(v * 1e6) / 1e6),
        max: bbox.max.map((v) => Math.round(v * 1e6) / 1e6),
        size: bbox.size.map((v) => Math.round(v * 1e6) / 1e6),
        diagonal: Math.round(bbox.diagonal * 1e6) / 1e6,
      },
      meshes: pose.perMesh,
      bake_ms: pose.bake_ms,
      npz: { file: path.basename(outBase) + '.npz', bytes: zip.zip_bytes, entries: zip.entries },
    };
    fs.writeFileSync(`${outBase}.json`, JSON.stringify(meta, null, 2) + '\n');
    records.push(meta);

    if (!opts.quiet) {
      console.log(`${path.basename(outBase)}.npz  verts=${meta.vertices} tris=${meta.triangles} ` +
        `clip=${meta.clip} t=${meta.time_seconds} bake=${meta.bake_ms}ms ` +
        `npz=${(zip.zip_bytes / 1024).toFixed(1)}KiB bbox_size=[${meta.bbox.size.join(', ')}]`);
    }
  }
  console.log(JSON.stringify(opts.quiet ? records : records.map((r) => ({
    npz: r.npz.file, clip: r.clip, time: r.time_seconds, vertices: r.vertices,
    triangles: r.triangles, bbox_size: r.bbox.size, bake_ms: r.bake_ms,
  })), null, 2));
})().catch((err) => {
  console.error(`bake_pose failed: ${err && err.stack ? err.stack : err}`);
  process.exit(1);
});