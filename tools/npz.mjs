// Minimal .npz (zip of .npy) writer: numpy.load() reads the result directly.
//
// Node ships zlib but no zip container, and we deliberately keep the tool free
// of npm dependencies, so entries are stored uncompressed (ZIP method 0) with a
// hand-rolled CRC32 -- exactly what numpy.savez does by default.  Data that
// Python needs is a couple of MB at most, so the size cost is irrelevant and
// the format stays trivially inspectable by both sides.

import fs from 'node:fs';

const CRC_TABLE = (() => {
  const table = new Int32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c;
  }
  return table;
})();

function crc32(buf) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

const DESCR = { Float32Array: '<f4', Float64Array: '<f8', Int32Array: '<i4', Int16Array: '<i2',
                Int8Array: '|i1', Uint8Array: '|u1', Uint16Array: '<u2', Uint32Array: '<u4' };

/** .npy v1.0 byte string for a C-contiguous array, shape = [rows, cols]. */
export function npyBytes(data, shape) {
  const descr = DESCR[data.constructor.name];
  if (!descr) throw new Error(`unsupported dtype ${data.constructor.name}`);
  const shapeText = shape.length === 1 ? `(${shape[0]},)` : `(${shape.join(', ')})`;
  let header = `{'descr': '${descr}', 'fortran_order': False, 'shape': ${shapeText}, }`;
  const padding = 64 - ((10 + header.length + 1) % 64);
  header = header + ' '.repeat(padding) + '\n';
  const headerBytes = Buffer.from(header, 'latin1');
  if (headerBytes.length > 0xffff) throw new Error('npy header too long');
  const out = Buffer.alloc(10 + headerBytes.length);
  out.write('\x93NUMPY', 0, 'latin1');
  out[6] = 1; // major version
  out[7] = 0; // minor version
  out.writeUInt16LE(headerBytes.length, 8);
  headerBytes.copy(out, 10);
  return Buffer.concat([out, Buffer.from(data.buffer, data.byteOffset, data.byteLength)]);
}

/**
 * Write { name: {data: TypedArray, shape: [...]} } as a .npz.
 * Returns { zip_bytes, entries: [...] }.
 */
export function writeNpz(file, arrays, { date = new Date() } = {}) {
  const chunks = [];
  const central = [];
  let offset = 0;
  const dosTime = ((date.getHours() << 11) | (date.getMinutes() << 5) | (date.getSeconds() / 2)) & 0xffff;
  const dosDate = (((date.getFullYear() - 1980) << 9) | ((date.getMonth() + 1) << 5) | date.getDate()) & 0xffff;
  const entries = [];

  for (const [name, { data, shape }] of Object.entries(arrays)) {
    const payload = npyBytes(data, shape);
    const nameBytes = Buffer.from(`${name}.npy`, 'utf8');
    const crc = crc32(payload);

    const local = Buffer.alloc(30 + nameBytes.length);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4);
    local.writeUInt16LE(0, 6);
    local.writeUInt16LE(0, 8); // stored
    local.writeUInt16LE(dosTime, 10);
    local.writeUInt16LE(dosDate, 12);
    local.writeUInt32LE(crc, 14);
    local.writeUInt32LE(payload.length, 18);
    local.writeUInt32LE(payload.length, 22);
    local.writeUInt16LE(nameBytes.length, 26);
    local.writeUInt16LE(0, 28);
    nameBytes.copy(local, 30);

    chunks.push(local, payload);

    const cd = Buffer.alloc(46 + nameBytes.length);
    cd.writeUInt32LE(0x02014b50, 0);
    cd.writeUInt16LE(20, 4);
    cd.writeUInt16LE(20, 6);
    cd.writeUInt16LE(0, 8);
    cd.writeUInt16LE(0, 10);
    cd.writeUInt16LE(dosTime, 12);
    cd.writeUInt16LE(dosDate, 14);
    cd.writeUInt32LE(crc, 16);
    cd.writeUInt32LE(payload.length, 20);
    cd.writeUInt32LE(payload.length, 24);
    cd.writeUInt16LE(nameBytes.length, 28);
    cd.writeUInt16LE(0, 30);
    cd.writeUInt16LE(0, 32);
    cd.writeUInt16LE(0, 34);
    cd.writeUInt16LE(0, 36);
    cd.writeUInt32LE(0, 38);
    cd.writeUInt32LE(offset, 42);
    nameBytes.copy(cd, 46);
    central.push(cd);

    offset += local.length + payload.length;
    entries.push({ name: `${name}.npy`, dtype: data.constructor.name, shape,
                   values: data.length, crc32: crc.toString(16), npy_bytes: payload.length });
  }

  const centralBuf = Buffer.concat(central);
  const eocd = Buffer.alloc(22);
  eocd.writeUInt32LE(0x06054b50, 0);
  eocd.writeUInt16LE(central.length, 8);
  eocd.writeUInt16LE(central.length, 10);
  eocd.writeUInt32LE(centralBuf.length, 12);
  eocd.writeUInt32LE(offset, 16);
  eocd.writeUInt16LE(0, 20);

  const zip = Buffer.concat([...chunks, centralBuf, eocd]);
  fs.writeFileSync(file, zip);
  return { zip_bytes: zip.length, entries };
}

export { crc32 };