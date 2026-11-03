//! The mesh as a binary glTF 2.0 file (.glb), which Blender, game engines and browsers open.
use crate::mesh::Mesh;
use serde_json::json;

pub fn write(mesh: &Mesh, color: [f64; 3], name: &str) -> Vec<u8> {
    let mut bin = Vec::new();
    for p in &mesh.pos {
        for c in p {
            bin.extend_from_slice(&c.to_le_bytes());
        }
    }
    let nrm_at = bin.len();
    for n in &mesh.nrm {
        for c in n {
            bin.extend_from_slice(&c.to_le_bytes());
        }
    }
    let idx_at = bin.len();
    for i in &mesh.idx {
        bin.extend_from_slice(&i.to_le_bytes());
    }
    while bin.len() % 4 != 0 {
        bin.push(0);
    }
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for p in &mesh.pos {
        for a in 0..3 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    }
    let n = mesh.pos.len();
    let doc = json!({
        "asset": { "version": "2.0", "generator": "xPersona v0" },
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "mesh": 0, "name": name }],
        "meshes": [{ "name": name, "primitives": [{ "attributes": { "POSITION": 0, "NORMAL": 1 }, "indices": 2, "material": 0 }] }],
        "materials": [{ "name": "clay", "pbrMetallicRoughness": { "baseColorFactor": [color[0], color[1], color[2], 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.65 } }],
        "buffers": [{ "byteLength": bin.len() }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": nrm_at, "target": 34962 },
            { "buffer": 0, "byteOffset": nrm_at, "byteLength": idx_at - nrm_at, "target": 34962 },
            { "buffer": 0, "byteOffset": idx_at, "byteLength": mesh.idx.len() * 4, "target": 34963 }
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": n, "type": "VEC3", "min": lo, "max": hi },
            { "bufferView": 1, "componentType": 5126, "count": n, "type": "VEC3" },
            { "bufferView": 2, "componentType": 5125, "count": mesh.idx.len(), "type": "SCALAR" }
        ]
    });
    let mut text = serde_json::to_vec(&doc).expect("json");
    while text.len() % 4 != 0 {
        text.push(b' ');
    }
    let total = 12 + 8 + text.len() + 8 + bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(text.len() as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&text);
    out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(&bin);
    out
}
