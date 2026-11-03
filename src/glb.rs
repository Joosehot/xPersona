//! The meshes as one binary glTF 2.0 file (.glb), which Blender, game engines and browsers open.
use crate::mesh::Mesh;
use serde_json::{json, Value};

/// one mesh of the file, with its colour, or a colour per vertex
pub struct Part<'a> {
    pub name: &'a str,
    pub mesh: &'a Mesh,
    pub color: [f64; 3],
    pub colors: Option<&'a [[f32; 3]]>,
}

pub fn write(parts: &[Part]) -> Vec<u8> {
    let mut bin = Vec::new();
    let (mut views, mut accessors, mut meshes, mut materials, mut nodes) = (Vec::<Value>::new(), Vec::<Value>::new(), Vec::new(), Vec::new(), Vec::new());
    let mut view = |bin: &mut Vec<u8>, bytes: Vec<u8>, target: u32| {
        let at = bin.len();
        bin.extend_from_slice(&bytes);
        while bin.len() % 4 != 0 {
            bin.push(0);
        }
        views.push(json!({ "buffer": 0, "byteOffset": at, "byteLength": bytes.len(), "target": target }));
        views.len() - 1
    };
    let floats = |xs: &[[f32; 3]]| xs.iter().flatten().flat_map(|c| c.to_le_bytes()).collect::<Vec<u8>>();
    for (k, p) in parts.iter().enumerate() {
        let m = p.mesh;
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for q in &m.pos {
            for a in 0..3 {
                lo[a] = lo[a].min(q[a]);
                hi[a] = hi[a].max(q[a]);
            }
        }
        let n = m.pos.len();
        let pv = view(&mut bin, floats(&m.pos), 34962);
        accessors.push(json!({ "bufferView": pv, "componentType": 5126, "count": n, "type": "VEC3", "min": lo, "max": hi }));
        let nv = view(&mut bin, floats(&m.nrm), 34962);
        accessors.push(json!({ "bufferView": nv, "componentType": 5126, "count": n, "type": "VEC3" }));
        let mut attributes = json!({ "POSITION": accessors.len() - 2, "NORMAL": accessors.len() - 1 });
        if let Some(cs) = p.colors {
            let cv = view(&mut bin, floats(cs), 34962);
            accessors.push(json!({ "bufferView": cv, "componentType": 5126, "count": n, "type": "VEC3" }));
            attributes["COLOR_0"] = json!(accessors.len() - 1);
        }
        let iv = view(&mut bin, m.idx.iter().flat_map(|i| i.to_le_bytes()).collect(), 34963);
        accessors.push(json!({ "bufferView": iv, "componentType": 5125, "count": m.idx.len(), "type": "SCALAR" }));
        let c = if p.colors.is_some() { [1.0, 1.0, 1.0] } else { p.color };
        materials.push(json!({ "name": p.name, "pbrMetallicRoughness": { "baseColorFactor": [c[0], c[1], c[2], 1.0], "metallicFactor": 0.0, "roughnessFactor": if p.colors.is_some() { 0.3 } else { 0.65 } } }));
        meshes.push(json!({ "name": p.name, "primitives": [{ "attributes": attributes, "indices": accessors.len() - 1, "material": k }] }));
        nodes.push(json!({ "mesh": k, "name": p.name }));
    }
    let doc = json!({
        "asset": { "version": "2.0", "generator": "xPersona v0" },
        "scene": 0,
        "scenes": [{ "nodes": (0..parts.len()).collect::<Vec<_>>() }],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "buffers": [{ "byteLength": bin.len() }],
        "bufferViews": views,
        "accessors": accessors
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
