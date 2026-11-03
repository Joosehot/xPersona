//! MakeHuman's base mesh and targets (CC0 1.0, makehumancommunity/makehuman), read as they are: a body is the base
//! mesh plus a weighted sum of targets, each target a list of vertex offsets. Only young adults are made, so the age
//! weights are fixed at "young".
use crate::v3::{v, V3};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const DATA: &str = "data/makehuman/makehuman/data";

pub struct Base {
    /// every vertex, helpers and joints included, in metres
    pub verts: Vec<V3>,
    /// the body's quads (vertex indices below `body_n`)
    pub quads: Vec<[u32; 4]>,
    pub body_n: usize,
    /// each joint helper's vertices
    pub joints: HashMap<String, Vec<u32>>,
}

/// decimetres to metres
const UNIT: f64 = 0.1;

pub fn load_base(root: &Path) -> Result<Base, String> {
    let path = root.join("3dobjs/base.obj");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e} (fetch MakeHuman's data first, see README)", path.display()))?;
    let mut verts = Vec::new();
    let mut quads = Vec::new();
    let mut joints: HashMap<String, Vec<u32>> = HashMap::new();
    let mut group = String::new();
    for line in text.lines() {
        let mut w = line.split_whitespace();
        match w.next() {
            Some("v") => {
                let c: Vec<f64> = w.take(3).map(|x| x.parse().unwrap_or(0.0)).collect();
                verts.push(v(c[0] * UNIT, c[1] * UNIT, c[2] * UNIT));
            }
            Some("g") => group = w.next().unwrap_or("").to_string(),
            Some("f") => {
                let idx: Vec<u32> = w.map(|t| t.split('/').next().unwrap().parse::<u32>().unwrap() - 1).collect();
                if group == "body" && idx.len() == 4 {
                    quads.push([idx[0], idx[1], idx[2], idx[3]]);
                } else if let Some(j) = group.strip_prefix("joint-") {
                    let e = joints.entry(j.to_string()).or_default();
                    for i in idx {
                        if !e.contains(&i) {
                            e.push(i);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    let body_n = quads.iter().flatten().map(|&i| i as usize + 1).max().unwrap_or(0);
    Ok(Base { verts, quads, body_n, joints })
}

/// a target: (vertex, offset in metres)
pub type Target = Vec<(u32, V3)>;

pub fn load_target(root: &Path, rel: &str) -> Result<Target, String> {
    let path: PathBuf = root.join("targets").join(rel);
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = Vec::new();
    for line in text.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let p: Vec<&str> = line.split_whitespace().collect();
        if p.len() == 4 {
            let f = |s: &str| s.parse::<f64>().unwrap_or(0.0) * UNIT;
            out.push((p[0].parse().map_err(|_| format!("{rel}: bad line {line}"))?, v(f(p[1]), f(p[2]), f(p[3]))));
        }
    }
    Ok(out)
}

/// what a body is made from: MakeHuman's own macro values and modifiers
#[derive(Clone, Debug)]
pub struct Params {
    /// 0 female, 1 male
    pub gender: f64,
    /// 0..1, 0.5 average
    pub muscle: f64,
    pub weight: f64,
    pub height: f64,
    /// 0.5..1, toward ideal
    pub proportions: f64,
    /// african, asian, caucasian; summing to 1
    pub ethnic: [f64; 3],
    /// measure modifiers, -1..1 each, in MEASURES order
    pub measures: Vec<f64>,
    /// head shapes, 0..1 each, in HEADS order
    pub heads: [f64; 4],
}

pub const MEASURES: &[&str] = &[
    "shoulder-dist", "hips-circ", "waist-circ", "bust-circ", "upperarm-length", "lowerarm-length", "upperleg-height", "lowerleg-height", "neck-height", "neck-circ", "napetowaist-dist", "waisttohip-dist", "thigh-circ", "upperarm-circ", "calf-circ",
];
pub const HEADS: [&str; 4] = ["oval", "round", "square", "rectangular"];
const ETHNIC: [&str; 3] = ["african", "asian", "caucasian"];

/// the three weights of a 0..1 value over min / average / max
fn tri(x: f64) -> [f64; 3] {
    if x < 0.5 {
        let lo = (0.5 - x) * 2.0;
        [lo, 1.0 - lo, 0.0]
    } else {
        let hi = (x - 0.5) * 2.0;
        [0.0, 1.0 - hi, hi]
    }
}

/// every target a body can use, loaded once
pub struct Targets {
    by_name: HashMap<String, Target>,
}

impl Targets {
    pub fn load(root: &Path) -> Result<Targets, String> {
        let mut names = Vec::new();
        for g in ["female", "male"] {
            for e in ETHNIC {
                names.push(format!("macrodetails/{e}-{g}-young.target"));
            }
            for m in ["minmuscle", "averagemuscle", "maxmuscle"] {
                for w in ["minweight", "averageweight", "maxweight"] {
                    names.push(format!("macrodetails/universal-{g}-young-{m}-{w}.target"));
                    for h in ["minheight", "maxheight"] {
                        names.push(format!("macrodetails/height/{g}-young-{m}-{w}-{h}.target"));
                    }
                    for p in ["idealproportions", "uncommonproportions"] {
                        names.push(format!("macrodetails/proportions/{g}-young-{m}-{w}-{p}.target"));
                    }
                }
            }
        }
        for m in MEASURES {
            for d in ["decr", "incr"] {
                names.push(format!("measure/measure-{m}-{d}.target"));
            }
        }
        for h in HEADS {
            names.push(format!("head/head-{h}.target"));
        }
        let mut by_name = HashMap::new();
        for n in names {
            let t = load_target(root, &n)?;
            by_name.insert(n, t);
        }
        Ok(Targets { by_name })
    }

    /// the body's vertices for these values: the base plus every target at its weight
    pub fn compose(&self, base: &Base, p: &Params) -> Vec<V3> {
        let mut out = base.verts.clone();
        let mut add = |name: &str, w: f64| {
            if w.abs() < 1e-9 {
                return;
            }
            for (i, d) in &self.by_name[name] {
                let x = &mut out[*i as usize];
                *x = *x + *d * w;
            }
        };
        let genders = [("female", 1.0 - p.gender), ("male", p.gender)];
        let (mw, ww) = (tri(p.muscle), tri(p.weight));
        let mn = ["minmuscle", "averagemuscle", "maxmuscle"];
        let wn = ["minweight", "averageweight", "maxweight"];
        let hw = [("minheight", ((0.5 - p.height) * 2.0).max(0.0)), ("maxheight", ((p.height - 0.5) * 2.0).max(0.0))];
        let pw = [("idealproportions", ((p.proportions - 0.5) * 2.0).max(0.0)), ("uncommonproportions", ((0.5 - p.proportions) * 2.0).max(0.0))];
        for (g, gw) in genders {
            for (e, ew) in ETHNIC.iter().zip(p.ethnic) {
                add(&format!("macrodetails/{e}-{g}-young.target"), gw * ew);
            }
            for (mi, m) in mn.iter().enumerate() {
                for (wi, w) in wn.iter().enumerate() {
                    let k = gw * mw[mi] * ww[wi];
                    add(&format!("macrodetails/universal-{g}-young-{m}-{w}.target"), k);
                    for (h, x) in hw {
                        add(&format!("macrodetails/height/{g}-young-{m}-{w}-{h}.target"), k * x);
                    }
                    for (q, x) in pw {
                        add(&format!("macrodetails/proportions/{g}-young-{m}-{w}-{q}.target"), k * x);
                    }
                }
            }
        }
        for (m, x) in MEASURES.iter().zip(&p.measures) {
            let d = if *x < 0.0 { "decr" } else { "incr" };
            add(&format!("measure/measure-{m}-{d}.target"), x.abs());
        }
        for (h, x) in HEADS.iter().zip(p.heads) {
            add(&format!("head/head-{h}.target"), x);
        }
        out
    }
}
