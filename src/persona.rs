//! The decided body made by the learned model: the rules' decisions become the network's inputs (sex, MakeHuman's
//! muscle and weight, nine measures, the head's shape), the network gives the shape, and the shape is measured back.
use crate::config::Config;
use crate::learn::{inputs, Learned};
use crate::measure::{measure, Measures, NAMES};
use crate::mesh::Mesh;
use crate::parser::Request;
use crate::rules::{self, Spec};
use crate::v3::v;

/// the same request with nothing asked of single parts, every variant at its plain one: what the factors are against
pub fn neutral(r: &Request, c: &Config) -> Spec {
    let mut r0 = r.clone();
    r0.parts.clear();
    let mut s = Spec::new(c);
    for rule in rules::all() {
        let v = match rule.name() {
            "proportions" | "fat" => "even",
            "torso" => "straight",
            "tone" => "smooth",
            "head" => "oval",
            "pose" => "a_pose",
            _ => "as_asked",
        };
        rule.apply(v, &r0, c, &mut s);
    }
    s
}

/// how much each measure was asked to change, against the neutral body
fn factors(s: &Spec, n: &Spec) -> [f64; 9] {
    let legs = |x: &Spec| x.chain.ankle + x.chain.shank + x.chain.thigh;
    let arm = |x: &Spec| x.chain.upper_arm + x.chain.forearm + x.chain.hand * x.hand;
    [
        legs(s) / legs(n),
        arm(s) / arm(n),
        s.w.shoulders / n.w.shoulders,
        s.w.chest / n.w.chest,
        s.w.waist / n.w.waist,
        s.w.waist_depth / n.w.waist_depth,
        s.w.hips / n.w.hips,
        s.w.neck / n.w.neck,
        s.chain.head / n.chain.head,
    ]
}

pub struct Made {
    pub mesh: Mesh,
    pub asked: [f64; 9],
    pub got: Measures,
    pub lines: Vec<String>,
}

pub fn make(r: &Request, s: &Spec, head: &str, c: &Config, l: &Learned) -> Made {
    let build = format!("{:?}", r.build()).to_lowercase();
    let strength = r.build.as_ref().map_or(1.0, |b| b.value.1);
    let [m0, w0] = c.build[&build].makehuman;
    let muscle = (0.5 + (m0 - 0.5) * strength + if s.defined { c.learned.defined_muscle } else { 0.0 }).clamp(0.0, 1.0);
    let weight = (0.5 + (w0 - 0.5) * strength).clamp(0.0, 1.0);
    let gender = 1.0 - s.female;
    let expected = l.expected(gender, muscle, weight, s.ideal);
    let f = factors(s, &neutral(r, c));
    let asked: [f64; 9] = std::array::from_fn(|i| expected[i] * f[i]);
    let mut heads = [0.0; 4];
    let hi = ["oval", "round", "square", "long"].iter().position(|x| *x == head).unwrap_or(0);
    heads[hi] = c.learned.head_weight;
    let shape = l.shape(&inputs(gender, muscle, weight, &asked, &heads, s.ideal));
    let body = l.body(&shape, s.h);
    let got = measure(&body, &body.verts);

    // MakeHuman's quads as triangles, the normals from the faces around each vertex
    let pos: Vec<[f32; 3]> = body.verts[..body.body_n].iter().map(|p| [p.x as f32, p.y as f32, p.z as f32]).collect();
    let mut idx = Vec::with_capacity(body.quads.len() * 6);
    for q in &body.quads {
        idx.extend_from_slice(&[q[0], q[1], q[2], q[0], q[2], q[3]]);
    }
    let mut nrm = vec![v(0.0, 0.0, 0.0); pos.len()];
    for t in idx.chunks(3) {
        let p = |i: u32| body.verts[i as usize];
        let n = (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0]));
        for &i in t {
            nrm[i as usize] = nrm[i as usize] + n;
        }
    }
    let nrm = nrm.iter().map(|n| {
        let u = if n.len() > 0.0 { n.unit() } else { v(0.0, 1.0, 0.0) };
        [u.x as f32, u.y as f32, u.z as f32]
    }).collect();
    let lines = vec![format!(
        "Learned body: MakeHuman muscle {muscle:.2}, weight {weight:.2} (\"{build}\"{}), {} head; asked {}.",
        if s.defined { ", defined" } else { "" },
        head,
        NAMES.iter().zip(&asked).map(|(n, a)| format!("{n} {a:.3}")).collect::<Vec<_>>().join(", ")
    )];
    Made { mesh: Mesh { pos, nrm, idx }, asked, got, lines }
}
