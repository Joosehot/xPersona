//! The decided body made by the learned model: the rules' decisions become the network's inputs (sex, MakeHuman's
//! muscle and weight, nine measures, the head's shape), the network gives the shape, and the shape is measured back.
use crate::config::Config;
use crate::learn::{inputs, random_face, Learned, BODY_INPUTS};
use crate::measure::{measure, Measures, NAMES};
use crate::mesh::Mesh;
use crate::parser::Request;
use crate::rules::{self, Spec};
use crate::v3::{v, V3};
use std::collections::HashMap;

/// the same request with nothing asked of single parts, every variant at its plain one: what the factors are against.
/// Fat is placed as the body is: the learned body takes its fat from MakeHuman's weight, so where the rules put it is
/// not asked of the network a second time
pub fn neutral(r: &Request, decided: &Spec, c: &Config) -> Spec {
    let mut r0 = r.clone();
    r0.parts.clear();
    let mut s = Spec::new(c);
    for rule in rules::all() {
        let v = match rule.name() {
            "proportions" => "even",
            "fat" if decided.fat_waist > 1.0 => "central",
            "fat" if decided.fat_hips > 1.0 => "lower",
            "fat" => "even",
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
    /// the eyeballs, with a colour per vertex
    pub eyes: Mesh,
    pub eye_colors: Vec<[f32; 3]>,
    pub asked: [f64; 9],
    pub got: Measures,
    /// the face's mean distance from the average face, mm; and the same person's face unasked, when it was asked
    pub face_mm: f64,
    pub plain_face_mm: Option<f64>,
    pub lines: Vec<String>,
}

/// one level of Catmull-Clark subdivision on a closed quad mesh: each quad becomes four, the surface smooth
pub fn subdivide(verts: &[V3], quads: &[[u32; 4]]) -> (Vec<V3>, Vec<[u32; 4]>) {
    let n = verts.len();
    let mut edges: HashMap<(u32, u32), usize> = HashMap::new();
    let mut edge_faces: Vec<Vec<usize>> = Vec::new();
    let mut edge_ends: Vec<(u32, u32)> = Vec::new();
    for (fi, q) in quads.iter().enumerate() {
        for k in 0..4 {
            let (a, b) = (q[k], q[(k + 1) % 4]);
            let key = (a.min(b), a.max(b));
            let e = *edges.entry(key).or_insert_with(|| {
                edge_faces.push(Vec::new());
                edge_ends.push(key);
                edge_ends.len() - 1
            });
            edge_faces[e].push(fi);
        }
    }
    let face_pt: Vec<V3> = quads.iter().map(|q| (verts[q[0] as usize] + verts[q[1] as usize] + verts[q[2] as usize] + verts[q[3] as usize]) * 0.25).collect();
    let edge_pt: Vec<V3> = edge_ends
        .iter()
        .zip(&edge_faces)
        .map(|(&(a, b), fs)| {
            let mid = verts[a as usize] + verts[b as usize];
            if fs.len() == 2 {
                (mid + face_pt[fs[0]] + face_pt[fs[1]]) * 0.25
            } else {
                mid * 0.5
            }
        })
        .collect();
    // each old vertex: (F + 2R + (k - 3)P) / k
    let mut f_sum = vec![v(0.0, 0.0, 0.0); n];
    let mut r_sum = vec![v(0.0, 0.0, 0.0); n];
    let mut fk = vec![0usize; n];
    let mut rk = vec![0usize; n];
    for (fi, q) in quads.iter().enumerate() {
        for &i in q {
            f_sum[i as usize] = f_sum[i as usize] + face_pt[fi];
            fk[i as usize] += 1;
        }
    }
    for &(a, b) in &edge_ends {
        let mid = (verts[a as usize] + verts[b as usize]) * 0.5;
        for i in [a, b] {
            r_sum[i as usize] = r_sum[i as usize] + mid;
            rk[i as usize] += 1;
        }
    }
    let mut out: Vec<V3> = (0..n)
        .map(|i| {
            if fk[i] == 0 {
                return verts[i];
            }
            let k = fk[i] as f64;
            let f = f_sum[i] * (1.0 / k);
            let r = r_sum[i] * (1.0 / rk[i] as f64);
            (f + r * 2.0 + verts[i] * (k - 3.0)) * (1.0 / k)
        })
        .collect();
    let e0 = out.len() as u32;
    out.extend(edge_pt);
    let f0 = out.len() as u32;
    out.extend(face_pt);
    let e = |a: u32, b: u32| e0 + edges[&(a.min(b), a.max(b))] as u32;
    let mut q2 = Vec::with_capacity(quads.len() * 4);
    for (fi, q) in quads.iter().enumerate() {
        for k in 0..4 {
            let (prev, here, next) = (q[(k + 3) % 4], q[k], q[(k + 1) % 4]);
            q2.push([here, e(here, next), f0 + fi as u32, e(prev, here)]);
        }
    }
    (out, q2)
}

/// quads as triangles, with normals from the faces around each vertex
fn triangles(verts: &[V3], quads: &[[u32; 4]]) -> Mesh {
    let pos: Vec<[f32; 3]> = verts.iter().map(|p| [p.x as f32, p.y as f32, p.z as f32]).collect();
    let mut idx = Vec::with_capacity(quads.len() * 6);
    for q in quads {
        idx.extend_from_slice(&[q[0], q[1], q[2], q[0], q[2], q[3]]);
    }
    let mut nrm = vec![v(0.0, 0.0, 0.0); pos.len()];
    for t in idx.chunks(3) {
        let p = |i: u32| verts[i as usize];
        let n = (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0]));
        for &i in t {
            nrm[i as usize] = nrm[i as usize] + n;
        }
    }
    let nrm = nrm
        .iter()
        .map(|n| {
            let u = if n.len() > 0.0 { n.unit() } else { v(0.0, 1.0, 0.0) };
            [u.x as f32, u.y as f32, u.z as f32]
        })
        .collect();
    Mesh { pos, nrm, idx }
}

/// an eyeball in MakeHuman's socket: centred on the eye joint, looking at the eye target; pupil, iris, white
fn eyeball(centre: V3, gaze: V3, radius: f64, iris: [f64; 3], out: &mut Mesh, colors: &mut Vec<[f32; 3]>) {
    const AROUND: usize = 32;
    const ALONG: usize = 24;
    let g = gaze.unit();
    let up = if g.y.abs() < 0.9 { v(0.0, 1.0, 0.0) } else { v(1.0, 0.0, 0.0) };
    let u = g.cross(up).unit();
    let w = u.cross(g);
    let first = out.pos.len() as u32;
    for i in 0..=ALONG {
        let theta = std::f64::consts::PI * i as f64 / ALONG as f64;
        for j in 0..AROUND {
            let phi = std::f64::consts::TAU * j as f64 / AROUND as f64;
            let n = g * theta.cos() + (u * phi.cos() + w * phi.sin()) * theta.sin();
            let p = centre + n * radius;
            out.pos.push([p.x as f32, p.y as f32, p.z as f32]);
            out.nrm.push([n.x as f32, n.y as f32, n.z as f32]);
            let c = if theta < 0.17 {
                [0.01, 0.01, 0.01]
            } else if theta < 0.42 {
                iris
            } else {
                [0.86, 0.84, 0.80]
            };
            colors.push([c[0] as f32, c[1] as f32, c[2] as f32]);
        }
    }
    for i in 0..ALONG as u32 {
        for j in 0..AROUND as u32 {
            let a = first + i * AROUND as u32 + j;
            let b = first + i * AROUND as u32 + (j + 1) % AROUND as u32;
            let (c, d) = (a + AROUND as u32, b + AROUND as u32);
            // wound to face out (counter-clockwise seen from outside)
            out.idx.extend_from_slice(&[a, d, c, a, b, d]);
        }
    }
}

/// the face region of a unit-height shape: the skin in front of the ears above the chin
fn face_distance(a: &[f32], b: &[f32], body_n: usize, h: f64) -> f64 {
    let (mut sum, mut k) = (0.0, 0usize);
    for (p, q) in a[..body_n * 3].chunks(3).zip(b.chunks(3)) {
        if q[1] > 0.88 && q[2] > 0.03 {
            sum += ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt() as f64;
            k += 1;
        }
    }
    sum / k.max(1) as f64 * h * 1000.0
}

pub fn make(r: &Request, s: &Spec, head: &str, c: &Config, l: &Learned) -> Made {
    let build = format!("{:?}", r.build()).to_lowercase();
    let strength = r.build.as_ref().map_or(1.0, |b| b.value.1);
    let [m0, w0] = c.build[&build].makehuman;
    let muscle = (0.5 + (m0 - 0.5) * strength + if s.defined { c.learned.defined_muscle } else { 0.0 }).clamp(0.0, 1.0);
    let weight = (0.5 + (w0 - 0.5) * strength).clamp(0.0, 1.0);
    let gender = 1.0 - s.female;
    let expected = l.expected(gender, muscle, weight, s.ideal);
    let f = factors(s, &neutral(r, s, c));
    // what the rules ask, held to what MakeHuman's bodies of this sex, muscle and weight measure (5th to 95th percentile)
    let wanted: [f64; 9] = std::array::from_fn(|i| expected[i] * f[i]);
    let reach = &l.reach[crate::learn::bin(gender, muscle, weight)];
    let asked: [f64; 9] = std::array::from_fn(|i| wanted[i].clamp(reach[i][0], reach[i][1]));
    let held: Vec<String> = (0..9).filter(|&i| (asked[i] - wanted[i]).abs() > 0.0005).map(|i| format!("{} {:.3} held to {:.3}", NAMES[i], wanted[i], asked[i])).collect();
    let mut heads = [0.0; 4];
    let hi = ["oval", "round", "square", "long"].iter().position(|x| *x == head).unwrap_or(0);
    heads[hi] = c.learned.head_weight;
    let individual = random_face(s.face_seed, l.in_mean.len() - BODY_INPUTS);
    let shape_for = |scale: f64| {
        let face: Vec<f64> = individual.iter().map(|x| x * scale).collect();
        move |aim: &[f64; 9]| l.shape(&inputs(gender, muscle, weight, aim, &heads, s.ideal, &face))
    };
    // aim: the network's measures input moved by what the made body missed, until it measures as asked
    let make_at = shape_for(s.face_scale);
    let mut aim = asked;
    let mut best: Option<(f64, Vec<f32>, crate::makehuman::Base, Measures)> = None;
    for _ in 0..AIMS {
        let shape = make_at(&aim);
        let body = l.body(&shape, s.h);
        let got = measure(&body, &body.verts);
        let miss = (0..9).map(|i| (asked[i] - got.ratios[i]).abs()).fold(0.0, f64::max);
        for i in 0..9 {
            aim[i] += asked[i] - got.ratios[i];
        }
        if best.as_ref().is_none_or(|b| miss < b.0) {
            best = Some((miss, shape, body, got));
        }
    }
    let (_, shape, body, got) = best.expect("at least one aim");
    let average = shape_for(0.0)(&aim);
    let face_mm = face_distance(&shape, &average, l.body_n, s.h);
    let plain_face_mm = (s.face_scale != 1.0).then(|| face_distance(&shape_for(1.0)(&aim), &average, l.body_n, s.h));

    // the skin, smoothed once
    let (verts, quads) = subdivide(&body.verts[..body.body_n], &body.quads);
    let mesh = triangles(&verts, &quads);

    // the eyes, in the sockets
    let joint = |name: &str| {
        let ids = &body.joints.iter().find(|j| j.0 == name).expect("a joint the model keeps").1;
        ids.iter().fold(v(0.0, 0.0, 0.0), |acc, &i| acc + body.verts[i as usize]) * (1.0 / ids.len() as f64)
    };
    let (eye, target) = (joint("l-eye"), joint("l-eye-target"));
    let gaze = target - eye;
    let mut eyes = Mesh { pos: Vec::new(), nrm: Vec::new(), idx: Vec::new() };
    let mut eye_colors = Vec::new();
    for side in [1.0, -1.0] {
        eyeball(v(eye.x * side, eye.y, eye.z), v(gaze.x * side, gaze.y, gaze.z), EYE_RADIUS * s.h, c.learned.iris, &mut eyes, &mut eye_colors);
    }

    let mut lines = vec![format!(
        "Learned body: MakeHuman muscle {muscle:.2}, weight {weight:.2} (\"{build}\"{}), {} head; asked {}.",
        if s.defined { ", defined" } else { "" },
        head,
        NAMES.iter().zip(&asked).map(|(n, a)| format!("{n} {a:.3}")).collect::<Vec<_>>().join(", ")
    )];
    if !held.is_empty() {
        lines.push(format!("Beyond what MakeHuman's bodies of this sex, muscle and weight measure (5th to 95th percentile): {}.", held.join(", ")));
    }
    Made { mesh, eyes, eye_colors, asked, got, face_mm, plain_face_mm, lines }
}

/// how many times the network's input is moved by what the made body missed
const AIMS: usize = 6;

/// MakeHuman's eye helper (a sphere of 0.15 dm on a 1.666 m body), 6% smaller so the iris stays under the lids when
/// seen from the side
const EYE_RADIUS: f64 = 0.94 * 0.015 / 1.666;
