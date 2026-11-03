//! What a body with these measures looks like, learned once from MakeHuman's bodies and kept frozen.
//!
//! Training (`xpersona --train N`): N bodies are made from MakeHuman's targets with random values and measured; their
//! shapes, scaled to unit height, are reduced to their K main components; a Neuras network learns the components from
//! sex, muscle, weight, the nine measures and the head shape. One body in ten is held out and judged unseen.
//! Making: the rules give those inputs, the network gives the components, the components give the vertices.
use crate::makehuman::{self, Base, Params, Targets, HEADS, MEASURES};
use crate::measure::{measure, NAMES};
use crate::v3::V3;
use neuras::model::{Layer, Net, Shape};
use neuras::train::net::Model;
use std::path::Path;

pub const MODEL: &str = "model/persona.bin";
/// sex, muscle, weight, the measures, the head shapes
pub const INPUTS: usize = 3 + NAMES.len() + HEADS.len();
const K: usize = 48;
const OVERSAMPLE: usize = 16;
const HIDDEN: usize = 256;
const EPOCHS: usize = 120;
const BATCH: usize = 64;
const JOINTS: [&str; 6] = ["l-shoulder", "l-elbow", "l-hand", "l-upper-leg", "neck", "jaw"];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (a, b) = (self.unit().max(1e-12), self.unit());
        (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
    }
}

/// the vertices the network gives: the body's, then the joints the measures read
pub fn shape_ids(b: &Base) -> Vec<u32> {
    let mut ids: Vec<u32> = (0..b.body_n as u32).collect();
    for j in JOINTS {
        ids.extend(&b.joints[j]);
    }
    ids
}

fn random_params(seed: u64) -> Params {
    let mut r = Rng(seed.wrapping_mul(0x2545F4914F6CDD1D) ^ 0xA5A5);
    let e = [r.unit() + 0.05, r.unit() + 0.05, r.unit() + 0.05];
    let s: f64 = e.iter().sum();
    let mut heads = [0.0; 4];
    heads[(r.next() % 4) as usize] = r.unit();
    Params {
        gender: r.unit(),
        muscle: r.unit(),
        weight: r.unit(),
        height: r.unit(),
        proportions: 0.5 + 0.5 * r.unit(),
        ethnic: e.map(|x| x / s),
        measures: MEASURES.iter().map(|_| (r.unit() * 2.0 - 1.0) * 0.6).collect(),
        heads,
    }
}

/// a body standing on the floor, centred, at unit height
fn normalised(verts: &[V3], ids: &[u32], body_n: usize) -> (Vec<f32>, f64) {
    let floor = verts[..body_n].iter().map(|p| p.y).fold(f64::MAX, f64::min);
    let top = verts[..body_n].iter().map(|p| p.y).fold(f64::MIN, f64::max);
    let h = top - floor;
    let mut out = Vec::with_capacity(ids.len() * 3);
    for &i in ids {
        let p = verts[i as usize];
        out.extend_from_slice(&[(p.x / h) as f32, ((p.y - floor) / h) as f32, (p.z / h) as f32]);
    }
    (out, h)
}

pub fn inputs(gender: f64, muscle: f64, weight: f64, ratios: &[f64; 9], heads: &[f64; 4]) -> Vec<f64> {
    let mut x = vec![gender, muscle, weight];
    x.extend_from_slice(ratios);
    x.extend_from_slice(heads);
    x
}

fn par<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let threads = std::thread::available_parallelism().map_or(4, |t| t.get());
    let per = n.div_ceil(threads).max(1);
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..n).step_by(per).map(|s| {
            let f = &f;
            sc.spawn(move || (s..(s + per).min(n)).map(f).collect::<Vec<T>>())
        }).collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    })
}

/// a symmetric matrix's eigenvalues and vectors by Jacobi rotations, largest first
fn eigen(mut a: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut vecs: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect();
    for _ in 0..100 {
        let off: f64 = (0..n).flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j))).map(|(i, j)| a[i][j] * a[i][j]).sum();
        if off < 1e-22 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for row in vecs.iter_mut() {
                    let (vp, vq) = (row[p], row[q]);
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[j][j].total_cmp(&a[i][i]));
    (order.iter().map(|&i| a[i][i]).collect(), order.iter().map(|&i| (0..n).map(|k| vecs[k][i]).collect()).collect())
}

/// the network's design: inputs, two hidden layers, the components
fn net() -> Net {
    Net::build(Shape::Vector(INPUTS), vec![Layer::Dense { out: HIDDEN }, Layer::Relu, Layer::Dense { out: HIDDEN }, Layer::Relu, Layer::Dense { out: K }]).expect("an mlp")
}

/// the frozen model
pub struct Learned {
    pub mean: Vec<f32>,
    /// K rows of the shape's length
    pub basis: Vec<Vec<f32>>,
    /// each component's spread, so the network learns numbers near 1
    pub scale: Vec<f32>,
    pub in_mean: Vec<f64>,
    pub in_std: Vec<f64>,
    /// the measures expected for sex, muscle and weight: per measure, coefficients on [1, g, m, w, gm, gw, mw]
    pub baseline: Vec<[f64; 7]>,
    pub model: Model,
    pub report: String,
    /// the body's quads, and the joints the measures read, as positions in the shape
    pub quads: Vec<[u32; 4]>,
    pub joints: Vec<(String, Vec<u32>)>,
    pub body_n: usize,
}

fn terms(g: f64, m: f64, w: f64) -> [f64; 7] {
    [1.0, g, m, w, g * m, g * w, m * w]
}

impl Learned {
    pub fn expected(&self, g: f64, m: f64, w: f64) -> [f64; 9] {
        let t = terms(g, m, w);
        let mut out = [0.0; 9];
        for (o, c) in out.iter_mut().zip(&self.baseline) {
            *o = c.iter().zip(t).map(|(a, b)| a * b).sum();
        }
        out
    }

    /// the shape (unit height) for these inputs
    pub fn shape(&self, x: &[f64]) -> Vec<f32> {
        let z: Vec<f32> = x.iter().zip(&self.in_mean).zip(&self.in_std).map(|((x, m), s)| ((x - m) / s) as f32).collect();
        let c = self.model.outputs(&z);
        let mut out = self.mean.clone();
        for ((ck, row), s) in c.iter().zip(&self.basis).zip(&self.scale) {
            let k = ck * s;
            for (o, b) in out.iter_mut().zip(row) {
                *o += k * b;
            }
        }
        out
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let mut b: Vec<u8> = Vec::new();
        let mut put = |xs: &[f32]| {
            b.extend_from_slice(&(xs.len() as u32).to_le_bytes());
            for x in xs {
                b.extend_from_slice(&x.to_le_bytes());
            }
        };
        put(&self.mean);
        for r in &self.basis {
            put(r);
        }
        put(&self.scale);
        put(&self.in_mean.iter().map(|x| *x as f32).collect::<Vec<_>>());
        put(&self.in_std.iter().map(|x| *x as f32).collect::<Vec<_>>());
        put(&self.baseline.iter().flatten().map(|x| *x as f32).collect::<Vec<_>>());
        for p in &self.model.params {
            put(p);
        }
        // whole numbers below 2^24 are exact as f32
        put(&self.quads.iter().flatten().map(|&i| i as f32).collect::<Vec<_>>());
        put(&[self.body_n as f32]);
        for (_, ids) in &self.joints {
            put(&ids.iter().map(|&i| i as f32).collect::<Vec<_>>());
        }
        let report = self.report.as_bytes();
        b.extend_from_slice(&(report.len() as u32).to_le_bytes());
        b.extend_from_slice(report);
        if let Some(d) = path.parent() {
            std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
        }
        std::fs::write(path, b).map_err(|e| e.to_string())
    }

    pub fn load(path: &Path) -> Result<Learned, String> {
        let b = std::fs::read(path).map_err(|e| format!("{}: {e} (train it first: xpersona --train 3000)", path.display()))?;
        let mut at = 0;
        let mut get = || -> Result<Vec<f32>, String> {
            let n = u32::from_le_bytes(b.get(at..at + 4).ok_or("the model file is cut short")?.try_into().unwrap()) as usize;
            at += 4;
            let xs = b.get(at..at + 4 * n).ok_or("the model file is cut short")?.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            at += 4 * n;
            Ok(xs)
        };
        let mean = get()?;
        let basis = (0..K).map(|_| get()).collect::<Result<Vec<_>, _>>()?;
        let scale = get()?;
        let in_mean = get()?.into_iter().map(|x| x as f64).collect();
        let in_std = get()?.into_iter().map(|x| x as f64).collect();
        let flat = get()?;
        let baseline = flat.chunks(7).map(|c| std::array::from_fn(|i| c[i] as f64)).collect();
        let mut model = Model::new(&net(), 0);
        for p in model.params.iter_mut() {
            let x = get()?;
            if x.len() != p.len() {
                return Err("the model file does not fit the network's design".into());
            }
            *p = x;
        }
        let quads = get()?.chunks(4).map(|c| [c[0] as u32, c[1] as u32, c[2] as u32, c[3] as u32]).collect();
        let body_n = get()?[0] as usize;
        let joints = JOINTS.iter().map(|j| Ok((j.to_string(), get()?.into_iter().map(|x| x as u32).collect()))).collect::<Result<Vec<_>, String>>()?;
        let n = u32::from_le_bytes(b.get(at..at + 4).ok_or("the model file is cut short")?.try_into().unwrap()) as usize;
        let report = String::from_utf8_lossy(&b[at + 4..at + 4 + n]).into_owned();
        Ok(Learned { mean, basis, scale, in_mean, in_std, baseline, model, report, quads, joints, body_n })
    }

    /// the vertices at this height (metres), as a Base the measures can read
    pub fn body(&self, shape: &[f32], h: f64) -> Base {
        let verts = shape.chunks(3).map(|c| crate::v3::v(c[0] as f64 * h, c[1] as f64 * h, c[2] as f64 * h)).collect();
        Base { verts, quads: self.quads.clone(), body_n: self.body_n, joints: self.joints.iter().cloned().collect() }
    }
}

/// make N bodies, learn, write the model; the report says how well it does on bodies it never saw
pub fn train(n: usize, log: &mut dyn FnMut(String)) -> Result<Learned, String> {
    let t0 = std::time::Instant::now();
    let root = Path::new(makehuman::DATA);
    let base = makehuman::load_base(root)?;
    let targets = Targets::load(root)?;
    let ids = shape_ids(&base);
    let d = ids.len() * 3;

    // the bodies, measured
    let made: Vec<(Vec<f32>, Vec<f64>, f64)> = par(n, |i| {
        let p = random_params(i as u64);
        let verts = targets.compose(&base, &p);
        let m = measure(&base, &verts);
        let (shape, h) = normalised(&verts, &ids, base.body_n);
        (shape, inputs(p.gender, p.muscle, p.weight, &m.ratios, &p.heads), h)
    });
    log(format!("{n} bodies made and measured from MakeHuman's targets ({:.0} s)", t0.elapsed().as_secs_f64()));
    let held = n / 10;
    let learn = n - held;

    // the main components of the learning bodies' shapes (randomised: project, sharpen twice, solve the small problem)
    let mut mean = vec![0f32; d];
    for (s, _, _) in &made[..learn] {
        for (m, x) in mean.iter_mut().zip(s) {
            *m += x / learn as f32;
        }
    }
    let x: Vec<Vec<f32>> = made[..learn].iter().map(|(s, _, _)| s.iter().zip(&mean).map(|(a, b)| a - b).collect()).collect();
    let l = K + OVERSAMPLE;
    let mut r = Rng(42);
    let omega: Vec<f32> = (0..d * l).map(|_| r.normal() as f32).collect();
    let times = |m: &[f32]| -> Vec<Vec<f64>> {
        // X (learn x d) times m (d x l)
        par(learn, |i| {
            let mut y = vec![0f64; l];
            for (k, &xv) in x[i].iter().enumerate() {
                let row = &m[k * l..(k + 1) * l];
                for (yy, w) in y.iter_mut().zip(row) {
                    *yy += (xv * w) as f64;
                }
            }
            y
        })
    };
    let back = |y: &[Vec<f64>]| -> Vec<f32> {
        // X^T (d x learn) times y (learn x l), as d x l
        let parts: Vec<Vec<f32>> = par(16, |c| {
            let lo = learn * c / 16;
            let hi = learn * (c + 1) / 16;
            let mut z = vec![0f32; d * l];
            for i in lo..hi {
                let yi: Vec<f32> = y[i].iter().map(|v| *v as f32).collect();
                for (k, &xv) in x[i].iter().enumerate() {
                    for (zz, w) in z[k * l..(k + 1) * l].iter_mut().zip(&yi) {
                        *zz += xv * w;
                    }
                }
            }
            z
        });
        let mut z = vec![0f32; d * l];
        for p in parts {
            for (a, b) in z.iter_mut().zip(p) {
                *a += b;
            }
        }
        z
    };
    let orth = |mut y: Vec<Vec<f64>>| -> Vec<Vec<f64>> {
        // Gram-Schmidt over the columns
        for j in 0..l {
            for k in 0..j {
                let dot: f64 = y.iter().map(|r| r[j] * r[k]).sum();
                for r in y.iter_mut() {
                    r[j] -= dot * r[k];
                }
            }
            let norm: f64 = y.iter().map(|r| r[j] * r[j]).sum::<f64>().sqrt().max(1e-30);
            for r in y.iter_mut() {
                r[j] /= norm;
            }
        }
        y
    };
    let mut q = orth(times(&omega));
    for _ in 0..2 {
        let z = back(&q);
        q = orth(times(&z));
    }
    // B = Q^T X, as l rows of d
    let bt = back(&q);
    let brow: Vec<Vec<f64>> = (0..l).map(|j| (0..d).map(|k| bt[k * l + j] as f64).collect()).collect();
    let c: Vec<Vec<f64>> = (0..l).map(|i| (0..l).map(|j| brow[i].iter().zip(&brow[j]).map(|(a, b)| a * b).sum()).collect()).collect();
    let (vals, vecs) = eigen(c);
    let basis: Vec<Vec<f32>> = (0..K)
        .map(|k| {
            let s = vals[k].max(1e-30).sqrt();
            (0..d).map(|i| (brow.iter().zip(&vecs[k]).map(|(row, u)| row[i] * u).sum::<f64>() / s) as f32).collect()
        })
        .collect();
    let total: f64 = x.iter().map(|r| r.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>()).sum();
    let kept: f64 = vals[..K].iter().sum();
    let scale: Vec<f32> = vals[..K].iter().map(|v| (v / learn as f64).sqrt() as f32).collect();
    log(format!("{K} main components of {d} numbers keep {:.3}% of the shapes' variation ({:.0} s)", kept / total * 100.0, t0.elapsed().as_secs_f64()));

    let coeffs = |s: &[f32]| -> Vec<f32> { basis.iter().zip(&scale).map(|(b, sc)| b.iter().zip(s.iter().zip(&mean)).map(|(bv, (x, m))| bv * (x - m)).sum::<f32>() / sc).collect() };
    let targets_k: Vec<Vec<f32>> = par(n, |i| coeffs(&made[i].0));

    // inputs, standardised on the learning bodies
    let in_mean: Vec<f64> = (0..INPUTS).map(|j| made[..learn].iter().map(|m| m.1[j]).sum::<f64>() / learn as f64).collect();
    let in_std: Vec<f64> = (0..INPUTS).map(|j| (made[..learn].iter().map(|m| (m.1[j] - in_mean[j]).powi(2)).sum::<f64>() / learn as f64).sqrt().max(1e-9)).collect();
    let z: Vec<Vec<f32>> = made.iter().map(|m| m.1.iter().zip(&in_mean).zip(&in_std).map(|((x, mu), s)| ((x - mu) / s) as f32).collect()).collect();

    // the measures expected for sex, muscle and weight (least squares)
    let mut baseline = Vec::new();
    for j in 0..NAMES.len() {
        let mut a = [[0f64; 8]; 7];
        for m in &made[..learn] {
            let t = terms(m.1[0], m.1[1], m.1[2]);
            for p in 0..7 {
                for q in 0..7 {
                    a[p][q] += t[p] * t[q];
                }
                a[p][7] += t[p] * m.1[3 + j];
            }
        }
        for p in 0..7 {
            let piv = (p..7).max_by(|&i, &k| a[i][p].abs().total_cmp(&a[k][p].abs())).unwrap();
            a.swap(p, piv);
            for i in 0..7 {
                if i != p {
                    let f = a[i][p] / a[p][p];
                    for q in p..8 {
                        a[i][q] -= f * a[p][q];
                    }
                }
            }
        }
        baseline.push(std::array::from_fn(|p| a[p][7] / a[p][p]));
    }

    // the network, designed and trained with Neuras
    let design = net();
    let mut model = Model::new(&design, 7);
    let mut order: Vec<usize> = (0..learn).collect();
    let mut rr = Rng(99);
    let held_loss = |model: &Model| -> f64 {
        (learn..n).map(|i| model.outputs(&z[i]).iter().zip(&targets_k[i]).map(|(o, t)| ((o - t) as f64).powi(2)).sum::<f64>()).sum::<f64>() / held as f64
    };
    for e in 0..EPOCHS {
        for i in (1..order.len()).rev() {
            order.swap(i, (rr.next() % (i as u64 + 1)) as usize);
        }
        let lr = 0.001 * 0.97f64.powi(e as i32);
        for b in order.chunks(BATCH) {
            let parts: Vec<Vec<Vec<f32>>> = par(b.len().min(8), |c| {
                let mut g = model.zero_grads();
                for &i in b.iter().skip(c).step_by(b.len().min(8)) {
                    let t = model.forward(&z[i]);
                    model.backward_mse(&t, &targets_k[i], &mut g);
                }
                g
            });
            let mut g = model.zero_grads();
            for p in parts {
                for (a, b2) in g.iter_mut().zip(p) {
                    for (x, y) in a.iter_mut().zip(b2) {
                        *x += y;
                    }
                }
            }
            let k = 1.0 / b.len() as f32;
            for a in g.iter_mut() {
                for v in a.iter_mut() {
                    *v *= k;
                }
            }
            model.adam(&g, lr);
        }
        if e % 20 == 19 || e == 0 {
            log(format!("epoch {}: unseen loss {:.4} ({:.0} s)", e + 1, held_loss(&model), t0.elapsed().as_secs_f64()));
        }
    }

    // the joints' vertices as positions in the shape (the body's come first, in order)
    let mut at = base.body_n as u32;
    let joints = JOINTS
        .iter()
        .map(|j| {
            let k = base.joints[*j].len() as u32;
            let ids = (at..at + k).collect();
            at += k;
            (j.to_string(), ids)
        })
        .collect();
    let mut learned = Learned { mean, basis, scale, in_mean, in_std, baseline, model, report: String::new(), quads: base.quads.clone(), joints, body_n: base.body_n };
    // judged on the held-out bodies, in millimetres at their own height
    let errs: Vec<(f64, f64, f64)> = par(held, |j| {
        let i = learn + j;
        let (s, x, h) = (&made[i].0, &made[i].1, made[i].2);
        let pred = learned.shape(x);
        let floor: Vec<f32> = {
            let c: Vec<f32> = learned.basis.iter().zip(&learned.scale).map(|(b, sc)| b.iter().zip(s.iter().zip(&learned.mean)).map(|(bv, (x, m))| bv * (x - m)).sum::<f32>() / sc).collect();
            let mut o = learned.mean.clone();
            for ((ck, row), sc) in c.iter().zip(&learned.basis).zip(&learned.scale) {
                for (oo, bb) in o.iter_mut().zip(row) {
                    *oo += ck * sc * bb;
                }
            }
            o
        };
        let err = |a: &[f32]| a.chunks(3).zip(s.chunks(3)).map(|(p, q)| ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt() as f64).sum::<f64>() / (a.len() / 3) as f64 * h * 1000.0;
        (err(&pred), err(&floor), err(&learned.mean))
    });
    let avg = |f: fn(&(f64, f64, f64)) -> f64| errs.iter().map(f).sum::<f64>() / held as f64;
    learned.report = format!(
        "trained on {learn} MakeHuman bodies, judged on {held} it never saw: mean vertex error {:.1} mm (the {K} components alone could reach {:.1} mm; the average body is {:.1} mm off)",
        avg(|e| e.0),
        avg(|e| e.1),
        avg(|e| e.2)
    );
    log(learned.report.clone());
    Ok(learned)
}
