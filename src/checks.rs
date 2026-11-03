//! The finished mesh, checked with no trust in the rules that made it: its shape as a solid, then the body measured
//! from the triangles themselves.
use crate::config::Config;
use crate::mesh::Mesh;
use crate::rules::Spec;
use std::collections::{HashMap, HashSet};

pub struct Check {
    pub ok: bool,
    pub name: &'static str,
    pub text: String,
}

fn check(ok: bool, name: &'static str, text: String) -> Check {
    Check { ok, name, text }
}

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

pub fn run(m: &Mesh, s: &Spec, c: &Config, cell: f64) -> Vec<Check> {
    let mut out = Vec::new();
    let tris: Vec<[u32; 3]> = m.idx.chunks(3).map(|t| [t[0], t[1], t[2]]).collect();

    // a closed surface facing one way: each directed edge once, and its reverse once
    let mut directed: HashMap<(u32, u32), u32> = HashMap::new();
    let mut degenerate = 0;
    for t in &tris {
        if t[0] == t[1] || t[1] == t[2] || t[0] == t[2] {
            degenerate += 1;
        }
        for k in 0..3 {
            *directed.entry((t[k], t[(k + 1) % 3])).or_insert(0) += 1;
        }
    }
    let bad = directed.iter().filter(|(e, n)| **n != 1 || directed.get(&(e.1, e.0)) != Some(&1)).count();
    out.push(check(bad == 0 && degenerate == 0, "closed", if bad == 0 && degenerate == 0 { "every edge is shared by exactly two triangles, facing the same way".into() } else { format!("{bad} edges open or doubled, {degenerate} degenerate triangles") }));

    // one piece
    let mut parent: Vec<u32> = (0..m.pos.len() as u32).collect();
    for t in &tris {
        let (a, b, cc) = (find(&mut parent, t[0]), find(&mut parent, t[1]), find(&mut parent, t[2]));
        parent[b as usize] = a;
        parent[cc as usize] = a;
    }
    let used: HashSet<u32> = m.idx.iter().copied().collect();
    let roots: HashSet<u32> = used.iter().map(|&x| find(&mut parent, x)).collect();
    out.push(check(roots.len() == 1, "one piece", format!("{} connected piece{}", roots.len(), if roots.len() == 1 { "" } else { "s" })));

    // no handles: a limb that touched the body would make a loop
    let edges = directed.len() / 2;
    let euler = used.len() as i64 - edges as i64 + tris.len() as i64;
    out.push(check(euler == 2, "no loops", format!("Euler characteristic {euler} (2: no limb touches the body, no hole)")));

    // facing out: the enclosed volume is positive
    let p = |i: u32| m.pos[i as usize].map(|x| x as f64);
    let mut vol = 0.0;
    for t in &tris {
        let (a, b, d) = (p(t[0]), p(t[1]), p(t[2]));
        vol += a[0] * (b[1] * d[2] - b[2] * d[1]) - a[1] * (b[0] * d[2] - b[2] * d[0]) + a[2] * (b[0] * d[1] - b[1] * d[0]);
    }
    vol /= 6.0;
    out.push(check(vol > 0.0, "facing out", format!("encloses {:.1} litres", vol * 1000.0)));

    // measured from the mesh
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for q in &m.pos {
        for a in 0..3 {
            lo[a] = lo[a].min(q[a] as f64);
            hi[a] = hi[a].max(q[a] as f64);
        }
    }
    let height = hi[1] - lo[1];
    let tol = cell.max(0.01 * s.h);
    out.push(check((height - s.h).abs() <= tol, "height", format!("measured {height:.3} m, asked {:.3} m (within {:.0} mm)", s.h, tol * 1000.0)));
    out.push(check(lo[1].abs() <= cell, "on the ground", format!("lowest point {:.1} mm from the floor", lo[1] * 1000.0)));

    // left and right the same: every vertex has its mirror image
    let key = |x: f64, y: f64, z: f64| ((x * 2000.0).round() as i64, (y * 2000.0).round() as i64, (z * 2000.0).round() as i64);
    let set: HashSet<(i64, i64, i64)> = m.pos.iter().map(|q| key(q[0] as f64, q[1] as f64, q[2] as f64)).collect();
    let mirrored = m.pos.iter().filter(|q| set.contains(&key(-(q[0] as f64), q[1] as f64, q[2] as f64))).count();
    let share = mirrored as f64 / m.pos.len() as f64;
    out.push(check(share > 0.99, "symmetric", format!("{:.2}% of the vertices have their mirror image", share * 100.0)));

    // the legs: the crotch is the lowest point of the body's middle above the knees
    let crotch = m.pos.iter().filter(|q| (q[0] as f64).abs() < cell * 0.75 && (q[1] as f64) > 0.3 * height).map(|q| q[1] as f64).fold(f64::MAX, f64::min);
    let legs = crotch / height;
    let [a, b] = c.ranges.legs;
    out.push(check((a..=b).contains(&legs), "legs", format!("crotch at {legs:.3} of the height (a body: {a} to {b})")));

    // the decided proportions against what a human body can be
    let span = (s.w.shoulders - 0.06 * s.arm_girth) + 2.0 * (s.chain.upper_arm + s.chain.forearm + s.chain.hand * s.hand);
    let [a, b] = c.ranges.arm_span;
    out.push(check((a..=b).contains(&span), "arm span", format!("{span:.3} of the height (a body: {a} to {b})")));
    let heads = 1.0 / s.chain.head;
    let [a, b] = c.ranges.heads;
    out.push(check((a..=b).contains(&heads), "heads", format!("{heads:.2} heads tall (a body: {a} to {b})")));
    out
}
