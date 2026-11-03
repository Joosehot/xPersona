//! The skin as triangles: surface nets over a grid symmetric about x = 0.
//!
//! Each grid cell the skin passes through gets one vertex (where its edges cross the skin, then moved onto it); each
//! grid edge the skin crosses gets the quad of the four cells around it. The quads close up by construction.
use crate::sdf::Field;
use crate::v3::v;

pub struct Mesh {
    pub pos: Vec<[f32; 3]>,
    pub nrm: Vec<[f32; 3]>,
    pub idx: Vec<u32>,
}

pub fn build(field: &Field, cell: f64) -> Mesh {
    let (lo, hi) = field.extent();
    let half = (hi.x / cell).ceil() as i64 + 2;
    let y0 = (lo.y / cell).floor() as i64 - 2;
    let z0 = (lo.z / cell).floor() as i64 - 2;
    let (nx, ny, nz) = ((2 * half + 1) as usize, ((hi.y / cell).ceil() as i64 + 2 - y0 + 1) as usize, ((hi.z / cell).ceil() as i64 + 2 - z0 + 1) as usize);
    let at = |i: usize, j: usize, k: usize| v((i as i64 - half) as f64 * cell, (j as i64 + y0) as f64 * cell, (k as i64 + z0) as f64 * cell);

    // the field at every grid point, one z-slice per task
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut vals = vec![0f64; nx * ny * nz];
    std::thread::scope(|sc| {
        let per = nz.div_ceil(threads);
        for (t, chunk) in vals.chunks_mut(per * nx * ny).enumerate() {
            let at = &at;
            sc.spawn(move || {
                for (n, out) in chunk.iter_mut().enumerate() {
                    let k = t * per + n / (nx * ny);
                    let r = n % (nx * ny);
                    *out = field.eval(at(r % nx, r / nx, k));
                }
            });
        }
    });
    let val = |i: usize, j: usize, k: usize| vals[i + nx * (j + ny * k)];

    // one vertex per cell the skin passes through
    let (cx, cy) = (nx - 1, ny - 1);
    let cell_id = |i: usize, j: usize, k: usize| i + cx * (j + cy * k);
    let mut vert = vec![u32::MAX; cx * cy * (nz - 1)];
    let mut pos = Vec::new();
    let mut nrm = Vec::new();
    const CORNERS: [(usize, usize, usize); 8] = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)];
    const EDGES: [(usize, usize); 12] = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)];
    for k in 0..nz - 1 {
        for j in 0..ny - 1 {
            for i in 0..nx - 1 {
                let f: Vec<f64> = CORNERS.iter().map(|c| val(i + c.0, j + c.1, k + c.2)).collect();
                let inside = f.iter().filter(|x| **x < 0.0).count();
                if inside == 0 || inside == 8 {
                    continue;
                }
                let mut sum = v(0.0, 0.0, 0.0);
                let mut n = 0.0;
                for (a, b) in EDGES {
                    if (f[a] < 0.0) != (f[b] < 0.0) {
                        let t = f[a] / (f[a] - f[b]);
                        let (pa, pb) = (at(i + CORNERS[a].0, j + CORNERS[a].1, k + CORNERS[a].2), at(i + CORNERS[b].0, j + CORNERS[b].1, k + CORNERS[b].2));
                        sum = sum + pa + (pb - pa) * t;
                        n += 1.0;
                    }
                }
                let mut p = sum * (1.0 / n);
                // onto the skin, staying in the cell
                let g = field.grad(p, cell * 0.25);
                let gg = g.dot(g);
                if gg > 0.0 {
                    p = p - g * (field.eval(p) / gg);
                }
                let base = at(i, j, k);
                p = v(p.x.clamp(base.x, base.x + cell), p.y.clamp(base.y, base.y + cell), p.z.clamp(base.z, base.z + cell));
                let g = field.grad(p, cell * 0.25).unit();
                vert[cell_id(i, j, k)] = pos.len() as u32;
                pos.push([p.x as f32, p.y as f32, p.z as f32]);
                nrm.push([g.x as f32, g.y as f32, g.z as f32]);
            }
        }
    }

    // one quad per grid edge the skin crosses, wound so it faces out
    let mut idx = Vec::new();
    let mut quad = |q: [u32; 4], flip: bool| {
        let q = if flip { [q[0], q[3], q[2], q[1]] } else { q };
        let p = |n: u32| {
            let a = pos[n as usize];
            v(a[0] as f64, a[1] as f64, a[2] as f64)
        };
        if (p(q[0]) - p(q[2])).len() <= (p(q[1]) - p(q[3])).len() {
            idx.extend_from_slice(&[q[0], q[1], q[2], q[0], q[2], q[3]]);
        } else {
            idx.extend_from_slice(&[q[0], q[1], q[3], q[1], q[2], q[3]]);
        }
    };
    for k in 1..nz - 1 {
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let inside = val(i, j, k) < 0.0;
                if i + 1 < nx && inside != (val(i + 1, j, k) < 0.0) {
                    let c = |a: usize, b: usize| vert[cell_id(i, j - 1 + a, k - 1 + b)];
                    quad([c(0, 0), c(1, 0), c(1, 1), c(0, 1)], !inside);
                }
                if j + 1 < ny && inside != (val(i, j + 1, k) < 0.0) {
                    let c = |a: usize, b: usize| vert[cell_id(i - 1 + b, j, k - 1 + a)];
                    quad([c(0, 0), c(1, 0), c(1, 1), c(0, 1)], !inside);
                }
                if k + 1 < nz && inside != (val(i, j, k + 1) < 0.0) {
                    let c = |a: usize, b: usize| vert[cell_id(i - 1 + a, j - 1 + b, k)];
                    quad([c(0, 0), c(1, 0), c(1, 1), c(0, 1)], !inside);
                }
            }
        }
    }
    Mesh { pos, nrm, idx }
}
