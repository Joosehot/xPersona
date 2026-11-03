//! The body as a distance field: negative inside, positive outside, 0 on the skin.
//!
//! Every shape is defined on the body's left side (x >= 0) or on the middle, and the field is read at |x|, so the body
//! is exactly symmetric. Shapes are joined one after another with a smooth minimum, each with its own blend radius.
use crate::v3::{lerp, v, V3};

#[derive(Clone, Debug)]
pub enum Shape {
    /// a cone with rounded ends: radius ra at a, rb at b
    RoundCone { a: V3, b: V3, ra: f64, rb: f64 },
    /// an ellipsoid with radii r along the three axes
    Ellipsoid { c: V3, r: V3, axes: [V3; 3] },
    /// an upright cone with elliptic cross-sections and flat ends, centred on x = 0
    Frustum { y0: f64, y1: f64, z0: f64, z1: f64, w0: f64, w1: f64, d0: f64, d1: f64 },
}

#[derive(Clone, Debug)]
pub struct Prim {
    pub name: &'static str,
    pub shape: Shape,
    /// how far the shape melts into what is already there
    pub blend: f64,
}

pub const AXES: [V3; 3] = [v(1.0, 0.0, 0.0), v(0.0, 1.0, 0.0), v(0.0, 0.0, 1.0)];

impl Shape {
    pub fn dist(&self, p: V3) -> f64 {
        match self {
            Shape::RoundCone { a, b, ra, rb } => round_cone(p, *a, *b, *ra, *rb),
            Shape::Ellipsoid { c, r, axes } => {
                let q = p - *c;
                let l = v(q.dot(axes[0]) / r.x, q.dot(axes[1]) / r.y, q.dot(axes[2]) / r.z);
                let k0 = l.len();
                let l2 = v(l.x / r.x, l.y / r.y, l.z / r.z);
                let k1 = l2.len();
                if k1 == 0.0 {
                    return -r.x.min(r.y).min(r.z);
                }
                k0 * (k0 - 1.0) / k1
            }
            Shape::Frustum { y0, y1, z0, z1, w0, w1, d0, d1 } => {
                let t = ((p.y - y0) / (y1 - y0)).clamp(0.0, 1.0);
                let (w, d, cz) = (lerp(*w0, *w1, t), lerp(*d0, *d1, t), lerp(*z0, *z1, t));
                let e = ((p.x / w).powi(2) + ((p.z - cz) / d).powi(2)).sqrt();
                let radial = (e - 1.0) * w.min(d);
                let axial = (y0 - p.y).max(p.y - y1);
                if radial > 0.0 && axial > 0.0 {
                    (radial * radial + axial * axial).sqrt()
                } else {
                    radial.max(axial)
                }
            }
        }
    }

    /// a sphere the shape lies in
    pub fn bound(&self) -> (V3, f64) {
        match self {
            Shape::RoundCone { a, b, ra, rb } => ((*a + *b) * 0.5, (*b - *a).len() * 0.5 + ra.max(*rb)),
            Shape::Ellipsoid { c, r, .. } => (*c, r.max_part()),
            Shape::Frustum { y0, y1, z0, z1, w0, w1, d0, d1 } => {
                let c = v(0.0, (y0 + y1) * 0.5, (z0 + z1) * 0.5);
                let half = v(w0.max(*w1), (y1 - y0) * 0.5, d0.max(*d1) + (z1 - z0).abs() * 0.5);
                (c, half.len())
            }
        }
    }
}

/// Inigo Quilez's exact distance to a round cone
fn round_cone(p: V3, a: V3, b: V3, r1: f64, r2: f64) -> f64 {
    let ba = b - a;
    let l2 = ba.dot(ba);
    let rr = r1 - r2;
    let a2 = l2 - rr * rr;
    let il2 = 1.0 / l2;
    let pa = p - a;
    let y = pa.dot(ba);
    let z = y - l2;
    let xv = pa * l2 - ba * y;
    let x2 = xv.dot(xv);
    let y2 = y * y * l2;
    let z2 = z * z * l2;
    let k = rr.signum() * rr * rr * x2;
    if z.signum() * a2 * z2 > k {
        return (x2 + z2).sqrt() * il2 - r2;
    }
    if y.signum() * a2 * y2 < k {
        return (x2 + y2).sqrt() * il2 - r1;
    }
    ((x2 * a2 * il2).sqrt() + y * rr) * il2 - r1
}

/// the polynomial smooth minimum: a and b melt together within k of each other
fn smin(a: f64, b: f64, k: f64) -> f64 {
    if !a.is_finite() {
        return b;
    }
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.min(b) - h * h * h * k / 6.0
}

pub struct Field {
    prims: Vec<Prim>,
    bounds: Vec<(V3, f64)>,
}

impl Field {
    pub fn new(prims: Vec<Prim>) -> Field {
        let bounds = prims.iter().map(|p| p.shape.bound()).collect();
        Field { prims, bounds }
    }

    pub fn eval(&self, p: V3) -> f64 {
        let q = v(p.x.abs(), p.y, p.z);
        let mut f = f64::INFINITY;
        for (pr, (c, r)) in self.prims.iter().zip(&self.bounds) {
            // a shape farther than its blend can change nothing
            if (q - *c).len() - r > f + pr.blend {
                continue;
            }
            f = smin(f, pr.shape.dist(q), pr.blend);
        }
        f
    }

    pub fn grad(&self, p: V3, h: f64) -> V3 {
        let d = |o: V3| self.eval(p + o) - self.eval(p - o);
        v(d(v(h, 0.0, 0.0)), d(v(0.0, h, 0.0)), d(v(0.0, 0.0, h))) * (0.5 / h)
    }

    /// the box the body lies in, for x >= 0 (it is mirrored)
    pub fn extent(&self) -> (V3, V3) {
        let mut lo = v(0.0, f64::MAX, f64::MAX);
        let mut hi = v(f64::MIN, f64::MIN, f64::MIN);
        for (c, r) in &self.bounds {
            lo = v(0.0, lo.y.min(c.y - r), lo.z.min(c.z - r));
            hi = v(hi.x.max(c.x + r), hi.y.max(c.y + r), hi.z.max(c.z + r));
        }
        (lo, hi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_measure_from_their_surface() {
        let s = Shape::Ellipsoid { c: v(0.0, 0.0, 0.0), r: v(1.0, 2.0, 3.0), axes: AXES };
        assert!(s.dist(v(0.0, 2.0, 0.0)).abs() < 1e-9);
        assert!(s.dist(v(0.0, 0.0, 0.0)) < 0.0);
        let c = Shape::RoundCone { a: v(0.0, 0.0, 0.0), b: v(0.0, 1.0, 0.0), ra: 0.2, rb: 0.1 };
        assert!((c.dist(v(0.0, 1.5, 0.0)) - 0.4).abs() < 1e-9);
        let f = Shape::Frustum { y0: 0.0, y1: 1.0, z0: 0.0, z1: 0.0, w0: 0.5, w1: 0.5, d0: 0.25, d1: 0.25 };
        assert!(f.dist(v(0.5, 0.5, 0.0)).abs() < 1e-9);
    }

    #[test]
    fn the_field_is_symmetric() {
        let f = Field::new(vec![Prim { name: "ball", shape: Shape::Ellipsoid { c: v(0.3, 0.0, 0.0), r: v(0.1, 0.1, 0.1), axes: AXES }, blend: 0.01 }]);
        assert_eq!(f.eval(v(0.3, 0.05, 0.0)), f.eval(v(-0.3, 0.05, 0.0)));
        assert!(f.eval(v(-0.3, 0.0, 0.0)) < 0.0);
    }
}
