//! A body measured from its vertices: the joints MakeHuman carries inside it, and widths across bands of the skin.
//! The same measures describe the training bodies and check the bodies made, so asked and made can be compared.
use crate::makehuman::Base;
use crate::v3::{v, V3};

/// the measures, each a fraction of the height
pub const NAMES: [&str; 9] = ["legs", "arm", "shoulders", "chest", "waist", "waist depth", "hips", "neck", "head"];

#[derive(Clone, Debug)]
pub struct Measures {
    pub height: f64,
    pub ratios: [f64; 9],
}

fn joint(b: &Base, verts: &[V3], name: &str) -> V3 {
    let ids = &b.joints[name];
    ids.iter().fold(v(0.0, 0.0, 0.0), |s, &i| s + verts[i as usize]) * (1.0 / ids.len() as f64)
}

/// the widest |x| of the skin between two heights, nearer the middle than `reach`
fn half_width(b: &Base, verts: &[V3], y0: f64, y1: f64, reach: f64) -> f64 {
    verts[..b.body_n].iter().filter(|p| p.y >= y0 && p.y <= y1 && p.x.abs() < reach).map(|p| p.x.abs()).fold(0.0, f64::max)
}

fn depth(b: &Base, verts: &[V3], y0: f64, y1: f64, reach: f64) -> f64 {
    let zs = verts[..b.body_n].iter().filter(|p| p.y >= y0 && p.y <= y1 && p.x.abs() < reach).map(|p| p.z);
    let (lo, hi) = zs.fold((f64::MAX, f64::MIN), |(a, c), z| (a.min(z), c.max(z)));
    hi - lo
}

pub fn measure(b: &Base, verts: &[V3]) -> Measures {
    let body = &verts[..b.body_n];
    let floor = body.iter().map(|p| p.y).fold(f64::MAX, f64::min);
    let top = body.iter().map(|p| p.y).fold(f64::MIN, f64::max);
    let h = top - floor;
    let (sh, el, hand, hip, neck, jaw) = (joint(b, verts, "l-shoulder"), joint(b, verts, "l-elbow"), joint(b, verts, "l-hand"), joint(b, verts, "l-upper-leg"), joint(b, verts, "neck"), joint(b, verts, "jaw"));
    // the fingertip: the skin farthest from the shoulder out to the left, above the hips (the toes are farther still)
    let tip = body.iter().filter(|p| p.x > sh.x && p.y > hip.y).map(|p| (*p - sh).len()).fold(0.0, f64::max);
    let arm = (el - sh).len() + (hand - el).len() + (tip - (hand - sh).len()).max(0.0);
    let band = 0.015 * h;
    let reach = sh.x * 1.05;
    let chest_y = hip.y + (neck.y - hip.y) * 0.72;
    let waist_y = hip.y + (neck.y - hip.y) * 0.38;
    let ratios = [
        (hip.y - floor) / h,
        arm / h,
        2.0 * sh.x / h,
        2.0 * half_width(b, verts, chest_y - band, chest_y + band, reach) / h,
        2.0 * half_width(b, verts, waist_y - band, waist_y + band, reach * 1.4) / h,
        depth(b, verts, waist_y - band, waist_y + band, sh.x * 0.6) / h,
        2.0 * half_width(b, verts, hip.y - band, hip.y + band, reach * 1.4) / h,
        2.0 * half_width(b, verts, neck.y - band, neck.y + band, sh.x * 0.5) / h,
        (top - jaw.y) / h,
    ];
    Measures { height: h, ratios }
}
