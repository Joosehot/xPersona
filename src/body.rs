//! The decided proportions as shapes: torso slices, limbs as round cones, joints, muscles, fat and the head's forms.
//! Shapes are placed on the body's left side and in the middle; the field mirrors them.
use crate::config::Config;
use crate::rules::Spec;
use crate::sdf::{Prim, Shape, AXES};
use crate::v3::{v, V3};

fn cone(name: &'static str, a: V3, b: V3, ra: f64, rb: f64, blend: f64) -> Prim {
    Prim { name, shape: Shape::RoundCone { a, b, ra, rb }, blend }
}

fn ball(name: &'static str, c: V3, r: V3, blend: f64) -> Prim {
    Prim { name, shape: Shape::Ellipsoid { c, r, axes: AXES }, blend }
}

fn ellipsoid(name: &'static str, c: V3, r: V3, axes: [V3; 3], blend: f64) -> Prim {
    Prim { name, shape: Shape::Ellipsoid { c, r, axes }, blend }
}

/// the heights of the landmarks, metres from the floor
pub struct Marks {
    pub ankle: f64,
    pub knee: f64,
    pub hip: f64,
    pub crotch: f64,
    pub waist: f64,
    pub chest: f64,
    pub shoulder: f64,
    pub chin: f64,
    pub top: f64,
}

pub fn marks(s: &Spec) -> Marks {
    let (c, h) = (&s.chain, s.h);
    let ankle = c.ankle * h;
    let knee = ankle + c.shank * h;
    let hip = knee + c.thigh * h;
    let shoulder = hip + c.torso * h;
    let chin = shoulder + c.neck * h;
    Marks { ankle, knee, hip, crotch: hip - 0.06 * h, waist: hip + 0.32 * c.torso * h, chest: hip + 0.66 * c.torso * h, shoulder, chin, top: h }
}

pub fn prims(s: &Spec, c: &Config) -> Vec<Prim> {
    let h = s.h;
    let f = |x: f64| x * h;
    let m = marks(s);
    let (bt, bl, bd) = (f(c.blend.torso), f(c.blend.limb), f(c.blend.detail));
    let w = &s.w;
    let (mu, fat) = (s.muscle, s.fat);
    let shown = if s.defined { mu } else { mu * 0.4 };
    let mut p = Vec::new();

    // the torso as stacked elliptic slices: (height, half width, half depth, centre forward)
    let levels = [
        (m.crotch + f(0.01), f(w.hips) * 0.36, f(w.hips_depth) * 0.40, f(-0.008)),
        (m.hip - f(0.02), f(w.hips) * 0.5, f(w.hips_depth) * 0.5, f(-0.01)),
        (m.waist, f(w.waist) * 0.5, f(w.waist_depth) * 0.5, 0.0),
        (m.chest, f(w.chest) * 0.5, f(w.chest_depth) * 0.5, f(0.004)),
        (m.shoulder - f(0.045), f(w.chest) * 0.5, f(w.chest_depth) * 0.47, 0.0),
        (m.shoulder - f(0.004), f(w.shoulders) * 0.30, f(w.chest_depth) * 0.30, f(-0.012)),
    ];
    for pair in levels.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        p.push(Prim { name: "torso", shape: Shape::Frustum { y0: a.0, y1: b.0, w0: a.1, w1: b.1, d0: a.2, d1: b.2, z0: a.3, z1: b.3 }, blend: bt });
    }
    p.push(ball("pelvis", v(0.0, m.hip - f(0.025), f(-0.01)), v(f(w.hips) * 0.47, f(0.06), f(w.hips_depth) * 0.5), bt));
    p.push(ball("glute", v(f(0.042), m.hip - f(0.045), f(-w.hips_depth * 0.5 + 0.032)), v(f(0.05), f(0.056), f(0.04)) * (1.0 + 0.15 * fat * s.fat_hips), bl));
    if fat > 0.3 {
        let k = (fat - 0.3) * s.fat_waist;
        p.push(ball("belly", v(0.0, m.waist - f(0.015), f(w.waist_depth * 0.5 - 0.03)), v(f(0.075), f(0.07), f(0.045)) * (1.0 + 0.4 * k), bt));
    }
    // chest: breasts on a woman's proportions, pectorals when muscle shows
    if s.female > 0.5 {
        let r = f(0.038) * (0.9 + 0.3 * fat);
        p.push(ball("breast", v(f(0.045), m.chest + f(0.005), f(w.chest_depth * 0.5 - 0.012)), v(r, r * 0.92, r * 0.9), bl));
    }
    if shown > 0.2 {
        p.push(ball("pectoral", v(f(0.045), m.chest + f(0.02), f(w.chest_depth * 0.5 - 0.022)), v(f(0.05), f(0.03), f(0.022)) * (0.8 + 0.3 * shown), bl));
    }

    // neck and shoulders
    let neck_r = f(w.neck);
    p.push(cone("neck", v(0.0, m.shoulder - f(0.03), f(-0.012)), v(0.0, m.chin + f(0.01), f(-0.004)), neck_r * 1.15, neck_r, bl));
    p.push(ball("trapezius", v(f(0.035), m.shoulder, f(-0.016)), v(f(0.055), f(0.02) * (1.0 + 0.4 * shown), f(0.032)), bl));
    let delt = f(0.03) * s.arm_girth * (1.0 + 0.15 * shown);
    let sh = v(f(w.shoulders) * 0.5 - delt, m.shoulder - f(0.022), f(-0.006));
    p.push(cone("collarbone", v(f(0.02), m.shoulder - f(0.012), f(0.018)), sh, f(0.012), f(0.02), bl));
    p.push(ball("deltoid", sh + v(f(0.006), 0.0, 0.0), v(delt * 1.08, delt * 1.05, delt * 1.12), bl));

    // arms: down from the horizontal by the pose's angle
    let a = s.pose_deg.to_radians();
    let dir = v(a.cos(), -a.sin(), 0.0);
    let side = v(a.sin(), a.cos(), 0.0);
    let fwd = v(0.0, 0.0, 1.0);
    let (ag, ch) = (s.arm_girth, &s.chain);
    let elbow = sh + dir * f(ch.upper_arm);
    let wrist = elbow + dir * f(ch.forearm);
    p.push(cone("upper arm", sh, elbow, f(0.029) * ag, f(0.021) * ag, bl));
    if shown > 0.2 {
        p.push(ellipsoid("biceps", sh + dir * f(ch.upper_arm * 0.55) + fwd * f(0.008), v(f(0.06), f(0.02) * ag, f(0.022) * ag) * (0.85 + 0.25 * shown), [dir, side, fwd], bd * 2.0));
    }
    p.push(ball("elbow", elbow, v(f(0.021), f(0.021), f(0.021)) * ag, bd));
    p.push(cone("forearm", elbow + dir * f(0.012), wrist, f(0.024) * ag, f(0.0155) * (0.7 + 0.3 * ag), bl));
    let hs = s.hand * ch.hand / 0.108;
    p.push(ellipsoid("palm", wrist + dir * f(0.032 * hs), v(f(0.036), f(0.012), f(0.026)) * hs, [dir, side, fwd], bd));
    p.push(ellipsoid("fingers", wrist + dir * f(0.082 * hs), v(f(0.028), f(0.009), f(0.022)) * hs, [dir, side, fwd], bd));
    p.push(cone("thumb", wrist + dir * f(0.022 * hs) + fwd * f(0.016 * hs), wrist + dir * f(0.06 * hs) + fwd * f(0.03 * hs), f(0.008) * hs, f(0.0065) * hs, bd));

    // legs
    let lg = s.leg_girth;
    let hip_joint = v(f(0.052) * (w.hips / 0.191).sqrt(), m.hip, f(-0.006));
    let knee = v(f(0.047), m.knee, 0.0);
    let ankle = v(f(0.045), m.ankle, f(-0.012));
    p.push(cone("thigh", hip_joint, knee, f(0.05) * lg, f(0.031) * lg, bt));
    if shown > 0.2 {
        p.push(ellipsoid("quadriceps", hip_joint + (knee - hip_joint) * 0.45 + fwd * f(0.016), v(f(0.034) * lg, f(0.08), f(0.03) * lg) * (0.85 + 0.25 * shown), AXES, bl));
    }
    p.push(ball("knee", knee + fwd * f(0.004), v(f(0.03), f(0.03), f(0.03)) * lg, bd));
    p.push(cone("shank", knee, ankle, f(0.03) * lg, f(0.014) * (0.7 + 0.3 * lg), bl));
    p.push(ball("calf", v(f(0.047), m.knee - f(0.075), f(-0.014)), v(f(0.031), f(0.065), f(0.032)) * lg * (1.0 + 0.15 * shown), bl));
    let fs = s.foot * ch.foot / 0.152;
    let fr = v(f(0.027), f(0.021), f(0.078)) * fs;
    p.push(ball("foot", v(f(0.047), fr.y, f(0.05) * fs), fr, bd * 2.0));
    let hr = v(f(0.022), f(0.021), f(0.03)) * fs;
    p.push(ball("heel", v(f(0.045), hr.y, f(-0.012)), hr, bd * 2.0));

    // the head: skull, jaw, nose, brow, lips, ears
    let skull = v(f(w.head) * 0.5, f(0.068), f(w.head_depth) * 0.5);
    let sc = v(0.0, m.top - skull.y, f(-0.006));
    p.push(ball("skull", sc, skull, bd * 2.0));
    let face_z = sc.z + skull.z;
    let jaw_r = v(f(s.jaw), (sc.y - m.chin) * 0.45, f(0.042));
    p.push(ball("jaw", v(0.0, m.chin + jaw_r.y, face_z - jaw_r.z * 0.85), jaw_r, bd * 3.0));
    p.push(ball("cheek", v(f(0.028), sc.y - f(0.035), face_z - f(0.03)), v(f(0.022), f(0.018), f(0.025)), bd * 2.0));
    p.push(cone("nose", v(0.0, sc.y - f(0.01), face_z - f(0.004)), v(0.0, sc.y - f(0.036), face_z + f(0.014)), f(0.005), f(0.008), bd));
    p.push(ball("brow", v(0.0, sc.y - f(0.002), face_z - f(0.008)), v(f(0.034), f(0.008), f(0.01)), bd));
    p.push(ball("lips", v(0.0, m.chin + jaw_r.y * 1.25, face_z - f(0.004)), v(f(0.013), f(0.006), f(0.007)), bd));
    p.push(ball("ear", v(skull.x - f(0.002), sc.y - f(0.018), sc.z - f(0.008)), v(f(0.006), f(0.017), f(0.011)), bd));
    p
}
