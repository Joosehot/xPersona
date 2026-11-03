//! The torso's silhouette: straight, V-tapered, pear or hourglass, each a change to chest, waist and hips.
use super::{Rule, Spec};
use crate::config::Config;
use crate::parser::Request;

pub struct Torso;

impl Rule for Torso {
    fn name(&self) -> &'static str {
        "torso"
    }
    fn variants(&self, _: &Request, _: &Config) -> Vec<&'static str> {
        vec!["straight", "v_taper", "pear", "hourglass"]
    }
    fn apply(&self, v: &str, _: &Request, c: &Config, s: &mut Spec) {
        let [chest, waist, hips] = c.torso[v];
        s.w.chest *= chest;
        s.w.waist *= waist;
        s.w.hips *= hips;
        s.say(format!("Torso {}: chest x{chest:.2}, waist x{waist:.2}, hips x{hips:.2}.", v.replace('_', "-")), "torso");
    }
}

#[cfg(test)]
mod tests {
    use crate::rules::spec_of;

    #[test]
    fn hourglass_narrows_the_waist() {
        assert!(spec_of("a woman", &[("torso", "hourglass")]).w.waist < spec_of("a woman", &[("torso", "straight")]).w.waist);
    }

    #[test]
    fn straight_keeps_the_hips() {
        let a = spec_of("a woman", &[("torso", "straight")]).w.hips;
        let b = spec_of("a woman", &[("torso", "pear")]).w.hips;
        assert!(a < b);
    }
}
