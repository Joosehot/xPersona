//! Where fat sits: evenly, around the middle, or on the hips and thighs. Only a body with fat to place has the choice.
use super::{Rule, Spec};
use crate::config::Config;
use crate::parser::Request;

pub struct Fat;

/// below this there is too little fat to place
const LITTLE: f64 = 0.3;

impl Rule for Fat {
    fn name(&self) -> &'static str {
        "fat"
    }
    fn variants(&self, r: &Request, c: &Config) -> Vec<&'static str> {
        let fat = c.build[&format!("{:?}", r.build()).to_lowercase()].fat;
        if fat < LITTLE {
            vec!["even"]
        } else {
            vec!["even", "central", "lower"]
        }
    }
    fn apply(&self, v: &str, _: &Request, _: &Config, s: &mut Spec) {
        let (waist, hips) = match v {
            "central" => (1.5, 0.7),
            "lower" => (0.7, 1.5),
            _ => (1.0, 1.0),
        };
        s.fat_waist = waist;
        s.fat_hips = hips;
        s.w.waist *= 1.0 + 0.22 * s.fat * waist;
        s.w.waist_depth *= 1.0 + 0.30 * s.fat * waist;
        s.w.hips *= 1.0 + 0.08 * s.fat * hips;
        s.w.hips_depth *= 1.0 + 0.10 * s.fat * hips;
        s.leg_girth *= 1.0 + 0.10 * s.fat * hips;
        if s.fat >= LITTLE {
            s.say(format!("Fat {}: waist x{:.2}, hips x{:.2}.", if v == "even" { "spread evenly" } else if v == "central" { "around the middle" } else { "on the hips and thighs" }, 1.0 + 0.22 * s.fat * waist, 1.0 + 0.08 * s.fat * hips), "fat");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    #[test]
    fn a_heavy_body_has_the_choice() {
        assert_eq!(Fat.variants(&parse("a heavy man").unwrap(), &crate::config::Config::load(crate::config::DEFAULT).unwrap()).len(), 3);
    }

    #[test]
    fn a_slim_body_does_not() {
        assert_eq!(Fat.variants(&parse("a slim man").unwrap(), &crate::config::Config::load(crate::config::DEFAULT).unwrap()), vec!["even"]);
    }
}
