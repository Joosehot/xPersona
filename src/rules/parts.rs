//! The widths and thicknesses asked of single parts: broad shoulders, narrow hips, thick arms, big hands, ...
use super::{factor, Rule, Spec};
use crate::config::Config;
use crate::lexicon::{Adj, Part};
use crate::parser::Request;

pub struct Parts;

impl Rule for Parts {
    fn name(&self) -> &'static str {
        "parts"
    }
    fn variants(&self, _: &Request, _: &Config) -> Vec<&'static str> {
        vec!["as_asked"]
    }
    fn apply(&self, _: &str, r: &Request, c: &Config, s: &mut Spec) {
        for m in &r.parts {
            let k = factor(m, c);
            let done = match (m.part, m.adj) {
                (Part::Shoulders, _) => {
                    s.w.shoulders *= k;
                    true
                }
                (Part::Hips, _) => {
                    s.w.hips *= k;
                    s.w.hips_depth *= k.sqrt();
                    true
                }
                (Part::Waist, _) => {
                    s.w.waist *= k;
                    s.w.waist_depth *= k.sqrt();
                    true
                }
                (Part::Chest, _) => {
                    s.w.chest *= k;
                    true
                }
                (Part::Arms, Adj::Thick | Adj::Thin) => {
                    s.arm_girth *= k;
                    true
                }
                (Part::Legs, Adj::Thick | Adj::Thin) => {
                    s.leg_girth *= k;
                    true
                }
                (Part::Neck, Adj::Thick | Adj::Thin) => {
                    s.w.neck *= k;
                    true
                }
                (Part::Hands, _) => {
                    s.hand *= k;
                    true
                }
                (Part::Feet, _) => {
                    s.foot *= k;
                    true
                }
                _ => false,
            };
            if done {
                s.say(format!("\"{}\": x{k:.2}.", m.words), "parts");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::rules::spec_of;

    #[test]
    fn broad_shoulders_widen_them() {
        assert!(spec_of("a man with broad shoulders", &[]).w.shoulders > spec_of("a man", &[]).w.shoulders);
    }

    #[test]
    fn a_length_is_not_a_width() {
        assert_eq!(spec_of("a man with long arms", &[]).arm_girth, spec_of("a man", &[]).arm_girth);
    }
}
