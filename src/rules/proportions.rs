//! The lengths of the segments: the measured ratios, a natural variation, the parts asked longer or shorter, and then
//! the floor-to-crown chain brought back to the height.
use super::{factor, Rule, Spec};
use crate::config::Config;
use crate::lexicon::Part;
use crate::parser::Request;

pub struct Proportions;

impl Rule for Proportions {
    fn name(&self) -> &'static str {
        "proportions"
    }
    fn variants(&self, _: &Request, _: &Config) -> Vec<&'static str> {
        vec!["even", "long_legs", "long_torso"]
    }
    fn apply(&self, v: &str, r: &Request, c: &Config, s: &mut Spec) {
        let ch = &mut s.chain;
        let mut said = Vec::new();
        match v {
            "long_legs" => {
                ch.shank *= c.proportions["long_legs"];
                ch.thigh *= c.proportions["long_legs"];
            }
            "long_torso" => ch.torso *= c.proportions["long_torso"],
            _ => {}
        }
        for m in &r.parts {
            let k = factor(m, c);
            match m.part {
                Part::Legs => {
                    if matches!(m.adj, crate::lexicon::Adj::Long | crate::lexicon::Adj::Short) {
                        ch.shank *= k;
                        ch.thigh *= k;
                        said.push(m.words.clone());
                    }
                }
                Part::Torso => {
                    ch.torso *= k;
                    said.push(m.words.clone());
                }
                Part::Neck => {
                    if matches!(m.adj, crate::lexicon::Adj::Long | crate::lexicon::Adj::Short) {
                        ch.neck *= k;
                        said.push(m.words.clone());
                    }
                }
                Part::Arms => {
                    if matches!(m.adj, crate::lexicon::Adj::Long | crate::lexicon::Adj::Short) {
                        ch.upper_arm *= k;
                        ch.forearm *= k;
                        said.push(m.words.clone());
                    }
                }
                Part::Head => {
                    if matches!(m.adj, crate::lexicon::Adj::Big | crate::lexicon::Adj::Small) {
                        ch.head *= k;
                        said.push(m.words.clone());
                    }
                }
                _ => {}
            }
        }
        // the floor-to-crown chain is the height: what grew takes its room from the rest
        let total = ch.ankle + ch.shank + ch.thigh + ch.torso + ch.neck + ch.head;
        for x in [&mut ch.ankle, &mut ch.shank, &mut ch.thigh, &mut ch.torso, &mut ch.neck, &mut ch.head] {
            *x /= total;
        }
        let legs = ch.ankle + ch.shank + ch.thigh;
        let heads = 1.0 / ch.head;
        let arm = ch.upper_arm + ch.forearm + ch.hand;
        let asked = if said.is_empty() { String::new() } else { format!("; asked: \"{}\"", said.join("\", \"")) };
        s.say(format!("Proportions {}: hip joint at {legs:.3} of the height, {heads:.1} heads tall, arm {arm:.3}{asked}.", v.replace('_', " ")), "proportions");
    }
}

#[cfg(test)]
mod tests {
    use crate::rules::spec_of;

    #[test]
    fn long_legs_take_room_from_the_rest() {
        let a = spec_of("a man with long legs", &[("proportions", "even")]).chain;
        let b = spec_of("a man", &[("proportions", "even")]).chain;
        assert!(a.thigh > b.thigh && a.torso < b.torso);
        assert!((a.ankle + a.shank + a.thigh + a.torso + a.neck + a.head - 1.0).abs() < 1e-9);
    }

    #[test]
    fn even_changes_nothing() {
        let a = spec_of("a man", &[("proportions", "even")]).chain;
        assert!((a.thigh - 0.245).abs() < 1e-9);
    }
}
