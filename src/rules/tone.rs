//! Muscle tone: smooth (muscle under a layer, shapes soft) or defined (the major muscles drawn as their own forms).
use super::{Rule, Spec};
use crate::config::Config;
use crate::parser::Request;

pub struct Tone;

impl Rule for Tone {
    fn name(&self) -> &'static str {
        "tone"
    }
    fn variants(&self, _: &Request, _: &Config) -> Vec<&'static str> {
        vec!["smooth", "defined"]
    }
    fn apply(&self, v: &str, _: &Request, _: &Config, s: &mut Spec) {
        s.defined = v == "defined";
        s.say(format!("Tone {v}: {}.", if s.defined { "deltoids, chest, biceps, quadriceps and calves shaped as their own forms" } else { "the muscles under a smooth layer" }), "tone");
    }
}

#[cfg(test)]
mod tests {
    use crate::rules::spec_of;

    #[test]
    fn defined_draws_the_muscles() {
        assert!(spec_of("a man", &[("tone", "defined")]).defined);
    }

    #[test]
    fn smooth_does_not() {
        assert!(!spec_of("a man", &[("tone", "smooth")]).defined);
    }
}
