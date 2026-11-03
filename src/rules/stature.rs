//! How tall: the number asked, or the adult mean for the sex moved by "tall" or "short".
use super::{female, Rule, Spec};
use crate::config::Config;
use crate::lexicon::Sex;
use crate::parser::Request;

pub struct Stature;

impl Rule for Stature {
    fn name(&self) -> &'static str {
        "stature"
    }
    fn variants(&self, _: &Request, _: &Config) -> Vec<&'static str> {
        vec!["as_asked"]
    }
    fn apply(&self, _: &str, r: &Request, c: &Config, s: &mut Spec) {
        let st = &c.stature;
        let (mean, who) = match r.sex() {
            Sex::Male => (st.male, "a man"),
            Sex::Female => (st.female, "a woman"),
            Sex::Neutral => (st.neutral, "a person"),
        };
        s.female = female(r.sex());
        s.h = match (&r.height, &r.stature) {
            (Some(h), _) => {
                s.say(format!("Height {:.2} m, as asked (\"{}\").", h.value, h.words), "stature");
                h.value
            }
            (None, Some(k)) => {
                let h = mean + k.value * st.step;
                s.say(format!("Height {h:.2} m: {who} averages {mean:.2} m, \"{}\" moves it {:+.2} m.", k.words, k.value * st.step), "stature");
                h
            }
            (None, None) => {
                s.say(format!("Height {mean:.2} m, the adult mean for {who}."), "stature");
                mean
            }
        };
    }
}

#[cfg(test)]
mod tests {
    use crate::rules::spec_of;

    #[test]
    fn tall_moves_the_mean() {
        assert!((spec_of("a very tall woman", &[]).h - (1.63 + 2.0 * 0.09)).abs() < 1e-9);
    }

    #[test]
    fn without_a_word_the_mean_stays() {
        assert!((spec_of("a man", &[]).h - 1.76).abs() < 1e-9);
    }
}
