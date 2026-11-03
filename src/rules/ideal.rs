//! "Beautiful": proportions moved toward MakeHuman's ideal ones, a canon its artists drew (longer legs, a smaller head
//! against the height, a balanced torso). One reading of beauty, said as such; it is not a claim about what beauty is.
use super::{Rule, Spec};
use crate::config::Config;
use crate::parser::Request;

pub struct Ideal;

impl Rule for Ideal {
    fn name(&self) -> &'static str {
        "ideal"
    }
    fn variants(&self, _: &Request, _: &Config) -> Vec<&'static str> {
        vec!["as_asked"]
    }
    fn apply(&self, _: &str, r: &Request, c: &Config, s: &mut Spec) {
        let [plain, beautiful, very] = c.learned.proportions;
        s.ideal = match &r.ideal {
            None => plain,
            Some(k) if k.value >= 2.0 => very,
            Some(_) => beautiful,
        };
        if let Some(k) = &r.ideal {
            s.say(format!("Proportions toward MakeHuman's ideal canon: {:.2} of the way from uncommon (0.5) to ideal (1), unasked {plain:.2} (\"{}\"). One reading of beauty: the canon its artists drew.", s.ideal, k.words), "ideal");
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::rules::spec_of;

    #[test]
    fn beautiful_moves_toward_the_ideal() {
        assert!(spec_of("a beautiful woman", &[]).ideal > spec_of("a woman", &[]).ideal);
    }

    #[test]
    fn unasked_it_stays_at_the_mean() {
        assert_eq!(spec_of("a woman", &[]).ideal, 0.75);
    }
}
