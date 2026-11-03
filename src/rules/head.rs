//! The head's shape: oval, round, long or square. A shape asked is the only variant.
use super::{Rule, Spec};
use crate::config::Config;
use crate::lexicon::{Adj, Part};
use crate::parser::Request;

pub struct Head;

impl Rule for Head {
    fn name(&self) -> &'static str {
        "head"
    }
    fn variants(&self, r: &Request, _: &Config) -> Vec<&'static str> {
        match r.part(Part::Head).map(|m| m.adj) {
            Some(Adj::Oval) => vec!["oval"],
            Some(Adj::Round) => vec!["round"],
            Some(Adj::Long) => vec!["long"],
            Some(Adj::Square) => vec!["square"],
            _ => vec!["oval", "round", "long", "square"],
        }
    }
    fn apply(&self, v: &str, r: &Request, c: &Config, s: &mut Spec) {
        let [dw, dd, jaw] = c.head[v];
        let size = r.part(Part::Head).filter(|m| matches!(m.adj, Adj::Big | Adj::Small)).map_or(1.0, |m| super::factor(m, c));
        s.w.head = (s.w.head + dw) * size;
        s.w.head_depth = (s.w.head_depth + dd) * size;
        s.jaw = jaw * size;
        let why = r.part(Part::Head).map_or(String::new(), |m| format!(" (\"{}\")", m.words));
        s.say(format!("Head {v}: width {:.3}, depth {:.3}, jaw {:.3} of the height{why}.", s.w.head, s.w.head_depth, s.jaw * 2.0), "head");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    #[test]
    fn an_asked_shape_is_the_only_one() {
        assert_eq!(Head.variants(&parse("a man with a round head").unwrap(), &crate::config::Config::load(crate::config::DEFAULT).unwrap()), vec!["round"]);
    }

    #[test]
    fn unasked_all_shapes_compete() {
        assert_eq!(Head.variants(&parse("a man").unwrap(), &crate::config::Config::load(crate::config::DEFAULT).unwrap()).len(), 4);
    }
}
