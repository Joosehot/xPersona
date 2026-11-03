//! The pose a model is delivered in: arms down at an angle (A-pose) or straight out (T-pose).
use super::{Rule, Spec};
use crate::config::Config;
use crate::lexicon::Pose;
use crate::parser::Request;

pub struct PoseRule;

impl Rule for PoseRule {
    fn name(&self) -> &'static str {
        "pose"
    }
    fn variants(&self, r: &Request, _: &Config) -> Vec<&'static str> {
        match r.pose.as_ref().map(|p| p.value) {
            Some(Pose::A) => vec!["a_pose"],
            Some(Pose::T) => vec!["t_pose"],
            None => vec!["a_pose", "t_pose"],
        }
    }
    fn apply(&self, v: &str, r: &Request, c: &Config, s: &mut Spec) {
        s.pose_deg = c.pose[v];
        let why = r.pose.as_ref().map_or(String::new(), |p| format!(" (\"{}\")", p.words));
        s.say(format!("Pose {}: arms {:.0} degrees below the horizontal{why}.", v.replace('_', "-").to_uppercase().replace("-POSE", "-pose"), s.pose_deg), "pose");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    #[test]
    fn an_asked_pose_is_the_only_one() {
        assert_eq!(PoseRule.variants(&parse("a man in a t-pose").unwrap(), &crate::config::Config::load(crate::config::DEFAULT).unwrap()), vec!["t_pose"]);
    }

    #[test]
    fn unasked_both_compete() {
        assert_eq!(PoseRule.variants(&parse("a man").unwrap(), &crate::config::Config::load(crate::config::DEFAULT).unwrap()).len(), 2);
    }
}
