//! Beam search over every rule's variants: each scored against the request's profile, the judges adding what the
//! choices so far say together. Ties break on the rules' fixed order, so the result is deterministic.
use crate::config::Config;
use crate::lexicon::{Adj, Axis, Build, Part, Sex};
use crate::parser::Request;
use crate::rules::Rule;

#[derive(Clone)]
pub struct Path {
    pub choices: Vec<&'static str>,
    pub score: f64,
    pub notes: Vec<String>,
}

/// definition · softness · balance, moved by the words
pub fn profile(r: &Request, c: &Config) -> [f64; 3] {
    let mut p = c.profile.default;
    for w in &r.profile {
        let i = match w.value {
            Axis::Definition => 0,
            Axis::Softness => 1,
            Axis::Balance => 2,
        };
        p[i] += c.profile.word;
    }
    match r.build() {
        Build::Athletic | Build::Muscular | Build::Lean => p[0] += c.profile.build,
        Build::Heavy | Build::Curvy | Build::Stocky => p[1] += c.profile.build,
        _ => {}
    }
    let sum: f64 = p.iter().sum();
    p.map(|x| x / sum)
}

/// what the choices so far say together
fn judges(r: &Request, c: &Config, rules: &[Box<dyn Rule>], choices: &[&'static str]) -> (f64, Vec<String>) {
    let chose = |name: &str| rules.iter().position(|x| x.name() == name).and_then(|i| choices.get(i).copied());
    let asked = |part: Part, adj: Adj| r.part(part).is_some_and(|m| m.adj == adj);
    let j = |k: &str| c.judges[k];
    let mut score = 0.0;
    let mut notes = Vec::new();
    let mut add = |k: &str, why: &str| {
        score += j(k);
        notes.push(format!("{why} ({:+.1})", j(k)));
    };
    if let Some(t) = chose("torso") {
        match (t, r.sex()) {
            ("hourglass", Sex::Male) => add("hourglass_on_a_man", "an hourglass on a man's frame"),
            ("pear", Sex::Male) => add("pear_on_a_man", "a pear shape on a man's frame"),
            ("v_taper", Sex::Female) if r.build() != Build::Muscular => add("v_taper_on_a_woman", "a V-taper on a woman's frame"),
            _ => {}
        }
        if t == "v_taper" && asked(Part::Shoulders, Adj::Broad) {
            add("asked_broad_shoulders", "a V-taper for the broad shoulders asked");
        }
        if matches!(t, "pear" | "hourglass") && asked(Part::Hips, Adj::Broad) {
            add("asked_broad_hips", "wide hips asked");
        }
        if matches!(t, "hourglass" | "v_taper") && (asked(Part::Waist, Adj::Narrow) || asked(Part::Waist, Adj::Thin)) {
            add("asked_narrow_waist", "a narrow waist asked");
        }
    }
    if let Some(t) = chose("tone") {
        match (t, r.build()) {
            ("defined", Build::Heavy) => add("defined_and_heavy", "defined muscle under heavy fat"),
            ("smooth", Build::Muscular) => add("smooth_and_muscular", "a muscular body drawn smooth"),
            _ => {}
        }
    }
    if let Some(p) = chose("proportions") {
        if (p == "long_torso" && asked(Part::Legs, Adj::Long)) || (p == "long_legs" && asked(Part::Legs, Adj::Short)) {
            add("against_the_asked_legs", "a variation against the legs asked");
        }
    }
    (score, notes)
}

pub fn search(r: &Request, c: &Config, rules: &[Box<dyn Rule>]) -> Vec<Path> {
    let prof = profile(r, c);
    // a variant's own score: its axes against the profile (a rule with no choice scores nothing)
    let local = |rule: &dyn Rule, v: &str| {
        c.variants.get(rule.name()).map_or(0.0, |_| {
            let a = c.variant(rule.name(), v);
            a[0] * prof[0] + a[1] * prof[1] + a[2] * prof[2]
        })
    };
    let mut beam = vec![(0.0, Path { choices: Vec::new(), score: 0.0, notes: Vec::new() })];
    for rule in rules {
        let mut next = Vec::new();
        for (sum, path) in &beam {
            for v in rule.variants(r, c) {
                let mut choices = path.choices.clone();
                choices.push(v);
                let sum = sum + local(rule.as_ref(), v);
                // the judges look at the whole path, so they are counted afresh, never added up
                let (judged, notes) = judges(r, c, rules, &choices);
                next.push((sum, Path { choices, score: sum + judged, notes }));
            }
        }
        next.sort_by(|a, b| b.1.score.total_cmp(&a.1.score));
        next.truncate(c.search.beam);
        beam = next;
    }
    beam.into_iter().map(|x| x.1).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    fn best(sentence: &str) -> Vec<&'static str> {
        let c = Config::load(crate::config::DEFAULT).unwrap();
        let rules = crate::rules::all();
        search(&parse(sentence).unwrap(), &c, &rules).remove(0).choices
    }

    #[test]
    fn a_man_gets_no_hourglass() {
        assert_ne!(best("a man")[4], "hourglass");
    }

    #[test]
    fn a_muscular_body_is_drawn_defined() {
        assert_eq!(best("a muscular man")[6], "defined");
    }

    #[test]
    fn the_same_request_gives_the_same_path() {
        assert_eq!(best("a tall slim woman with long arms"), best("a tall slim woman with long arms"));
    }
}
