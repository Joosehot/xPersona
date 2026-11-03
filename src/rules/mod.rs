//! One decision per file. A rule says which variants this request allows and carries out the one chosen; every number
//! it uses comes from rules.toml.
use crate::config::{Chain, Config, Width};
use crate::lexicon::{Adj, Sex};
use crate::parser::{PartMod, Request};

mod build;
mod face;
mod fat;
mod head;
mod ideal;
mod parts;
mod pose;
mod proportions;
mod stature;
mod tone;
mod torso;

/// what the rules decide, in fractions of the height h unless said otherwise
#[derive(Clone)]
pub struct Spec {
    /// metres
    pub h: f64,
    /// 0 a man's proportions, 1 a woman's, 0.5 between
    pub female: f64,
    pub chain: Chain,
    pub w: Width,
    pub jaw: f64,
    pub arm_girth: f64,
    pub leg_girth: f64,
    pub hand: f64,
    pub foot: f64,
    pub muscle: f64,
    pub fat: f64,
    /// where fat sits: waist and hips multipliers on `fat`
    pub fat_waist: f64,
    pub fat_hips: f64,
    pub defined: bool,
    /// arms below the horizontal, degrees
    pub pose_deg: f64,
    /// MakeHuman's proportions value, 0.5 uncommon to 1 ideal
    pub ideal: f64,
    /// the face: which individual one, and how far from the average face (1 as drawn)
    pub face_seed: u64,
    pub face_scale: f64,
    /// what each rule did, with the rule's name
    pub steps: Vec<(String, &'static str)>,
}

impl Spec {
    pub fn new(c: &Config) -> Spec {
        Spec {
            h: 0.0,
            female: 0.5,
            chain: c.chain.clone(),
            w: c.width.male.clone(),
            jaw: 0.0,
            arm_girth: 1.0,
            leg_girth: 1.0,
            hand: 1.0,
            foot: 1.0,
            muscle: 0.0,
            fat: 0.0,
            fat_waist: 1.0,
            fat_hips: 1.0,
            defined: false,
            pose_deg: 0.0,
            ideal: 0.75,
            face_seed: 0,
            face_scale: 1.0,
            steps: Vec::new(),
        }
    }
    pub fn say(&mut self, text: String, rule: &'static str) {
        self.steps.push((text, rule));
    }
}

pub fn female(sex: Sex) -> f64 {
    match sex {
        Sex::Male => 0.0,
        Sex::Female => 1.0,
        Sex::Neutral => 0.5,
    }
}

/// a modifier's factor: the step from rules.toml, taken once, or twice with "very"
pub fn factor(m: &PartMod, c: &Config) -> f64 {
    let step = match m.adj {
        Adj::Long => c.modifiers.long,
        Adj::Short => c.modifiers.short,
        Adj::Broad => c.modifiers.broad,
        Adj::Narrow => c.modifiers.narrow,
        Adj::Thick => c.modifiers.thick,
        Adj::Thin => c.modifiers.thin,
        Adj::Big => c.modifiers.big,
        Adj::Small => c.modifiers.small,
        Adj::Round | Adj::Oval | Adj::Square => 1.0,
    };
    step.powf(m.strength)
}

pub trait Rule: Sync {
    /// its name, and its table in [variants] when it has a choice
    fn name(&self) -> &'static str;
    /// the variants this request allows, in a fixed order
    fn variants(&self, r: &Request, _: &Config) -> Vec<&'static str>;
    fn apply(&self, variant: &str, r: &Request, c: &Config, s: &mut Spec);
}

/// in the order they are carried out
pub fn all() -> Vec<Box<dyn Rule>> {
    vec![
        Box::new(stature::Stature),
        Box::new(build::BuildRule),
        Box::new(proportions::Proportions),
        Box::new(ideal::Ideal),
        Box::new(fat::Fat),
        Box::new(torso::Torso),
        Box::new(parts::Parts),
        Box::new(tone::Tone),
        Box::new(head::Head),
        Box::new(face::Face),
        Box::new(pose::PoseRule),
    ]
}

#[cfg(test)]
pub fn spec_of(sentence: &str, choices: &[(&str, &str)]) -> Spec {
    let c = Config::load(crate::config::DEFAULT).unwrap();
    let r = crate::parser::parse(sentence).unwrap();
    let mut s = Spec::new(&c);
    for rule in all() {
        let v = choices.iter().find(|x| x.0 == rule.name()).map_or(rule.variants(&r, &c)[0], |x| x.1);
        rule.apply(v, &r, &c, &mut s);
    }
    s
}
