//! rules.toml, read and checked at load: a misspelt key fails at once.
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub search: Search,
    pub profile: Profile,
    pub stature: Stature,
    pub chain: Chain,
    pub width: Widths,
    pub modifiers: Modifiers,
    pub build: BTreeMap<String, BuildRow>,
    pub blend: Blend,
    pub mesh: MeshCfg,
    pub variants: BTreeMap<String, BTreeMap<String, [f64; 3]>>,
    pub torso: BTreeMap<String, [f64; 3]>,
    pub head: BTreeMap<String, [f64; 3]>,
    pub pose: BTreeMap<String, f64>,
    pub proportions: BTreeMap<String, f64>,
    pub judges: BTreeMap<String, f64>,
    pub ranges: Ranges,
    pub learned: LearnedCfg,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Search {
    pub beam: usize,
    pub finalists: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub default: [f64; 3],
    pub word: f64,
    pub build: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stature {
    pub male: f64,
    pub female: f64,
    pub neutral: f64,
    pub step: f64,
    pub least: f64,
    pub most: f64,
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct Chain {
    pub ankle: f64,
    pub shank: f64,
    pub thigh: f64,
    pub torso: f64,
    pub neck: f64,
    pub head: f64,
    pub upper_arm: f64,
    pub forearm: f64,
    pub hand: f64,
    pub foot: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Widths {
    pub male: Width,
    pub female: Width,
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct Width {
    pub shoulders: f64,
    pub chest: f64,
    pub chest_depth: f64,
    pub waist: f64,
    pub waist_depth: f64,
    pub hips: f64,
    pub hips_depth: f64,
    pub neck: f64,
    pub head: f64,
    pub head_depth: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Modifiers {
    pub long: f64,
    pub short: f64,
    pub broad: f64,
    pub narrow: f64,
    pub thick: f64,
    pub thin: f64,
    pub big: f64,
    pub small: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildRow {
    pub girth: f64,
    pub muscle: f64,
    pub fat: f64,
    /// MakeHuman's muscle and weight values (0..1, 0.5 average) for the learned body
    pub makehuman: [f64; 2],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearnedCfg {
    pub head_weight: f64,
    pub defined_muscle: f64,
    pub tolerance_mm: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Blend {
    pub torso: f64,
    pub limb: f64,
    pub detail: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshCfg {
    pub cells: Vec<f64>,
    pub color: [f64; 3],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ranges {
    pub legs: [f64; 2],
    pub arm_span: [f64; 2],
    pub heads: [f64; 2],
}

pub const DEFAULT: &str = include_str!("../rules.toml");

/// the names the code reads: missing any of them is an error at load
const NEEDED: &[(&str, &[&str])] = &[
    ("variants.proportions", &["even", "long_legs", "long_torso"]),
    ("variants.fat", &["even", "central", "lower"]),
    ("variants.torso", &["straight", "v_taper", "pear", "hourglass"]),
    ("variants.tone", &["smooth", "defined"]),
    ("variants.head", &["oval", "round", "long", "square"]),
    ("variants.pose", &["a_pose", "t_pose"]),
    ("torso", &["straight", "v_taper", "pear", "hourglass"]),
    ("head", &["oval", "round", "long", "square"]),
    ("pose", &["a_pose", "t_pose"]),
    ("proportions", &["long_legs", "long_torso"]),
    ("build", &["thin", "slim", "lean", "average", "athletic", "muscular", "stocky", "curvy", "heavy"]),
    ("judges", &["hourglass_on_a_man", "pear_on_a_man", "v_taper_on_a_woman", "asked_broad_shoulders", "asked_broad_hips", "asked_narrow_waist", "defined_and_heavy", "smooth_and_muscular", "against_the_asked_legs"]),
];

impl Config {
    pub fn load(text: &str) -> Result<Config, String> {
        let c: Config = toml::from_str(text).map_err(|e| format!("rules.toml: {e}"))?;
        for (table, names) in NEEDED {
            let have: Vec<&String> = match *table {
                "torso" => c.torso.keys().collect(),
                "head" => c.head.keys().collect(),
                "pose" => c.pose.keys().collect(),
                "proportions" => c.proportions.keys().collect(),
                "build" => c.build.keys().collect(),
                "judges" => c.judges.keys().collect(),
                t => c.variants.get(&t["variants.".len()..]).map(|m| m.keys().collect()).unwrap_or_default(),
            };
            for n in *names {
                if !have.iter().any(|h| h.as_str() == *n) {
                    return Err(format!("rules.toml: [{table}] has no \"{n}\""));
                }
            }
            for h in &have {
                if !names.contains(&h.as_str()) {
                    return Err(format!("rules.toml: [{table}] \"{h}\" is read by no rule (misspelt?)"));
                }
            }
        }
        if c.mesh.cells.is_empty() || c.mesh.cells.iter().any(|x| *x <= 0.0) {
            return Err("rules.toml: [mesh] cells must be positive".into());
        }
        Ok(c)
    }

    pub fn variant(&self, rule: &str, name: &str) -> [f64; 3] {
        self.variants[rule][name]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shipped_rules_load() {
        Config::load(DEFAULT).unwrap();
    }

    #[test]
    fn a_misspelt_name_fails_at_load() {
        let bad = DEFAULT.replace("hourglass = [1.03", "hourglas = [1.03");
        assert!(Config::load(&bad).err().unwrap().contains("hourglas"));
    }

    #[test]
    fn the_chain_sums_to_the_height() {
        let c = Config::load(DEFAULT).unwrap().chain;
        assert!((c.ankle + c.shank + c.thigh + c.torso + c.neck + c.head - 1.0).abs() < 1e-9);
    }
}
