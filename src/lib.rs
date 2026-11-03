//! xPersona: a sentence in, a realistic 3D body out. A HotMotor engine: closed vocabulary, parser, one rule per
//! decision, beam search with judges, checks before output. No neural network runs while it works.
pub mod body;
pub mod checks;
pub mod config;
pub mod glb;
pub mod learn;
pub mod lexicon;
pub mod makehuman;
pub mod measure;
pub mod mesh;
pub mod parser;
pub mod persona;
pub mod rules;
pub mod sdf;
pub mod search;
pub mod v3;

use config::Config;

pub struct Made {
    /// the worked steps and the checks, as printed
    pub text: String,
    pub glb: Vec<u8>,
}

pub struct Failure {
    pub what: String,
    pub hint: String,
}

/// `learned`: the frozen network's body; without it, the body sculpted from shapes by the rules alone
pub fn make(sentence: &str, c: &Config, learned: Option<&learn::Learned>, explain: bool) -> Result<Made, Failure> {
    let r = parser::parse(sentence).map_err(|e| Failure { what: e.what, hint: e.hint })?;
    let rules = rules::all();
    let paths = search::search(&r, c, &rules);
    let mut why_not = Vec::new();
    let chose = |path: &search::Path, name: &str| rules.iter().position(|x| x.name() == name).map_or("", |i| path.choices[i]);
    for path in paths.iter().take(c.search.finalists) {
        let mut s = rules::Spec::new(c);
        for (rule, v) in rules.iter().zip(&path.choices) {
            rule.apply(v, &r, c, &mut s);
        }
        if !(c.stature.least..=c.stature.most).contains(&s.h) {
            return Err(Failure { what: format!("a height of {:.2} m", s.h), hint: format!("v0 makes adults from {:.2} to {:.2} m", c.stature.least, c.stature.most) });
        }
        if let Some(l) = learned {
            if chose(path, "pose") == "t_pose" {
                why_not.push("the learned body stands in MakeHuman's A-pose; a T-pose needs a rig, which v0 does not have".to_string());
                continue;
            }
            let made = persona::make(&r, &s, chose(path, "head"), c, l);
            let mut checks = checks::run(&made.mesh, &s, c, 0.01);
            // "beautiful": the same body made without the word, measured against this one
            if let Some(k) = &r.ideal {
                let mut plain = s.clone();
                plain.ideal = c.learned.proportions[0];
                let other = persona::make(&r, &plain, chose(path, "head"), c, l);
                let moved = made.mesh.pos.iter().zip(&other.mesh.pos).map(|(a, b)| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt() as f64).sum::<f64>() / made.mesh.pos.len() as f64 * 1000.0;
                let d = |i: usize| (made.got.ratios[i] - other.got.ratios[i]) * s.h * 1000.0;
                checks.push(checks::Check {
                    ok: moved >= 1.0,
                    name: "toward the ideal",
                    text: format!("\"{}\" moved the skin {moved:.1} mm on average from the same body unasked; legs {:+.0} mm, shoulders {:+.0} mm, chest {:+.0} mm, waist {:+.0} mm, hips {:+.0} mm", k.words, d(0), d(2), d(3), d(4), d(6)),
                });
            }
            // the arm span as the made body has it: shoulder to shoulder, and each arm to the fingertip
            let span = made.got.ratios[2] + 2.0 * made.got.ratios[1];
            let [a, b] = c.ranges.arm_span;
            if let Some(k) = checks.iter_mut().find(|k| k.name == "arm span") {
                *k = checks::Check { ok: (a..=b).contains(&span), name: "arm span", text: format!("measured {span:.3} of the height (a body: {a} to {b})") };
            }
            // the body made, measured, against what was asked of it
            let devs: Vec<(usize, f64)> = (0..measure::NAMES.len()).map(|i| (i, (made.got.ratios[i] - made.asked[i]) * s.h * 1000.0)).collect();
            let worst = devs.iter().cloned().fold((0, 0.0f64), |a, b| if b.1.abs() > a.1.abs() { b } else { a });
            checks.push(checks::Check {
                ok: worst.1.abs() <= c.learned.tolerance_mm,
                name: "as asked",
                text: format!("the made body measured: every measure within {:.0} mm of the asked one, the farthest {} at {:+.1} mm", c.learned.tolerance_mm, measure::NAMES[worst.0], worst.1),
            });
            if let Some(bad) = checks.iter().find(|k| !k.ok) {
                why_not.push(format!("{}: {}: {}", path.choices.join(" "), bad.name, bad.text));
                continue;
            }
            let mut s = s;
            for line in made.lines {
                s.say(line, "learned");
            }
            let mut text = describe(&r, &s, &made.mesh, &checks, path, &paths, 0.0, explain);
            text += &format!("  model: {}\n", l.report);
            return Ok(Made { text, glb: glb::write(&made.mesh, c.mesh.color, "xPersona") });
        }
        if let Some(k) = &r.ideal {
            return Err(Failure { what: format!("\"{}\" without the learned body", k.words), hint: "the ideal proportions are MakeHuman's; drop --sculpt".into() });
        }
        let field = sdf::Field::new(body::prims(&s, c));
        for &cell in &c.mesh.cells {
            let m = mesh::build(&field, cell);
            let checks = checks::run(&m, &s, c, cell);
            if let Some(bad) = checks.iter().find(|k| !k.ok) {
                why_not.push(format!("{} at a {:.0} mm grid: {}: {}", path.choices.join(" "), cell * 1000.0, bad.name, bad.text));
                // a body outside human proportions stays outside at any grid
                if matches!(bad.name, "arm span" | "heads" | "legs" | "height") {
                    break;
                }
                continue;
            }
            return Ok(Made { text: describe(&r, &s, &m, &checks, path, &paths, cell, explain), glb: glb::write(&m, c.mesh.color, "xPersona") });
        }
    }
    Err(Failure { what: format!("no way to build \"{}\" passes the checks", r.sentence), hint: why_not.last().cloned().unwrap_or_default() })
}

#[allow(clippy::too_many_arguments)]
fn describe(r: &parser::Request, s: &rules::Spec, m: &mesh::Mesh, checks: &[checks::Check], path: &search::Path, paths: &[search::Path], cell: f64, explain: bool) -> String {
    let mut o = format!("Model {}\n\n", r.sentence);
    for (i, (text, rule)) in s.steps.iter().enumerate() {
        o += &format!("{:>3}.  {text}   [{rule}]\n", i + 1);
    }
    if !path.notes.is_empty() {
        o += &format!("       judges: {}\n", path.notes.join("; "));
    }
    let grid = if cell > 0.0 { format!(", {:.0} mm grid", cell * 1000.0) } else { ", MakeHuman's topology".to_string() };
    o += &format!("\nModel: {} triangles, {} vertices{grid}\n", m.idx.len() / 3, m.pos.len());
    for k in checks {
        o += &format!("  ok {}: {}\n", k.name, k.text);
    }
    if explain {
        o += "\nFinalists:\n";
        for p in paths {
            o += &format!("  {:.3}  {}{}\n", p.score, p.choices.join(" "), if p.notes.is_empty() { String::new() } else { format!("  ({})", p.notes.join("; ")) });
        }
    }
    o
}
