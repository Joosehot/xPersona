//! xPersona: a sentence in, a realistic 3D body out. A HotMotor engine: closed vocabulary, parser, one rule per
//! decision, beam search with judges, checks before output. No neural network runs while it works.
pub mod body;
pub mod checks;
pub mod config;
pub mod glb;
pub mod lexicon;
pub mod mesh;
pub mod parser;
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

pub fn make(sentence: &str, c: &Config, explain: bool) -> Result<Made, Failure> {
    let r = parser::parse(sentence).map_err(|e| Failure { what: e.what, hint: e.hint })?;
    let rules = rules::all();
    let paths = search::search(&r, c, &rules);
    let mut why_not = Vec::new();
    for path in paths.iter().take(c.search.finalists) {
        let mut s = rules::Spec::new(c);
        for (rule, v) in rules.iter().zip(&path.choices) {
            rule.apply(v, &r, c, &mut s);
        }
        if !(c.stature.least..=c.stature.most).contains(&s.h) {
            return Err(Failure { what: format!("a height of {:.2} m", s.h), hint: format!("v0 makes adults from {:.2} to {:.2} m", c.stature.least, c.stature.most) });
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
    o += &format!("\nModel: {} triangles, {} vertices, {:.0} mm grid\n", m.idx.len() / 3, m.pos.len(), cell * 1000.0);
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
