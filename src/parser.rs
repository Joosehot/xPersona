//! A sentence becomes a request. Every value remembers the words that produced it.
use crate::lexicon::{self, Adj, Axis, Build, Part, Pose, Sex, Tok};

#[derive(Clone, Debug)]
pub struct Said<T> {
    pub value: T,
    pub words: String,
}

#[derive(Clone, Debug)]
pub struct PartMod {
    pub part: Part,
    pub adj: Adj,
    /// 1, or 2 with "very"
    pub strength: f64,
    pub words: String,
}

#[derive(Clone, Debug, Default)]
pub struct Request {
    pub sentence: String,
    pub sex: Option<Said<Sex>>,
    /// metres, as written
    pub height: Option<Said<f64>>,
    /// tall +1, short -1, doubled by "very"
    pub stature: Option<Said<f64>>,
    pub build: Option<Said<(Build, f64)>>,
    pub parts: Vec<PartMod>,
    pub pose: Option<Said<Pose>>,
    pub profile: Vec<Said<Axis>>,
    /// "beautiful": 1, or 2 with "very"
    pub ideal: Option<Said<f64>>,
    /// "a beautiful face", or "beautiful" for the whole person: 1, or 2 with "very"
    pub face_ideal: Option<Said<f64>>,
}

impl Request {
    pub fn sex(&self) -> Sex {
        self.sex.as_ref().map_or(Sex::Neutral, |s| s.value)
    }
    pub fn part(&self, part: Part) -> Option<&PartMod> {
        self.parts.iter().find(|m| m.part == part)
    }
    pub fn build(&self) -> Build {
        self.build.as_ref().map_or(Build::Average, |b| b.value.0)
    }
}

#[derive(Debug)]
pub struct Error {
    pub what: String,
    pub hint: String,
}

fn err(what: impl Into<String>, hint: impl Into<String>) -> Error {
    Error { what: what.into(), hint: hint.into() }
}

/// the adjectives each part takes
pub fn takes(part: Part) -> &'static [Adj] {
    use Adj::*;
    match part {
        Part::Arms | Part::Legs | Part::Neck => &[Long, Short, Thick, Thin],
        Part::Torso => &[Long, Short],
        Part::Shoulders | Part::Hips | Part::Chest => &[Broad, Narrow],
        Part::Waist => &[Broad, Narrow, Thin, Thick],
        Part::Head => &[Round, Oval, Long, Square, Big, Small],
        Part::Hands | Part::Feet => &[Big, Small, Long],
        Part::Face => &[],
    }
}

fn split(sentence: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in sentence.to_lowercase().replace(',', " and ").split_whitespace() {
        let w = raw.trim_end_matches(['.', '!', '?']);
        // "180cm" is "180 cm"
        let digits: String = w.chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
        if !digits.is_empty() && digits.len() < w.len() && w[digits.len()..].chars().all(|c| c.is_ascii_alphabetic()) {
            out.push(digits.clone());
            out.push(w[digits.len()..].to_string());
        } else if !w.is_empty() {
            out.push(w.to_string());
        }
    }
    out
}

enum Item {
    Word(String, Tok),
    Num(String, f64),
}

pub fn parse(sentence: &str) -> Result<Request, Error> {
    let words = split(sentence);
    let mut items = Vec::new();
    for (n, w) in words.iter().enumerate() {
        if let Ok(x) = w.parse::<f64>() {
            items.push(Item::Num(w.clone(), x));
            continue;
        }
        let after_number = n > 0 && words[n - 1].parse::<f64>().is_ok();
        let tok = match w.as_str() {
            "feet" | "foot" | "ft" | "in" | "inches" | "cm" | "m" | "metres" | "meters" | "centimetres" | "centimeters" if after_number => {
                lexicon::lookup(if w == "in" { "inches" } else { w }).filter(|t| matches!(t, Tok::Unit(_))).or_else(|| {
                    // "feet" is first a part in the table
                    Some(Tok::Unit(0.3048))
                })
            }
            "in" | "cm" | "m" | "ft" | "inches" | "metres" | "meters" | "centimetres" | "centimeters" if !after_number => {
                if w == "in" {
                    Some(Tok::Filler)
                } else {
                    return Err(err(format!("\"{w}\" without a number"), "write a height like \"180 cm\""));
                }
            }
            _ => lexicon::lookup(w),
        };
        match tok {
            Some(Tok::Declined(why)) => return Err(err(format!("\"{w}\" is not something v0 can make"), why)),
            Some(t) => items.push(Item::Word(w.clone(), t)),
            None => {
                let hint = match lexicon::nearest(w) {
                    Some(n) => format!("did you mean \"{n}\"? (--vocabulary lists every word)"),
                    None => "--vocabulary lists every word".into(),
                };
                return Err(err(format!("unknown word \"{w}\""), hint));
            }
        }
    }

    let mut r = Request { sentence: sentence.trim().to_string(), ..Default::default() };
    let mut i = 0;
    let mut height_m = 0.0;
    let mut height_words: Vec<String> = Vec::new();
    while i < items.len() {
        // "very" doubles what follows it
        let mut strength = 1.0;
        let mut said = Vec::new();
        while let Item::Word(w, Tok::Very) = &items[i] {
            strength *= 2.0;
            said.push(w.clone());
            i += 1;
            if i == items.len() {
                return Err(err("\"very\" at the end", "\"very\" goes before a word it makes stronger: \"very tall\""));
            }
        }
        match &items[i] {
            Item::Num(w, x) => {
                let Some(Item::Word(u, Tok::Unit(per))) = items.get(i + 1) else {
                    return Err(err(format!("{w} what?"), "a number needs its unit: \"180 cm\", \"5 feet 10 inches\""));
                };
                height_m += x * per;
                height_words.push(format!("{w} {u}"));
                i += 2;
                if let Some(Item::Word(t, Tok::Stature(1))) = items.get(i) {
                    height_words.push(t.clone());
                    i += 1;
                }
                continue;
            }
            Item::Word(w, tok) => {
                said.push(w.clone());
                let next_part = |k: usize| match items.get(k) {
                    Some(Item::Word(p, Tok::Part(part))) => Some((p.clone(), *part)),
                    _ => None,
                };
                // an adjective (or "short", "thin") before a part belongs to it
                let as_adj = match tok {
                    Tok::Adj(a) => Some(*a),
                    Tok::Stature(-1) => Some(Adj::Short),
                    Tok::Build(Build::Thin) => Some(Adj::Thin),
                    _ => None,
                };
                if let (Some(adj), Some((pw, part))) = (as_adj, next_part(i + 1)) {
                    if !takes(part).contains(&adj) {
                        let ok: Vec<String> = takes(part).iter().map(|a| format!("{a:?}").to_lowercase()).collect();
                        return Err(err(format!("\"{w} {pw}\": v0 has no rule for it"), format!("{pw} can be: {}", ok.join(", "))));
                    }
                    if r.part(part).is_some() {
                        return Err(err(format!("{pw} described twice"), "say once how they are"));
                    }
                    said.push(pw);
                    r.parts.push(PartMod { part, adj, strength, words: said.join(" ") });
                    i += 2;
                    continue;
                }
                match tok {
                    Tok::Filler | Tok::And => {}
                    Tok::Sex(s) => {
                        if r.sex.is_some() {
                            return Err(err(format!("\"{w}\": a second person"), "v0 makes one body at a time"));
                        }
                        r.sex = Some(Said { value: *s, words: said.join(" ") });
                    }
                    Tok::Stature(k) => {
                        if r.stature.is_some() {
                            return Err(err(format!("\"{w}\": the height is said twice"), "say it once"));
                        }
                        r.stature = Some(Said { value: *k as f64 * strength, words: said.join(" ") });
                    }
                    Tok::Build(b) => {
                        if r.build.is_some() {
                            return Err(err(format!("\"{w}\": a second build"), "say one build: slim, athletic, muscular, heavy, ..."));
                        }
                        r.build = Some(Said { value: (*b, strength), words: said.join(" ") });
                    }
                    Tok::Pose(p) => {
                        if r.pose.is_some() {
                            return Err(err("two poses", "say one: a-pose or t-pose"));
                        }
                        r.pose = Some(Said { value: *p, words: said.join(" ") });
                    }
                    Tok::Profile(a) => r.profile.push(Said { value: *a, words: said.join(" ") }),
                    Tok::Ideal => {
                        // "a beautiful face" is the face's; "beautiful" alone is the whole person's
                        if let Some((pw, Part::Face)) = next_part(i + 1) {
                            if r.face_ideal.is_some() {
                                return Err(err(format!("\"{w} {pw}\": the face is described twice"), "say it once"));
                            }
                            said.push(pw);
                            r.face_ideal = Some(Said { value: strength, words: said.join(" ") });
                            i += 2;
                            continue;
                        }
                        if r.ideal.is_some() {
                            return Err(err(format!("\"{w}\": said twice"), "say it once"));
                        }
                        r.ideal = Some(Said { value: strength, words: said.join(" ") });
                        if r.face_ideal.is_none() {
                            r.face_ideal = Some(Said { value: strength, words: said.join(" ") });
                        }
                    }
                    Tok::Adj(_) => return Err(err(format!("\"{w}\" what?"), "an adjective goes before a part: \"long arms\", \"broad shoulders\"")),
                    Tok::Part(_) => return Err(err(format!("\"{w}\" without saying how"), format!("say how, e.g. \"long {w}\""))),
                    Tok::Unit(_) => return Err(err(format!("\"{w}\" without a number"), "write a height like \"180 cm\"")),
                    Tok::Very | Tok::Declined(_) => unreachable!(),
                }
                if strength != 1.0 && !matches!(tok, Tok::Stature(_) | Tok::Build(_) | Tok::Ideal) {
                    return Err(err(format!("\"{}\": very what?", said.join(" ")), "\"very\" goes with tall, short, a build or a part: \"very long arms\""));
                }
                i += 1;
            }
        }
    }
    if !height_words.is_empty() {
        if r.stature.is_some() {
            return Err(err("the height is said twice", "say \"tall\" or a number, not both"));
        }
        r.height = Some(Said { value: height_m, words: height_words.join(" ") });
    }
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sentence_becomes_a_request() {
        let r = parse("a very tall slim woman with long arms and a round head").unwrap();
        assert_eq!(r.sex(), Sex::Female);
        assert_eq!(r.stature.as_ref().unwrap().value, 2.0);
        assert_eq!(r.build(), Build::Slim);
        assert_eq!(r.part(Part::Arms).unwrap().adj, Adj::Long);
        assert_eq!(r.part(Part::Head).unwrap().adj, Adj::Round);
        assert_eq!(r.part(Part::Arms).unwrap().words, "long arms");
    }

    #[test]
    fn heights_in_units_add_up() {
        let r = parse("a man 5 feet 10 inches tall").unwrap();
        assert!((r.height.unwrap().value - 1.778).abs() < 1e-9);
        assert!((parse("a 180cm woman").unwrap().height.unwrap().value - 1.8).abs() < 1e-9);
    }

    #[test]
    fn beautiful_is_read_with_its_strength() {
        assert_eq!(parse("a very beautiful woman").unwrap().ideal.unwrap().value, 2.0);
        assert!(parse("a woman").unwrap().ideal.is_none());
    }

    #[test]
    fn a_beautiful_face_is_the_face_alone() {
        let r = parse("a woman with a beautiful face").unwrap();
        assert!(r.ideal.is_none() && r.face_ideal.is_some());
        let r = parse("a beautiful woman").unwrap();
        assert!(r.ideal.is_some() && r.face_ideal.is_some());
    }

    #[test]
    fn short_belongs_to_the_part_after_it() {
        let r = parse("a short man with short legs").unwrap();
        assert_eq!(r.stature.as_ref().unwrap().value, -1.0);
        assert_eq!(r.part(Part::Legs).unwrap().adj, Adj::Short);
    }

    #[test]
    fn what_cannot_be_read_is_refused() {
        assert!(parse("a woamn").unwrap_err().hint.contains("woman"));
        assert!(parse("a man 180").unwrap_err().what.contains("what?"));
        assert!(parse("a round neck").is_err());
        assert!(parse("a girl").is_err());
        assert!(parse("a tall man 190 cm").is_err());
    }
}
