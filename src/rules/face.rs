//! The face. Unasked, an individual face drawn from the sentence's own words: the same words, the same face. Asked
//! beautiful or harmonious, that face is pulled toward the average of all faces and kept symmetric, the two traits
//! research ties most firmly to attractiveness (averaged faces: Langlois & Roggman 1990; symmetry: Rhodes 2006).
use super::{Rule, Spec};
use crate::config::Config;
use crate::parser::Request;

pub struct Face;

/// the sentence's words without the ones asking for beauty and the ones carrying nothing, hashed: so "a woman",
/// "a beautiful woman" and "a woman with a beautiful face" are one person
pub fn seed(r: &Request) -> u64 {
    let mut drop: Vec<String> = ["a", "an", "the", "with", "has", "having", "who", "is", "of", "and", "make", "create", "build", "model", "me", "please"].iter().map(|w| w.to_string()).collect();
    for said in r.ideal.iter().chain(r.face_ideal.iter()) {
        drop.extend(said.words.split_whitespace().map(String::from));
    }
    let mut h: u64 = 0xCBF29CE484222325;
    for w in r.sentence.to_lowercase().replace(',', " ").split_whitespace() {
        let w = w.trim_end_matches(['.', '!', '?']);
        if drop.iter().any(|d| d == w) {
            continue;
        }
        for b in w.bytes().chain([b' ']) {
            h = (h ^ b as u64).wrapping_mul(0x100000001B3);
        }
    }
    h
}

impl Rule for Face {
    fn name(&self) -> &'static str {
        "face"
    }
    fn variants(&self, _: &Request, _: &Config) -> Vec<&'static str> {
        vec!["as_asked"]
    }
    fn apply(&self, _: &str, r: &Request, c: &Config, s: &mut Spec) {
        s.face_seed = seed(r);
        s.face_scale = match &r.face_ideal {
            None => 1.0,
            Some(k) if k.value >= 2.0 => c.learned.face_average * c.learned.face_average,
            Some(_) => c.learned.face_average,
        };
        match &r.face_ideal {
            None => s.say(format!("Face: an individual face drawn from the words (face {:04x}).", s.face_seed & 0xFFFF), "face"),
            Some(k) => s.say(format!("Face {:04x} pulled to {:.0}% of its distance from the average face, both sides alike (\"{}\"): averageness and symmetry, what research ties most to attractive faces.", s.face_seed & 0xFFFF, s.face_scale * 100.0, k.words), "face"),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::parser::parse;
    use crate::rules::spec_of;

    #[test]
    fn beauty_words_keep_the_person() {
        assert_eq!(super::seed(&parse("a beautiful woman").unwrap()), super::seed(&parse("a woman").unwrap()));
        assert_eq!(super::seed(&parse("a woman with a very beautiful face").unwrap()), super::seed(&parse("a woman").unwrap()));
        assert_ne!(super::seed(&parse("a tall woman").unwrap()), super::seed(&parse("a woman").unwrap()));
    }

    #[test]
    fn a_beautiful_face_is_nearer_the_average() {
        assert!(spec_of("a woman with a beautiful face", &[]).face_scale < spec_of("a woman", &[]).face_scale);
    }
}
