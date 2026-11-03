//! The body's build: the base widths for the sex, then flesh (girth), muscle and fat from the build word.
use super::{Rule, Spec};
use crate::config::{Config, Width};
use crate::lexicon::Build;
use crate::parser::Request;
use crate::v3::lerp;

pub struct BuildRule;

fn mix(a: &Width, b: &Width, t: f64) -> Width {
    Width {
        shoulders: lerp(a.shoulders, b.shoulders, t),
        chest: lerp(a.chest, b.chest, t),
        chest_depth: lerp(a.chest_depth, b.chest_depth, t),
        waist: lerp(a.waist, b.waist, t),
        waist_depth: lerp(a.waist_depth, b.waist_depth, t),
        hips: lerp(a.hips, b.hips, t),
        hips_depth: lerp(a.hips_depth, b.hips_depth, t),
        neck: lerp(a.neck, b.neck, t),
        head: lerp(a.head, b.head, t),
        head_depth: lerp(a.head_depth, b.head_depth, t),
    }
}

impl Rule for BuildRule {
    fn name(&self) -> &'static str {
        "build"
    }
    fn variants(&self, _: &Request, _: &Config) -> Vec<&'static str> {
        vec!["as_asked"]
    }
    fn apply(&self, _: &str, r: &Request, c: &Config, s: &mut Spec) {
        let build = r.build();
        let name = format!("{build:?}").to_lowercase();
        let row = &c.build[&name];
        let strength = r.build.as_ref().map_or(1.0, |b| b.value.1);
        // "very" takes the build twice as far from average
        let avg = &c.build["average"];
        let girth = avg.girth + (row.girth - avg.girth) * strength;
        s.w = mix(&c.width.male, &c.width.female, s.female);
        // bone does not grow with flesh: shoulders take half of it
        s.w.shoulders *= 0.5 + 0.5 * girth;
        for x in [&mut s.w.chest, &mut s.w.chest_depth, &mut s.w.waist, &mut s.w.waist_depth, &mut s.w.hips, &mut s.w.hips_depth, &mut s.w.neck] {
            *x *= girth;
        }
        s.arm_girth = girth;
        s.leg_girth = girth;
        s.muscle = (avg.muscle + (row.muscle - avg.muscle) * strength).clamp(0.0, 1.5);
        s.fat = (avg.fat + (row.fat - avg.fat) * strength).clamp(0.0, 1.5);
        if build == Build::Curvy {
            s.w.hips *= 1.0 + 0.04 * strength;
        }
        let why = r.build.as_ref().map_or("no build was asked, so average".to_string(), |b| format!("\"{}\"", b.words));
        s.say(format!("Build {name}: flesh x{girth:.2}, muscle {:.2}, fat {:.2} ({why}).", s.muscle, s.fat), "build");
    }
}

#[cfg(test)]
mod tests {
    use crate::rules::spec_of;

    #[test]
    fn a_build_word_changes_the_flesh() {
        assert!(spec_of("a heavy man", &[]).w.waist > spec_of("a man", &[]).w.waist);
    }

    #[test]
    fn without_a_build_it_is_average() {
        assert_eq!(spec_of("a man", &[]).arm_girth, 1.0);
    }
}
