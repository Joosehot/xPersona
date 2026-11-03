//! The closed vocabulary. A word not here is an error with the nearest known word, never a guess.

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sex {
    Male,
    Female,
    Neutral,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Build {
    Thin,
    Slim,
    Lean,
    Average,
    Athletic,
    Muscular,
    Stocky,
    Curvy,
    Heavy,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Part {
    Arms,
    Legs,
    Neck,
    Torso,
    Shoulders,
    Hips,
    Waist,
    Chest,
    Head,
    Hands,
    Feet,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Adj {
    Long,
    Short,
    Broad,
    Narrow,
    Thick,
    Thin,
    Big,
    Small,
    Round,
    Oval,
    Square,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pose {
    A,
    T,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Axis {
    Definition,
    Softness,
    Balance,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tok {
    /// a word that carries nothing: "a", "with", "make"
    Filler,
    And,
    Very,
    Sex(Sex),
    /// tall +1, short -1 (short before a part is that part's)
    Stature(i8),
    Build(Build),
    Part(Part),
    Adj(Adj),
    Pose(Pose),
    Profile(Axis),
    /// metres per unit
    Unit(f64),
    /// a word xPersona knows and declines, with why
    Declined(&'static str),
}

const WORDS: &[(&str, Tok)] = &[
    ("a", Tok::Filler),
    ("an", Tok::Filler),
    ("the", Tok::Filler),
    ("with", Tok::Filler),
    ("has", Tok::Filler),
    ("having", Tok::Filler),
    ("who", Tok::Filler),
    ("is", Tok::Filler),
    ("of", Tok::Filler),
    ("in", Tok::Filler),
    ("me", Tok::Filler),
    ("please", Tok::Filler),
    ("make", Tok::Filler),
    ("create", Tok::Filler),
    ("build", Tok::Filler),
    ("model", Tok::Filler),
    ("generate", Tok::Filler),
    ("character", Tok::Filler),
    ("body", Tok::Filler),
    ("figure", Tok::Filler),
    ("standing", Tok::Filler),
    ("pose", Tok::Filler),
    ("height", Tok::Filler),
    ("and", Tok::And),
    ("very", Tok::Very),
    ("really", Tok::Very),
    ("man", Tok::Sex(Sex::Male)),
    ("male", Tok::Sex(Sex::Male)),
    ("woman", Tok::Sex(Sex::Female)),
    ("female", Tok::Sex(Sex::Female)),
    ("person", Tok::Sex(Sex::Neutral)),
    ("human", Tok::Sex(Sex::Neutral)),
    ("adult", Tok::Sex(Sex::Neutral)),
    ("tall", Tok::Stature(1)),
    ("short", Tok::Stature(-1)),
    ("thin", Tok::Build(Build::Thin)),
    ("skinny", Tok::Build(Build::Thin)),
    ("slim", Tok::Build(Build::Slim)),
    ("slender", Tok::Build(Build::Slim)),
    ("lean", Tok::Build(Build::Lean)),
    ("average", Tok::Build(Build::Average)),
    ("athletic", Tok::Build(Build::Athletic)),
    ("fit", Tok::Build(Build::Athletic)),
    ("muscular", Tok::Build(Build::Muscular)),
    ("strong", Tok::Build(Build::Muscular)),
    ("stocky", Tok::Build(Build::Stocky)),
    ("curvy", Tok::Build(Build::Curvy)),
    ("heavy", Tok::Build(Build::Heavy)),
    ("heavyset", Tok::Build(Build::Heavy)),
    ("overweight", Tok::Build(Build::Heavy)),
    ("arms", Tok::Part(Part::Arms)),
    ("legs", Tok::Part(Part::Legs)),
    ("neck", Tok::Part(Part::Neck)),
    ("torso", Tok::Part(Part::Torso)),
    ("shoulders", Tok::Part(Part::Shoulders)),
    ("hips", Tok::Part(Part::Hips)),
    ("waist", Tok::Part(Part::Waist)),
    ("chest", Tok::Part(Part::Chest)),
    ("head", Tok::Part(Part::Head)),
    ("hands", Tok::Part(Part::Hands)),
    ("feet", Tok::Part(Part::Feet)),
    ("long", Tok::Adj(Adj::Long)),
    ("broad", Tok::Adj(Adj::Broad)),
    ("wide", Tok::Adj(Adj::Broad)),
    ("narrow", Tok::Adj(Adj::Narrow)),
    ("thick", Tok::Adj(Adj::Thick)),
    ("big", Tok::Adj(Adj::Big)),
    ("large", Tok::Adj(Adj::Big)),
    ("small", Tok::Adj(Adj::Small)),
    ("round", Tok::Adj(Adj::Round)),
    ("oval", Tok::Adj(Adj::Oval)),
    ("square", Tok::Adj(Adj::Square)),
    ("a-pose", Tok::Pose(Pose::A)),
    ("apose", Tok::Pose(Pose::A)),
    ("t-pose", Tok::Pose(Pose::T)),
    ("tpose", Tok::Pose(Pose::T)),
    ("defined", Tok::Profile(Axis::Definition)),
    ("toned", Tok::Profile(Axis::Definition)),
    ("soft", Tok::Profile(Axis::Softness)),
    ("natural", Tok::Profile(Axis::Balance)),
    ("balanced", Tok::Profile(Axis::Balance)),
    ("cm", Tok::Unit(0.01)),
    ("centimetres", Tok::Unit(0.01)),
    ("centimeters", Tok::Unit(0.01)),
    ("m", Tok::Unit(1.0)),
    ("metres", Tok::Unit(1.0)),
    ("meters", Tok::Unit(1.0)),
    ("ft", Tok::Unit(0.3048)),
    ("feet", Tok::Unit(0.3048)),
    ("foot", Tok::Unit(0.3048)),
    ("in", Tok::Unit(0.0254)),
    ("inches", Tok::Unit(0.0254)),
    ("child", Tok::Declined("v0 makes adults; a child's proportions are not in its rules")),
    ("kid", Tok::Declined("v0 makes adults; a child's proportions are not in its rules")),
    ("boy", Tok::Declined("v0 makes adults; say \"man\" for an adult")),
    ("girl", Tok::Declined("v0 makes adults; say \"woman\" for an adult")),
    ("baby", Tok::Declined("v0 makes adults; a child's proportions are not in its rules")),
    ("teenager", Tok::Declined("v0 makes adults; a teenager's proportions are not in its rules")),
    ("old", Tok::Declined("v0 has no rules for age yet")),
    ("elderly", Tok::Declined("v0 has no rules for age yet")),
    ("young", Tok::Declined("v0 has no rules for age yet")),
    ("hair", Tok::Declined("v0 models the body only: no hair, clothes or colours")),
    ("beard", Tok::Declined("v0 models the body only: no hair, clothes or colours")),
    ("clothes", Tok::Declined("v0 models the body only: no hair, clothes or colours")),
    ("shirt", Tok::Declined("v0 models the body only: no hair, clothes or colours")),
    ("dress", Tok::Declined("v0 models the body only: no hair, clothes or colours")),
    ("eyes", Tok::Declined("v0 shapes the head as a whole; no separate eyes yet")),
    ("fingers", Tok::Declined("v0 hands are mittens with a thumb; no separate fingers yet")),
];

/// "in" is a unit after a number and a filler elsewhere; the parser decides
pub fn lookup(word: &str) -> Option<Tok> {
    WORDS.iter().find(|w| w.0 == word).map(|w| w.1)
}

pub fn all() -> Vec<&'static str> {
    let mut w: Vec<&str> = WORDS.iter().map(|w| w.0).collect();
    w.sort_unstable();
    w.dedup();
    w
}

fn distance(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut prev = row[0];
        row[0] = i;
        for j in 1..=b.len() {
            let keep = row[j];
            row[j] = (row[j] + 1).min(row[j - 1] + 1).min(prev + usize::from(a[i - 1] != b[j - 1]));
            prev = keep;
        }
    }
    row[b.len()]
}

pub fn nearest(word: &str) -> Option<&'static str> {
    let best = all().into_iter().min_by_key(|w| (distance(word, w), *w))?;
    (distance(word, best) <= 2).then_some(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_typo_gets_the_nearest_word() {
        assert_eq!(nearest("woamn"), Some("woman"));
        assert_eq!(nearest("xylophone"), None);
    }
}
