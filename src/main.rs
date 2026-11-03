//! xpersona "a tall slim woman with long arms" [-o out.glb] [--explain] [--rules my.toml] [--vocabulary]
use xpersona::config::{Config, DEFAULT};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut out = "out/persona.glb".to_string();
    let mut rules: Option<String> = None;
    let mut explain = false;
    let mut words = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--out" => {
                i += 1;
                out = args.get(i).cloned().unwrap_or(out);
            }
            "--rules" => {
                i += 1;
                rules = args.get(i).cloned();
            }
            "--explain" => explain = true,
            "--vocabulary" => {
                println!("{}", xpersona::lexicon::all().join(" "));
                return;
            }
            a => words.push(a.to_string()),
        }
        i += 1;
    }
    let text = match &rules {
        Some(p) => std::fs::read_to_string(p).unwrap_or_else(|e| fail(&format!("{p}: {e}"), "")),
        None => DEFAULT.to_string(),
    };
    let c = Config::load(&text).unwrap_or_else(|e| fail(&e, ""));
    if words.is_empty() {
        fail("nothing to make", "xpersona \"a tall slim woman with long arms\"");
    }
    match xpersona::make(&words.join(" "), &c, explain) {
        Ok(m) => {
            if let Some(dir) = std::path::Path::new(&out).parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            std::fs::write(&out, &m.glb).unwrap_or_else(|e| fail(&format!("{out}: {e}"), ""));
            print!("{}", m.text);
            println!("\nWritten: {out}");
        }
        Err(f) => fail(&f.what, &f.hint),
    }
}

fn fail(what: &str, hint: &str) -> ! {
    eprintln!("error: {what}");
    if !hint.is_empty() {
        eprintln!("  hint: {hint}");
    }
    std::process::exit(1)
}
