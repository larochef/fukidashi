//! Embeds the shipped profiles and styles (`profiles/*.toml`, `styles/*.toml`)
//! in the binary: `$OUT_DIR/builtin.rs` lists them as `(file name, text)`.

use std::{fmt::Write, path::Path};

fn main() {
    let mut code = String::new();
    for (constant, dir) in [("PROFILES", "profiles"), ("STYLES", "styles")] {
        println!("cargo:rerun-if-changed={dir}");
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "toml"))
            .collect();
        files.sort();
        writeln!(code, "pub const {constant}: &[(&str, &str)] = &[").unwrap();
        for file in files {
            let name = file.file_name().unwrap().to_string_lossy();
            let path = Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap()).join(&file);
            writeln!(code, "    ({name:?}, include_str!({:?})),", path.display().to_string()).unwrap();
        }
        writeln!(code, "];").unwrap();
    }
    let out = Path::new(&std::env::var("OUT_DIR").unwrap()).join("builtin.rs");
    std::fs::write(out, code).unwrap();
}
