//! What the translator remembers about a game across sessions: every block
//! already translated, and the romaji chosen for each name.
//!
//! Files live in `<data_dir>/<game label>/`:
//! - `memory.jsonl`: one `{"lang", "ja", "tr"}` line per translated block, appended;
//! - `glossary.toml`: names learned from the translations, meant to be read and
//!   corrected by hand (`[glossary]` table, same format as game profiles).

use std::{
    collections::{BTreeMap, HashMap},
    io::Write,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

#[derive(Default)]
pub struct GameMemory {
    dir: Option<PathBuf>,
    /// Translation per (target language, Japanese text).
    translations: HashMap<(String, String), String>,
    /// Names learned from earlier translations: Japanese → romaji.
    pub names: BTreeMap<String, String>,
}

#[derive(Serialize, Deserialize)]
struct Entry {
    lang: String,
    ja: String,
    tr: String,
}

#[derive(Default, Serialize, Deserialize)]
struct GlossaryFile {
    #[serde(default)]
    glossary: BTreeMap<String, String>,
}

impl GameMemory {
    /// Loads a game's memory; `None` keeps everything in RAM (replay, bench).
    pub fn load(data_dir: Option<&Path>, label: &str) -> Self {
        let Some(data_dir) = data_dir else {
            return Self::default();
        };
        let dir = data_dir.join(safe_name(label));
        let mut memory = Self { dir: Some(dir.clone()), ..Self::default() };

        if let Ok(text) = std::fs::read_to_string(dir.join("memory.jsonl")) {
            for entry in text.lines().filter_map(|l| serde_json::from_str::<Entry>(l).ok()) {
                memory.translations.insert((entry.lang, entry.ja), entry.tr);
            }
        }
        if let Ok(text) = std::fs::read_to_string(dir.join("glossary.toml")) {
            match toml::from_str::<GlossaryFile>(&text) {
                Ok(file) => memory.names = file.glossary,
                Err(error) => tracing::warn!(%error, "invalid learned glossary, ignoring it"),
            }
        }
        tracing::info!(label, translations = memory.translations.len(), names = memory.names.len(), "game memory loaded");
        memory
    }

    pub fn get(&self, lang: &str, ja: &str) -> Option<&String> {
        self.translations.get(&(lang.to_owned(), ja.to_owned()))
    }

    pub fn remember(&mut self, lang: &str, ja: &str, tr: &str) {
        if tr.is_empty() {
            return;
        }
        self.translations.insert((lang.to_owned(), ja.to_owned()), tr.to_owned());
        let line = serde_json::to_string(&Entry { lang: lang.into(), ja: ja.into(), tr: tr.into() }).unwrap_or_default();
        self.write(|dir| {
            let mut file = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("memory.jsonl"))?;
            writeln!(file, "{line}")
        });
    }

    /// Records names not known yet; returns how many were added.
    pub fn learn_names(&mut self, names: impl IntoIterator<Item = (String, String)>) -> usize {
        let before = self.names.len();
        for (ja, romaji) in names {
            self.names.entry(ja).or_insert(romaji);
        }
        let added = self.names.len() - before;
        if added > 0 {
            let text = toml::to_string(&GlossaryFile { glossary: self.names.clone() }).unwrap_or_default();
            self.write(|dir| {
                let header = "# Names learned from translations. Fix a reading here and restart the server.\n\n";
                std::fs::write(dir.join("glossary.toml"), format!("{header}{text}"))
            });
        }
        added
    }

    fn write(&self, op: impl FnOnce(&Path) -> std::io::Result<()>) {
        let Some(dir) = &self.dir else { return };
        if let Err(error) = std::fs::create_dir_all(dir).and_then(|()| op(dir)) {
            tracing::warn!(dir = %dir.display(), %error, "cannot save game memory");
        }
    }
}

fn safe_name(label: &str) -> String {
    label.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect()
}
