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

/// One translated block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    pub lang: String,
    pub ja: String,
    pub tr: String,
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

        for entry in entries(&dir) {
            memory.translations.insert((entry.lang, entry.ja), entry.tr);
        }
        memory.names = names(&dir);
        // The directory name loses the label's spaces and punctuation, which
        // profiles match against.
        if label_of(&dir).as_deref() != Some(label) {
            memory.write(|dir| std::fs::write(dir.join("label.txt"), label));
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
            self.write(|dir| save_names(dir, &self.names));
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

/// The translations saved in a game's directory, one per (language, text):
/// the latest wins, in the order they were last written.
pub fn entries(dir: &Path) -> Vec<Entry> {
    let Ok(text) = std::fs::read_to_string(dir.join("memory.jsonl")) else {
        return Vec::new();
    };
    let mut latest: HashMap<(String, String), usize> = HashMap::new();
    let mut entries: Vec<Option<Entry>> = Vec::new();
    for entry in text.lines().filter_map(|l| serde_json::from_str::<Entry>(l).ok()) {
        if let Some(previous) = latest.insert((entry.lang.clone(), entry.ja.clone()), entries.len()) {
            entries[previous] = None;
        }
        entries.push(Some(entry));
    }
    entries.into_iter().flatten().collect()
}

/// Rewrites a game's translations (after a correction or a deletion).
pub fn save_entries(dir: &Path, entries: &[Entry]) -> std::io::Result<()> {
    let mut text = String::new();
    for entry in entries {
        text.push_str(&serde_json::to_string(entry).map_err(std::io::Error::other)?);
        text.push('\n');
    }
    std::fs::create_dir_all(dir)?;
    let tmp = dir.join("memory.jsonl.tmp");
    std::fs::write(&tmp, text)?;
    std::fs::rename(tmp, dir.join("memory.jsonl"))
}

/// The names learned for a game: Japanese → romaji.
pub fn names(dir: &Path) -> BTreeMap<String, String> {
    let Ok(text) = std::fs::read_to_string(dir.join("glossary.toml")) else {
        return BTreeMap::new();
    };
    match toml::from_str::<GlossaryFile>(&text) {
        Ok(file) => file.glossary,
        Err(error) => {
            tracing::warn!(%error, "invalid learned glossary, ignoring it");
            BTreeMap::new()
        }
    }
}

pub fn save_names(dir: &Path, names: &BTreeMap<String, String>) -> std::io::Result<()> {
    let text = toml::to_string(&GlossaryFile { glossary: names.clone() }).map_err(std::io::Error::other)?;
    let header = "# Names learned from translations; fix a reading here or in the settings page.\n\n";
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join("glossary.toml"), format!("{header}{text}"))
}

/// RetroArch's label for the game whose memory is in `dir`, once recorded.
pub fn label_of(dir: &Path) -> Option<String> {
    std::fs::read_to_string(dir.join("label.txt")).ok().map(|l| l.trim().to_owned()).filter(|l| !l.is_empty())
}

/// The directory name of a game: its label with only safe characters.
pub fn safe_name(label: &str) -> String {
    label.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect()
}
