//! What the translator remembers about a game across sessions: every block
//! already translated, and the romaji chosen for each name.
//!
//! Files live in `<data_dir>/<game label>/`:
//! - `memory.jsonl`: one `{"lang", "ja", "tr", "by"}` line per translated block,
//!   appended (`by`: the model that translated it, or `user` once corrected);
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
    /// The model that made the translation, or [`BY_USER`]; absent in older memories.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub by: Option<String>,
}

/// `by` of a translation corrected by hand.
pub const BY_USER: &str = "user";

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

    /// The translation of `ja`, or of a text it differs from by a misread
    /// character or two (see [`close`]).
    pub fn get(&self, lang: &str, ja: &str) -> Option<&String> {
        if let Some(tr) = self.translations.get(&(lang.to_owned(), ja.to_owned())) {
            return Some(tr);
        }
        let mut best: Option<(usize, &String, bool)> = None;
        for ((l, known), tr) in &self.translations {
            let Some(distance) = (l == lang).then(|| close(ja, known)).flatten() else { continue };
            match best {
                Some((d, _, _)) if d < distance => {}
                // Two texts as close: no telling which one it is.
                Some((d, _, _)) if d == distance => best = Some((d, tr, true)),
                _ => best = Some((distance, tr, false)),
            }
        }
        let (distance, tr, tied) = best?;
        if tied {
            return None;
        }
        tracing::debug!(ja, distance, "close translation reused");
        Some(tr)
    }

    pub fn remember(&mut self, lang: &str, ja: &str, tr: &str, by: &str) {
        if tr.is_empty() {
            return;
        }
        self.translations.insert((lang.to_owned(), ja.to_owned()), tr.to_owned());
        let line = serde_json::to_string(&Entry { lang: lang.into(), ja: ja.into(), tr: tr.into(), by: Some(by.into()) }).unwrap_or_default();
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

/// The edit distance between two texts, when small enough to be the same
/// text read differently by the OCR: line breaks and spaces aside, at most one
/// character in ten differs, the numbers are the same, and short texts (where
/// one character changes the meaning) must be equal.
fn close(a: &str, b: &str) -> Option<usize> {
    const MIN_CHARS: usize = 10;
    let chars = |s: &str| -> Vec<char> { s.chars().filter(|c| *c != '⏎' && !c.is_whitespace()).collect() };
    let (a, b) = (chars(a), chars(b));
    let allowed = a.len().min(b.len()) / 10;
    if a.len().min(b.len()) < MIN_CHARS || a.len().abs_diff(b.len()) > allowed {
        return None;
    }
    let digits = |s: &[char]| -> Vec<char> { s.iter().filter_map(|c| c.to_digit(10).and(Some(*c))).collect() };
    if digits(&a) != digits(&b) {
        return None;
    }
    let distance = levenshtein(&a, &b);
    (distance <= allowed).then_some(distance)
}

fn levenshtein(a: &[char], b: &[char]) -> usize {
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let substitution = diagonal + usize::from(ca != cb);
            diagonal = row[j + 1];
            row[j + 1] = substitution.min(row[j] + 1).min(row[j + 1] + 1);
        }
    }
    row[b.len()]
}

/// A game's translations and names as shared between players: exported from
/// the games page, imported into another player's memory of the same game.
#[derive(Serialize, Deserialize)]
pub struct Pack {
    /// Format version.
    pub fukidashi: u32,
    /// RetroArch's label of the game it was made from, and its shown name.
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub names: BTreeMap<String, String>,
    #[serde(default)]
    pub translations: Vec<Entry>,
}

pub const PACK_VERSION: u32 = 1;

/// What an import changed.
#[derive(Debug, Default, PartialEq, Serialize)]
pub struct Merged {
    pub added: usize,
    pub replaced: usize,
    pub kept: usize,
    pub names: usize,
}

/// Adds a pack's translations to a game's: a translation corrected by hand
/// (`by` [`BY_USER`]) replaces a model's, and is never replaced; otherwise the
/// one already there stays. Names already known keep their reading.
pub fn merge(entries: &mut Vec<Entry>, names: &mut BTreeMap<String, String>, pack: Pack) -> Merged {
    let mut merged = Merged::default();
    let by_user = |e: &Entry| e.by.as_deref() == Some(BY_USER);
    let mut index: HashMap<(String, String), usize> =
        entries.iter().enumerate().map(|(i, e)| ((e.lang.clone(), e.ja.clone()), i)).collect();
    for entry in pack.translations {
        let entry = Entry { lang: entry.lang.trim().to_owned(), ja: entry.ja, tr: entry.tr.trim().to_owned(), by: entry.by };
        if entry.lang.is_empty() || entry.ja.trim().is_empty() || entry.tr.is_empty() {
            continue;
        }
        match index.get(&(entry.lang.clone(), entry.ja.clone())) {
            Some(&i) if by_user(&entry) && !by_user(&entries[i]) && entries[i].tr != entry.tr => {
                entries[i] = entry;
                merged.replaced += 1;
            }
            Some(_) => merged.kept += 1,
            None => {
                index.insert((entry.lang.clone(), entry.ja.clone()), entries.len());
                entries.push(entry);
                merged.added += 1;
            }
        }
    }
    for (ja, romaji) in pack.names {
        let (ja, romaji) = (ja.trim().to_owned(), romaji.trim().to_owned());
        if !ja.is_empty() && !romaji.is_empty() && !names.contains_key(&ja) {
            names.insert(ja, romaji);
            merged.names += 1;
        }
    }
    merged
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{BY_USER, Entry, GameMemory, Merged, PACK_VERSION, Pack, merge};

    fn entry(ja: &str, tr: &str, by: &str) -> Entry {
        Entry { lang: "fr".into(), ja: ja.into(), tr: tr.into(), by: Some(by.into()) }
    }

    #[test]
    fn reuses_a_translation_the_ocr_read_slightly_differently() {
        let mut memory = GameMemory::default();
        let line = "今日は一緒に帰ろうって言ったじゃない、忘れたの？";
        memory.translations.insert(("fr".into(), line.into()), "Tu avais dit qu'on rentrerait ensemble".into());
        memory.translations.insert(("fr".into(), "所持金は100円です".into()), "Tu as 100 yens".into());
        memory.translations.insert(("fr".into(), "はい".into()), "Oui".into());
        // A kanji misread, another line break.
        assert!(memory.get("fr", "今日は一緒に帰ろうって言ったじゃない、忌れたの？").is_some());
        assert!(memory.get("fr", "今日は一緒に帰ろうって⏎言ったじゃない、忘れたの？").is_some());
        assert!(memory.get("en", line).is_none());
        // Short texts and numbers must be the same.
        assert!(memory.get("fr", "いい").is_none());
        assert!(memory.get("fr", "所持金は200円です").is_none());
        // Too many differences.
        assert!(memory.get("fr", "明日は一人で帰ろうって言ったじゃない、忘れたの？").is_none());
    }

    #[test]
    fn imports_a_pack_without_losing_corrections() {
        let mut entries = vec![entry("A", "a (model)", "gemma"), entry("B", "b (mine)", BY_USER), entry("C", "c (model)", "gemma")];
        let mut names = BTreeMap::from([("春香".to_owned(), "Haruka".to_owned())]);
        let pack = || Pack {
            fukidashi: PACK_VERSION,
            label: String::new(),
            name: String::new(),
            names: BTreeMap::from([("春香".to_owned(), "Haruka".to_owned()), ("美咲".to_owned(), "Misaki".to_owned())]),
            translations: vec![
                entry("A", "a (theirs)", BY_USER),
                entry("B", "b (theirs)", BY_USER),
                entry("C", "c (their model)", "hy-mt2-7b"),
                entry("D", "d (theirs)", "hy-mt2-7b"),
                entry("", "empty", "x"),
            ],
        };
        let merged = merge(&mut entries, &mut names, pack());
        assert_eq!(merged, Merged { added: 1, replaced: 1, kept: 2, names: 1 });
        let tr: Vec<&str> = entries.iter().map(|e| e.tr.as_str()).collect();
        assert_eq!(tr, ["a (theirs)", "b (mine)", "c (model)", "d (theirs)"]);
        assert_eq!(names["春香"], "Haruka");
        // Twice: nothing changes.
        let again = merge(&mut entries, &mut names, pack());
        assert_eq!(again, Merged { added: 0, replaced: 0, kept: 4, names: 0 });
    }
}
