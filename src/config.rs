use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The settings as shipped (the repository's `config.toml`): the user's file
/// only holds what differs from them.
pub const DEFAULTS: &str = include_str!("../config.toml");

/// The shipped profiles and styles, as `(file name, text)`.
mod builtin {
    include!(concat!(env!("OUT_DIR"), "/builtin.rs"));
}

#[derive(Debug, Deserialize)]
pub struct Config {
    pub listen: String,
    /// Frames received are saved here; empty: not saved.
    #[serde(default)]
    pub capture_dir: String,
    /// Per-game memory (past translations, learned names); empty to disable.
    #[serde(default = "default_data_dir")]
    pub data_dir: String,
    pub default_target_lang: String,
    pub ocr: OcrConfig,
    pub translate: TranslateConfig,
    pub render: RenderConfig,
    /// The shipped profiles, and the user's (which replace a shipped one of the
    /// same file name).
    #[serde(skip)]
    pub profiles: Vec<Profile>,
}

/// Expands a leading `~/` to `$HOME`, so the config needs no user-specific path.
pub fn expand_home(path: &str) -> String {
    match (path.strip_prefix("~/"), std::env::var("HOME")) {
        (Some(rest), Ok(home)) => format!("{home}/{rest}"),
        _ => path.to_owned(),
    }
}

/// The app's cache: `$XDG_CACHE_HOME/fukidashi`, by default
/// `~/.cache/fukidashi` (OCR models, server log).
pub fn cache_dir() -> PathBuf {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .unwrap_or_else(|| PathBuf::from(expand_home("~/.cache")));
    base.join("fukidashi")
}

/// The user's settings: `$XDG_CONFIG_HOME/fukidashi`, by default
/// `~/.config/fukidashi` (`config.toml`, `profiles/`, `styles/`, choices made
/// in the web page, secrets).
pub fn config_dir() -> PathBuf {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .unwrap_or_else(|| PathBuf::from(expand_home("~/.config")));
    base.join("fukidashi")
}

/// The app's data: `$XDG_DATA_HOME/fukidashi`, by default
/// `~/.local/share/fukidashi` (games' memory).
pub fn data_home() -> PathBuf {
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .unwrap_or_else(|| PathBuf::from(expand_home("~/.local/share")));
    base.join("fukidashi")
}

/// The user's profiles, which add to the shipped ones or replace them.
pub fn user_profiles_dir() -> PathBuf {
    config_dir().join("profiles")
}

fn user_styles_dir() -> PathBuf {
    config_dir().join("styles")
}

fn default_data_dir() -> String {
    data_home().join("data").to_string_lossy().into_owned()
}

/// Settings shared by both OpenAI-compatible backends.
#[derive(Debug, Clone, Deserialize)]
pub struct Backend {
    pub url: String,
    pub model: String,
    pub temperature: f32,
    pub max_tokens: u32,
    /// Merged as-is into the request body (e.g. `chat_template_kwargs`, `reasoning_effort`).
    #[serde(default)]
    pub extra: serde_json::Map<String, Value>,
    /// Sent as a bearer token; never read from the config file.
    #[serde(skip)]
    pub api_key: Option<String>,
}

/// PP-OCR models: registered names are downloaded on first use, paths are used as is.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct OcrConfig {
    /// Frames are resized to this width before OCR.
    pub width: u32,
    pub detection: String,
    pub recognition: String,
    pub dictionary: String,
    pub min_confidence: f32,
    /// Minimum mean text score inside a detected box (PP-OCR's default is 0.6).
    #[serde(default = "default_box_threshold")]
    pub box_threshold: f32,
}

fn default_box_threshold() -> f32 {
    0.5
}

#[derive(Debug, Clone, Deserialize)]
pub struct TranslateConfig {
    /// The local model (`url`, served by `server`) or a remote provider.
    #[serde(default)]
    pub provider: Provider,
    #[serde(flatten)]
    pub backend: Backend,
    #[serde(default)]
    pub remote: RemoteConfig,
    pub history: usize,
    /// `{lang}` is replaced by the target language name.
    pub system_prompt: String,
    /// When present, the app starts this server if nothing answers on `url`.
    pub server: Option<ServerConfig>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    #[default]
    Local,
    Remote,
}

/// An OpenAI-compatible provider (OpenRouter, a server on another machine):
/// only the text of the screen is sent, never the image.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct RemoteConfig {
    /// Base URL (`https://openrouter.ai/api/v1`) or the full chat completions URL.
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub model: String,
    /// Merged as-is into the request body, like `translate.extra`.
    #[serde(default)]
    pub extra: serde_json::Map<String, Value>,
}

/// A `llama-server` the app launches itself. Models are `owner/repo:file`
/// specs resolved through the Hugging Face cache.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Program to run (`~/` allowed) instead of a downloaded build; the
    /// `LLAMA_SERVER` environment variable overrides it.
    pub command: Option<String>,
    /// Prebuilt build to download: "auto" (from the GPU), "strix", "vulkan",
    /// "cuda", "cuda12", "cuda13" or "cpu".
    #[serde(default = "default_runtime")]
    pub runtime: String,
    /// "latest", or a release tag to stay on ("b11856").
    #[serde(default = "default_release")]
    pub release: String,
    #[serde(default)]
    pub update: Update,
    pub model: String,
    /// Model for one kind of build ("cuda", "vulkan", "strix", "cpu"), when
    /// another quantization suits it better.
    #[serde(default)]
    pub models: BTreeMap<String, String>,
    /// Optional draft model for speculative decoding (dropped if the server rejects it).
    pub draft: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    /// Extra arguments used only with the draft model.
    #[serde(default)]
    pub draft_args: Vec<String>,
    /// Minutes without a translation after which the server is stopped,
    /// freeing the GPU memory; the next translation starts it again. 0: never.
    #[serde(default = "default_idle_minutes")]
    pub idle_minutes: u64,
}

fn default_idle_minutes() -> u64 {
    30
}

/// What to do when a newer llama.cpp build is out.
#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Update {
    /// Install it in the background, use it from the next start.
    #[default]
    Auto,
    /// Only say so in the log.
    Notify,
    Off,
}

fn default_runtime() -> String {
    "auto".into()
}

fn default_release() -> String {
    "latest".into()
}

#[derive(Debug, Deserialize)]
pub struct RenderConfig {
    /// Font file for the translations; the bundled Noto Sans Bold when absent.
    /// The bundled Noto Sans JP Bold draws the characters it lacks.
    #[serde(default)]
    pub font: Option<String>,
    pub width: u32,
    pub background: [u8; 4],
    pub text_color: [u8; 4],
    /// Backdrop in the colour of the game's own box when it is plain, with
    /// black or white text, whichever reads better; `background` otherwise.
    #[serde(default = "default_true")]
    pub sample_background: bool,
    /// Outline around the letters, as a fraction of the font size (0: none).
    #[serde(default = "default_outline")]
    pub outline: f32,
    /// What RetroArch stretches the overlay over: the whole screen for 1.22
    /// and older ("auto": the primary screen's current mode, or a ratio such
    /// as "16:9"), the game alone for newer versions ("game").
    #[serde(default, deserialize_with = "screen")]
    pub screen_aspect: Screen,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum Screen {
    #[default]
    Auto,
    Game,
    Aspect(f32),
}

/// "auto", "game", "16:9" or a plain ratio such as 1.78.
fn screen<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Screen, D::Error> {
    let text = String::deserialize(deserializer)?;
    let ratio = match text.trim() {
        "auto" => return Ok(Screen::Auto),
        "game" => return Ok(Screen::Game),
        other => match other.split_once(':') {
            Some((w, h)) => w.trim().parse::<f32>().ok().zip(h.trim().parse::<f32>().ok()).map(|(w, h)| w / h),
            None => other.parse().ok(),
        },
    };
    match ratio {
        Some(r) if r.is_finite() && r > 0.0 => Ok(Screen::Aspect(r)),
        _ => Err(serde::de::Error::custom(format!("invalid screen_aspect {text:?}, expected \"auto\", \"game\" or e.g. \"16:9\""))),
    }
}

fn default_true() -> bool {
    true
}

fn default_outline() -> f32 {
    0.07
}

#[derive(Debug, Default, Clone, Deserialize)]
pub struct Profile {
    /// Substring matched against RetroArch's `system__game` label.
    #[serde(rename = "match")]
    pub pattern: String,
    /// Genre styles (files in `styles/`) whose instructions and glossary come
    /// before the game's own.
    #[serde(default)]
    pub styles: Vec<String>,
    #[serde(default)]
    pub instructions: String,
    #[serde(default)]
    pub glossary: BTreeMap<String, String>,
    /// Misreadings of the game's font, fixed before translation ("ヒンン" = "ヒント").
    #[serde(default)]
    pub ocr: BTreeMap<String, String>,
    /// Its file name (`kimikiss.toml`), shipped or in the user's profiles.
    #[serde(skip)]
    pub name: String,
    /// The file as written (without its styles merged in), to edit it.
    #[serde(skip)]
    pub text: String,
}

impl Profile {
    /// Whether RetroArch's label is this profile's game: `match` is part of
    /// it, spaces and punctuation aside ("Tokimeki Memorial 4" and
    /// "Tokimeki_Memorial_4" both match "Tokimeki_Memorial_4_JPN").
    pub fn matches(&self, label: &str) -> bool {
        use crate::memory::safe_name;
        !self.pattern.is_empty() && (label.contains(&self.pattern) || safe_name(label).contains(&safe_name(&self.pattern)))
    }
}

/// Instructions and glossary shared by every game of a genre.
#[derive(Debug, Deserialize)]
struct Style {
    #[serde(default)]
    instructions: String,
    #[serde(default)]
    glossary: BTreeMap<String, String>,
}

impl Config {
    /// The shipped settings with the user's file (if any) over them.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = user_text(path)?;
        Self::parse(&text).with_context(|| format!("invalid config {}", path.display()))
    }

    /// The shipped settings with `text` (the user's file) over them, and the
    /// profiles and styles, shipped and the user's.
    pub fn parse(text: &str) -> anyhow::Result<Self> {
        let mut config: Config = toml::Value::Table(merged(text)?).try_into()?;
        let ocr = &mut config.ocr;
        for path in [&mut ocr.detection, &mut ocr.recognition, &mut ocr.dictionary] {
            *path = expand_home(path);
        }
        for path in [&mut config.data_dir, &mut config.capture_dir] {
            *path = expand_home(path);
        }
        // An emptied optional setting is the same as none.
        let none_if_empty = |value: &mut Option<String>| *value = value.take().filter(|v| !v.trim().is_empty());
        none_if_empty(&mut config.render.font);
        config.render.font = config.render.font.as_deref().map(expand_home);
        if let Some(server) = &mut config.translate.server {
            none_if_empty(&mut server.command);
            none_if_empty(&mut server.draft);
        }

        let styles = files(builtin::STYLES, &user_styles_dir())?;
        for (name, text) in files(builtin::PROFILES, &user_profiles_dir())? {
            let mut profile: Profile = toml::from_str(&text).with_context(|| format!("invalid profile {name}"))?;
            // The game's own instructions and glossary entries come after its
            // styles', so they take precedence.
            let mut instructions = String::new();
            let mut glossary = BTreeMap::new();
            for style_name in &profile.styles {
                let file = format!("{style_name}.toml");
                let style_text = styles.get(&file).with_context(|| format!("profile {name}: no style {style_name}"))?;
                let style: Style = toml::from_str(style_text).with_context(|| format!("invalid style {file}"))?;
                instructions.push_str(style.instructions.trim_end());
                instructions.push('\n');
                glossary.extend(style.glossary);
            }
            instructions.push_str(&profile.instructions);
            glossary.append(&mut profile.glossary);
            profile.instructions = instructions;
            profile.glossary = glossary;
            profile.name = name;
            profile.text = text;
            config.profiles.push(profile);
        }
        Ok(config)
    }

    pub fn profile_for(&self, label: &str) -> Option<&Profile> {
        self.profiles.iter().find(|p| p.matches(label))
    }
}

/// The user's settings file, empty when there is none yet.
pub fn user_text(path: &Path) -> anyhow::Result<String> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(text),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(String::new()),
        Err(error) => Err(error).with_context(|| format!("cannot read {}", path.display())),
    }
}

/// The shipped settings with `text` over them: tables merge key by key, any
/// other value replaces the shipped one.
pub fn merged(text: &str) -> anyhow::Result<toml::Table> {
    fn merge(base: &mut toml::Table, top: toml::Table) {
        for (key, value) in top {
            match (base.get_mut(&key), value) {
                (Some(toml::Value::Table(base)), toml::Value::Table(top)) => merge(base, top),
                (_, value) => {
                    base.insert(key, value);
                }
            }
        }
    }
    let mut table: toml::Table = toml::from_str(DEFAULTS).context("invalid shipped config.toml")?;
    merge(&mut table, toml::from_str(text)?);
    Ok(table)
}

/// The shipped files and the user's `*.toml` in `dir`, by file name: a user
/// file replaces the shipped one of the same name.
fn files(shipped: &[(&str, &str)], dir: &Path) -> anyhow::Result<BTreeMap<String, String>> {
    let mut files: BTreeMap<String, String> = shipped.iter().map(|(name, text)| ((*name).to_owned(), (*text).to_owned())).collect();
    for entry in std::fs::read_dir(dir).into_iter().flatten().filter_map(Result::ok) {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "toml") {
            let text = std::fs::read_to_string(&path).with_context(|| format!("cannot read {}", path.display()))?;
            files.insert(entry.file_name().to_string_lossy().into_owned(), text);
        }
    }
    Ok(files)
}

/// The styles profiles can use, shipped and the user's.
pub fn style_names() -> Vec<String> {
    files(builtin::STYLES, &user_styles_dir())
        .unwrap_or_default()
        .into_keys()
        .filter_map(|name| name.strip_suffix(".toml").map(str::to_owned))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{Config, merged};

    #[test]
    fn user_settings_override_the_shipped_ones() {
        let table = merged("default_target_lang = \"fr\"\n[render]\nwidth = 1280\n").unwrap();
        assert_eq!(table["default_target_lang"].as_str(), Some("fr"));
        assert_eq!(table["render"]["width"].as_integer(), Some(1280));
        // The rest of the table is still the shipped one.
        assert!(table["render"].get("background").is_some());
        assert!(table["translate"]["system_prompt"].as_str().is_some_and(|p| !p.is_empty()));
    }

    #[test]
    fn shipped_settings_and_profiles_load() {
        let config = Config::parse("").unwrap();
        assert!(config.capture_dir.is_empty(), "captures are off by default");
        assert!(config.data_dir.ends_with("fukidashi/data"), "{}", config.data_dir);
        assert!(config.profiles.iter().any(|p| p.name == "kimikiss.toml" && !p.text.is_empty()));
        // Styles are merged into the profiles that use them.
        let kimikiss = config.profile_for("playstation2__KimiKiss (Japan)").unwrap();
        assert!(kimikiss.instructions.len() > kimikiss.text.len() / 4);
    }

    #[test]
    fn emptied_optional_settings_are_none() {
        let config = Config::parse("[translate.server]\ndraft = \"\"\n[render]\nfont = \"\"\n").unwrap();
        assert_eq!(config.translate.server.unwrap().draft, None);
        assert_eq!(config.render.font, None);
    }
}
