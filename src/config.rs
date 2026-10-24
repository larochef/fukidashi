use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub listen: String,
    #[serde(default)]
    pub capture_dir: String,
    /// Per-game memory (past translations, learned names); empty to disable.
    #[serde(default = "default_data_dir")]
    pub data_dir: String,
    pub default_target_lang: String,
    pub ocr: OcrConfig,
    pub translate: TranslateConfig,
    pub render: RenderConfig,
    /// Directory of per-game profiles, one TOML file each, relative to the
    /// working directory.
    #[serde(default = "default_profiles_dir")]
    pub profiles_dir: String,
    /// Directory of genre styles that profiles can reference.
    #[serde(default = "default_styles_dir")]
    pub styles_dir: String,
    #[serde(default)]
    pub profiles: Vec<Profile>,
}

/// Expands a leading `~/` to `$HOME`, so the config needs no user-specific path.
pub fn expand_home(path: &str) -> String {
    match (path.strip_prefix("~/"), std::env::var("HOME")) {
        (Some(rest), Ok(home)) => format!("{home}/{rest}"),
        _ => path.to_owned(),
    }
}

/// The app's cache: `$XDG_CACHE_HOME/game-translator`, by default
/// `~/.cache/game-translator` (OCR models, server log).
pub fn cache_dir() -> PathBuf {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .unwrap_or_else(|| PathBuf::from(expand_home("~/.cache")));
    base.join("game-translator")
}

/// The app's per-machine settings: `$XDG_CONFIG_HOME/game-translator`, by
/// default `~/.config/game-translator` (choices made in the web page, secrets).
pub fn config_dir() -> PathBuf {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .unwrap_or_else(|| PathBuf::from(expand_home("~/.config")));
    base.join("game-translator")
}

fn default_data_dir() -> String {
    "data".into()
}

fn default_profiles_dir() -> String {
    "profiles".into()
}

fn default_styles_dir() -> String {
    "styles".into()
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
#[derive(Debug, Deserialize)]
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
#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Deserialize)]
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
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
        let mut config: Config = toml::from_str(&text).with_context(|| format!("invalid config {}", path.display()))?;
        let ocr = &mut config.ocr;
        for path in [&mut ocr.detection, &mut ocr.recognition, &mut ocr.dictionary] {
            *path = expand_home(path);
        }
        config.render.font = config.render.font.as_deref().map(expand_home);

        let dir = Path::new(&config.profiles_dir);
        if dir.is_dir() {
            let mut files: Vec<_> = std::fs::read_dir(dir)?
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "toml"))
                .collect();
            files.sort();
            for file in files {
                let text = std::fs::read_to_string(&file)?;
                let profile = toml::from_str(&text).with_context(|| format!("invalid profile {}", file.display()))?;
                config.profiles.push(profile);
            }
        }

        // Resolve styles once, at load: the game's own instructions and glossary
        // entries come last so they take precedence.
        for profile in &mut config.profiles {
            let mut instructions = String::new();
            let mut glossary = BTreeMap::new();
            for name in &profile.styles {
                let file = Path::new(&config.styles_dir).join(format!("{name}.toml"));
                let text = std::fs::read_to_string(&file).with_context(|| format!("cannot read style {}", file.display()))?;
                let style: Style = toml::from_str(&text).with_context(|| format!("invalid style {}", file.display()))?;
                instructions.push_str(style.instructions.trim_end());
                instructions.push('\n');
                glossary.extend(style.glossary);
            }
            instructions.push_str(&profile.instructions);
            glossary.append(&mut profile.glossary);
            profile.instructions = instructions;
            profile.glossary = glossary;
        }
        Ok(config)
    }

    pub fn profile_for(&self, label: &str) -> Option<&Profile> {
        self.profiles.iter().find(|p| label.contains(&p.pattern))
    }
}
