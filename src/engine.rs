//! The translation engine in use: the local model (a `llama-server` the app
//! runs) or a remote OpenAI-compatible provider. It can be switched while the
//! app runs, from the web page: the local server is stopped when a remote
//! provider takes over (freeing the GPU memory) and started again on return.
//!
//! Choices made in the page are saved per machine, apart from the versioned
//! `config.toml`: `~/.config/game-translator/engine.toml`, and the API key in
//! `secrets.toml` next to it (readable by the user only).

use std::{
    path::PathBuf,
    sync::{Arc, RwLock},
};

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing::{info, warn};

use crate::{
    config::{Backend, Provider, RemoteConfig, ServerConfig, TranslateConfig, Update, config_dir},
    server::{self, ManagedServer},
};

/// Environment variable holding the remote provider's API key; it takes
/// precedence over the key saved from the page.
pub const API_KEY_ENV: &str = "GAME_TRANSLATOR_API_KEY";

/// What the page can change.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub provider: Provider,
    pub local: LocalSettings,
    pub remote: RemoteConfig,
}

/// The llama.cpp build to run (see `runtime`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalSettings {
    pub runtime: String,
    pub release: String,
    pub update: Update,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "lowercase", tag = "state", content = "detail")]
pub enum Status {
    /// No local server (none configured, or a remote provider in use).
    Stopped,
    /// Downloading the build or the model, or loading the model.
    Starting,
    /// Running the given program or build.
    Ready(String),
    /// Something else already answers on the local endpoint.
    External,
    Failed(String),
}

pub struct Engine {
    client: reqwest::Client,
    /// `[translate]` of config.toml, replaced when it is edited.
    config: RwLock<TranslateConfig>,
    settings: RwLock<Settings>,
    server: Mutex<Option<ManagedServer>>,
    /// The settings the running server started with (endpoint, server config).
    started_with: RwLock<Option<(String, ServerConfig)>>,
    status: RwLock<Status>,
}

impl Engine {
    pub fn new(client: reqwest::Client, config: &TranslateConfig) -> anyhow::Result<Arc<Self>> {
        let settings = match load_settings()? {
            Some(saved) => saved,
            None => Settings {
                provider: config.provider,
                local: config.server.as_ref().map_or_else(
                    || LocalSettings { runtime: "auto".into(), release: "latest".into(), update: Update::Auto },
                    |s| LocalSettings { runtime: s.runtime.clone(), release: s.release.clone(), update: s.update },
                ),
                remote: config.remote.clone(),
            },
        };
        Ok(Arc::new(Self {
            client,
            config: RwLock::new(config.clone()),
            settings: RwLock::new(settings),
            server: Mutex::new(None),
            started_with: RwLock::new(None),
            status: RwLock::new(Status::Stopped),
        }))
    }

    /// Takes an edited config; a local server applies it when restarted.
    pub fn set_config(&self, config: &TranslateConfig) {
        *self.config.write().unwrap() = config.clone();
    }

    pub fn config(&self) -> TranslateConfig {
        self.config.read().unwrap().clone()
    }

    pub fn settings(&self) -> Settings {
        self.settings.read().unwrap().clone()
    }

    pub fn status(&self) -> Status {
        self.status.read().unwrap().clone()
    }

    fn set_status(&self, status: Status) {
        *self.status.write().unwrap() = status;
    }

    /// The endpoint translations go to, with its key.
    pub fn backend(&self) -> Backend {
        let settings = self.settings.read().unwrap();
        let config = self.config.read().unwrap();
        match settings.provider {
            Provider::Local => config.backend.clone(),
            Provider::Remote => remote_backend(&config.backend, &settings.remote, api_key().map(|(key, _)| key)),
        }
    }

    /// The `[translate.server]` settings with the page's choices.
    pub fn server_config(&self) -> Option<ServerConfig> {
        let local = self.settings.read().unwrap().local.clone();
        self.config.read().unwrap().server.clone().map(|mut server| {
            server.runtime = local.runtime;
            server.release = local.release;
            server.update = local.update;
            server
        })
    }

    pub fn local_port(&self) -> Option<u16> {
        server::port(&self.config.read().unwrap().backend.url).ok()
    }

    /// Starts the local server if the local model is in use and none runs.
    pub async fn start(&self) {
        if self.settings.read().unwrap().provider != Provider::Local {
            self.set_status(Status::Stopped);
            return;
        }
        let Some(config) = self.server_config() else {
            self.set_status(Status::External);
            return;
        };
        if matches!(self.status(), Status::Starting) || self.server.lock().await.is_some() {
            return;
        }
        self.set_status(Status::Starting);
        let url = self.config.read().unwrap().backend.url.clone();
        match server::ensure(&self.client, &url, &config).await {
            Ok(Some(started)) => {
                // Switched to a remote provider while it was loading.
                if self.settings.read().unwrap().provider != Provider::Local {
                    info!("remote provider selected meanwhile, stopping the local server");
                    self.set_status(Status::Stopped);
                    return;
                }
                self.set_status(Status::Ready(started.program.clone()));
                *self.started_with.write().unwrap() = Some((url, config));
                *self.server.lock().await = Some(started);
            }
            Ok(None) => self.set_status(Status::External),
            Err(error) => {
                warn!("cannot start the translation server: {error:#}");
                self.set_status(Status::Failed(format!("{error:#}")));
            }
        }
    }

    /// Whether the running local server started with other settings than
    /// the current ones (model, arguments, build).
    pub fn needs_restart(&self) -> bool {
        let current = self.server_config().map(|server| (self.config.read().unwrap().backend.url.clone(), server));
        match &*self.started_with.read().unwrap() {
            Some(started) => matches!(self.status(), Status::Ready(_)) && current.as_ref() != Some(started),
            None => false,
        }
    }

    /// Stops the local server we started (killing it frees its memory).
    pub async fn stop(&self) {
        *self.started_with.write().unwrap() = None;
        if let Some(server) = self.server.lock().await.take() {
            info!(program = server.program, "stopping the translation server");
            drop(server);
        }
        if !matches!(self.status(), Status::Starting) {
            self.set_status(Status::Stopped);
        }
    }

    pub fn start_in_background(self: &Arc<Self>) {
        let engine = self.clone();
        tokio::spawn(async move { engine.start().await });
    }

    pub async fn restart(self: &Arc<Self>) {
        self.stop().await;
        self.start_in_background();
    }

    /// Applies and saves new settings; `api_key`: `None` keeps the saved key,
    /// an empty one removes it.
    pub async fn update(self: &Arc<Self>, settings: Settings, api_key: Option<String>) -> anyhow::Result<()> {
        validate(&settings)?;
        if let Some(key) = api_key {
            save_api_key(key.trim())?;
        }
        save_settings(&settings)?;
        let previous = std::mem::replace(&mut *self.settings.write().unwrap(), settings.clone());
        info!(provider = ?settings.provider, "engine settings saved");
        match (previous.provider, settings.provider) {
            (Provider::Local, Provider::Remote) => self.stop().await,
            (Provider::Remote, Provider::Local) => self.start_in_background(),
            _ => {}
        }
        Ok(())
    }
}

/// The remote provider as a backend: the sampling settings of the local one,
/// its own model and options.
pub fn remote_backend(local: &Backend, remote: &RemoteConfig, api_key: Option<String>) -> Backend {
    let url = remote.url.trim().trim_end_matches('/');
    let url = if url.ends_with("/chat/completions") { url.to_owned() } else { format!("{url}/chat/completions") };
    Backend {
        url,
        model: remote.model.trim().to_owned(),
        temperature: local.temperature,
        max_tokens: local.max_tokens,
        extra: remote.extra.clone(),
        api_key,
    }
}

fn validate(settings: &Settings) -> anyhow::Result<()> {
    crate::runtime::variant(&settings.local.runtime)?;
    let release = &settings.local.release;
    if release != "latest" && !(release.starts_with('b') && release[1..].parse::<u64>().is_ok()) {
        bail!("release must be \"latest\" or a tag like b11856, got \"{release}\"");
    }
    if settings.provider == Provider::Remote {
        let url = reqwest::Url::parse(settings.remote.url.trim()).context("invalid provider URL")?;
        if !matches!(url.scheme(), "http" | "https") {
            bail!("the provider URL must start with http:// or https://");
        }
        if settings.remote.model.trim().is_empty() {
            bail!("the provider needs a model name");
        }
    }
    Ok(())
}

fn settings_path() -> PathBuf {
    config_dir().join("engine.toml")
}

fn secrets_path() -> PathBuf {
    config_dir().join("secrets.toml")
}

fn load_settings() -> anyhow::Result<Option<Settings>> {
    let path = settings_path();
    match std::fs::read_to_string(&path) {
        Ok(text) => Ok(Some(toml::from_str(&text).with_context(|| format!("invalid {}", path.display()))?)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error).with_context(|| format!("cannot read {}", path.display())),
    }
}

fn save_settings(settings: &Settings) -> anyhow::Result<()> {
    let text = format!(
        "# Engine choices made in the web page; they take precedence over config.toml.\n{}",
        toml::to_string(settings)?
    );
    write_private(&settings_path(), &text)
}

#[derive(Default, Serialize, Deserialize)]
struct Secrets {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    api_key: Option<String>,
}

/// Where the key comes from, for the page (the key itself never leaves).
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum KeySource {
    Env,
    File,
}

/// The API key: the environment variable first, then the saved one.
pub fn api_key() -> Option<(String, KeySource)> {
    if let Ok(key) = std::env::var(API_KEY_ENV)
        && !key.trim().is_empty()
    {
        return Some((key.trim().to_owned(), KeySource::Env));
    }
    let text = std::fs::read_to_string(secrets_path()).ok()?;
    let secrets: Secrets = toml::from_str(&text).ok()?;
    secrets.api_key.filter(|k| !k.is_empty()).map(|key| (key, KeySource::File))
}

fn save_api_key(key: &str) -> anyhow::Result<()> {
    let secrets = Secrets { api_key: (!key.is_empty()).then(|| key.to_owned()) };
    write_private(&secrets_path(), &toml::to_string(&secrets)?)
}

/// Writes a file only the user can read, in a directory only they can enter.
fn write_private(path: &PathBuf, text: &str) -> anyhow::Result<()> {
    use std::{
        io::Write,
        os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt},
    };
    let dir = path.parent().context("no parent directory")?;
    std::fs::DirBuilder::new().recursive(true).mode(0o700).create(dir)?;
    let tmp = path.with_extension("tmp");
    let mut file = std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(&tmp)?;
    file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    file.write_all(text.as_bytes())?;
    file.sync_all()?;
    std::fs::rename(&tmp, path).with_context(|| format!("cannot write {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::remote_backend;
    use crate::config::{Backend, RemoteConfig};

    #[test]
    fn completes_the_provider_url() {
        let local = Backend {
            url: "http://127.0.0.1:8194/v1/chat/completions".into(),
            model: "local".into(),
            temperature: 0.1,
            max_tokens: 512,
            extra: Default::default(),
            api_key: None,
        };
        let remote = |url: &str| RemoteConfig { url: url.into(), model: " google/gemma-4-26b-a4b-it ".into(), extra: Default::default() };
        let backend = remote_backend(&local, &remote("https://openrouter.ai/api/v1/"), Some("key".into()));
        assert_eq!(backend.url, "https://openrouter.ai/api/v1/chat/completions");
        assert_eq!(backend.model, "google/gemma-4-26b-a4b-it");
        assert_eq!(backend.temperature, 0.1);
        assert!(backend.extra.is_empty());
        let full = remote_backend(&local, &remote("http://host:8080/v1/chat/completions"), None);
        assert_eq!(full.url, "http://host:8080/v1/chat/completions");
    }
}
