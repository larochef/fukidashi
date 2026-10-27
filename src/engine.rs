//! The translation engine in use: the local model (a `llama-server` the app
//! runs) or a remote OpenAI-compatible provider. It can be switched while the
//! app runs, from the web page: the local server is stopped when a remote
//! provider takes over (freeing the GPU memory) and started again on return.
//!
//! Choices made in the page are saved per machine, apart from the versioned
//! `config.toml`: `~/.config/fukidashi/engine.toml`, and the API key in
//! `secrets.toml` next to it (readable by the user only). Until a choice is
//! saved, nothing is downloaded: the model is ~16 GB, and a remote provider
//! needs none.
//!
//! The local server starts with the first translation that needs it, and
//! stops after `idle_minutes` without one, freeing the GPU memory.

use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{
        Arc, RwLock,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing::{info, warn};

use crate::{
    config::{Backend, LocalModel, Mode, Provider, RemoteConfig, ServerConfig, TranslateConfig, Update, config_dir},
    server::{self, ManagedServer},
};

/// Environment variable holding the remote provider's API key; it takes
/// precedence over the key saved from the page.
pub const API_KEY_ENV: &str = "FUKIDASHI_API_KEY";

/// What the page can change.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub provider: Provider,
    pub local: LocalSettings,
    pub remote: RemoteConfig,
}

/// The local model and the llama.cpp build to run it (see `runtime`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalSettings {
    /// A model of `[translate.models]`; absent: `[translate.server] model`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub runtime: String,
    pub release: String,
    pub update: Update,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case", tag = "state", content = "detail")]
pub enum Status {
    /// No engine chosen yet in the page (or with `fukidashi setup`).
    NotConfigured,
    /// No local server (none configured, or a remote provider in use).
    Stopped,
    /// Not running, started by the next translation (at first, or stopped
    /// after a while without one).
    Sleeping,
    /// Downloading the build or the model, or loading the model: what it is
    /// doing.
    Starting(String),
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
    /// An engine was chosen (saved settings): until then nothing starts.
    configured: AtomicBool,
    /// The last translation that needed the local server.
    last_used: std::sync::Mutex<Instant>,
}

impl Engine {
    pub fn new(client: reqwest::Client, config: &TranslateConfig) -> anyhow::Result<Arc<Self>> {
        let saved = load_settings()?;
        let configured = saved.is_some();
        let settings = match saved {
            Some(saved) => saved,
            None => Settings {
                provider: config.provider,
                local: config.server.as_ref().map_or_else(
                    || LocalSettings { model: None, runtime: "auto".into(), release: "latest".into(), update: Update::Auto },
                    |s| LocalSettings { model: None, runtime: s.runtime.clone(), release: s.release.clone(), update: s.update },
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
            status: RwLock::new(if configured { Status::Sleeping } else { Status::NotConfigured }),
            configured: AtomicBool::new(configured),
            last_used: std::sync::Mutex::new(Instant::now()),
        }))
    }

    pub fn configured(&self) -> bool {
        self.configured.load(Ordering::Relaxed)
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
        match &*self.status.read().unwrap() {
            Status::Starting(_) => Status::Starting(server::phase()),
            status => status.clone(),
        }
    }

    fn set_status(&self, status: Status) {
        *self.status.write().unwrap() = status;
    }

    /// The endpoint translations go to, with its key.
    pub fn backend(&self) -> Backend {
        let settings = self.settings.read().unwrap();
        if settings.provider == Provider::Remote {
            let config = self.config.read().unwrap();
            return remote_backend(&config.backend, &settings.remote, api_key().map(|(key, _)| key));
        }
        drop(settings);
        self.local_backend()
    }

    /// The local model's endpoint, asked as the model in use wants.
    pub fn local_backend(&self) -> Backend {
        let settings = self.settings.read().unwrap();
        let config = self.config.read().unwrap();
        let mut backend = config.backend.clone();
        let id = local_model_id(&config, &settings);
        if let Some(model) = config.models.get(&id) {
            // The id names the model in the requests, and in the memory.
            backend.model = id;
            backend.temperature = model.temperature.unwrap_or(backend.temperature);
            backend.extra.extend(model.request.clone());
            backend.mode = model.mode;
            backend.prompt = model.prompt.clone();
        }
        backend
    }

    /// The local model in use: an id of `[translate.models]`, or a file of
    /// the user's own.
    pub fn local_model(&self) -> String {
        local_model_id(&self.config.read().unwrap(), &self.settings.read().unwrap())
    }

    /// The `[translate.server]` settings with the page's choices and the
    /// chosen model's files and arguments.
    pub fn server_config(&self) -> Option<ServerConfig> {
        let settings = self.settings.read().unwrap();
        let config = self.config.read().unwrap();
        let local = &settings.local;
        config.server.clone().map(|mut server| {
            server.runtime = local.runtime.clone();
            server.release = local.release.clone();
            server.update = local.update;
            if let Some(model) = config.models.get(&local_model_id(&config, &settings)) {
                server.model = model.file.clone();
                server.models = model.files.clone();
                server.draft = model.draft.clone();
                server.args.extend(model.args.iter().cloned());
            }
            server
        })
    }

    pub fn local_port(&self) -> Option<u16> {
        server::port(&self.config.read().unwrap().backend.url).ok()
    }

    /// Starts the local server if the local model is in use and none runs.
    pub async fn start(&self) {
        if self.begin_start() {
            self.launch().await;
        }
    }

    pub fn start_in_background(self: &Arc<Self>) {
        if self.begin_start() {
            let engine = self.clone();
            tokio::spawn(async move { engine.launch().await });
        }
    }

    /// Moves to `Starting` at once if a start is due (so that no one reads the
    /// state before it), or says why not.
    fn begin_start(&self) -> bool {
        if self.settings.read().unwrap().provider != Provider::Local {
            self.set_status(Status::Stopped);
            return false;
        }
        if self.server_config().is_none() {
            self.set_status(Status::External);
            return false;
        }
        // Running already, or being stopped.
        if self.server.try_lock().map_or(true, |server| server.is_some()) {
            return false;
        }
        let mut status = self.status.write().unwrap();
        if matches!(*status, Status::Starting(_)) {
            return false;
        }
        server::set_phase("getting ready".into());
        *status = Status::Starting(String::new());
        *self.last_used.lock().unwrap() = Instant::now();
        true
    }

    async fn launch(&self) {
        let Some(config) = self.server_config() else { return };
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
        if !matches!(self.status(), Status::Starting(_)) {
            self.set_status(Status::Stopped);
        }
    }

    /// Whether the next translation can be made, starting the local server if
    /// it sleeps; otherwise the message to show instead (no engine chosen, the
    /// model downloading). Waits for the model to load, which takes seconds:
    /// RetroArch waits up to a minute for an answer.
    pub async fn ready(self: &Arc<Self>, page: &str) -> Result<(), String> {
        if self.settings.read().unwrap().provider != Provider::Local {
            return Ok(());
        }
        if !self.configured() {
            return Err(format!("Fukidashi needs a translation engine: open {page} to choose one."));
        }
        *self.last_used.lock().unwrap() = Instant::now();
        let failed = |error: &str| format!("The translation model could not start ({error}). See {page}.");
        match self.status() {
            Status::Ready(_) | Status::External => return Ok(()),
            Status::Starting(_) => {}
            // A failed start is tried again: it may have been a network error.
            Status::Stopped | Status::Sleeping | Status::NotConfigured | Status::Failed(_) => {
                info!("starting the translation server for this translation");
                self.start_in_background();
            }
        }
        let deadline = Instant::now() + Duration::from_secs(45);
        while Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(250)).await;
            match self.status() {
                Status::Ready(_) | Status::External => return Ok(()),
                Status::Failed(error) => return Err(failed(&error)),
                Status::Starting(phase) if phase.starts_with("downloading") => {
                    return Err(format!("The translation model is getting ready ({phase}). Try again once it is done: {page}"));
                }
                _ => {}
            }
        }
        Err("The translation model is still loading. Try again in a moment.".into())
    }

    /// Stops the local server after `idle_minutes` without a translation; the
    /// next one starts it again.
    pub fn stop_when_idle(self: &Arc<Self>) {
        let engine = self.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(30)).await;
                let minutes = engine.server_config().map_or(0, |s| s.idle_minutes);
                let idle = engine.last_used.lock().unwrap().elapsed();
                let running = matches!(engine.status(), Status::Ready(_)) && engine.server.lock().await.is_some();
                if minutes > 0 && running && idle >= Duration::from_secs(minutes * 60) {
                    info!(minutes, "no translation for a while: stopping the translation server, the next one starts it again");
                    engine.stop().await;
                    engine.set_status(Status::Sleeping);
                }
            }
        });
    }

    pub async fn restart(self: &Arc<Self>) {
        self.stop().await;
        self.start_in_background();
    }

    /// Applies and saves new settings; `api_key`: `None` keeps the saved key,
    /// an empty one removes it.
    pub async fn update(self: &Arc<Self>, settings: Settings, api_key: Option<String>) -> anyhow::Result<()> {
        save_choice(&settings, api_key, &self.config.read().unwrap().models)?;
        let model = self.local_model();
        let previous = std::mem::replace(&mut *self.settings.write().unwrap(), settings.clone());
        let first = !self.configured.swap(true, Ordering::Relaxed);
        let model_changed = self.local_model() != model;
        info!(provider = ?settings.provider, model = self.local_model(), "engine settings saved");
        match (previous.provider, settings.provider) {
            (Provider::Local, Provider::Remote) => self.stop().await,
            // The local model just chosen: downloaded and loaded now, rather
            // than during the first translation.
            (_, Provider::Local) if first || previous.provider == Provider::Remote => self.start_in_background(),
            // Another local model: the running one makes way for it.
            (Provider::Local, Provider::Local) if model_changed => self.restart().await,
            _ if first => self.set_status(Status::Stopped),
            _ => {}
        }
        Ok(())
    }
}

/// Checks and saves an engine choice (`api_key` as for `Engine::update`),
/// without applying it: for `fukidashi setup` while the app is not running.
pub fn save_choice(settings: &Settings, api_key: Option<String>, models: &BTreeMap<String, LocalModel>) -> anyhow::Result<()> {
    validate(settings, models)?;
    if let Some(key) = api_key {
        save_api_key(key.trim())?;
    }
    save_settings(settings)
}

impl Engine {
    /// Downloads what the local model needs (build, model files) without
    /// starting it.
    pub async fn download(&self) -> anyhow::Result<()> {
        let config = self.server_config().context("no [translate.server] in the settings: nothing to download")?;
        server::prepare(&self.client, &config).await.map(|_| ())
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
        mode: Mode::Chat,
        prompt: String::new(),
    }
}

/// The local model of these settings: the chosen one, else the configured one.
fn local_model_id(config: &TranslateConfig, settings: &Settings) -> String {
    settings.local.model.clone().or_else(|| config.server.as_ref().map(|s| s.model.clone())).unwrap_or_default()
}

fn validate(settings: &Settings, models: &BTreeMap<String, LocalModel>) -> anyhow::Result<()> {
    crate::runtime::variant(&settings.local.runtime)?;
    if let Some(model) = &settings.local.model
        && !models.contains_key(model)
    {
        let known: Vec<&str> = models.keys().map(String::as_str).collect();
        bail!("unknown model \"{model}\": expected one of {}", known.join(", "));
    }
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
            mode: Default::default(),
            prompt: String::new(),
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
