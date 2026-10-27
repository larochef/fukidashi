//! Settings page (`GET /`) and its JSON API (`/api/...`), served next to
//! RetroArch's endpoint (`POST /`).

use std::{
    collections::{HashMap, VecDeque},
    io::Write,
    sync::{Arc, LazyLock, Mutex},
    time::Instant,
};

use axum::{
    Json, Router,
    extract::{Query, State},
    http::{StatusCode, header},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use regex::Regex;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    App,
    config::{self, Provider, RemoteConfig},
    engine::{self, Settings},
    runtime, server,
    settings::{self, Edit},
    system,
    translate::Translator,
};

const PAGE: &str = include_str!("../assets/ui/index.html");
const FAVICON: &str = include_str!("../assets/icons/fukidashi.svg");
/// Lines of the app's log kept for the page.
const LOG_LINES: usize = 500;

pub fn routes() -> Router<Arc<App>> {
    Router::new()
        .route("/favicon.svg", get(favicon))
        .route("/api/engine", get(engine_state).put(update_engine))
        .route("/api/engine/test", post(test_engine))
        .route("/api/engine/restart", post(restart_engine))
        .route("/api/engine/check", post(check_update))
        .route("/api/logs", get(logs))
        .route("/api/config", get(config_state).put(update_config))
        .route("/api/config/raw", axum::routing::put(update_raw_config))
        .route("/api/config/reload", post(reload_config))
        .route("/api/quit", post(quit))
        .route("/api/system", get(system_state).put(update_system))
        .merge(crate::games::routes())
}

/// The settings page; RetroArch only ever posts to `/`.
pub async fn page() -> Html<&'static str> {
    Html(PAGE)
}

/// The app's icon, as the page's favicon.
async fn favicon() -> ([(header::HeaderName, &'static str); 1], &'static str) {
    ([(header::CONTENT_TYPE, "image/svg+xml")], FAVICON)
}

/// An error as `{ "error": ... }`, with a status code the page can show.
pub struct ApiError(pub StatusCode, pub String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "error": self.1 }))).into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(error: anyhow::Error) -> Self {
        ApiError(StatusCode::BAD_REQUEST, format!("{error:#}"))
    }
}

async fn engine_state(State(app): State<Arc<App>>) -> Json<Value> {
    let settings = app.engine.settings();
    let server = app.engine.server_config();
    let variant = runtime::variant(&settings.local.runtime).ok();
    let key = engine::api_key();
    // A program given by the user replaces the downloaded builds.
    let program = std::env::var("LLAMA_SERVER")
        .map(|p| format!("{p} (LLAMA_SERVER)"))
        .ok()
        .or_else(|| server.as_ref().and_then(|s| s.command.clone()));
    let model = server.as_ref().map(|s| server::model_spec(s, variant.as_ref()).to_owned());
    let config = app.engine.config();
    let memory = runtime::Memory::detect();
    // Without a GPU build, the GPU does not count.
    let memory = runtime::Memory { gpu: memory.gpu.filter(|_| !matches!(variant, Some(runtime::Variant::Cpu))), ..memory };
    // Best first, then the bigger ones.
    let mut sorted: Vec<_> = config.models.iter().collect();
    sorted.sort_by(|(_, a), (_, b)| {
        let need = |m: &config::LocalModel| m.gpu_memory.or(m.ram).unwrap_or_default();
        b.quality.cmp(&a.quality).then(need(b).total_cmp(&need(a)))
    });
    let models: Vec<Value> = sorted
        .into_iter()
        .map(|(id, m)| {
            let file = variant.as_ref().and_then(|v| m.files.get(v.family())).unwrap_or(&m.file);
            json!({
                "id": id,
                "name": m.name,
                "quality": m.quality,
                "summary": m.summary,
                "mode": m.mode,
                "gpu_memory": m.gpu_memory,
                "ram": m.ram,
                "size": m.size,
                "cached": server::is_cached(file),
            })
        })
        .collect();
    let model_size = match &model {
        Some(spec) => server::file_size(&app.client, spec).await,
        None => None,
    };
    Json(json!({
        "provider": settings.provider,
        "configured": app.engine.configured(),
        "status": app.engine.status(),
        "restart_needed": app.engine.needs_restart(),
        "local": {
            "managed": server.is_some(),
            "runtime": settings.local.runtime,
            "release": settings.local.release,
            "update": settings.local.update,
            "program": program,
            "model": model,
            "model_id": app.engine.local_model(),
            "models": models,
            "memory": memory,
            "recommended": config::recommend(&config.models, &memory),
            "model_size": model_size,
            "model_cached": model.as_deref().is_some_and(server::is_cached),
            "gpus": runtime::Gpus::detect(),
            "variant": variant.as_ref().map(runtime::Variant::name),
            "builds": variant.as_ref().map(runtime::slot),
        },
        "remote": {
            "url": settings.remote.url,
            "model": settings.remote.model,
            "extra": settings.remote.extra,
            "key": key.map(|(_, source)| source),
            "key_env": engine::API_KEY_ENV,
        },
    }))
}

#[derive(Deserialize)]
struct EngineUpdate {
    #[serde(flatten)]
    settings: Settings,
    /// Absent: keep the saved key; empty: remove it.
    api_key: Option<String>,
}

async fn update_engine(State(app): State<Arc<App>>, Json(update): Json<EngineUpdate>) -> Result<Json<Value>, ApiError> {
    app.engine.update(update.settings, update.api_key).await?;
    Ok(engine_state(State(app)).await)
}

#[derive(Deserialize)]
struct TestRequest {
    provider: Provider,
    #[serde(default)]
    remote: RemoteConfig,
    /// Absent: the saved key.
    api_key: Option<String>,
}

/// Translates one line with the real request (system prompt, JSON schema):
/// what a provider rejects shows up here rather than in the game.
async fn test_engine(State(app): State<Arc<App>>, Json(test): Json<TestRequest>) -> Result<Json<Value>, ApiError> {
    let config = app.engine.config();
    let backend = match test.provider {
        Provider::Local => app.engine.local_backend(),
        Provider::Remote => {
            let key = test.api_key.filter(|k| !k.trim().is_empty()).map(|k| k.trim().to_owned()).or_else(|| engine::api_key().map(|(k, _)| k));
            engine::remote_backend(&config.backend, &test.remote, key)
        }
    };
    let started = Instant::now();
    let text = "相原「おはよう、星乃さん。今日もいい天気だね」".to_owned();
    let result = Translator::new(app.client.clone(), None)
        .translate(&config, &backend, None, "test", "en", std::slice::from_ref(&text))
        .await;
    let ms = started.elapsed().as_millis() as u64;
    Ok(Json(match result {
        Ok((translations, _)) => json!({ "ok": true, "ms": ms, "model": backend.model, "source": text, "translation": translations.first() }),
        Err(error) => json!({ "ok": false, "ms": ms, "model": backend.model, "error": format!("{error:#}") }),
    }))
}

async fn restart_engine(State(app): State<Arc<App>>) -> Result<Json<Value>, ApiError> {
    if app.engine.settings().provider != Provider::Local {
        return Err(ApiError(StatusCode::CONFLICT, "a remote provider is in use".into()));
    }
    app.engine.restart().await;
    Ok(engine_state(State(app)).await)
}

/// Looks for a newer llama.cpp build now; it installs in the background and
/// is used from the next start (or restart).
async fn check_update(State(app): State<Arc<App>>) -> Result<Json<Value>, ApiError> {
    let server = app.engine.server_config().ok_or_else(|| ApiError(StatusCode::CONFLICT, "no local server configured".into()))?;
    let variant = runtime::variant(&server.runtime)?;
    tokio::spawn(runtime::check_for_update(app.client.clone(), server.update, server.release, variant, true));
    Ok(Json(json!({ "started": true })))
}

/// The settings in use (shipped ones with the user's over them), the user's
/// file, and the shipped settings for reference.
async fn config_state(State(app): State<Arc<App>>) -> Result<Json<Value>, ApiError> {
    let text = config::user_text(&app.config_path)?;
    Ok(Json(json!({
        "path": app.config_path,
        "values": serde_json::to_value(config::merged(&text)?).map_err(anyhow::Error::from)?,
        "text": text,
        "defaults": config::DEFAULTS,
    })))
}

#[derive(Deserialize)]
struct ConfigEdits {
    edits: Vec<Edit>,
}

/// Changes some settings in the user's file (comments kept) and applies them.
/// A setting set back to its shipped value leaves the file, which keeps only
/// what differs.
async fn update_config(State(app): State<Arc<App>>, Json(body): Json<ConfigEdits>) -> Result<Json<Value>, ApiError> {
    let text = config::user_text(&app.config_path)?;
    let shipped = serde_json::to_value(config::merged("")?).map_err(anyhow::Error::from)?;
    let edits: Vec<Edit> = body
        .edits
        .into_iter()
        .map(|edit| {
            let default = edit.key.split('.').try_fold(&shipped, |value, part| value.get(part));
            let unset = default.is_none() && edit.value.as_str() == Some("");
            if default == Some(&edit.value) || unset { Edit { value: Value::Null, ..edit } } else { edit }
        })
        .collect();
    let edited = settings::apply(&text, &edits)?;
    let reloaded = app.reload(&edited).await?;
    let mut state = config_state(State(app)).await?.0;
    state["reloaded"] = json!(reloaded);
    Ok(Json(state))
}

#[derive(Deserialize)]
struct RawConfig {
    text: String,
}

/// Replaces the user's settings file as a whole, once it is known to be valid.
async fn update_raw_config(State(app): State<Arc<App>>, Json(body): Json<RawConfig>) -> Result<Json<Value>, ApiError> {
    let reloaded = app.reload(&body.text).await?;
    let mut state = config_state(State(app)).await?.0;
    state["reloaded"] = json!(reloaded);
    Ok(Json(state))
}

/// Applies the user's settings as they are on disk (after editing them by
/// hand), with the profiles and styles.
async fn reload_config(State(app): State<Arc<App>>) -> Result<Json<Value>, ApiError> {
    let text = config::user_text(&app.config_path)?;
    let reloaded = app.reload(&text).await?;
    let mut state = config_state(State(app)).await?.0;
    state["reloaded"] = json!(reloaded);
    Ok(Json(state))
}

/// Starting with the session, the GNOME extension.
async fn system_state() -> Json<Value> {
    Json(system::state().await)
}

async fn update_system(Json(change): Json<system::Change>) -> Result<Json<Value>, ApiError> {
    system::apply(change).await?;
    Ok(Json(system::state().await))
}

/// Stops the app (`fukidashi stop`, the menu entry's Stop action).
async fn quit(State(app): State<Arc<App>>) -> Json<Value> {
    app.quit.notify_one();
    Json(json!({ "stopping": true }))
}

/// The end of the app's log, or of the local server's (`?source=server`).
async fn logs(State(app): State<Arc<App>>, Query(query): Query<HashMap<String, String>>) -> Response {
    let lines: usize = query.get("lines").and_then(|n| n.parse().ok()).unwrap_or(200).min(LOG_LINES);
    let text = match query.get("source").map(String::as_str) {
        Some("server") => {
            let path = app.engine.local_port().map(server::log_path);
            let content = path.and_then(|p| std::fs::read_to_string(p).ok()).unwrap_or_default();
            let all: Vec<&str> = content.lines().collect();
            all[all.len().saturating_sub(lines)..].join("\n")
        }
        _ => {
            let buffer = LOG.lock().unwrap();
            buffer.iter().skip(buffer.len().saturating_sub(lines)).cloned().collect::<Vec<_>>().join("\n")
        }
    };
    ([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], text).into_response()
}

static LOG: LazyLock<Mutex<VecDeque<String>>> = LazyLock::new(|| Mutex::new(VecDeque::with_capacity(LOG_LINES)));

/// Log output for `tracing_subscriber`: to stderr, and the last lines kept
/// (without colours) for the page.
#[derive(Clone, Copy)]
pub struct LogWriter;

impl Write for LogWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        static ANSI: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\x1b\[[0-9;]*m").expect("valid pattern"));
        let text = String::from_utf8_lossy(bytes);
        let mut buffer = LOG.lock().unwrap();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            if buffer.len() == LOG_LINES {
                buffer.pop_front();
            }
            buffer.push_back(ANSI.replace_all(line, "").into_owned());
        }
        drop(buffer);
        std::io::stderr().write_all(bytes)?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        std::io::stderr().flush()
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogWriter {
    type Writer = LogWriter;

    fn make_writer(&'a self) -> Self::Writer {
        *self
    }
}
