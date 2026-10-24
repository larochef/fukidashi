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
    config::{Provider, RemoteConfig},
    engine::{self, Settings},
    runtime, server,
    translate::Translator,
};

const PAGE: &str = include_str!("../assets/ui/index.html");
/// Lines of the app's log kept for the page.
const LOG_LINES: usize = 500;

pub fn routes() -> Router<Arc<App>> {
    Router::new()
        .route("/api/engine", get(engine_state).put(update_engine))
        .route("/api/engine/test", post(test_engine))
        .route("/api/engine/restart", post(restart_engine))
        .route("/api/engine/check", post(check_update))
        .route("/api/logs", get(logs))
}

/// The settings page; RetroArch only ever posts to `/`.
pub async fn page() -> Html<&'static str> {
    Html(PAGE)
}

/// An error as `{ "error": ... }`, with a status code the page can show.
struct ApiError(StatusCode, String);

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
    let model = server
        .as_ref()
        .map(|s| variant.as_ref().and_then(|v| s.models.get(v.family())).unwrap_or(&s.model).clone());
    Json(json!({
        "provider": settings.provider,
        "status": app.engine.status(),
        "local": {
            "managed": server.is_some(),
            "runtime": settings.local.runtime,
            "release": settings.local.release,
            "update": settings.local.update,
            "program": program,
            "model": model,
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
    let backend = match test.provider {
        Provider::Local => app.config.translate.backend.clone(),
        Provider::Remote => {
            let key = test.api_key.filter(|k| !k.trim().is_empty()).map(|k| k.trim().to_owned()).or_else(|| engine::api_key().map(|(k, _)| k));
            engine::remote_backend(&app.config.translate.backend, &test.remote, key)
        }
    };
    let started = Instant::now();
    let text = "相原「おはよう、星乃さん。今日もいい天気だね」".to_owned();
    let result = Translator::new(app.client.clone(), None)
        .translate(&app.config.translate, &backend, None, "test", "en", std::slice::from_ref(&text))
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
