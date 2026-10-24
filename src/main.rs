//! RetroArch AI Service endpoint: OCR → translation → overlay, all local.
//!
//! Usage:
//!   game-translator [--config FILE] [serve]
//!   game-translator [--config FILE] replay [--lang CODE] [--label LABEL] [--ocr-only] [--out DIR] CAPTURE.png...
//!   game-translator [--config FILE] bench [--lang CODE] CASES.toml

mod config;
mod llm;
mod memory;
mod ocr;
mod render;
mod runtime;
mod server;
mod translate;

use std::{
    collections::HashMap,
    io::Cursor,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, bail};
use axum::{
    Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Query, State},
    http::{StatusCode, header},
    routing::post,
};
use base64::{Engine, engine::general_purpose::STANDARD as B64};
use image::{DynamicImage, ImageFormat, RgbaImage};
use serde_json::{Value, json};
use tracing::{info, warn};

use crate::{
    config::Config,
    ocr::{Block, Ocr},
    render::Renderer,
    translate::Translator,
};

/// RetroArch treats this exact message specially (it unpauses the game).
const NO_TEXT: &str = "No text found.";

struct App {
    config: Config,
    client: reqwest::Client,
    ocr: Ocr,
    translator: Translator,
    renderer: Renderer,
    last_frame: Mutex<Option<LastFrame>>,
}

/// The previous answer, reused when the same screen is requested again.
struct LastFrame {
    key: String,
    thumbnail: Vec<u8>,
    response: Value,
}

struct Outcome {
    blocks: Vec<Block>,
    translations: Vec<String>,
    overlay: Option<RgbaImage>,
}

impl App {
    async fn run(&self, frame: &DynamicImage, label: &str, lang: &str, ocr_only: bool) -> anyhow::Result<Outcome> {
        let started = Instant::now();
        let blocks = self.ocr.detect(&self.config.ocr, frame).await.context("OCR failed")?;
        let ocr_ms = started.elapsed().as_millis() as u64;
        if blocks.is_empty() || ocr_only {
            info!(ocr_ms, blocks = blocks.len(), "ocr done");
            return Ok(Outcome { blocks, translations: Vec::new(), overlay: None });
        }

        let profile = self.config.profile_for(label);
        let texts: Vec<String> = blocks
            .iter()
            .map(|block| {
                let text = block.joined();
                match profile {
                    Some(p) => p.ocr.iter().fold(text, |text, (from, to)| text.replace(from, to)),
                    None => text,
                }
            })
            .collect();
        let (translations, cached) = self
            .translator
            .translate(&self.client, &self.config.translate, profile, label, lang, &texts)
            .await
            .context("translation failed")?;
        let translate_ms = started.elapsed().as_millis() as u64 - ocr_ms;

        let items: Vec<(&Block, &str)> = blocks.iter().zip(translations.iter().map(String::as_str)).collect();
        let overlay = self.renderer.render(frame, &items);
        let total_ms = started.elapsed().as_millis() as u64;
        info!(
            ocr_ms,
            translate_ms,
            render_ms = total_ms - ocr_ms - translate_ms,
            total_ms,
            blocks = blocks.len(),
            cached,
            profile = profile.map(|p| p.pattern.as_str()),
            "pipeline done"
        );
        Ok(Outcome { blocks, translations, overlay: Some(overlay) })
    }
}

fn main() -> anyhow::Result<()> {
    // PP-OCR models are downloaded into the app's cache unless OAR_HOME says
    // otherwise. Set before the runtime starts any thread: changing the
    // environment is only sound while the process is single-threaded.
    if std::env::var_os("OAR_HOME").is_none() {
        unsafe { std::env::set_var("OAR_HOME", config::cache_dir().join("ocr")) };
    }
    tokio::runtime::Builder::new_multi_thread().enable_all().build()?.block_on(run())
}

async fn run() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "info,ort=warn".into()).as_str())
        .init();

    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let config_path = take_option(&mut args, "--config").unwrap_or_else(|| "config.toml".into());
    let config = Config::load(Path::new(&config_path))?;
    // Only the server builds the games' persistent memory; replay and bench runs
    // must not mix test translations into it.
    let serving = matches!(args.first().map(String::as_str), None | Some("serve"));
    let data_dir = (serving && !config.data_dir.is_empty()).then(|| PathBuf::from(&config.data_dir));
    let app = Arc::new(App {
        renderer: Renderer::new(&config.render).context("cannot load font")?,
        ocr: Ocr::new(&config.ocr)?,
        client: reqwest::Client::new(),
        translator: Translator::new(data_dir),
        last_frame: Mutex::new(None),
        config,
    });

    // Held until the command returns: dropping it stops the server we started.
    let translates = !args.iter().any(|a| a == "--ocr-only");
    let _server = match &app.config.translate.server {
        Some(server) if translates => server::ensure(&app.client, &app.config.translate.backend.url, server).await?,
        _ => None,
    };

    match args.first().map(String::as_str) {
        None | Some("serve") => serve(app).await,
        Some("replay") => replay(app, args.split_off(1)).await,
        Some("bench") => bench(app, args.split_off(1)).await,
        Some(other) => bail!("unknown command {other}"),
    }
}

fn take_option(args: &mut Vec<String>, name: &str) -> Option<String> {
    let i = args.iter().position(|a| a == name)?;
    args.remove(i);
    (i < args.len()).then(|| args.remove(i))
}

fn take_flag(args: &mut Vec<String>, name: &str) -> bool {
    let found = args.iter().position(|a| a == name);
    if let Some(i) = found {
        args.remove(i);
    }
    found.is_some()
}

async fn serve(app: Arc<App>) -> anyhow::Result<()> {
    if !app.config.capture_dir.is_empty() {
        std::fs::create_dir_all(&app.config.capture_dir)?;
    }
    let listen = app.config.listen.clone();
    let router = Router::new()
        .route("/", post(handle_request))
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024))
        .with_state(app);
    let listener = tokio::net::TcpListener::bind(&listen).await?;
    info!("listening on http://{listen}/");
    // Ctrl+C returns normally, so the translation server we started gets stopped.
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            info!("shutting down");
        })
        .await?;
    Ok(())
}

async fn handle_request(
    State(app): State<Arc<App>>,
    Query(query): Query<HashMap<String, String>>,
    body: Bytes,
) -> (StatusCode, [(header::HeaderName, &'static str); 1], String) {
    let response = match respond(&app, &query, &body).await {
        Ok(value) => value,
        Err(error) => {
            warn!("{error:#}");
            json!({ "error": format!("{error:#}") })
        }
    };
    (StatusCode::OK, [(header::CONTENT_TYPE, "application/json")], response.to_string())
}

async fn respond(app: &App, query: &HashMap<String, String>, body: &[u8]) -> anyhow::Result<Value> {
    // RetroArch builds the body by hand: parse it ourselves rather than trust the Content-Type.
    let request: Value = serde_json::from_slice(body).context("invalid JSON")?;
    let label = request.get("label").and_then(Value::as_str).unwrap_or("unknown");
    let lang = query.get("target_lang").map_or(app.config.default_target_lang.as_str(), String::as_str);

    let encoded = request.get("image").and_then(Value::as_str).context("missing image")?;
    let png = B64.decode(encoded).context("invalid base64")?;
    let frame = image::load_from_memory(&png).context("invalid image")?;
    info!(label, lang, width = frame.width(), height = frame.height(), "request");

    let key = format!("{label}|{lang}");
    let thumb = thumbnail(&frame);
    if let Some(last) = app.last_frame.lock().unwrap().as_ref()
        && last.key == key && same_screen(&last.thumbnail, &thumb) {
            info!("same screen as last request, reusing overlay");
            return Ok(last.response.clone());
        }
    save_capture(&app.config.capture_dir, label, &png);

    let outcome = app.run(&frame, label, lang, false).await?;
    for (block, translation) in outcome.blocks.iter().zip(&outcome.translations) {
        info!(ja = block.joined(), tr = translation.as_str(), "block");
    }
    let response = match outcome.overlay {
        Some(overlay) => json!({ "image": B64.encode(encode_png(&overlay)?) }),
        None => json!({ "error": NO_TEXT }),
    };
    *app.last_frame.lock().unwrap() = Some(LastFrame { key, thumbnail: thumb, response: response.clone() });
    Ok(response)
}

/// Small greyscale version of the frame, to recognize a screen already translated.
fn thumbnail(frame: &DynamicImage) -> Vec<u8> {
    frame.resize_exact(160, 90, image::imageops::FilterType::Triangle).to_luma8().into_raw()
}

/// Tolerates a small animation (blinking cursor) but not a new line of text,
/// which changes far more pixels.
fn same_screen(a: &[u8], b: &[u8]) -> bool {
    let changed = a.iter().zip(b).filter(|(x, y)| x.abs_diff(**y) > 24).count();
    a.len() == b.len() && changed * 1000 < a.len() * 3
}

fn save_capture(dir: &str, label: &str, png: &[u8]) {
    if dir.is_empty() {
        return;
    }
    let millis = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis();
    let safe_label: String = label
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    let path = format!("{dir}/{millis}_{safe_label}.png");
    if let Err(error) = std::fs::write(&path, png) {
        warn!(path, %error, "cannot save capture");
    }
}

fn encode_png(image: &RgbaImage) -> anyhow::Result<Vec<u8>> {
    let mut out = Cursor::new(Vec::new());
    image.write_to(&mut out, ImageFormat::Png)?;
    Ok(out.into_inner())
}

/// Runs the pipeline on saved captures and writes `<name>.overlay.png` and
/// `<name>.composite.png` (or `<name>.boxes.png` with `--ocr-only`).
async fn replay(app: Arc<App>, mut args: Vec<String>) -> anyhow::Result<()> {
    let lang = take_option(&mut args, "--lang").unwrap_or_else(|| app.config.default_target_lang.clone());
    let forced_label = take_option(&mut args, "--label");
    let out_dir = PathBuf::from(take_option(&mut args, "--out").unwrap_or_else(|| "replay".into()));
    let ocr_only = take_flag(&mut args, "--ocr-only");
    std::fs::create_dir_all(&out_dir)?;

    for path in args.iter().map(PathBuf::from) {
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("capture").to_owned();
        // Captures are named `<millis>_<label>.png`.
        let label = forced_label.clone().unwrap_or_else(|| stem.split_once('_').map_or(stem.clone(), |(_, l)| l.into()));
        let frame = image::open(&path).with_context(|| format!("cannot open {}", path.display()))?;
        println!("== {} ({label})", path.display());

        let outcome = app.run(&frame, &label, &lang, ocr_only).await?;
        for (i, block) in outcome.blocks.iter().enumerate() {
            println!("  [{i}] {}", block.joined());
            if let Some(translation) = outcome.translations.get(i) {
                println!("      → {translation}");
            }
        }
        if ocr_only {
            let boxes = render::draw_boxes(&frame, &outcome.blocks, app.config.render.width);
            boxes.save(out_dir.join(format!("{stem}.boxes.png")))?;
        } else if let Some(overlay) = &outcome.overlay {
            overlay.save(out_dir.join(format!("{stem}.overlay.png")))?;
            let layout = app.renderer.layout(frame.width(), frame.height());
            render::composite(&frame, overlay, layout).save(out_dir.join(format!("{stem}.composite.png")))?;
        }
    }
    Ok(())
}

#[derive(serde::Deserialize)]
struct BenchFile {
    label: String,
    case: Vec<BenchCase>,
}

#[derive(serde::Deserialize)]
struct BenchCase {
    #[serde(default)]
    note: String,
    blocks: Vec<String>,
}

/// Translates the cases of a benchmark file in order (so history builds up as
/// in a play session) and prints each translation with its latency.
async fn bench(app: Arc<App>, mut args: Vec<String>) -> anyhow::Result<()> {
    let lang = take_option(&mut args, "--lang").unwrap_or_else(|| app.config.default_target_lang.clone());
    let path = args.first().context("missing benchmark file")?;
    let file: BenchFile = toml::from_str(&std::fs::read_to_string(path)?)?;
    let profile = app.config.profile_for(&file.label);
    println!("model: {} ({})", app.config.translate.backend.model, app.config.translate.backend.url);

    let mut total_ms = 0;
    for case in &file.case {
        let started = Instant::now();
        let (translations, _) = app
            .translator
            .translate(&app.client, &app.config.translate, profile, &file.label, &lang, &case.blocks)
            .await?;
        let ms = started.elapsed().as_millis();
        total_ms += ms;
        println!("\n# {} ({ms} ms)", case.note);
        for (ja, tr) in case.blocks.iter().zip(&translations) {
            println!("  {ja}\n    → {tr}");
        }
    }
    println!("\ntotal: {total_ms} ms for {} cases", file.case.len());
    Ok(())
}
