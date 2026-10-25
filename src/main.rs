//! RetroArch AI Service endpoint: OCR → translation → overlay, all local.
//!
//! Usage:
//!   game-translator [--config FILE] [serve]
//!   game-translator [--config FILE] replay [--lang CODE] [--label LABEL] [--ocr-only] [--out DIR] [--fixtures DIR] CAPTURE.png...
//!   game-translator [--config FILE] bench [--lang CODE] CASES.toml

mod config;
mod engine;
mod games;
mod llm;
mod memory;
mod ocr;
mod render;
mod runtime;
mod server;
mod settings;
mod translate;
mod web;

use std::{
    collections::HashMap,
    io::Cursor,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, RwLock},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, bail};
use axum::{
    Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Query, State},
    http::{StatusCode, header},
    routing::get,
};
use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use image::{DynamicImage, ImageFormat, RgbaImage};
use serde_json::{Value, json};
use tracing::{info, warn};

use crate::{
    config::Config,
    engine::Engine,
    ocr::{Block, Ocr},
    render::Renderer,
    translate::Translator,
};

/// RetroArch treats this exact message specially (it unpauses the game).
const NO_TEXT: &str = "No text found.";

struct App {
    config_path: PathBuf,
    /// `listen` and `data_dir` as the app started: changing them needs a restart.
    started_with: (String, String),
    /// Replaced as a whole when the settings change.
    live: RwLock<Arc<Live>>,
    client: reqwest::Client,
    /// The local model or a remote provider, switchable from the web page.
    engine: Arc<Engine>,
    translator: Translator,
    last_frame: Mutex<Option<LastFrame>>,
}

/// What the settings shape: a request works on one snapshot from start to end.
struct Live {
    config: Config,
    ocr: Arc<Ocr>,
    renderer: Renderer,
}

/// What a settings change needs beyond its hot application.
#[derive(Debug, Default, serde::Serialize)]
struct Reloaded {
    /// Settings read only at start (`listen`, `data_dir`).
    restart_app: Vec<&'static str>,
    /// The local server's settings changed: a server restart applies them.
    restart_server: bool,
}

impl App {
    fn live(&self) -> Arc<Live> {
        self.live.read().unwrap().clone()
    }

    /// Validates a new config text, applies it (rebuilding the OCR only if its
    /// settings changed) and, if it is new, saves it.
    async fn reload(&self, text: &str) -> anyhow::Result<Reloaded> {
        let config = Config::parse(text)?;
        let old = self.live();
        let ocr = if config.ocr == old.config.ocr {
            old.ocr.clone()
        } else {
            let ocr_config = config.ocr.clone();
            Arc::new(tokio::task::spawn_blocking(move || Ocr::new(&ocr_config)).await??)
        };
        let renderer = Renderer::new(&config.render).context("cannot load font")?;

        let before = std::fs::read_to_string(&self.config_path).unwrap_or_default();
        if before != text {
            let tmp = self.config_path.with_extension("toml.tmp");
            std::fs::write(&tmp, text)?;
            std::fs::rename(&tmp, &self.config_path).with_context(|| format!("cannot write {}", self.config_path.display()))?;
        }

        let mut reloaded = Reloaded::default();
        if config.listen != self.started_with.0 {
            reloaded.restart_app.push("listen");
        }
        if config.data_dir != self.started_with.1 {
            reloaded.restart_app.push("data_dir");
        }

        self.engine.set_config(&config.translate);
        reloaded.restart_server = self.engine.needs_restart();
        *self.live.write().unwrap() = Arc::new(Live { config, ocr, renderer });
        // The overlay of the last screen may look different now.
        *self.last_frame.lock().unwrap() = None;
        info!(?reloaded, "settings applied");
        Ok(reloaded)
    }
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
        let live = self.live();
        let blocks = live.ocr.detect(&live.config.ocr, frame).await.context("OCR failed")?;
        let ocr_ms = started.elapsed().as_millis() as u64;
        if blocks.is_empty() || ocr_only {
            info!(ocr_ms, blocks = blocks.len(), "ocr done");
            return Ok(Outcome { blocks, translations: Vec::new(), overlay: None });
        }

        let profile = live.config.profile_for(label);
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
            .translate(&live.config.translate, &self.engine.backend(), profile, label, lang, &texts)
            .await
            .context("translation failed")?;
        let translate_ms = started.elapsed().as_millis() as u64 - ocr_ms;

        let items: Vec<(&Block, &str)> = blocks.iter().zip(translations.iter().map(String::as_str)).collect();
        let overlay = live.renderer.render(frame, &items);
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
        .with_writer(web::LogWriter)
        .init();

    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let config_path = take_option(&mut args, "--config").unwrap_or_else(|| "config.toml".into());
    let config = Config::load(Path::new(&config_path))?;
    // Only the server builds the games' persistent memory; replay and bench runs
    // must not mix test translations into it.
    let serving = matches!(args.first().map(String::as_str), None | Some("serve"));
    let data_dir = (serving && !config.data_dir.is_empty()).then(|| PathBuf::from(&config.data_dir));
    let client = reqwest::Client::new();
    let app = Arc::new(App {
        config_path: PathBuf::from(&config_path),
        started_with: (config.listen.clone(), config.data_dir.clone()),
        engine: Engine::new(client.clone(), &config.translate)?,
        translator: Translator::new(client.clone(), data_dir),
        client,
        last_frame: Mutex::new(None),
        live: RwLock::new(Arc::new(Live {
            renderer: Renderer::new(&config.render).context("cannot load font")?,
            ocr: Arc::new(Ocr::new(&config.ocr)?),
            config,
        })),
    });

    // The server starts in the background so that the page shows its progress
    // (first downloads); replay and bench need it before going on.
    let translates = !args.iter().any(|a| a == "--ocr-only");
    if serving {
        app.engine.start_in_background();
    } else if translates {
        app.engine.start().await;
        if let engine::Status::Failed(error) = app.engine.status() {
            bail!("translation server: {error}");
        }
    }

    let result = match args.first().map(String::as_str) {
        None | Some("serve") => serve(app.clone()).await,
        Some("replay") => replay(app.clone(), args.split_off(1)).await,
        Some("bench") => bench(app.clone(), args.split_off(1)).await,
        Some(other) => Err(anyhow::anyhow!("unknown command {other}")),
    };
    // Stops the server we started.
    app.engine.stop().await;
    result
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
    let listen = app.live().config.listen.clone();
    let router = Router::new()
        .route("/", get(web::page).post(handle_request))
        .merge(web::routes())
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024))
        .with_state(app);
    let listener = tokio::net::TcpListener::bind(&listen).await?;
    info!("listening on http://{listen}/ (open it in a browser for the settings)");
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
    let live = app.live();
    let lang = query.get("target_lang").map_or(live.config.default_target_lang.as_str(), String::as_str);

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
    save_capture(&live.config.capture_dir, label, &png);

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
    if let Err(error) = std::fs::create_dir_all(dir).and_then(|()| std::fs::write(&path, png)) {
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
    let lang = take_option(&mut args, "--lang").unwrap_or_else(|| app.live().config.default_target_lang.clone());
    let forced_label = take_option(&mut args, "--label");
    let out_dir = PathBuf::from(take_option(&mut args, "--out").unwrap_or_else(|| "replay".into()));
    let ocr_only = take_flag(&mut args, "--ocr-only");
    // Layout test cases: the blocks and translations of each capture.
    let fixtures = take_option(&mut args, "--fixtures").map(PathBuf::from);
    std::fs::create_dir_all(&out_dir)?;
    if let Some(dir) = &fixtures {
        std::fs::create_dir_all(dir)?;
    }

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
        if let Some(dir) = fixtures.as_ref().filter(|_| !ocr_only) {
            let fixture = render::Fixture {
                frame: [frame.width(), frame.height()],
                blocks: outcome
                    .blocks
                    .iter()
                    .zip(&outcome.translations)
                    .map(|(block, translation)| render::FixtureBlock { translation: translation.clone(), block: block.clone() })
                    .collect(),
            };
            let text = format!("# {}\n{}", path.display(), toml::to_string(&fixture)?);
            std::fs::write(dir.join(format!("{stem}.toml")), text)?;
        }
        if ocr_only {
            let boxes = render::draw_boxes(&frame, &outcome.blocks, app.live().config.render.width);
            boxes.save(out_dir.join(format!("{stem}.boxes.png")))?;
        } else if let Some(overlay) = &outcome.overlay {
            overlay.save(out_dir.join(format!("{stem}.overlay.png")))?;
            let layout = app.live().renderer.layout(frame.width(), frame.height());
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
    let lang = take_option(&mut args, "--lang").unwrap_or_else(|| app.live().config.default_target_lang.clone());
    let path = args.first().context("missing benchmark file")?;
    let file: BenchFile = toml::from_str(&std::fs::read_to_string(path)?)?;
    let live = app.live();
    let profile = live.config.profile_for(&file.label);
    let backend = app.engine.backend();
    println!("model: {} ({})", backend.model, backend.url);

    let mut total_ms = 0;
    for case in &file.case {
        let started = Instant::now();
        let (translations, _) = app
            .translator
            .translate(&live.config.translate, &app.engine.backend(), profile, &file.label, &lang, &case.blocks)
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
