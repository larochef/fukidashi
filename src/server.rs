//! Starts the translation model server (`llama-server`) when it isn't running:
//! a prebuilt llama.cpp build fitting the GPU (see `runtime`) unless a program
//! is given, model files from the Hugging Face cache (downloaded once, shared
//! with other tools). The server is stopped when the app exits.

use std::{path::PathBuf, process::Stdio, time::Duration};

use anyhow::{Context, bail};
use tokio::process::{Child, Command};
use tracing::{info, warn};

use crate::{
    config::{ServerConfig, cache_dir, expand_home},
    runtime::{self, Build},
};

/// Keeps the server alive as long as it is held; killed on drop.
pub struct ManagedServer {
    _child: Child,
    /// The program or build that runs.
    pub program: String,
}

/// The output of the server listening on `port`.
pub fn log_path(port: u16) -> PathBuf {
    cache_dir().join(format!("llama-server-{port}.log"))
}

/// The port of the translation endpoint.
pub fn port(endpoint: &str) -> anyhow::Result<u16> {
    reqwest::Url::parse(endpoint)?.port_or_known_default().context("no port in translate.url")
}

/// Makes sure something answers on the translation endpoint. Returns the
/// spawned server, or `None` when one was already running.
pub async fn ensure(client: &reqwest::Client, endpoint: &str, config: &ServerConfig) -> anyhow::Result<Option<ManagedServer>> {
    let health = health_url(endpoint)?;
    if is_up(client, &health).await {
        info!(%health, "translation server already running");
        return Ok(None);
    }

    let port = port(endpoint)?;

    // LLAMA_SERVER lets each machine pick its program without touching the config.
    let explicit = std::env::var("LLAMA_SERVER").ok().or_else(|| config.command.as_deref().map(expand_home));
    let (variant, builds) = match explicit {
        Some(program) => (None, vec![Program::Path(program.into())]),
        None => {
            let variant = runtime::variant(&config.runtime)?;
            info!(gpus = ?runtime::Gpus::detect(), variant = variant.name(), "llama.cpp build");
            let builds = runtime::builds(client, config, &variant).await?;
            (Some(variant), builds.into_iter().map(Program::Build).collect())
        }
    };

    let spec = variant.as_ref().and_then(|v| config.models.get(v.family())).unwrap_or(&config.model);
    let model = fetch(spec).await?;
    let draft = match &config.draft {
        Some(draft) => Some(fetch(draft).await?),
        None => None,
    };

    // Newest build first; an older one is the fallback when a new one fails.
    let mut failed: Vec<&Build> = Vec::new();
    let mut last_error = None;
    for program in &builds {
        match start(client, &health, config, port, program, &model, draft.as_ref()).await {
            Ok(server) => {
                if let (Some(variant), Program::Build(build)) = (&variant, program) {
                    runtime::started(variant, build, &failed);
                    tokio::spawn(runtime::check_for_update(client.clone(), config.update, config.release.clone(), variant.clone(), false));
                }
                return Ok(Some(server));
            }
            Err(error) => {
                warn!("{error:#}");
                if let Program::Build(build) = program {
                    failed.push(build);
                }
                last_error = Some(error);
            }
        }
    }
    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("no llama.cpp build to start")))
}

/// What to run: a program given by the user, or a downloaded build.
enum Program {
    Path(PathBuf),
    Build(Build),
}

/// Starts the server, with speculative decoding when configured: it needs a
/// recent llama.cpp, so plain decoding is the fallback if the build rejects it.
async fn start(
    client: &reqwest::Client,
    health: &str,
    config: &ServerConfig,
    port: u16,
    program: &Program,
    model: &PathBuf,
    draft: Option<&PathBuf>,
) -> anyhow::Result<ManagedServer> {
    if let Some(draft) = draft {
        match spawn(client, health, config, port, program, model, Some(draft)).await {
            Ok(server) => return Ok(server),
            Err(error) => warn!("{error:#}; retrying without the draft model"),
        }
    }
    spawn(client, health, config, port, program, model, None).await
}

async fn spawn(
    client: &reqwest::Client,
    health: &str,
    config: &ServerConfig,
    port: u16,
    program: &Program,
    model: &PathBuf,
    draft: Option<&PathBuf>,
) -> anyhow::Result<ManagedServer> {
    let log_path = log_path(port);
    std::fs::create_dir_all(cache_dir())?;
    let log = std::fs::File::create(&log_path)?;

    let mut command = match program {
        Program::Path(path) => Command::new(path),
        Program::Build(build) => {
            let mut command = Command::new(&build.program);
            command.env("LD_LIBRARY_PATH", runtime::library_path(build));
            command
        }
    };
    let program = match program {
        Program::Path(path) => path.display().to_string(),
        Program::Build(build) => build.id.clone(),
    };
    command.arg("-m").arg(model).args(["--host", "127.0.0.1", "--port", &port.to_string()]).args(&config.args);
    if let Some(draft) = draft {
        command.arg("-md").arg(draft).args(&config.draft_args);
    }
    let mut child = command
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log))
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("cannot start `{program}`"))?;
    info!(program, model = %model.display(), draft = draft.is_some(), log = %log_path.display(), "starting translation server");

    // Loading a ~20 GB model can take a while on first start.
    for _ in 0..600 {
        if let Some(status) = child.try_wait()? {
            bail!("translation server exited ({status}), see {}", log_path.display());
        }
        if is_up(client, health).await {
            info!("translation server ready");
            return Ok(ManagedServer { _child: child, program });
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    bail!("translation server did not start within 5 minutes, see {}", log_path.display())
}

/// Resolves `owner/repo:path/in/repo` through the Hugging Face cache,
/// downloading it the first time.
async fn fetch(spec: &str) -> anyhow::Result<PathBuf> {
    let (repo, file) = spec.split_once(':').with_context(|| format!("expected `owner/repo:file`, got `{spec}`"))?;
    info!(repo, file, "resolving model file (downloads once into the Hugging Face cache)");
    let api = hf_hub::api::tokio::ApiBuilder::from_env().with_progress(true).build()?;
    api.model(repo.to_owned())
        .get(file)
        .await
        .with_context(|| format!("cannot get {file} from {repo}"))
}

fn health_url(endpoint: &str) -> anyhow::Result<String> {
    let mut url = reqwest::Url::parse(endpoint)?;
    url.set_path("/health");
    Ok(url.to_string())
}

async fn is_up(client: &reqwest::Client, health: &str) -> bool {
    client.get(health).timeout(Duration::from_secs(2)).send().await.is_ok_and(|r| r.status().is_success())
}
