//! Starts the translation model server (`llama-server`) when it isn't running:
//! model files come from the Hugging Face cache (downloaded once, shared with
//! other tools), and the server is stopped when the app exits.

use std::{path::PathBuf, process::Stdio, time::Duration};

use anyhow::{Context, bail};
use tokio::process::{Child, Command};
use tracing::{info, warn};

use crate::config::{ServerConfig, cache_dir, expand_home};

/// Keeps the server alive as long as it is held; killed on drop.
pub struct ManagedServer {
    _child: Child,
}

/// Makes sure something answers on the translation endpoint. Returns the
/// spawned server, or `None` when one was already running.
pub async fn ensure(client: &reqwest::Client, endpoint: &str, config: &ServerConfig) -> anyhow::Result<Option<ManagedServer>> {
    let health = health_url(endpoint)?;
    if is_up(client, &health).await {
        info!(%health, "translation server already running");
        return Ok(None);
    }

    let model = fetch(&config.model).await?;
    let draft = match &config.draft {
        Some(draft) => Some(fetch(draft).await?),
        None => None,
    };
    let port = reqwest::Url::parse(endpoint)?.port_or_known_default().context("no port in translate.url")?;

    // Speculative decoding needs a recent llama.cpp; fall back to plain decoding
    // if this build rejects the draft options.
    if let Some(draft) = &draft {
        match spawn(client, &health, config, port, &model, Some(draft)).await {
            Ok(server) => return Ok(Some(server)),
            Err(error) => warn!("{error:#}; retrying without the draft model"),
        }
    }
    spawn(client, &health, config, port, &model, None).await.map(Some)
}

async fn spawn(
    client: &reqwest::Client,
    health: &str,
    config: &ServerConfig,
    port: u16,
    model: &PathBuf,
    draft: Option<&PathBuf>,
) -> anyhow::Result<ManagedServer> {
    // LLAMA_SERVER lets each machine pick its build without touching the config.
    let program = std::env::var("LLAMA_SERVER").unwrap_or_else(|_| expand_home(&config.command));
    let log_path = cache_dir().join("llama-server.log");
    std::fs::create_dir_all(cache_dir())?;
    let log = std::fs::File::create(&log_path)?;

    let mut command = Command::new(&program);
    command.arg("-m").arg(model).args(["--host", "127.0.0.1", "--port", &port.to_string()]).args(&config.args);
    if let Some(draft) = draft {
        command.arg("-md").arg(draft).args(&config.draft_args);
    }
    let mut child = command
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log))
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("cannot start `{program}`: install llama.cpp or set LLAMA_SERVER"))?;
    info!(program, model = %model.display(), draft = draft.is_some(), log = %log_path.display(), "starting translation server");

    // Loading a ~20 GB model can take a while on first start.
    for _ in 0..600 {
        if let Some(status) = child.try_wait()? {
            bail!("translation server exited ({status}), see {}", log_path.display());
        }
        if is_up(client, health).await {
            info!("translation server ready");
            return Ok(ManagedServer { _child: child });
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
