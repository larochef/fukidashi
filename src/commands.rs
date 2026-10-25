//! Commands beside the server: `open` (what the menu entry runs), `stop`, and
//! `setup` to choose the engine without the settings page.

use std::{process::Stdio, time::Duration};

use anyhow::{Context, bail};
use serde_json::json;

use crate::{
    config::{Config, Provider, RemoteConfig, cache_dir},
    engine::{self, API_KEY_ENV, Engine},
    is_running, take_option,
};

/// The user service the packages install.
const SERVICE: &str = "fukidashi.service";

/// Whether the user service is installed.
fn has_service() -> bool {
    systemctl(&["cat", SERVICE]).is_ok_and(|ok| ok)
}

fn systemctl(args: &[&str]) -> anyhow::Result<bool> {
    let status = std::process::Command::new("systemctl")
        .arg("--user")
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .context("cannot run systemctl")?;
    Ok(status.success())
}

/// Opens the settings page, starting the app first if it is not running: as
/// the user service when installed, else in the background (its log in the
/// cache directory).
pub async fn open(url: &str, config: Option<&str>) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    if !is_running(&client, url).await {
        if has_service() {
            println!("Starting the Fukidashi service…");
            if !systemctl(&["start", SERVICE])? {
                bail!("cannot start {SERVICE}: see systemctl --user status fukidashi");
            }
        } else {
            let log_path = cache_dir().join("fukidashi.log");
            std::fs::create_dir_all(cache_dir())?;
            let log = std::fs::File::create(&log_path)?;
            println!("Starting Fukidashi (log: {})…", log_path.display());
            let mut command = std::process::Command::new(std::env::current_exe()?);
            if let Some(config) = config {
                command.args(["--config", config]);
            }
            use std::os::unix::process::CommandExt;
            // Its own process group: it keeps running once this command ends.
            command
                .arg("serve")
                .stdin(Stdio::null())
                .stdout(Stdio::from(log.try_clone()?))
                .stderr(Stdio::from(log))
                .process_group(0)
                .spawn()
                .context("cannot start fukidashi serve")?;
        }
        let mut waited = Duration::ZERO;
        while !is_running(&client, url).await {
            if waited >= Duration::from_secs(60) {
                bail!("Fukidashi did not answer at {url}");
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
            waited += Duration::from_millis(250);
        }
    }
    println!("{url}");
    let opened = std::process::Command::new("xdg-open").arg(url).stdout(Stdio::null()).stderr(Stdio::null()).status();
    if !opened.is_ok_and(|s| s.success()) {
        println!("Open this address in a browser.");
    }
    Ok(())
}

/// Stops the app: the user service if it runs as one, else through its API.
pub async fn stop(url: &str) -> anyhow::Result<()> {
    if systemctl(&["is-active", "--quiet", SERVICE]).is_ok_and(|ok| ok) {
        systemctl(&["stop", SERVICE])?;
        println!("Fukidashi service stopped.");
        return Ok(());
    }
    let client = reqwest::Client::new();
    if !is_running(&client, url).await {
        println!("Fukidashi is not running.");
        return Ok(());
    }
    client.post(format!("{url}api/quit")).send().await?.error_for_status()?;
    println!("Fukidashi stopped.");
    Ok(())
}

/// `setup local`: the local model, downloaded now. `setup remote URL MODEL
/// [--key KEY]`: an OpenAI-compatible provider. A running app applies the
/// choice at once (and downloads the model itself, its page showing the
/// progress); otherwise it is saved, and the model downloaded here.
pub async fn setup(url: &str, config: &Config, mut args: Vec<String>) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    let key = take_option(&mut args, "--key");
    let engine = Engine::new(client.clone(), &config.translate)?;
    let mut settings = engine.settings();
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["local"] => settings.provider = Provider::Local,
        ["remote", provider_url, model] => {
            settings.provider = Provider::Remote;
            settings.remote = RemoteConfig { url: (*provider_url).to_owned(), model: (*model).to_owned(), extra: settings.remote.extra.clone() };
            if key.is_none() && engine::api_key().is_none() {
                println!("No API key given (--key or {API_KEY_ENV}): fine for a server that needs none.");
            }
        }
        _ => bail!("usage: fukidashi setup local | fukidashi setup remote URL MODEL [--key KEY]"),
    }

    if is_running(&client, url).await {
        let mut body = serde_json::to_value(&settings)?;
        if let Some(key) = key {
            body["api_key"] = json!(key);
        }
        let response = client.put(format!("{url}api/engine")).json(&body).send().await?;
        if !response.status().is_success() {
            let error: serde_json::Value = response.json().await.unwrap_or_default();
            bail!("{}", error["error"].as_str().unwrap_or("the app refused these settings"));
        }
        match settings.provider {
            Provider::Local => println!("Saved. The running app downloads and loads the model now: progress at {url}"),
            Provider::Remote => println!("Saved and applied."),
        }
        return Ok(());
    }

    engine::save_choice(&settings, key)?;
    match settings.provider {
        Provider::Local => {
            println!("Saved. Downloading the local model now (once; it can take a while)…");
            Engine::new(client, &config.translate)?.download().await?;
            println!("Done: the model loads with the first translation.");
        }
        Provider::Remote => println!("Saved: used from the next start."),
    }
    Ok(())
}
