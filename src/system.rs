//! What the app can set up in the user's session, from its page and only when
//! asked there: starting with the session (the user service the packages
//! install) and the GNOME Shell extension.

use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::{Value, json};
use tokio::process::Command;

const SERVICE: &str = "fukidashi.service";
const EXTENSION: &str = "fukidashi@larochef.github.io";

/// The output of a command, `None` if it could not run or failed.
async fn output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().await.ok()?;
    output.status.success().then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Whether the session is GNOME (the extension only applies there). A user
/// service started early may lack the variable: the session's manager has it.
async fn is_gnome() -> bool {
    let gnome = |desktop: &str| desktop.split(':').any(|d| d.eq_ignore_ascii_case("GNOME"));
    if let Ok(desktop) = std::env::var("XDG_CURRENT_DESKTOP") {
        return gnome(&desktop);
    }
    output("systemctl", &["--user", "show-environment"])
        .await
        .is_some_and(|env| env.lines().any(|l| l.strip_prefix("XDG_CURRENT_DESKTOP=").is_some_and(gnome)))
}

/// `enabled`, `disabled`, or `none` when the user service is not installed
/// (the app runs from a terminal).
async fn service_state() -> &'static str {
    match Command::new("systemctl").args(["--user", "is-enabled", SERVICE]).output().await {
        Ok(output) => match String::from_utf8_lossy(&output.stdout).trim() {
            "enabled" => "enabled",
            "disabled" => "disabled",
            _ => "none",
        },
        Err(_) => "none",
    }
}

/// `~/.local/share` (`$XDG_DATA_HOME`).
fn user_data_dir() -> PathBuf {
    let app = crate::config::data_home();
    app.parent().map_or(app.clone(), Path::to_path_buf)
}

/// The directories GNOME Shell reads extensions from, the user's first.
fn extension_dirs() -> Vec<PathBuf> {
    let data_dirs = std::env::var("XDG_DATA_DIRS").ok().filter(|d| !d.is_empty());
    let data_dirs = data_dirs.as_deref().unwrap_or("/usr/local/share:/usr/share");
    std::iter::once(user_data_dir())
        .chain(data_dirs.split(':').filter(|d| !d.is_empty()).map(PathBuf::from))
        .map(|dir| dir.join("gnome-shell/extensions").join(EXTENSION))
        .collect()
}

/// The extension as `gnome-extensions info` tells it: enabled, its state
/// (`ACTIVE` once running), and whether it is a user extension, which GNOME's
/// `disable-user-extensions` turns off. GNOME Shell only looks for extensions
/// at login: one installed since is on disk but unknown to it (`loaded`
/// false) until the user logs in again.
async fn extension_state() -> Value {
    let Some(info) = output("gnome-extensions", &["info", EXTENSION]).await else {
        return json!({ "installed": extension_dirs().iter().any(|d| d.join("metadata.json").is_file()), "loaded": false });
    };
    let field = |name: &str| info.lines().find_map(|l| l.trim().strip_prefix(name).map(|v| v.trim().to_owned()));
    let user = field("Path:").is_some_and(|path| Path::new(&path).starts_with(user_data_dir()));
    let user_extensions_off = user
        && output("gsettings", &["get", "org.gnome.shell", "disable-user-extensions"]).await.is_some_and(|v| v.trim() == "true");
    json!({
        "installed": true,
        "loaded": true,
        "enabled": field("Enabled:").is_some_and(|v| v == "Yes"),
        "state": field("State:"),
        "user_extensions_off": user_extensions_off,
    })
}

pub async fn state() -> Value {
    let gnome = is_gnome().await;
    json!({
        "service": service_state().await,
        "gnome": gnome,
        "extension": if gnome { extension_state().await } else { Value::Null },
    })
}

#[derive(Deserialize)]
pub struct Change {
    /// Start the app with the session (enable the user service) or not.
    autostart: Option<bool>,
    /// Enable the GNOME Shell extension.
    extension: Option<bool>,
}

pub async fn apply(change: Change) -> anyhow::Result<()> {
    if let Some(on) = change.autostart {
        let action = if on { "enable" } else { "disable" };
        if output("systemctl", &["--user", action, SERVICE]).await.is_none() {
            anyhow::bail!("systemctl --user {action} {SERVICE} failed: is the service installed?");
        }
        tracing::info!(action, "user service");
    }
    if let Some(on) = change.extension {
        let action = if on { "enable" } else { "disable" };
        if output("gnome-extensions", &[action, EXTENSION]).await.is_none() {
            anyhow::bail!("gnome-extensions {action} {EXTENSION} failed: is the extension installed?");
        }
        tracing::info!(action, "GNOME extension");
    }
    Ok(())
}
