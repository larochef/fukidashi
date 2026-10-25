//! What the app can set up in the user's session, from its page and only when
//! asked there: starting with the session (the user service the packages
//! install) and the GNOME Shell extension.

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

/// The extension as `gnome-extensions info` tells it: installed, enabled,
/// and its state (`ACTIVE` once loaded, which a new extension only is after
/// logging in again).
async fn extension_state() -> Value {
    let Some(info) = output("gnome-extensions", &["info", EXTENSION]).await else {
        return json!({ "installed": false });
    };
    let field = |name: &str| info.lines().find_map(|l| l.trim().strip_prefix(name).map(|v| v.trim().to_owned()));
    json!({
        "installed": true,
        "enabled": field("Enabled:").is_some_and(|v| v == "Yes"),
        "state": field("State:"),
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
