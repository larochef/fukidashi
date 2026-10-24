//! Prebuilt llama.cpp builds: picks the one that fits the GPU, downloads it
//! from GitHub releases (checksum verified) into the cache, and keeps it up to
//! date: a newer build is installed in the background and used from the next
//! start, the previous one staying as a fallback if the new one fails to start.
//!
//! Layout, under `~/.cache/game-translator/runtimes`:
//!   <variant>-<tag>/   unpacked build (llama-server and its libraries)
//!   state.json         current / previous / pending build of each variant
//!   downloads/         archives in flight (deleted once unpacked)

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::LazyLock,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, bail};
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;
use tracing::{info, warn};

use crate::config::{ServerConfig, Update, cache_dir};

/// How often the releases are checked for a newer build.
const CHECK_INTERVAL: Duration = Duration::from_secs(24 * 3600);
/// Interval between two progress lines while downloading.
const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);

/// A family of prebuilt builds, from one repository.
#[derive(Debug, Clone, PartialEq)]
pub enum Variant {
    /// halo-box/strix-llama.cpp, ROCm for one AMD GPU target, libraries included.
    Strix(String),
    Vulkan,
    /// CUDA 12 or 13, with the CUDA runtime libraries (only the driver is needed).
    Cuda(u32),
    Cpu,
}

impl Variant {
    pub fn name(&self) -> String {
        match self {
            Variant::Strix(gfx) => format!("strix-rocm-{gfx}"),
            Variant::Vulkan => "vulkan".into(),
            Variant::Cuda(major) => format!("cuda{major}"),
            Variant::Cpu => "cpu".into(),
        }
    }

    /// The key of `[translate.server] models` that applies to this build.
    pub fn family(&self) -> &'static str {
        match self {
            Variant::Strix(_) => "strix",
            Variant::Vulkan => "vulkan",
            Variant::Cuda(_) => "cuda",
            Variant::Cpu => "cpu",
        }
    }

    fn repository(&self) -> &'static str {
        match self {
            Variant::Strix(_) => "halo-box/strix-llama.cpp",
            _ => "ggml-org/llama.cpp",
        }
    }

    /// Names of the archives that make up the build, all required.
    fn archives(&self) -> Vec<Regex> {
        let pattern = |p: String| Regex::new(&format!("^{p}$")).expect("valid pattern");
        match self {
            Variant::Strix(gfx) => vec![pattern(format!(r"llama-b\d+-ubuntu-rocm-{gfx}-x64\.zip"))],
            Variant::Vulkan => vec![pattern(r"llama-b\d+-bin-ubuntu-vulkan-x64\.tar\.gz".into())],
            Variant::Cuda(major) => vec![
                pattern(format!(r"llama-b\d+-bin-ubuntu-cuda-{major}\.[\d.]+-x64\.tar\.gz")),
                pattern(format!(r"cudart-llama-b\d+-bin-ubuntu-cuda-{major}\.[\d.]+-x64\.tar\.gz")),
            ],
            Variant::Cpu => vec![pattern(r"llama-b\d+-bin-ubuntu-x64\.tar\.gz".into())],
        }
    }
}

/// The GPUs of this machine, as far as picking a build goes.
#[derive(Debug, Default)]
struct Gpus {
    /// AMD GPU targets (gfx1151), from the ROCm kernel driver.
    amd: Vec<String>,
    /// NVIDIA driver version (major, minor).
    nvidia: Option<(u32, u32)>,
}

impl Gpus {
    fn detect() -> Self {
        let amd = std::fs::read_dir("/sys/class/kfd/kfd/topology/nodes")
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .filter_map(|node| std::fs::read_to_string(node.path().join("properties")).ok())
            .filter_map(|properties| {
                let version: u32 = properties.lines().find_map(|l| l.strip_prefix("gfx_target_version "))?.trim().parse().ok()?;
                // 110501 is gfx1151: major, then minor and stepping in hexadecimal. CPUs report 0.
                (version != 0).then(|| format!("gfx{}{:x}{:x}", version / 10000, version / 100 % 100, version % 100))
            })
            .collect();
        // "NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.82.07  ..."
        static DRIVER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"Kernel Module\s+(\d+)\.(\d+)").expect("valid pattern"));
        let nvidia = std::fs::read_to_string("/proc/driver/nvidia/version").ok().and_then(|text| {
            let caps = DRIVER.captures(&text)?;
            Some((caps[1].parse().ok()?, caps[2].parse().ok()?))
        });
        Gpus { amd, nvidia }
    }
}

/// The build for the `runtime` setting: "auto" picks from the GPUs found.
pub fn variant(setting: &str) -> anyhow::Result<Variant> {
    let gpus = Gpus::detect();
    let cuda = || match gpus.nvidia {
        // CUDA 13 needs a 580 driver, CUDA 12.8 a 570 one.
        Some((major, _)) if major >= 580 => Some(Variant::Cuda(13)),
        Some((major, _)) if major >= 570 => Some(Variant::Cuda(12)),
        _ => None,
    };
    let variant = match setting {
        "auto" => {
            if gpus.amd.iter().any(|gfx| gfx == "gfx1151") {
                Variant::Strix("gfx1151".into())
            } else if let Some(cuda) = cuda() {
                cuda
            } else if !gpus.amd.is_empty() || gpus.nvidia.is_some() || Path::new("/dev/dri").exists() {
                Variant::Vulkan
            } else {
                Variant::Cpu
            }
        }
        "strix" => Variant::Strix(gpus.amd.first().cloned().unwrap_or_else(|| "gfx1151".into())),
        "vulkan" => Variant::Vulkan,
        "cuda" => cuda().with_context(|| format!("runtime \"cuda\" needs an NVIDIA driver 570 or newer, found {:?}", gpus.nvidia))?,
        "cuda12" => Variant::Cuda(12),
        "cuda13" => Variant::Cuda(13),
        "cpu" => Variant::Cpu,
        other => bail!("unknown runtime \"{other}\": expected auto, strix, vulkan, cuda, cuda12, cuda13 or cpu"),
    };
    info!(?gpus, variant = variant.name(), "llama.cpp build");
    Ok(variant)
}

/// An installed build, ready to run.
#[derive(Debug, Clone)]
pub struct Build {
    /// `<variant>-<tag>`, the name of its directory.
    pub id: String,
    pub program: PathBuf,
    /// Where its libraries are (for `LD_LIBRARY_PATH`).
    pub library_dir: PathBuf,
}

/// The builds to try, newest first: the current one (installed now if there
/// is none yet), then the previous one, kept as a fallback.
pub async fn builds(client: &reqwest::Client, config: &ServerConfig, variant: &Variant) -> anyhow::Result<Vec<Build>> {
    let root = root();
    let mut state = State::load();
    let slot = state.variants.entry(variant.name()).or_default();

    if config.release != "latest" {
        let id = format!("{}-{}", variant.name(), config.release);
        if !root.join(&id).is_dir() {
            let release = github_release(client, variant.repository(), &config.release).await?;
            install(client, variant, &release).await?;
        }
        if slot.current.as_deref() != Some(id.as_str()) {
            slot.previous = slot.current.replace(id);
        }
    } else {
        // A build installed in the background during the previous run.
        if let Some(pending) = slot.pending.take().filter(|id| root.join(id).is_dir()) {
            info!(build = pending, "switching to the newer llama.cpp build");
            slot.previous = slot.current.replace(pending);
        }
        if slot.current.as_ref().is_none_or(|id| !root.join(id).is_dir()) {
            let release = latest_release(client, variant, &slot.failed).await?;
            slot.current = Some(install(client, variant, &release).await?);
            slot.checked_at = now();
        }
    }

    let keep: Vec<String> = [&slot.current, &slot.previous].into_iter().flatten().cloned().collect();
    prune(&root, variant, &keep);
    state.save()?;
    keep.iter().map(|id| locate(&root, id)).collect()
}

/// Records which build started: the newer ones tried before it failed where
/// it worked, so they are dropped and never installed again.
pub fn started(variant: &Variant, build: &Build, failed: &[&Build]) {
    let mut state = State::load();
    let slot = state.variants.entry(variant.name()).or_default();
    for build in failed {
        warn!(build = build.id, "this llama.cpp build failed to start, back to the previous one");
        slot.failed.push(build.id.clone());
    }
    if !failed.is_empty() {
        slot.current = Some(build.id.clone());
        slot.previous = None;
        prune(&root(), variant, std::slice::from_ref(&build.id));
    }
    if let Err(error) = state.save() {
        warn!("cannot save the llama.cpp build state: {error:#}");
    }
}

/// Looks for a newer build (at most once a day) and, in `auto` mode, installs
/// it in the background for the next start.
pub async fn check_for_update(client: reqwest::Client, config_update: Update, release: String, variant: Variant) {
    if config_update == Update::Off || release != "latest" {
        return;
    }
    let mut state = State::load();
    let slot = state.variants.entry(variant.name()).or_default();
    if now().saturating_sub(slot.checked_at) < CHECK_INTERVAL.as_secs() {
        return;
    }
    let failed = slot.failed.clone();
    let current = slot.pending.clone().or_else(|| slot.current.clone());
    let result = async {
        let release = latest_release(&client, &variant, &failed).await?;
        let id = format!("{}-{}", variant.name(), release.tag_name);
        if current.as_deref().is_some_and(|current| build_number(current) >= build_number(&id)) {
            info!(build = current, "llama.cpp is up to date");
            return Ok(None);
        }
        if config_update == Update::Notify {
            info!(
                build = id,
                "a newer llama.cpp build is available: set update = \"auto\" in [translate.server] to install it"
            );
            return Ok(None);
        }
        info!(build = id, "installing a newer llama.cpp build in the background");
        install(&client, &variant, &release).await.map(Some)
    }
    .await;

    let mut state = State::load();
    let slot = state.variants.entry(variant.name()).or_default();
    slot.checked_at = now();
    match result {
        Ok(Some(id)) => {
            info!(build = id, "newer llama.cpp build installed, used from the next start");
            slot.pending = Some(id);
        }
        Ok(None) => {}
        Err(error) => warn!("cannot update llama.cpp: {error:#}"),
    }
    if let Err(error) = state.save() {
        warn!("cannot save the llama.cpp build state: {error:#}");
    }
}

fn root() -> PathBuf {
    cache_dir().join("runtimes")
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// 11856 for `vulkan-b11856`.
fn build_number(id: &str) -> u64 {
    id.rsplit_once("-b").and_then(|(_, n)| n.parse().ok()).unwrap_or(0)
}

#[derive(Default, Serialize, Deserialize)]
struct State {
    #[serde(default)]
    variants: BTreeMap<String, Slot>,
}

#[derive(Default, Serialize, Deserialize)]
struct Slot {
    current: Option<String>,
    previous: Option<String>,
    /// Installed in the background, used from the next start.
    pending: Option<String>,
    /// Builds that failed to start: never installed again.
    #[serde(default)]
    failed: Vec<String>,
    /// Last look for a newer build (Unix time).
    #[serde(default)]
    checked_at: u64,
}

impl State {
    fn path() -> PathBuf {
        root().join("state.json")
    }

    fn load() -> Self {
        std::fs::read_to_string(Self::path()).ok().and_then(|text| serde_json::from_str(&text).ok()).unwrap_or_default()
    }

    fn save(&self) -> anyhow::Result<()> {
        std::fs::create_dir_all(root())?;
        let tmp = Self::path().with_extension("tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(self)?)?;
        std::fs::rename(tmp, Self::path())?;
        Ok(())
    }
}

/// Deletes the builds of this variant that are not kept.
fn prune(root: &Path, variant: &Variant, keep: &[String]) {
    let prefix = format!("{}-b", variant.name());
    let entries = std::fs::read_dir(root).into_iter().flatten().filter_map(Result::ok);
    for entry in entries {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with(&prefix) && !keep.contains(&name) && entry.path().is_dir() {
            info!(build = name, "removing an old llama.cpp build");
            if let Err(error) = std::fs::remove_dir_all(entry.path()) {
                warn!("cannot remove {}: {error}", entry.path().display());
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct Release {
    tag_name: String,
    assets: Vec<Asset>,
}

#[derive(Debug, Deserialize)]
struct Asset {
    name: String,
    size: u64,
    browser_download_url: String,
    /// "sha256:<hex>".
    digest: Option<String>,
}

impl Release {
    /// The archives of the build, or `None` if this release lacks one.
    fn archives(&self, variant: &Variant) -> Option<Vec<&Asset>> {
        variant.archives().iter().map(|pattern| self.assets.iter().find(|a| pattern.is_match(&a.name))).collect()
    }
}

async fn github<T: serde::de::DeserializeOwned>(client: &reqwest::Client, path: &str) -> anyhow::Result<T> {
    let mut request = client
        .get(format!("https://api.github.com/repos/{path}"))
        .header("Accept", "application/vnd.github+json")
        .header("User-Agent", "game-translator");
    // Optional: lifts the limit of 60 requests an hour.
    if let Ok(token) = std::env::var("GITHUB_TOKEN") {
        request = request.bearer_auth(token);
    }
    let response = request.send().await?.error_for_status().with_context(|| format!("GitHub API {path}"))?;
    Ok(response.json().await?)
}

async fn github_release(client: &reqwest::Client, repository: &str, tag: &str) -> anyhow::Result<Release> {
    github(client, &format!("{repository}/releases/tags/{tag}")).await
}

/// The newest release that has every archive of the build, skipping the
/// builds that failed to start here.
async fn latest_release(client: &reqwest::Client, variant: &Variant, failed: &[String]) -> anyhow::Result<Release> {
    let releases: Vec<Release> = github(client, &format!("{}/releases?per_page=20", variant.repository())).await?;
    releases
        .into_iter()
        .filter(|r| !failed.contains(&format!("{}-{}", variant.name(), r.tag_name)))
        .find(|r| r.archives(variant).is_some())
        .with_context(|| format!("no recent {} release has a {} build", variant.repository(), variant.name()))
}

/// Downloads, unpacks and checks a build; returns its id.
async fn install(client: &reqwest::Client, variant: &Variant, release: &Release) -> anyhow::Result<String> {
    let id = format!("{}-{}", variant.name(), release.tag_name);
    let root = root();
    // Already here (kept as the previous build): nothing to download.
    if let Ok(build) = locate(&root, &id)
        && validate(&build).await.is_ok()
    {
        return Ok(id);
    }
    let archives = release.archives(variant).with_context(|| format!("release {} has no {} build", release.tag_name, variant.name()))?;
    let downloads = root.join("downloads");
    tokio::fs::create_dir_all(&downloads).await?;

    let mut files = Vec::new();
    for asset in archives {
        let file = downloads.join(&asset.name);
        download(client, asset, &file).await?;
        files.push(file);
    }

    let target = root.join(&id);
    let staging = root.join(format!(".{id}.tmp"));
    {
        let (files, staging, target) = (files.clone(), staging.clone(), target.clone());
        tokio::task::spawn_blocking(move || unpack(&files, &staging, &target)).await??;
    }
    for file in &files {
        let _ = tokio::fs::remove_file(file).await;
    }

    let build = locate(&root, &id)?;
    match validate(&build).await {
        Ok(version) => {
            info!(build = id, version, "llama.cpp build installed");
            Ok(id)
        }
        Err(error) => {
            let _ = tokio::fs::remove_dir_all(&target).await;
            Err(error.context(format!("the {id} build does not run on this machine")))
        }
    }
}

/// Streams an archive to disk, logging the progress, and checks its SHA-256.
async fn download(client: &reqwest::Client, asset: &Asset, file: &Path) -> anyhow::Result<()> {
    let part = file.with_file_name(format!("{}.part", asset.name));
    let mut response = client
        .get(&asset.browser_download_url)
        .header("User-Agent", "game-translator")
        .send()
        .await?
        .error_for_status()
        .with_context(|| format!("cannot download {}", asset.name))?;
    let total = response.content_length().unwrap_or(asset.size).max(1);
    let mb = |bytes: u64| bytes as f64 / 1_048_576.0;
    info!(archive = asset.name, size = format!("{:.0} MB", mb(total)), "downloading llama.cpp");

    let mut out = tokio::fs::File::create(&part).await?;
    let mut hasher = Sha256::new();
    let (start, mut logged, mut done) = (Instant::now(), Instant::now(), 0u64);
    while let Some(chunk) = response.chunk().await? {
        hasher.update(&chunk);
        out.write_all(&chunk).await?;
        done += chunk.len() as u64;
        if logged.elapsed() >= PROGRESS_INTERVAL {
            logged = Instant::now();
            let speed = mb(done) / start.elapsed().as_secs_f64();
            let left = (total.saturating_sub(done)) as f64 / 1_048_576.0 / speed.max(0.01);
            info!(
                "downloading {}: {:.0} % ({:.0} / {:.0} MB, {speed:.1} MB/s, {left:.0} s left)",
                asset.name,
                100.0 * done as f64 / total as f64,
                mb(done),
                mb(total)
            );
        }
    }
    out.flush().await?;
    drop(out);

    let actual: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();
    match asset.digest.as_deref().and_then(|d| d.strip_prefix("sha256:")) {
        Some(expected) if expected != actual => {
            let _ = tokio::fs::remove_file(&part).await;
            bail!("{}: checksum mismatch (expected {expected}, got {actual})", asset.name);
        }
        Some(_) => {}
        None => warn!(archive = asset.name, "no checksum published, not verified"),
    }
    info!(archive = asset.name, seconds = start.elapsed().as_secs(), "downloaded, checksum verified");
    tokio::fs::rename(&part, file).await?;
    Ok(())
}

/// Unpacks the archives together into `target`, through a staging directory
/// so that an interrupted unpack never leaves a half build behind. An archive
/// holding a single top-level directory is unwrapped.
fn unpack(files: &[PathBuf], staging: &Path, target: &Path) -> anyhow::Result<()> {
    let _ = std::fs::remove_dir_all(staging);
    std::fs::create_dir_all(staging)?;
    for (i, file) in files.iter().enumerate() {
        info!(archive = %file.display(), "unpacking");
        let into = staging.join(format!(".archive{i}"));
        std::fs::create_dir_all(&into)?;
        let name = file.to_string_lossy();
        if name.ends_with(".zip") {
            zip::ZipArchive::new(std::fs::File::open(file)?)?.extract(&into)?;
        } else if name.ends_with(".tar.gz") {
            tar::Archive::new(flate2::read::GzDecoder::new(std::fs::File::open(file)?)).unpack(&into)?;
        } else {
            bail!("unknown archive format: {name}");
        }
        let entries: Vec<_> = std::fs::read_dir(&into)?.filter_map(Result::ok).collect();
        let content = match entries.as_slice() {
            [single] if single.path().is_dir() => single.path(),
            _ => into.clone(),
        };
        for entry in std::fs::read_dir(&content)?.filter_map(Result::ok) {
            std::fs::rename(entry.path(), staging.join(entry.file_name()))?;
        }
        std::fs::remove_dir_all(&into)?;
    }
    let _ = std::fs::remove_dir_all(target);
    std::fs::rename(staging, target)?;
    Ok(())
}

/// Finds `llama-server` in an unpacked build.
fn locate(root: &Path, id: &str) -> anyhow::Result<Build> {
    let dir = root.join(id);
    let program = ["llama-server", "bin/llama-server", "build/bin/llama-server"]
        .iter()
        .map(|p| dir.join(p))
        .find(|p| p.is_file())
        .with_context(|| format!("no llama-server in {}", dir.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755))?;
    }
    let library_dir = program.parent().unwrap_or(&dir).to_path_buf();
    Ok(Build { id: id.to_owned(), program, library_dir })
}

/// `LD_LIBRARY_PATH` for a build: its own libraries first.
pub fn library_path(build: &Build) -> String {
    let mut dirs = vec![build.library_dir.to_string_lossy().into_owned()];
    let lib = build.library_dir.join("lib");
    if lib.is_dir() {
        dirs.push(lib.to_string_lossy().into_owned());
    }
    if let Ok(existing) = std::env::var("LD_LIBRARY_PATH") {
        dirs.push(existing);
    }
    dirs.join(":")
}

/// Runs `llama-server --version`: the build is usable when its libraries
/// load. Returns the version it reports.
async fn validate(build: &Build) -> anyhow::Result<String> {
    let output = tokio::time::timeout(
        Duration::from_secs(60),
        tokio::process::Command::new(&build.program).arg("--version").env("LD_LIBRARY_PATH", library_path(build)).output(),
    )
    .await
    .context("llama-server --version timed out")??;
    let text = format!("{}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    static VERSION: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"version: (\S+) \(build (\d+)").expect("valid pattern"));
    match VERSION.captures(&text) {
        Some(caps) if output.status.success() => Ok(format!("{} (build {})", &caps[1], &caps[2])),
        _ => bail!("llama-server --version failed ({}): {}", output.status, text.trim()),
    }
}

#[cfg(test)]
mod tests {
    use super::{Asset, Release, Variant, build_number};

    fn release(names: &[&str]) -> Release {
        let assets = names
            .iter()
            .map(|n| Asset { name: (*n).into(), size: 0, browser_download_url: String::new(), digest: None })
            .collect();
        Release { tag_name: "b11856".into(), assets }
    }

    #[test]
    fn finds_the_archives_of_each_build() {
        let official = release(&[
            "llama-b11856-bin-ubuntu-x64.tar.gz",
            "llama-b11856-bin-ubuntu-vulkan-x64.tar.gz",
            "llama-b11856-bin-ubuntu-vulkan-arm64.tar.gz",
            "llama-b11856-bin-ubuntu-rocm-10.0-x64.tar.gz",
            "llama-b11856-bin-ubuntu-cuda-12.8-x64.tar.gz",
            "llama-b11856-bin-ubuntu-cuda-13.4-x64.tar.gz",
            "cudart-llama-b11856-bin-ubuntu-cuda-13.4-x64.tar.gz",
        ]);
        let names = |v: &Variant| official.archives(v).map(|a| a.iter().map(|a| a.name.as_str()).collect::<Vec<_>>());
        assert_eq!(names(&Variant::Vulkan), Some(vec!["llama-b11856-bin-ubuntu-vulkan-x64.tar.gz"]));
        assert_eq!(names(&Variant::Cpu), Some(vec!["llama-b11856-bin-ubuntu-x64.tar.gz"]));
        assert_eq!(
            names(&Variant::Cuda(13)),
            Some(vec!["llama-b11856-bin-ubuntu-cuda-13.4-x64.tar.gz", "cudart-llama-b11856-bin-ubuntu-cuda-13.4-x64.tar.gz"])
        );
        // CUDA 12 without its runtime libraries is not a usable build.
        assert_eq!(names(&Variant::Cuda(12)), None);

        let strix = release(&["llama-b1030-ubuntu-rocm-gfx1151-x64.zip", "llama-b1030-ubuntu-vulkan-x64.zip"]);
        assert!(strix.archives(&Variant::Strix("gfx1151".into())).is_some());
        assert!(strix.archives(&Variant::Strix("gfx1100".into())).is_none());
    }

    #[test]
    fn reads_build_numbers() {
        assert_eq!(build_number("vulkan-b11856"), 11856);
        assert_eq!(build_number("strix-rocm-gfx1151-b1030"), 1030);
        assert!(build_number("cuda13-b11856") > build_number("cuda13-b9999"));
    }
}
