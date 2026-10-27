//! The games page's API: every game met (its saved memory) or set up (a
//! profile), with its translations, learned names and profile to fix by hand.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::Arc,
    time::UNIX_EPOCH,
};

use anyhow::{Context, bail};
use axum::{
    Json, Router,
    extract::{Path as UrlPath, Query, State},
    routing::{get, post, put},
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    App,
    config::{self, Profile},
    memory::{self, Entry, safe_name},
    settings::{self, Edit},
    web::ApiError,
};

/// Prefix of the id of a game known only from its profile.
const PROFILE_ID: &str = "profile:";

pub fn routes() -> Router<Arc<App>> {
    Router::new()
        .route("/api/games", get(list))
        .route("/api/games/{id}", get(detail))
        .route("/api/games/{id}/memory", get(translations).put(correct))
        .route("/api/games/{id}/memory/delete", post(forget))
        .route("/api/games/{id}/names", put(save_names))
        .route("/api/games/{id}/profile", put(save_profile))
}

/// A game's name as shown: RetroArch's label is `system__file name`, the
/// file name often ending with a region and release tags.
#[derive(Debug, PartialEq)]
struct Title {
    system: String,
    name: String,
    region: Option<&'static str>,
}

/// "playstation2__KimiKiss (Japan)" → PlayStation 2, "KimiKiss", Japan;
/// "playstation_portable__Tokimeki_Memorial_4_JPN_PSP-Caravan" → PSP,
/// "Tokimeki Memorial 4", Japan (what follows the region is a release tag).
/// Directory names, with `_` for spaces and brackets, read the same.
fn title(label: &str) -> Title {
    let (system, rest) = label.split_once("__").unwrap_or(("", label));
    // Brackets hold tags; in a bare file name, a region word ends the title.
    let (name, tags) = rest.split_at(rest.find(['(', '[']).unwrap_or(rest.len()));
    let split = |text: &str| -> Vec<String> {
        text.split(|c: char| c == '_' || c.is_whitespace() || "()[],".contains(c)).filter(|w| !w.is_empty()).map(str::to_owned).collect()
    };
    let region = |word: &str, tagged: bool| match word {
        "Japan" | "JPN" | "JAP" => Some("Japan"),
        "USA" => Some("USA"),
        "Europe" | "EUR" => Some("Europe"),
        "JP" | "J" if tagged => Some("Japan"),
        "US" | "U" if tagged => Some("USA"),
        "EU" | "E" if tagged => Some("Europe"),
        "World" if tagged => Some("World"),
        _ => None,
    };
    let name_words = split(name);
    // A region word only counts after the title's first word.
    let cut = name_words.iter().skip(1).position(|w| region(w, false).is_some()).map(|i| i + 1);
    let region = match cut {
        Some(i) => region(&name_words[i], false),
        None => split(tags).iter().find_map(|w| region(w, true)),
    };
    let name = name_words[..cut.unwrap_or(name_words.len())].join(" ");
    let system = match system {
        "playstation" => "PlayStation".to_owned(),
        "playstation2" => "PlayStation 2".to_owned(),
        "playstation_portable" => "PSP".to_owned(),
        other => other.replace('_', " "),
    };
    Title { system, name, region }
}

/// RetroArch's label of a game, recorded in its directory; the directory name
/// itself (spaces and punctuation as `_`) for memories older than that.
fn label(app: &App, dir: &str) -> String {
    app.translator.game_dir(dir).and_then(|path| memory::label_of(&path)).unwrap_or_else(|| dir.to_owned())
}

/// The profile that applies to a game, matched like the translator does.
fn profile_for<'a>(profiles: &'a [Profile], label: &str) -> Option<&'a Profile> {
    profiles.iter().find(|p| p.matches(label))
}

/// Games by directory name: those with a memory, and those with captures
/// (`<time>_<label>.png`), played before the memory existed or with it off.
fn game_dirs(data_dir: Option<&Path>, capture_dir: &str) -> BTreeSet<String> {
    let names = |dir: &Path| -> Vec<(String, bool)> {
        std::fs::read_dir(dir)
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .map(|e| (e.file_name().to_string_lossy().into_owned(), e.path().is_dir()))
            .collect()
    };
    let mut games: BTreeSet<String> = data_dir.map(names).unwrap_or_default().into_iter().filter(|(_, d)| *d).map(|(n, _)| n).collect();
    if !capture_dir.is_empty() {
        for (file, _) in names(Path::new(capture_dir)) {
            let label = file.strip_suffix(".png").and_then(|f| f.split_once('_')).filter(|(time, _)| time.chars().all(|c| c.is_ascii_digit()));
            if let Some((_, label)) = label {
                games.insert(safe_name(label));
            }
        }
    }
    games
}

async fn list(State(app): State<Arc<App>>) -> Json<Value> {
    let live = app.live();
    let profiles = &live.config.profiles;
    let mut games = Vec::new();
    let mut with_games = Vec::new();
    for dir in game_dirs(app.translator.data_dir(), &live.config.capture_dir) {
        let path = app.translator.game_dir(&dir).unwrap_or_default();
        let entries = memory::entries(&path);
        let mut langs: BTreeMap<String, usize> = BTreeMap::new();
        for entry in &entries {
            *langs.entry(entry.lang.clone()).or_default() += 1;
        }
        let updated = std::fs::metadata(path.join("memory.jsonl"))
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_secs());
        let label = label(&app, &dir);
        let profile = profile_for(profiles, &label);
        if let Some(p) = profile {
            with_games.push(p.name.clone());
        }
        let title = title(&label);
        games.push(json!({
            "id": dir,
            "system": title.system,
            "name": title.name,
            "region": title.region,
            "translations": entries.len(),
            "langs": langs,
            "names": memory::names(&path).len(),
            "profile": profile.map(|p| &p.name),
            "updated": updated,
        }));
    }
    // Profiles of games not played yet.
    for profile in profiles.iter().filter(|p| !with_games.contains(&p.name)) {
        let stem = profile.name.trim_end_matches(".toml");
        games.push(json!({
            "id": format!("{PROFILE_ID}{stem}"),
            "system": "",
            "name": profile.pattern.replace('_', " "),
            "translations": 0,
            "langs": {},
            "names": 0,
            "profile": profile.name,
            "updated": 0,
        }));
    }
    Json(json!({ "games": games, "memory": app.translator.data_dir().is_some() }))
}

/// The game's directory name, or `None` for a game known only from its profile.
fn memory_dir(id: &str) -> Option<&str> {
    (!id.starts_with(PROFILE_ID)).then_some(id)
}

/// The profile of a game, if it has one.
fn game_profile(app: &App, id: &str) -> Option<Profile> {
    let live = app.live();
    match id.strip_prefix(PROFILE_ID) {
        Some(stem) => live.config.profiles.iter().find(|p| p.name.trim_end_matches(".toml") == stem).cloned(),
        None => profile_for(&live.config.profiles, &label(app, id)).cloned(),
    }
}

/// The profile as written in its file (without the styles merged in).
fn read_profile(name: &str, text: &str) -> anyhow::Result<Value> {
    let values = settings::to_json(text)?;
    Ok(json!({
        "file": name,
        "match": values.get("match").cloned().unwrap_or(Value::Null),
        "styles": values.get("styles").cloned().unwrap_or(json!([])),
        "instructions": values.get("instructions").cloned().unwrap_or(json!("")),
        "glossary": values.get("glossary").cloned().unwrap_or(json!({})),
        "ocr": values.get("ocr").cloned().unwrap_or(json!({})),
    }))
}

async fn detail(State(app): State<Arc<App>>, UrlPath(id): UrlPath<String>) -> Result<Json<Value>, ApiError> {
    let title = match memory_dir(&id) {
        Some(dir) => title(&label(&app, dir)),
        None => Title { system: String::new(), name: id[PROFILE_ID.len()..].replace(['_', '-'], " "), region: None },
    };
    let names = memory_dir(&id).and_then(|dir| app.translator.game_dir(dir)).map(|path| memory::names(&path)).unwrap_or_default();
    let profile = game_profile(&app, &id).map(|p| read_profile(&p.name, &p.text)).transpose()?;
    Ok(Json(json!({
        "id": id,
        "system": title.system,
        "name": title.name,
        "region": title.region,
        "memory": memory_dir(&id).is_some() && app.translator.data_dir().is_some(),
        "names": names,
        "profile": profile,
        "styles": config::style_names(),
    })))
}

#[derive(Deserialize)]
struct MemoryQuery {
    lang: Option<String>,
    q: Option<String>,
    #[serde(default)]
    offset: usize,
    limit: Option<usize>,
}

/// A page of the game's translations, newest first, filtered by language and
/// by text (Japanese or translation).
async fn translations(
    State(app): State<Arc<App>>,
    UrlPath(id): UrlPath<String>,
    Query(query): Query<MemoryQuery>,
) -> Result<Json<Value>, ApiError> {
    let path = memory_dir(&id).and_then(|dir| app.translator.game_dir(dir)).context("no memory for this game")?;
    let needle = query.q.as_deref().map(str::trim).filter(|q| !q.is_empty()).map(str::to_lowercase);
    let entries: Vec<Entry> = memory::entries(&path)
        .into_iter()
        .rev()
        .filter(|e| query.lang.as_deref().is_none_or(|lang| lang.is_empty() || e.lang == lang))
        .filter(|e| needle.as_ref().is_none_or(|n| e.ja.to_lowercase().contains(n) || e.tr.to_lowercase().contains(n)))
        .collect();
    let limit = query.limit.unwrap_or(50).min(500);
    let page: Vec<&Entry> = entries.iter().skip(query.offset).take(limit).collect();
    Ok(Json(json!({ "total": entries.len(), "offset": query.offset, "entries": page })))
}

/// Replaces a translation: the next time the text shows, this one is used.
async fn correct(State(app): State<Arc<App>>, UrlPath(id): UrlPath<String>, Json(entry): Json<Entry>) -> Result<Json<Value>, ApiError> {
    let dir = memory_dir(&id).context("no memory for this game")?;
    if entry.tr.trim().is_empty() {
        return Err(anyhow::anyhow!("the translation is empty: delete it instead").into());
    }
    app.translator
        .edit_game(dir, |path| {
            let mut entries = memory::entries(path);
            match entries.iter_mut().find(|e| e.lang == entry.lang && e.ja == entry.ja) {
                Some(found) => {
                    found.tr = entry.tr.trim().to_owned();
                    found.by = Some(memory::BY_USER.into());
                }
                None => bail!("this translation is not in the memory any more"),
            }
            memory::save_entries(path, &entries).context("cannot save the memory")
        })
        .context("memory disabled")??;
    tracing::info!(game = dir, ja = entry.ja, tr = entry.tr, "translation corrected");
    Ok(Json(json!({ "ok": true })))
}

#[derive(Deserialize)]
struct Key {
    lang: String,
    ja: String,
}

#[derive(Deserialize)]
struct Forget {
    entries: Vec<Key>,
}

/// Deletes translations: their text is translated again next time it shows.
async fn forget(State(app): State<Arc<App>>, UrlPath(id): UrlPath<String>, Json(body): Json<Forget>) -> Result<Json<Value>, ApiError> {
    let dir = memory_dir(&id).context("no memory for this game")?;
    let removed = app
        .translator
        .edit_game(dir, |path| {
            let mut entries = memory::entries(path);
            let before = entries.len();
            entries.retain(|e| !body.entries.iter().any(|k| k.lang == e.lang && k.ja == e.ja));
            memory::save_entries(path, &entries).context("cannot save the memory")?;
            anyhow::Ok(before - entries.len())
        })
        .context("memory disabled")??;
    tracing::info!(game = dir, removed, "translations deleted");
    Ok(Json(json!({ "removed": removed })))
}

#[derive(Deserialize)]
struct Names {
    names: BTreeMap<String, String>,
}

/// Replaces the learned names (a reading fixed, a wrong name removed).
async fn save_names(State(app): State<Arc<App>>, UrlPath(id): UrlPath<String>, Json(body): Json<Names>) -> Result<Json<Value>, ApiError> {
    let dir = memory_dir(&id).context("no memory for this game")?;
    let names: BTreeMap<String, String> = body
        .names
        .into_iter()
        .map(|(ja, romaji)| (ja.trim().to_owned(), romaji.trim().to_owned()))
        .filter(|(ja, romaji)| !ja.is_empty() && !romaji.is_empty())
        .collect();
    app.translator.edit_game(dir, |path| memory::save_names(path, &names)).context("memory disabled")?.context("cannot save the names")?;
    tracing::info!(game = dir, names = names.len(), "learned names saved");
    Ok(Json(json!({ "names": names })))
}

#[derive(Deserialize)]
struct ProfileUpdate {
    #[serde(rename = "match")]
    pattern: String,
    #[serde(default)]
    styles: Vec<String>,
    #[serde(default)]
    instructions: String,
    #[serde(default)]
    glossary: BTreeMap<String, String>,
    #[serde(default)]
    ocr: BTreeMap<String, String>,
}

/// Writes the game's profile into the user's profiles (a new file if it has
/// none; comments kept otherwise; a shipped profile is copied there, edited)
/// and applies it.
async fn save_profile(
    State(app): State<Arc<App>>,
    UrlPath(id): UrlPath<String>,
    Json(update): Json<ProfileUpdate>,
) -> Result<Json<Value>, ApiError> {
    let pattern = update.pattern.trim();
    if pattern.is_empty() {
        return Err(anyhow::anyhow!("the profile needs a match: part of the game's name as RetroArch gives it").into());
    }
    if let Some(dir) = memory_dir(&id) {
        let label = label(&app, dir);
        if profile_for(std::slice::from_ref(&Profile { pattern: pattern.to_owned(), ..Profile::default() }), &label).is_none() {
            return Err(anyhow::anyhow!("\"{pattern}\" is not part of this game's name ({label})").into());
        }
    }
    let clean = |map: BTreeMap<String, String>| -> serde_json::Map<String, Value> {
        map.into_iter()
            .map(|(k, v)| (k.trim().to_owned(), v.trim().to_owned()))
            .filter(|(k, v)| !k.is_empty() && !v.is_empty())
            .map(|(k, v)| (k, Value::String(v)))
            .collect()
    };
    let glossary = clean(update.glossary);
    let ocr = clean(update.ocr);
    let instructions = update.instructions.trim();
    let edits = [
        Edit { key: "match".into(), value: json!(pattern) },
        Edit { key: "styles".into(), value: if update.styles.is_empty() { Value::Null } else { json!(update.styles) } },
        Edit {
            key: "instructions".into(),
            value: if instructions.is_empty() { Value::Null } else { json!(format!("{instructions}\n")) },
        },
        Edit { key: "glossary".into(), value: if glossary.is_empty() { Value::Null } else { Value::Object(glossary) } },
        Edit { key: "ocr".into(), value: if ocr.is_empty() { Value::Null } else { Value::Object(ocr) } },
    ];

    let (name, text) = match game_profile(&app, &id) {
        Some(profile) => (profile.name, profile.text),
        None => {
            let stem: String = pattern
                .to_lowercase()
                .chars()
                .map(|c| if c.is_alphanumeric() { c } else { '-' })
                .collect::<String>()
                .split('-')
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join("-");
            let name = format!("{stem}.toml");
            if app.live().config.profiles.iter().any(|p| p.name == name) {
                return Err(anyhow::anyhow!("a profile {name} already exists").into());
            }
            let header = format!(
                // A key to hold the comment, else it ends up below what is added.
                "# Profile for {}: applied when RetroArch's label (`system__game name`)\n# contains `match`. See tokimeki-memorial-4.toml.\n\nmatch = \"\"\n",
                memory_dir(&id).map_or(pattern.to_owned(), |dir| title(&label(&app, dir)).name)
            );
            (name, header)
        }
    };
    let edited = settings::apply(&text, &edits)?;
    // Checked as a profile before it replaces the file.
    toml::from_str::<Profile>(&edited).context("invalid profile")?;
    let dir = config::user_profiles_dir();
    std::fs::create_dir_all(&dir).map_err(anyhow::Error::from)?;
    let file = dir.join(&name);
    let tmp = file.with_extension("toml.tmp");
    std::fs::write(&tmp, &edited).map_err(anyhow::Error::from)?;
    std::fs::rename(&tmp, &file).map_err(anyhow::Error::from)?;

    // Profiles are read with the config.
    app.reload(&config::user_text(&app.config_path)?).await?;
    tracing::info!(file = %file.display(), "profile saved");
    Ok(Json(json!({ "profile": read_profile(&name, &edited)? })))
}

#[cfg(test)]
mod tests {
    use super::{Title, title};

    fn t(system: &str, name: &str, region: Option<&'static str>) -> Title {
        Title { system: system.into(), name: name.into(), region }
    }

    #[test]
    fn names_games_from_their_label() {
        assert_eq!(title("playstation2__KimiKiss (Japan)"), t("PlayStation 2", "KimiKiss", Some("Japan")));
        assert_eq!(title("playstation2__KimiKiss__Japan_"), t("PlayStation 2", "KimiKiss", Some("Japan")));
        assert_eq!(title("playstation_portable__Tokimeki_Memorial_4_JPN_PSP-Caravan"), t("PSP", "Tokimeki Memorial 4", Some("Japan")));
        assert_eq!(title("playstation__TOKIMEKI MEMORIAL 2"), t("PlayStation", "TOKIMEKI MEMORIAL 2", None));
        assert_eq!(
            title("playstation2__Tokimeki_Memorial_3_-_Yakusoku_no_Ano_Basho_de__Japan_"),
            t("PlayStation 2", "Tokimeki Memorial 3 - Yakusoku no Ano Basho de", Some("Japan"))
        );
        assert_eq!(title("snes__Super Mario World (USA)"), t("snes", "Super Mario World", Some("USA")));
    }
}
