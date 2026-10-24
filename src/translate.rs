//! Translation with per-game context: profile instructions, glossary (written
//! and learned), recent history and a persistent memory of past translations.

use std::{
    collections::{BTreeMap, HashMap},
    path::{Path, PathBuf},
    sync::{LazyLock, Mutex},
};

use regex::Regex;
use serde_json::json;

use crate::{
    config::{Backend, Profile, TranslateConfig},
    llm,
    memory::{GameMemory, safe_name},
};

pub struct Translator {
    client: reqwest::Client,
    /// Where game memories are saved; `None` keeps them in RAM only.
    data_dir: Option<PathBuf>,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    /// Past (request, answer) exchanges, per game label and target language.
    history: HashMap<(String, String), Vec<(String, String)>>,
    /// Translations and learned names, per game label, loaded on first use.
    games: HashMap<String, GameMemory>,
}

impl Translator {
    pub fn new(client: reqwest::Client, data_dir: Option<PathBuf>) -> Self {
        Self { client, data_dir, state: Mutex::default() }
    }

    /// Returns one translation per input text. `cached` in the result tells how
    /// many texts were served from memory.
    pub async fn translate(
        &self,
        config: &TranslateConfig,
        backend: &Backend,
        profile: Option<&Profile>,
        label: &str,
        lang: &str,
        texts: &[String],
    ) -> anyhow::Result<(Vec<String>, usize)> {
        let key = (label.to_owned(), lang.to_owned());
        let (mut results, history, learned) = {
            let mut state = self.state.lock().unwrap();
            let history = state.history.get(&key).cloned().unwrap_or_default();
            let game = self.game(&mut state, label);
            let results: Vec<Option<String>> = texts.iter().map(|t| game.get(lang, t).cloned()).collect();
            (results, history, game.names.clone())
        };

        let missing: Vec<usize> = (0..texts.len()).filter(|&i| results[i].is_none()).collect();
        let cached = texts.len() - missing.len();
        if missing.is_empty() {
            return Ok((results.into_iter().map(Option::unwrap_or_default).collect(), cached));
        }

        // Blocks are keyed by id: tying each answer to an id (rather than to a
        // position in a list) keeps the model from sliding text into the next slot.
        let blocks: serde_json::Map<String, serde_json::Value> =
            missing.iter().enumerate().map(|(n, &i)| ((n + 1).to_string(), json!(texts[i]))).collect();
        let request = serde_json::Value::Object(blocks).to_string();

        let system = system_prompt(config, profile, lang);
        let mut messages = vec![json!({ "role": "system", "content": system })];
        for (user, assistant) in &history {
            messages.push(json!({ "role": "user", "content": user }));
            messages.push(json!({ "role": "assistant", "content": assistant }));
        }
        let texts_sent: Vec<&str> = missing.iter().map(|&i| texts[i].as_str()).collect();
        messages.push(json!({ "role": "user", "content": with_known_names(&request, &learned, profile, &texts_sent) }));

        // The model sometimes loops on line breaks until max_tokens: stopping at a
        // long run of them saves seconds, and `parse` repairs the cut JSON. A few
        // blank lines are legitimate (`clean` drops them), and stopping on those cut
        // dialogues short.
        // An escaped quote before a brace means the model believes the JSON is over
        // while still inside a string: what follows is notes, sometimes looping too.
        let (breaks, quotes) = (r"\n".repeat(8), r#"\""#.repeat(8));
        let stops = [breaks.as_str(), quotes.as_str(), r#"\"}"#];
        let answer = llm::chat(&self.client, backend, messages, Some(response_schema(missing.len())), &stops).await?;
        tracing::debug!(answer, "translation answer");
        let (parsed, names) = parse(&answer, missing.len());

        // A name is learned only if it really is in the Japanese we sent, and never
        // overrides the game profile's own glossary.
        let names: Vec<(String, String)> = names
            .into_iter()
            .filter(|(ja, romaji)| {
                !romaji.is_empty()
                    && romaji.chars().count() <= 40
                    && ja.chars().any(is_japanese)
                    && missing.iter().any(|&i| texts[i].contains(ja.as_str()))
                    && !profile.is_some_and(|p| p.glossary.contains_key(ja))
            })
            .collect();

        let mut state = self.state.lock().unwrap();
        let game = self.game(&mut state, label);
        for (&i, translation) in missing.iter().zip(&parsed) {
            game.remember(lang, &texts[i], translation);
            results[i] = Some(translation.clone());
        }
        let added = game.learn_names(names);
        if added > 0 {
            tracing::info!(added, "learned names");
        }
        let history = state.history.entry(key).or_default();
        let keyed: serde_json::Map<String, serde_json::Value> =
            parsed.iter().enumerate().map(|(n, t)| ((n + 1).to_string(), json!(t))).collect();
        history.push((request, serde_json::Value::Object(keyed).to_string()));
        // Trim in bulk rather than sliding, so the prompt prefix stays cacheable between calls.
        if history.len() > config.history {
            history.drain(..history.len() - config.history / 2);
        }

        Ok((results.into_iter().map(Option::unwrap_or_default).collect(), cached))
    }

    /// Where a game's memory is saved (`None`: memory disabled).
    pub fn game_dir(&self, dir_name: &str) -> Option<PathBuf> {
        self.data_dir.as_ref().map(|d| d.join(dir_name))
    }

    pub fn data_dir(&self) -> Option<&Path> {
        self.data_dir.as_deref()
    }

    /// Runs `edit` on a game's saved files while no translation writes them,
    /// then forgets what was loaded from them, and the recent exchanges (which
    /// may repeat a corrected translation): the next request reads them again.
    pub fn edit_game<T>(&self, dir_name: &str, edit: impl FnOnce(&Path) -> T) -> Option<T> {
        let dir = self.game_dir(dir_name)?;
        let mut state = self.state.lock().unwrap();
        let result = edit(&dir);
        state.games.retain(|label, _| safe_name(label) != dir_name);
        state.history.retain(|(label, _), _| safe_name(label) != dir_name);
        Some(result)
    }

    fn game<'a>(&self, state: &'a mut State, label: &str) -> &'a mut GameMemory {
        state
            .games
            .entry(label.to_owned())
            .or_insert_with(|| GameMemory::load(self.data_dir.as_deref(), label))
    }
}

/// The system prompt only changes with the profile, so the server can reuse
/// its cache (system prompt and history) from one request to the next.
fn system_prompt(config: &TranslateConfig, profile: Option<&Profile>, lang: &str) -> String {
    let mut prompt = config.system_prompt.replace("{lang}", language_name(lang));
    if let Some(profile) = profile.filter(|p| !p.instructions.is_empty()) {
        prompt.push_str(&profile.instructions);
        prompt.push('\n');
    }
    if let Some(profile) = profile.filter(|p| !p.glossary.is_empty()) {
        prompt.push_str("Glossary (Japanese = translation), always use these:\n");
        for (ja, tr) in &profile.glossary {
            prompt.push_str(&format!("{ja} = {tr}\n"));
        }
    }
    prompt
}

/// Learned names go with the request, after the cached prefix, and only those
/// the texts use: the list grows during play and would otherwise invalidate the
/// cache at each new name.
fn with_known_names(request: &str, learned: &BTreeMap<String, String>, profile: Option<&Profile>, texts: &[&str]) -> String {
    let known: Vec<String> = learned
        .iter()
        .filter(|(ja, _)| !profile.is_some_and(|p| p.glossary.contains_key(*ja)))
        .filter(|(ja, _)| texts.iter().any(|t| t.contains(ja.as_str())))
        .map(|(ja, romaji)| format!("{ja} = {romaji}"))
        .collect();
    if known.is_empty() {
        return request.to_owned();
    }
    format!("Names already used in this game: {}\n{request}", known.join(", "))
}

fn is_japanese(c: char) -> bool {
    matches!(c, '\u{3040}'..='\u{30ff}' | '\u{3400}'..='\u{4dbf}' | '\u{4e00}'..='\u{9fff}')
}

/// RetroArch sends ISO-like codes; models do better with language names.
pub fn language_name(code: &str) -> &str {
    match code {
        "en" => "English",
        "fr" => "French",
        "de" => "German",
        "es" => "Spanish",
        "it" => "Italian",
        "pt" => "Portuguese",
        "nl" => "Dutch",
        "ru" => "Russian",
        "pl" => "Polish",
        "ko" => "Korean",
        "zh-CN" => "Simplified Chinese",
        "zh-TW" => "Traditional Chinese",
        other => other,
    }
}

/// Removes line-break markers the model sometimes copies from the input,
/// swaps Japanese quotes for Latin ones and drops a trailing loop and JSON debris.
fn clean(text: &str) -> String {
    // Line breaks the model kept on purpose (lists) survive; other whitespace collapses.
    // Japanese quotes have no glyph in the overlay font and read oddly in Latin text.
    let text = strip_loop(text)
        .replace('⏎', " ")
        .replace('「', "“")
        .replace('」', "”")
        .replace('『', "‘")
        .replace('』', "’")
        .lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    strip_json_debris(&text)
}

/// Drops the end of a translation where the model got stuck repeating anything
/// (`""""…`, `💡 💡 …`, a whole exchange) until the token limit. Real text repeats
/// short things ("......", "!!!!!!!!", "Akari! Akari!") but never 20 times in a
/// row, nor a phrase 5 times, nor a whole exchange 3 times. The first copy of a
/// repeated phrase is usually the real translation, so it is kept.
fn strip_loop(text: &str) -> &str {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    for period in 1..=400.min(n / 3) {
        // Start of the periodic tail: the text repeats itself every `period` chars from there.
        let mut start = n - period;
        while start > 0 && chars[start - 1] == chars[start - 1 + period] {
            start -= 1;
        }
        let min_repeats = match period {
            1..=8 => 20,
            9..=64 => 5,
            _ => 3,
        };
        if n - start >= min_repeats * period {
            // A repeated exchange is kept from the start of one of its lines.
            let first = (start..start + period).find(|&i| i > 0 && chars[i - 1] == '\n').unwrap_or(start);
            let keep = if period <= 8 { start } else { first + period };
            let end = text.char_indices().nth(keep).map_or(text.len(), |(i, _)| i);
            return text[..end].trim_end();
        }
    }
    text
}

/// Drops JSON debris the model sometimes leaves in a translation (`"}`, `` `} ,``,
/// the next key, `"names":`, or a closing `"}` followed by notes): quotes in the
/// text make it lose track of where the JSON string ends. Everything from the
/// debris on is cut; a closing quote the text needs is kept.
fn strip_json_debris(text: &str) -> String {
    static DEBRIS: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r#"(?s)([\s,]*\\?"(names|\d+)\\?"\s*:|[{}`]).*$"#).unwrap());
    let Some(debris) = DEBRIS.find(text) else {
        return text.to_owned();
    };
    let core = text[..debris.start()].trim_end_matches(|c: char| matches!(c, ',' | '"' | '\'') || c.is_whitespace());
    let mut cleaned = core.to_owned();
    if core.matches('"').count() % 2 == 1 {
        cleaned.push('"');
    }
    cleaned
}

/// JSON schema requiring one translation under each block id, plus the names
/// used. Servers that support grammar-constrained decoding (llama.cpp) then
/// cannot merge, split or drop blocks.
fn response_schema(blocks: usize) -> serde_json::Value {
    let ids: Vec<String> = (1..=blocks).map(|n| n.to_string()).collect();
    let mut properties: serde_json::Map<String, serde_json::Value> =
        ids.iter().map(|id| (id.clone(), json!({ "type": "string" }))).collect();
    properties.insert("names".into(), json!({ "type": "object", "additionalProperties": { "type": "string" } }));
    let mut required = ids;
    required.push("names".into());
    json!({
        "type": "json_schema",
        "json_schema": {
            "name": "translations",
            "strict": true,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": false,
            },
        },
    })
}

/// Reads `{"1": "...", "2": "...", "names": {...}}`; falls back to numbered
/// lines for servers that ignore the schema.
fn parse(answer: &str, expected: usize) -> (Vec<String>, Vec<(String, String)>) {
    let json = answer.trim().trim_start_matches("```json").trim_matches('`').trim();
    if let Some((serde_json::Value::Object(map), complete)) = parse_json(json) {
        let translations: Vec<String> = (1..=expected)
            .map(|n| clean(map.get(&n.to_string()).and_then(|v| v.as_str()).unwrap_or_default()))
            .collect();
        if translations.iter().any(String::is_empty) {
            tracing::warn!(answer, expected, "missing translations");
        }
        // A cut answer may end inside a name: none of its names are trusted.
        let names = map
            .get("names")
            .filter(|_| complete)
            .and_then(|v| v.as_object())
            .map(|names| {
                names
                    .iter()
                    .filter_map(|(ja, romaji)| Some((ja.trim().to_owned(), romaji.as_str()?.trim().to_owned())))
                    .collect()
            })
            .unwrap_or_default();
        return (translations, names);
    }
    (parse_numbered(answer, expected), Vec::new())
}

/// Parses the answer, or closes it when generation was cut (stop string or
/// max_tokens): the blocks written so far are kept. Also tells whether the
/// answer was complete.
fn parse_json(json: &str) -> Option<(serde_json::Value, bool)> {
    if let Ok(value) = serde_json::from_str(json) {
        return Some((value, true));
    }
    let cut = json.trim_end_matches(r"\n").trim_end_matches('\\');
    ["\"}", "}", "\"}}", "}}"].iter().find_map(|end| Some((serde_json::from_str(&format!("{cut}{end}")).ok()?, false)))
}

/// Maps `N. translation` lines back to blocks; falls back to line order when
/// the numbering is missing or inconsistent.
fn parse_numbered(answer: &str, expected: usize) -> Vec<String> {
    static NUMBERED: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\s*(\d+)\s*[.):]\s*(.*)$").unwrap());

    let lines: Vec<&str> = answer.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let mut by_number = vec![String::new(); expected];
    let mut numbered = 0;
    // Unnumbered lines continue the previous entry: models sometimes keep the
    // original line breaks of a multi-line block.
    let mut current: Option<usize> = None;
    for line in &lines {
        let entry = NUMBERED
            .captures(line)
            .and_then(|caps| Some((caps[1].parse::<usize>().ok()?, caps.get(2)?.as_str())))
            .filter(|(n, _)| (1..=expected).contains(n));
        match (entry, current) {
            (Some((n, text)), _) => {
                by_number[n - 1] = clean(text);
                numbered += 1;
                current = Some(n - 1);
            }
            (None, Some(i)) => {
                let joined = format!("{} {}", by_number[i], clean(line));
                by_number[i] = joined.trim().to_owned();
            }
            (None, None) => {}
        }
    }
    if numbered == expected {
        return by_number;
    }

    tracing::warn!(answer, expected, "translation numbering mismatch, using line order");
    let mut positional: Vec<String> = lines
        .iter()
        .map(|l| clean(NUMBERED.captures(l).map_or(*l, |c| c.get(2).unwrap().as_str())))
        .collect();
    positional.resize(expected, String::new());
    positional
}

#[cfg(test)]
mod tests {
    use super::{clean, parse, strip_json_debris};

    #[test]
    fn drops_loops() {
        let looped = format!("“Ignorance is a sin!”\n{}", "\"".repeat(300));
        assert_eq!(clean(&looped), "“Ignorance is a sin!”");
        assert_eq!(clean(&format!("“Your bag.” {}", "💡 ".repeat(50))), "“Your bag.”");
        assert_eq!(clean(&format!("Hiro! {}", "I said it's fine. ".repeat(6))), "Hiro! I said it's fine.");
        let exchange = "“Seriously, you really never change.”\n“Eh?\n“Everything.”\n“What do you mean by that?”\nAbout what?”\n";
        assert_eq!(
            clean(&format!("“Hey.”\n{}“Seriously", exchange.repeat(12))),
            format!("“Hey.”\n{}", exchange.trim_end())
        );
        assert_eq!(clean("“Wait......!!!!!!!!”"), "“Wait......!!!!!!!!”");
        assert_eq!(clean("“Akari! Akari! Akari! Akari! Akari!”"), "“Akari! Akari! Akari! Akari! Akari!”");
    }

    #[test]
    fn repairs_answers_cut_by_a_loop() {
        let answer = r#"{"1":"“Seriously.”\n“Eh?”","2":"“Everything.”\n\n\n"#;
        assert_eq!(parse(answer, 2).0, ["“Seriously.”\n“Eh?”", "“Everything.”"]);
        let answer = r#"{"1":"“There, your bag.”"#;
        assert_eq!(parse(answer, 1).0, ["“There, your bag.”"]);
        let answer = r#"{"1":"Hi.","names":{"志保":"Shiho","あかり":"Ak"#;
        let (translations, names) = parse(answer, 1);
        assert_eq!(translations, ["Hi."]);
        assert!(names.is_empty());
    }

    #[test]
    fn strips_json_debris_seen_in_play() {
        assert_eq!(strip_json_debris("in a panic. '\"`} ,"), "in a panic.");
        assert_eq!(strip_json_debris("\"What's for breakfast?\"}"), "\"What's for breakfast?\"");
        assert_eq!(strip_json_debris("“Really.”\n\"Heh, is that so?\"\"}"), "“Really.”\n\"Heh, is that so?\"");
        assert_eq!(strip_json_debris("“Especially in the morning.”\n\"names\":"), "“Especially in the morning.”");
        assert_eq!(strip_json_debris("“Hi.”\", \"2\": \"Bye.\"}"), "“Hi.”");
        assert_eq!(strip_json_debris("“There, your bag.”\"}``` (Note: a typo.) 💡 💡"), "“There, your bag.”");
    }

    #[test]
    fn keeps_clean_translations() {
        assert_eq!(strip_json_debris("Depending on my mood,"), "Depending on my mood,");
        assert_eq!(strip_json_debris("\"I-I can't...\""), "\"I-I can't...\"");
    }
}
