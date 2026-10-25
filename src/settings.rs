//! Edits of `config.toml` from the settings page: values are replaced in
//! place, so the file keeps its comments, order and layout.

use anyhow::{Context, bail};
use serde::Deserialize;
use serde_json::Value;
use toml_edit::{Array, DocumentMut, InlineTable, Item, TableLike};

/// One setting to change: a dotted key (`render.outline`) and its new value;
/// `null` removes the key (back to its default).
#[derive(Debug, Deserialize)]
pub struct Edit {
    pub key: String,
    pub value: Value,
}

/// The config text with the edits applied.
pub fn apply(text: &str, edits: &[Edit]) -> anyhow::Result<String> {
    let mut doc: DocumentMut = text.parse().context("config.toml is not valid TOML")?;
    for edit in edits {
        let parts: Vec<&str> = edit.key.split('.').collect();
        let (last, tables) = parts.split_last().context("empty key")?;
        let mut table: &mut dyn TableLike = doc.as_table_mut();
        for part in tables {
            table = table
                .entry(part)
                .or_insert_with(toml_edit::table)
                .as_table_like_mut()
                .with_context(|| format!("{} is not a table", edit.key))?;
        }
        if edit.value.is_null() {
            table.remove(last);
            continue;
        }
        match table.get_mut(last) {
            Some(Item::Value(old)) => {
                let mut new = to_toml(&edit.value, Some(old))?;
                // Keeps the spacing and a comment at the end of the line.
                *new.decor_mut() = old.decor().clone();
                *old = new;
            }
            // A table (a glossary) gets new contents under its header and comments.
            Some(Item::Table(old)) => {
                let Value::Object(map) = &edit.value else { bail!("{} is a table, not a value", edit.key) };
                old.clear();
                for (key, value) in map {
                    old.insert(key, Item::Value(to_toml(value, None)?));
                }
            }
            Some(_) => bail!("{} cannot be changed here", edit.key),
            None => match &edit.value {
                Value::Object(map) => {
                    let mut new = toml_edit::Table::new();
                    for (key, value) in map {
                        new.insert(key, Item::Value(to_toml(value, None)?));
                    }
                    table.insert(last, Item::Table(new));
                }
                value => {
                    table.insert(last, Item::Value(to_toml(value, None)?));
                }
            },
        }
    }
    tidy(doc.as_table_mut());
    // A removed first table leaves the blank line that separated it.
    Ok(doc.to_string().trim_start_matches('\n').to_owned())
}

/// Drops the tables left empty by removed keys, and the headers of tables
/// that only hold other tables (`[translate]` above `[translate.server]`).
fn tidy(table: &mut toml_edit::Table) {
    let mut empty = Vec::new();
    for (key, item) in table.iter_mut() {
        if let Item::Table(inner) = item {
            tidy(inner);
            if inner.is_empty() {
                empty.push(key.get().to_owned());
            } else {
                let only_tables = inner.iter().all(|(_, item)| item.is_table());
                inner.set_implicit(only_tables);
            }
        }
    }
    for key in empty {
        table.remove(&key);
    }
}

/// A JSON value as TOML; a number takes the type of the value it replaces, so
/// that `temperature = 1` stays a float.
fn to_toml(value: &Value, old: Option<&toml_edit::Value>) -> anyhow::Result<toml_edit::Value> {
    Ok(match value {
        Value::Bool(b) => (*b).into(),
        Value::Number(n) => {
            let float = old.is_some_and(toml_edit::Value::is_float) || n.as_i64().is_none();
            if float { n.as_f64().context("invalid number")?.into() } else { n.as_i64().context("invalid number")?.into() }
        }
        Value::String(s) => s.as_str().into(),
        Value::Array(items) => {
            let mut array = Array::new();
            for item in items {
                array.push(to_toml(item, None)?);
            }
            array.into()
        }
        Value::Object(map) => {
            let mut table = InlineTable::new();
            for (key, item) in map {
                table.insert(key, to_toml(item, None)?);
            }
            table.into()
        }
        Value::Null => bail!("null inside a value"),
    })
}

/// The config as JSON, for the page.
pub fn to_json(text: &str) -> anyhow::Result<Value> {
    let value: toml::Value = toml::from_str(text)?;
    Ok(serde_json::to_value(value)?)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{Edit, apply};

    const TEXT: &str = r#"listen = "127.0.0.1:4404"

[translate]
# Lower is more literal.
temperature = 0.1   # sampling
system_prompt = """
Line one.
"""

[render]
width = 1920
background = [16, 16, 24, 240]
"#;

    fn edit(key: &str, value: serde_json::Value) -> Edit {
        Edit { key: key.into(), value }
    }

    #[test]
    fn edits_values_and_keeps_comments() {
        let text = apply(
            TEXT,
            &[
                edit("translate.temperature", json!(1)),
                edit("render.width", json!(1280)),
                edit("render.background", json!([0, 0, 0, 200])),
                edit("render.font", json!("~/fonts/A.ttf")),
                edit("translate.system_prompt", json!("Line one.\nLine two.\n")),
            ],
        )
        .unwrap();
        assert!(text.contains("# Lower is more literal.\ntemperature = 1.0   # sampling\n"), "{text}");
        assert!(text.contains("width = 1280\n"), "{text}");
        assert!(text.contains("background = [0, 0, 0, 200]\n"), "{text}");
        assert!(text.contains("font = \"~/fonts/A.ttf\"\n"), "{text}");
        // Multi-line text stays readable.
        assert!(text.contains("Line one.\nLine two.\n"), "{text}");
        let parsed: toml::Value = toml::from_str(&text).unwrap();
        assert_eq!(parsed["translate"]["system_prompt"].as_str(), Some("Line one.\nLine two.\n"));
    }

    #[test]
    fn replaces_a_glossary_keeping_its_comments() {
        let profile = "match = \"KimiKiss\"\n\n# Names whose reading can't be guessed.\n[glossary]\n\"相原\" = \"Aihara\"\n";
        let text = apply(profile, &[edit("glossary", json!({ "星乃": "Hoshino", "相原": "Aihara" })), edit("ocr", json!({ "ヒンン": "ヒント" }))]).unwrap();
        assert!(text.contains("# Names whose reading can't be guessed.\n[glossary]\n"), "{text}");
        let parsed: toml::Value = toml::from_str(&text).unwrap();
        assert_eq!(parsed["glossary"]["星乃"].as_str(), Some("Hoshino"));
        assert_eq!(parsed["ocr"]["ヒンン"].as_str(), Some("ヒント"));
        assert_eq!(parsed["match"].as_str(), Some("KimiKiss"));
    }

    #[test]
    fn drops_tables_left_empty() {
        let text = apply("[render]\nwidth = 1280\n\n[translate.server]\ndraft = \"\"\n", &[edit("render.width", json!(null))]).unwrap();
        assert_eq!(text, "[translate.server]\ndraft = \"\"\n");
    }

    #[test]
    fn removes_and_adds_tables() {
        let text = apply(TEXT, &[edit("render.width", json!(null)), edit("translate.server.model", json!("a/b:c.gguf"))]).unwrap();
        assert!(!text.contains("width"), "{text}");
        let parsed: toml::Value = toml::from_str(&text).unwrap();
        assert_eq!(parsed["translate"]["server"]["model"].as_str(), Some("a/b:c.gguf"));
    }
}
