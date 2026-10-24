//! Minimal client for OpenAI-compatible chat completion endpoints.

use anyhow::{Context, anyhow};
use serde_json::{Value, json};

use crate::config::Backend;

/// `response_format` (e.g. a JSON schema) is sent as-is when given; generation
/// ends early on any of the `stop` strings.
pub async fn chat(
    client: &reqwest::Client,
    backend: &Backend,
    messages: Vec<Value>,
    response_format: Option<Value>,
    stop: &[&str],
) -> anyhow::Result<String> {
    let mut body = json!({
        "model": backend.model,
        "temperature": backend.temperature,
        "max_tokens": backend.max_tokens,
        "messages": messages,
        "stop": stop,
    });
    let object = body.as_object_mut().expect("body is an object");
    for (key, value) in &backend.extra {
        object.insert(key.clone(), value.clone());
    }
    if let Some(format) = response_format {
        object.insert("response_format".into(), format);
    }

    let mut request = client.post(&backend.url).json(&body);
    if let Some(key) = &backend.api_key {
        request = request.bearer_auth(key);
    }
    let response = request.send().await.with_context(|| format!("cannot reach {}", backend.url))?;
    // A provider explains a refusal in the body (unknown model, unsupported option).
    if let Err(error) = response.error_for_status_ref() {
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!("{error}: {}", body.trim()));
    }
    let response: Value = response.json().await?;

    response["choices"][0]["message"]["content"]
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| anyhow!("no content in response: {response}"))
}
