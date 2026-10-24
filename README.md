# game-translator

Translate Japanese games while you play, entirely on your own machine.

Press a hotkey in RetroArch: the game pauses, the screen is read (OCR), the
Japanese text is translated by a local LLM, and the translation is drawn over
the original text. Press the hotkey again to hide it and resume.

- **Local only**: no cloud service, no API key.
- **Placed like the game**: translations sit where the Japanese was, never
  overlap, and share one font size per kind of text.
- **Made for story-heavy games** (visual novels, RPGs, dating sims): keeps
  honorifics (-san, -senpai, onii-chan…), character voices, and remembers
  names and past lines across sessions.

It plugs into RetroArch's built-in **AI Service**, so it works with any
RetroArch core, in fullscreen, without screen capture or overlay windows.

## Requirements

- **Rust** (stable): <https://rustup.rs>
- **Linux x86-64** with a GPU driver: Mesa (AMD, Intel) or the NVIDIA driver (570 or
  newer for CUDA). Other systems: bring your own `llama-server` (see Configuration).
- **RetroArch** (a recent version, with the AI Service and the Vulkan or GL video driver).
- **Memory**: the default translation model (Gemma 4 26B-A4B, 4-bit) needs about
  20 GB of GPU or unified memory, on top of the game. OCR runs on the CPU.

Everything else is downloaded on first start: the OCR models (~90 MB) and a
prebuilt llama.cpp fitting the GPU (into `~/.cache/game-translator`), and the
translation model (~16 GB, into the Hugging Face cache, `~/.cache/huggingface`,
shared with other tools).

| GPU | llama.cpp build (`runtime = "auto"`) |
|---|---|
| AMD Strix Halo (gfx1151) | ROCm, from [strix-llama.cpp](https://github.com/halo-box/strix-llama.cpp) (~550 MB, ROCm libraries included) |
| NVIDIA, driver 570+ | CUDA 12 or 13, from [llama.cpp](https://github.com/ggml-org/llama.cpp) (~600 MB with the CUDA libraries); FP4 model |
| Other AMD or Intel, older NVIDIA driver | Vulkan, from llama.cpp (~35 MB) |
| None | CPU, from llama.cpp |

The build is kept up to date: at start (at most once a day) a newer release is
installed in the background and used from the next start, the previous one
staying as a fallback if the new one fails to start.

## Quick start

```sh
git clone <this repository> && cd game-translator
cargo run --release
```

The app starts `llama-server` with the translation model, then listens on
`http://localhost:4404/`. Stop it with Ctrl+C (it stops `llama-server` too).

### RetroArch setup

In the **AI Service** settings (under Settings, or Settings → Accessibility depending on the version):

| Setting | Value |
|---|---|
| AI Service Enabled | On |
| AI Service Output | Image Mode |
| AI Service URL | `http://localhost:4404/` |
| Pause During Translation | On |
| Source Language | Japanese |
| Target Language | English (or any other: switch it here at any time) |

Then bind a key in **Settings → Input → Hotkeys → AI Service**. The video
driver must be `gl`, `glcore` or `vulkan` for the transparent overlay.

## Settings page

<http://localhost:4404/>, opened in a browser, shows the translation engine and lets you switch it
while the app runs:

- **Local model**: the GPU found, the llama.cpp build in use and the ones kept,
  update settings, a button to look for a newer build, the server log.
- **Remote provider**: any OpenAI-compatible endpoint (OpenRouter, a server on
  another machine) with its model and API key; only the text of the screen is
  sent, never the image. Switching to it stops the local server and frees its memory.
- **Test the translation**: translates one line with the real request, so a
  provider that rejects it shows up here rather than in the game.

Choices made in the page are saved in `~/.config/game-translator/engine.toml`
and override `config.toml`; the API key goes to `secrets.toml` next to it,
readable by you only (the `GAME_TRANSLATOR_API_KEY` environment variable takes
precedence and is never written).

## Configuration

`config.toml` holds everything: OCR, translation model and prompt, layout.

- **Another model or server**: edit `[translate]` (any OpenAI-compatible
  endpoint works). Remove `[translate.server]` if you run the server yourself.
- **llama.cpp build**: `[translate.server]` sets the kind of build (`runtime`),
  pins a release (`release = "b11856"`) or turns updates off or to a log line
  (`update`). To run your own build: `command = "/path/to/llama-server"`, or
  `LLAMA_SERVER=/path/to/llama-server cargo run --release`.

### Game profiles and styles

A profile in `profiles/` applies when RetroArch's game label contains `match`.
It adds instructions and a glossary, which is the reliable way to fix the
reading of character names:

```toml
match = "Tokimeki_Memorial_4"
styles = ["dating-sim"]
instructions = "Game: Tokimeki Memorial 4 (PSP, 2009)."

[glossary]
"大倉都子" = "Ōkura Miyako"
```

Styles in `styles/` hold what a whole genre shares (tone, honorifics, common
terms); a profile lists the styles it uses.

### Game memory

While serving, every translated block and every name the model transcribed is
saved in `data/<game>/`:

- `memory.jsonl`: past translations, reused instantly (also across sessions);
- `glossary.toml`: names learned along the way. Fix a wrong reading here once
  and it is used from then on.

## Development

- `cargo run --release -- replay captures/<frame>.png`: run the whole pipeline
  on a saved frame (RetroArch frames are saved in `captures/` while serving) and
  write the overlay and a preview in `replay/`. Add `--ocr-only` to see the
  detected text boxes.
- `cargo run --release -- bench <cases.toml>`: translate a list of test cases
  and time them, to compare models.

## Credits

- OCR: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) PP-OCRv6 models
  through [oar-ocr](https://github.com/GreatV/oar-ocr).
- Translation: any model served by [llama.cpp](https://github.com/ggml-org/llama.cpp);
  default [Gemma 4 26B-A4B](https://huggingface.co/unsloth/gemma-4-26b-a4b-it-GGUF).
- Fonts: [Noto Sans](https://github.com/notofonts/latin-greek-cyrillic) (`assets/fonts/OFL.txt`)
  and [Noto Sans JP](https://github.com/notofonts/noto-cjk) for symbols and Japanese
  (`assets/fonts/OFL-NotoSansJP.txt`), SIL Open Font License 1.1, embedded in the binary.

## License

Apache License 2.0, see [LICENSE](LICENSE). The bundled Noto Sans and Noto Sans JP
fonts keep their own license (SIL Open Font License 1.1).
