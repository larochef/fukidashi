<img src="assets/icons/fukidashi.svg" alt="" width="96" align="right">

# Fukidashi

Translate Japanese games while you play, entirely on your own machine.
Fukidashi (吹き出し) is the speech bubble of manga: it puts the translation in
place of the original text.

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

- **Linux x86-64**, glibc 2.39 or newer (Ubuntu 24.04, Debian 13, Fedora 40 or later), with a GPU driver: Mesa (AMD, Intel) or the NVIDIA driver (570 or
  newer for CUDA). Other systems: bring your own `llama-server` (see Configuration).
- **RetroArch** (a recent version, with the AI Service and the Vulkan or GL video driver).
- **Memory**: depends on the translation model, which the first start proposes
  from what the computer has (see [Translation models](#translation-models)):
  from 16 GB of GPU or unified memory for the best one down to a 4 GB graphics
  card, or no GPU at all. OCR runs on the CPU.

The OCR models (~90 MB) are downloaded on first start. The local translation
model is only downloaded once you choose it (see First start): a prebuilt
llama.cpp fitting the GPU (into `~/.cache/fukidashi`) and the model (2 to 18 GB,
into the Hugging Face cache, `~/.cache/huggingface`, shared with other tools).
A remote provider needs neither.

| GPU | llama.cpp build (`runtime = "auto"`) |
|---|---|
| AMD Strix Halo (gfx1151) | ROCm, from [strix-llama.cpp](https://github.com/halo-box/strix-llama.cpp) (~550 MB, ROCm libraries included) |
| NVIDIA, driver 570+ | CUDA 12 or 13, from [llama.cpp](https://github.com/ggml-org/llama.cpp) (~600 MB with the CUDA libraries); FP4 model |
| Other AMD or Intel, older NVIDIA driver | Vulkan, from llama.cpp (~35 MB) |
| None | CPU, from llama.cpp |

The build is kept up to date: at start (at most once a day) a newer release is
installed in the background and used from the next start, the previous one
staying as a fallback if the new one fails to start.

## Install

Each [release](https://github.com/larochef/fukidashi/releases) has:

- `fukidashi_<version>_amd64.deb` (Debian, Ubuntu) and
  `fukidashi-<version>.x86_64.rpm` (Fedora, openSUSE): the app, a menu entry,
  a user service, and the GNOME extension;
- on Arch, the AUR package `fukidashi-bin` installs the same;
- `fukidashi-<version>-x86_64-linux.tar.gz`: the same files, to run
  `./fukidashi` from anywhere (its `fukidashi.service` expects
  `/usr/bin/fukidashi`).

From the sources, with [Rust](https://rustup.rs):

```sh
git clone https://github.com/larochef/fukidashi.git && cd fukidashi
cargo run --release
```

## First start

Open **Fukidashi** from the applications menu (or run `fukidashi open`): the
app starts and its page opens at <http://localhost:4404/>, on Getting started.
Choose where the text is translated:

- **On this computer**: the page shows the GPU and memory found and the local
  models, the one fitting this computer first, and downloads the chosen one,
  with its progress;
- **With a provider**: any OpenAI-compatible endpoint, nothing to download.

The same page offers to start Fukidashi when you log in, and, on GNOME, to
enable the extension (installed by a package: log out and back in, or restart, first; see
[GNOME extension](#gnome-extension)). In a terminal: `fukidashi setup local`
(the model for this computer) or `fukidashi setup local MODEL`, or
`fukidashi setup remote URL MODEL --key KEY`.

### Translation models

| Model | Proposed for | Quality | Download |
|---|---|---|---|
| Gemma 4 26B-A4B (`gemma-4-26b`) | 16 GB of GPU or unified memory and more | ★★★★★ | 17.5 GB |
| Hy-MT2 7B (`hy-mt2-7b`) | 6 to 16 GB graphics cards | ★★★★☆ | 4.6 GB |
| Hy-MT2 1.8B (`hy-mt2-1-8b`) | 3 to 6 GB graphics cards; no GPU and less than 28 GB of RAM | ★★☆☆☆ | 1.9 GB |
| Hy-MT2 30B-A3B (`hy-mt2-30b`) | no GPU, 28 GB of RAM or more | ★★★★☆ | 18.2 GB |

Gemma is a chat model: it translates the whole screen in one request, follows
the conversation (the previous lines) and learns the characters' names as it
reads them. The Hy-MT2 models only translate: each text alone, faster and
lighter, with no context, and they cannot tell names; they use the ones in the
game's profile and in its Names tab (learned by Gemma, or written there), and
keep forms of address (-san, Onii-chan, Senpai) as they are.

Measured on a Strix Halo, per screen: Gemma 1.1 s; Hy-MT2 7B 0.8 s; Hy-MT2
1.8B 0.4 s, 1.2 s without the GPU; Hy-MT2 30B-A3B 3 s without the GPU. A common
computer without a GPU is several times slower. The models are defined in
`[translate.models]` of `config.toml`, where you can add your own.

## Running

| | |
|---|---|
| Start, and open the page | the menu entry, or `fukidashi open` |
| Stop | the menu entry's Stop action, or `fukidashi stop` |
| With the session | the option on the page (`systemctl --user enable fukidashi`) |
| In a terminal | `fukidashi` (Ctrl+C stops it), or `cargo run --release` |

The app listens on `http://localhost:4404/` (`listen`); a second one says the
first already runs. The local model loads with the first translation that
needs it (a few seconds) and is unloaded after 30 minutes without one
(`idle_minutes`), freeing the GPU memory; the app itself stays, small. Until
the model is ready (no engine chosen, still downloading), a translation shows a
message saying so instead.

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

### Any game, outside RetroArch

On GNOME, the extension translates whatever is on screen: see
[GNOME extension](#gnome-extension).

## GNOME extension

It translates whatever is on screen, not only RetroArch: a PC game, a
standalone emulator, a video. Press **Super+T**: the monitor of the focused
window is captured and sent to the app (which must be running), and the
translation is shown over it. Any key, a click or a gamepad button hides it
(gamepads need libmanette, installed with WebKitGTK on most systems; the button
also reaches the game). The game is not paused, and the window's class
(`steam_app_1234`, `pcsx2-qt`…) stands for the game label, for its memory and
profile.

### Install

- **With a package** (`.deb`, `.rpm`, AUR): it is installed for every user,
  nothing more to install.
- **From a release**, without a package: download
  `fukidashi@larochef.github.io.zip`, then
  `gnome-extensions install --force fukidashi@larochef.github.io.zip`.
- **From a clone**: `glib-compile-schemas gnome-extension/schemas`, then link
  `gnome-extension` as
  `~/.local/share/gnome-shell/extensions/fukidashi@larochef.github.io`.

### Enable

GNOME never enables an extension by itself, and only looks for extensions when
you log in. So, once installed:

1. **Log out and back in**, or restart (GNOME hides "Log Out" when the
   computer has a single user account). Until then, GNOME does not know the extension:
   it is missing from the Extensions app, and `gnome-extensions` answers that
   it "doesn't exist".
2. **Enable it**: on Fukidashi's page (Getting started, or Settings → This
   computer), in the Extensions app, or:

   ```sh
   gnome-extensions enable fukidashi@larochef.github.io
   ```

3. **Check it**: `gnome-extensions info fukidashi@larochef.github.io` says
   `Enabled: Yes` and `State: ACTIVE`.

Fukidashi's page shows where the extension stands and the command for the next
step.

### Settings

The shortcut, the language to translate into (by default the system's) and the
app's address are in the extension's preferences, from the Extensions app or:

```sh
gnome-extensions prefs fukidashi@larochef.github.io
```

### When it does not work

| `gnome-extensions info` says | Why, and what to do |
|---|---|
| "doesn't exist" | Installed since you logged in: log out and back in, or restart. |
| `Enabled: No` | Enable it (above). |
| `State: INITIALIZED` or `INACTIVE` | User extensions are turned off (the switch at the top of the Extensions app): `gsettings set org.gnome.shell disable-user-extensions false`. Packages install it as a system extension, which this switch does not affect. |
| `State: OUT OF DATE` | GNOME is newer than the versions it was checked with (`shell-version` in `metadata.json`): a newer Fukidashi may support it. |
| `State: ERROR` | It failed to start; GNOME Shell's log says why: `journalctl -b -o cat /usr/bin/gnome-shell \| grep -i fukidashi`. |

Super+T showing an error notification means the extension works but the app
does not answer: start it (the menu entry, or `fukidashi open`).

## Settings page

<http://localhost:4404/>, opened in a browser, shows the translation engine and lets you switch it
while the app runs:

- **Getting started**: the engine to use (shown first until one is chosen),
  starting with the session, the GNOME extension.
- **Local model**: the model (see [Translation models](#translation-models)), the GPU found, the llama.cpp build in use and the ones kept,
  update settings, a button to look for a newer build, the server log.
- **Remote provider**: any OpenAI-compatible endpoint (OpenRouter, a server on
  another machine) with its model and API key; only the text of the screen is
  sent, never the image. Switching to it stops the local server and frees its memory.
- **Test the translation**: translates one line with the real request, so a
  provider that rejects it shows up here rather than in the game.
- **Settings**: the main settings of `config.toml` as a form (target language,
  OCR, prompt, sampling, model, overlay colours and screen), plus the whole file
  in an editor. Changes are checked, written back keeping the file's comments,
  and applied at once; the page says when the app (`listen`, `data_dir`) or the
  local server (model, arguments) needs a restart. "Reload from disk" applies a
  file edited by hand, profiles and styles included.
- **Games**: every game met (or with a profile). Its translations, newest
  first, by language and by text: correct one (it is used as is from then on)
  or delete it (translated again next time); the names it learned; its profile
  (instructions, styles, glossary, OCR fixes), created here when it has none.

Choices made in the page are saved in `~/.config/fukidashi/engine.toml`
and override `config.toml`; the API key goes to `secrets.toml` next to it,
readable by you only (the `FUKIDASHI_API_KEY` environment variable takes
precedence and is never written).

## Configuration

The settings ship with the app (the repository's `config.toml`: OCR,
translation model and prompt, layout). Yours go in
`~/.config/fukidashi/config.toml`, which the settings page writes: it holds
only what differs from the shipped settings, so improvements to the shipped
ones (the prompt above all) still reach you. `--config FILE` uses another file
of your settings instead.

- **Another model or server**: add a model to `[translate.models]` (or set
  `[translate.server] model` to an `owner/repo:file` of your own), or edit
  `[translate]` (any OpenAI-compatible endpoint works). A server of your own
  answering on `url` is used as is.
- **llama.cpp build**: `[translate.server]` sets the kind of build (`runtime`),
  pins a release (`release = "b11856"`) or turns updates off or to a log line
  (`update`). To run your own build: `command = "/path/to/llama-server"`, or
  `LLAMA_SERVER=/path/to/llama-server cargo run --release`.

### Game profiles and styles

A profile applies when the game's label (RetroArch's, or the window's for the
GNOME extension) contains `match`. Profiles ship in the repository's
`profiles/`; yours, written by the Games page, go in
`~/.config/fukidashi/profiles/` and replace a shipped one of the same file name.
It adds instructions and a glossary, which is the reliable way to fix the
reading of character names:

```toml
match = "Tokimeki_Memorial_4"
styles = ["dating-sim"]
instructions = "Game: Tokimeki Memorial 4 (PSP, 2009)."

[glossary]
"大倉都子" = "Ōkura Miyako"
```

Styles hold what a whole genre shares (tone, honorifics, common terms); a
profile lists the styles it uses. They ship in `styles/`; yours go in
`~/.config/fukidashi/styles/`.

### Game memory

While serving, every translated block and every name the model transcribed is
saved in `~/.local/share/fukidashi/data/<game>/` (`data_dir`):

- `memory.jsonl`: past translations, reused instantly (also across sessions);
- `glossary.toml`: names learned along the way. Fix a wrong reading here once
  and it is used from then on.

## Development

- The repository's `config.toml`, `profiles/` and `styles/` are the shipped
  settings, embedded at build time: a change there needs a rebuild
  (`cargo run` does it), and your own settings in `~/.config/fukidashi` still
  apply over them.
- `cargo run --release -- replay <frame>.png`: run the whole pipeline on a
  saved frame (frames received are saved while serving when `capture_dir` is
  set, e.g. `capture_dir = "~/.local/share/fukidashi/captures"` in your
  settings) and write the overlay and a preview in `replay/`. Add `--ocr-only`
  to see the detected text boxes.
- `cargo run --release -- bench <cases.toml>`: translate a list of test cases
  and time them, to compare models.
- `cargo test`: also checks the overlay layout against reviewed cases in
  `tests/layout/` (a screen's blocks as read by the OCR, with translations, and
  the expected panels). `replay --fixtures tests/layout captures/<frame>.png`
  adds a case; after a deliberate layout change, check the result, then accept
  it with `UPDATE_LAYOUT=1 cargo test layout`.
- `packaging/`: the user service, the menu entry, and the AUR package's
  PKGBUILD (`aur/`, published by the CI); `Cargo.toml` lists what the `.deb` (`cargo deb`) and the `.rpm`
  (`cargo generate-rpm`) install.
- CI (`.github/workflows/ci.yml`): every push and pull request runs clippy and
  the tests. Every push to `main` that passes them is published as a release with
  the packages and the Linux archive, numbered like llama.cpp's builds: `v<major.minor of
  Cargo.toml>.<number of commits>` (e.g. `v0.1.42`). Raise the version in
  `Cargo.toml` to mark a bigger step. Each release then updates the AUR package
  `fukidashi-bin`: `packaging/aur/PKGBUILD` with the release's version and
  checksum, pushed with the `AUR_SSH_KEY` secret (skipped without it).

## Credits

- OCR: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) PP-OCRv6 models
  through [oar-ocr](https://github.com/GreatV/oar-ocr).
- Translation: any model served by [llama.cpp](https://github.com/ggml-org/llama.cpp);
  [Gemma 4 26B-A4B](https://huggingface.co/unsloth/gemma-4-26b-a4b-it-GGUF) (Google) and the
  [Hy-MT2](https://huggingface.co/tencent/Hy-MT2-7B-GGUF) translation models (Tencent).
- Fonts: [Noto Sans](https://github.com/notofonts/latin-greek-cyrillic) (`assets/fonts/OFL.txt`)
  and [Noto Sans JP](https://github.com/notofonts/noto-cjk) for symbols and Japanese
  (`assets/fonts/OFL-NotoSansJP.txt`), SIL Open Font License 1.1, embedded in the binary.
  The icon's letters are outlines of these fonts.

## License

MIT License, see [LICENSE](LICENSE). The bundled Noto Sans and Noto Sans JP
fonts keep their own license (SIL Open Font License 1.1).
