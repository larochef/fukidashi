//! Draws translations over the original text boxes into a transparent overlay.

use ab_glyph::{Font, FontVec, GlyphId, PxScale, ScaleFont, point};
use image::{DynamicImage, GenericImageView, Rgba, RgbaImage, imageops::FilterType};

use crate::{
    config::{RenderConfig, Screen},
    ocr::Block,
};

/// Noto Sans Bold (SIL Open Font License 1.1, see `assets/fonts/OFL.txt`),
/// embedded so the overlay looks the same everywhere, without any system font.
const BUNDLED_FONT: &[u8] = include_bytes!("../assets/fonts/NotoSans-Bold.ttf");
/// Noto Sans JP Bold (SIL Open Font License 1.1, see
/// `assets/fonts/OFL-NotoSansJP.txt`), for what the main font lacks: Japanese
/// kept by the translator (【】, 〇, names) and symbols (♡, ♪, ★, →).
const FALLBACK_FONT: &[u8] = include_bytes!("../assets/fonts/NotoSansJP-Bold.otf");
const LINE_SPACING: f32 = 1.15;
/// The detector pads its boxes by the same distance on every side, about a
/// sixth of the line height: the mask takes back this much of it all around,
/// so it hugs the Japanese as closely above and below as on the sides.
const MASK_INSET: f32 = 0.12;
/// Smallest font, as a fraction of the overlay width (24 px at 1920): Japanese is
/// compact, so dense menus need translations much smaller than the original glyphs.
const MIN_SIZE_RATIO: f32 = 1.0 / 80.0;
/// Above this many texts on screen, labels use a smaller font.
const DENSE_ITEMS: usize = 12;
/// A box is plain when this share of its pixels is within `PLAIN_TOLERANCE`
/// (per channel) of its median colour: the text strokes make up the rest.
const PLAIN_SHARE: f32 = 0.55;
const PLAIN_TOLERANCE: u8 = 20;

pub struct Renderer {
    /// The main font, then the fallback for the characters it lacks.
    fonts: Vec<FontVec>,
    width: u32,
    background: Rgba<u8>,
    text_color: Rgba<u8>,
    sample_background: bool,
    outline: f32,
    screen: Screen,
}

/// Where the game sits on the overlay, in pixels.
#[derive(Debug, Clone, Copy)]
pub struct Layout {
    pub width: u32,
    pub height: u32,
    game_x: u32,
    game_y: u32,
    game_width: u32,
    game_height: u32,
}

/// Colours of one translation.
#[derive(Debug, Clone, Copy)]
struct Colors {
    backdrop: Rgba<u8>,
    text: Rgba<u8>,
    outline: Rgba<u8>,
}

impl Renderer {
    pub fn new(config: &RenderConfig) -> anyhow::Result<Self> {
        let data = match &config.font {
            Some(path) => std::fs::read(path)?,
            None => BUNDLED_FONT.to_vec(),
        };
        Ok(Self {
            fonts: vec![FontVec::try_from_vec(data)?, FontVec::try_from_vec(FALLBACK_FONT.to_vec())?],
            width: config.width,
            background: Rgba(config.background),
            text_color: Rgba(config.text_color),
            sample_background: config.sample_background,
            outline: config.outline.max(0.0),
            screen: config.screen_aspect,
        })
    }

    /// Overlay geometry for a frame, at a fixed width. The game is shown at its
    /// display aspect, 4:3 or 16:9, whichever the frame is closer to (a PS2
    /// frame of 2560x1792 is shown at 4:3). RetroArch 1.22 and older stretch
    /// the overlay over the whole screen: the game then sits in the middle,
    /// between black bars, as on screen.
    pub fn layout(&self, frame_width: u32, frame_height: u32) -> Layout {
        let game = game_aspect(frame_width, frame_height);
        let screen = match self.screen {
            Screen::Auto => primary_screen_aspect().unwrap_or(game),
            Screen::Game => game,
            Screen::Aspect(aspect) => aspect,
        };
        let (width, height) = (self.width, ((self.width as f32 / screen).round() as u32).max(1));
        let (game_width, game_height) = if game < screen {
            (((height as f32 * game).round() as u32).min(width), height)
        } else {
            (width, ((width as f32 / game).round() as u32).min(height))
        };
        Layout { width, height, game_x: (width - game_width) / 2, game_y: (height - game_height) / 2, game_width, game_height }
    }

    /// Lays the translations out over their original text, then draws them.
    ///
    /// 1. Each original box gets the free space around it, bounded by its neighbours.
    /// 2. The text is fitted into that space: largest font first, original width first.
    /// 3. Panels that still overlap are pushed apart along the shortest axis.
    /// 4. The Japanese is always masked; stacked panels that nearly touch share a backdrop.
    ///    All backdrops are drawn before any text, so none can hide a neighbour's text.
    pub fn render(&self, frame: &DynamicImage, items: &[(&Block, &str)]) -> RgbaImage {
        let layout = self.layout(frame.width(), frame.height());
        let game = self.render_game(frame, items, layout.game_width, layout.game_height);
        if (layout.width, layout.height) == (layout.game_width, layout.game_height) {
            return game;
        }
        let mut img = RgbaImage::from_pixel(layout.width, layout.height, Rgba([0, 0, 0, 0]));
        image::imageops::replace(&mut img, &game, layout.game_x as i64, layout.game_y as i64);
        img
    }

    /// The overlay of the game area alone.
    fn render_game(&self, frame: &DynamicImage, items: &[(&Block, &str)], ow: u32, oh: u32) -> RgbaImage {
        let mut img = RgbaImage::from_pixel(ow, oh, Rgba([0, 0, 0, 0]));
        let (owf, ohf) = (ow as f32, oh as f32);
        let pad = (owf / 240.0).max(2.0);
        let bounds = Rect { x0: 0.0, y0: 0.0, x1: owf, y1: ohf };

        // Japanese punctuation the translation kept, written with glyphs the font has.
        let texts: Vec<String> = items.iter().map(|(_, t)| self.drawable(t)).collect();
        let items: Vec<(&Block, &str)> = items.iter().zip(&texts).map(|((b, _), t)| (*b, t.as_str())).collect();
        // Colours come from the whole original block, so the lines of a list match.
        let (items, colors): (Vec<(Block, &str, bool)>, Vec<Colors>) = items
            .iter()
            .filter(|(_, t)| !t.trim().is_empty())
            .flat_map(|&(b, t)| {
                let colors = self.colors(frame, b);
                split_list(b, t).into_iter().map(move |item| (item, colors))
            })
            .unzip();
        let anchors: Vec<Rect> = items
            .iter()
            .map(|(b, _, _)| {
                let inset = b.line_height * ohf * MASK_INSET;
                Rect {
                    x0: b.x0 * owf + inset - pad,
                    y0: b.y0 * ohf + inset,
                    x1: b.x1 * owf - inset + pad,
                    y1: b.y1 * ohf - inset,
                }
                .clamp_to(&bounds)
            })
            .collect();

        let spaces: Vec<Rect> = (0..items.len()).map(|i| free_space(i, &anchors, &bounds)).collect();
        // A screen crowded with labels (a map, a menu grid) reads better in a smaller font.
        let label_cap = if items.len() > DENSE_ITEMS { owf / 40.0 } else { f32::MAX };
        let fit_all = |caps: &[f32]| -> Vec<Panel> {
            items
                .iter()
                .enumerate()
                .map(|(i, (block, text, listed))| {
                    let cap = if block.lines.len() == 1 { caps[i].min(label_cap) } else { caps[i] };
                    Panel { index: i, ..self.fit(block, text, *listed, &anchors[i], &spaces[i], cap, ohf, pad, &bounds) }
                })
                .collect()
        };
        // Same-looking text gets the same size: each group of similar original glyph
        // heights uses the smallest size any of its members needed.
        let first = fit_all(&vec![f32::MAX; items.len()]);
        let caps = harmonize(&items.iter().map(|(b, _, _)| b.line_height).collect::<Vec<_>>(), &first);
        let mut panels = fit_all(&caps);
        resolve_overlaps(&mut panels, &bounds);

        for (anchor, colors) in anchors.iter().zip(&colors) {
            // The Japanese must not show through, even under a see-through backdrop.
            let mut mask = colors.backdrop;
            mask.0[3] = 255;
            fill_rounded(&mut img, anchor.x0, anchor.y0, anchor.w(), anchor.h(), pad, mask);
        }
        for (a, b) in stacked_pairs(&panels) {
            let (upper, lower) = if a.rect.y0 <= b.rect.y0 { (a, b) } else { (b, a) };
            let x0 = upper.rect.x0.max(lower.rect.x0);
            let x1 = upper.rect.x1.min(lower.rect.x1);
            let (y, h) = (upper.rect.y1 - pad, lower.rect.y0 - upper.rect.y1 + 2.0 * pad);
            fill_rounded(&mut img, x0, y, x1 - x0, h, 0.0, colors[upper.index].backdrop);
        }
        for panel in &panels {
            let r = &panel.rect;
            fill_rounded(&mut img, r.x0, r.y0, r.w(), r.h(), pad * 1.5, colors[panel.index].backdrop);
        }
        for panel in &panels {
            let r = &panel.rect;
            let scaled = self.fonts[0].as_scaled(PxScale::from(panel.size));
            let text_h = panel.lines.len() as f32 * panel.size * LINE_SPACING;
            let mut baseline = r.y0 + (r.h() - text_h) / 2.0 + scaled.ascent();
            for line in &panel.lines {
                let x = if panel.centered { r.x0 + (r.w() - self.text_width(line, panel.size)) / 2.0 } else { r.x0 + pad };
                self.draw_line(&mut img, line, x, baseline, panel.size, &colors[panel.index]);
                baseline += panel.size * LINE_SPACING;
            }
        }
        img
    }

    /// Picks the largest font (and, at that size, the narrowest width starting
    /// from the original one) that fits the free space, then places the panel
    /// on the original box, starting where the original starts.
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &self,
        block: &Block,
        text: &str,
        listed: bool,
        anchor: &Rect,
        space: &Rect,
        cap: f32,
        ohf: f32,
        pad: f32,
        bounds: &Rect,
    ) -> Panel {
        let line_h = block.line_height * ohf;
        let min_size = bounds.w() * MIN_SIZE_RATIO;
        let label = block.lines.len() == 1;
        // OCR boxes are taller than the glyphs, and Latin letters look bigger than
        // kana at the same size: a label (often one of many on a dense screen)
        // matches the original at 70% of the box, running text at 85%.
        let ratio = if label { 0.7 } else { 0.85 };
        let size_for = |ceiling: f32| (line_h * ratio).clamp(min_size, ceiling).min(cap).max(min_size);
        // Up to `comfortable`, text may grow past the original box into the free space.
        // A text box (several lines) may also take larger sizes, up to the original
        // glyphs, when the text then fills the box without growing it: a short
        // translation of a big dialogue box reads at the game's own size instead of
        // floating in empty space. Labels keep the comfortable size, so menus don't
        // collide.
        let comfortable = size_for(bounds.w() / 32.0);
        let max_size = if label { comfortable } else { size_for(bounds.w() / 24.0) };

        let measure = |size: f32, width: f32| {
            let lines = self.wrap(text, size, width - 2.0 * pad);
            let widest = lines.iter().map(|l| self.text_width(l, size)).fold(0.0, f32::max);
            let height = lines.len() as f32 * size * LINE_SPACING + 2.0 * pad;
            (lines, widest + 2.0 * pad, height)
        };

        let mut size = max_size;
        let (lines, width, height) = loop {
            let widths = [0.0, 0.33, 0.66, 1.0].map(|t| anchor.w() + (space.w() - anchor.w()).max(0.0) * t);
            let candidates: Vec<_> = widths
                .iter()
                .map(|&w| measure(size, w))
                .filter(|(_, width, height)| {
                    let max_h = if size > comfortable { anchor.h().min(space.h()) } else { space.h() };
                    *width <= space.w() + 0.5 && *height <= max_h + 0.5
                })
                .collect();
            // Keep the original line count when some width allows it (a one-line label
            // stays on one line), otherwise take the narrowest fit.
            let original_lines = text.lines().count().max(block.lines.len());
            let found = candidates
                .iter()
                .find(|(lines, _, _)| lines.len() <= original_lines)
                .or(candidates.first())
                .cloned();
            if let Some(fitted) = found {
                break fitted;
            }
            if size <= min_size {
                // Nothing fits: use the whole free width and let overlap resolution make room.
                break measure(size, space.w().max(anchor.w()));
            }
            size = (size * 0.92).max(min_size);
        };

        // Panels cover the original box when they can, so the result reads like the game's layout.
        let w = width.max(anchor.w().min(space.w()));
        let h = height.max(if label { 0.0 } else { anchor.h().min(space.h()) });
        // A label starts where its original starts and grows to the right: centred
        // on the original, a longer translation would start well before the Japanese.
        let (x0, y0) = if label { (anchor.x0, anchor.cy() - h / 2.0) } else { (anchor.x0, anchor.y0) };
        let rect = Rect { x0, y0, x1: x0 + w, y1: y0 + h }.shift_into(space).shift_into(bounds);
        Panel { index: 0, rect, size, lines, centered: label && !listed }
    }

    fn text_width(&self, text: &str, size: f32) -> f32 {
        self.layout_line(text, size).last().map_or(0.0, |&(font, id, x)| x + self.fonts[font].as_scaled(self.scale(font, size)).h_advance(id))
    }

    /// The font and glyph of each character, with its x offset from the start
    /// of the line (kerning between glyphs of the same font).
    fn layout_line(&self, text: &str, size: f32) -> Vec<(usize, GlyphId, f32)> {
        let mut caret = 0.0;
        let mut previous: Option<(usize, GlyphId)> = None;
        let mut glyphs = Vec::new();
        for c in text.chars() {
            let (font, id) = self.glyph(c);
            let scaled = self.fonts[font].as_scaled(self.scale(font, size));
            if let Some((prev_font, prev)) = previous {
                let prev_scaled = self.fonts[prev_font].as_scaled(self.scale(prev_font, size));
                caret += prev_scaled.h_advance(prev);
                if prev_font == font {
                    caret += scaled.kern(prev, id);
                }
            }
            glyphs.push((font, id, caret));
            previous = Some((font, id));
        }
        glyphs
    }

    /// The first font that has a glyph for `c` (the main one, drawing its
    /// empty box, when none has).
    fn glyph(&self, c: char) -> (usize, GlyphId) {
        self.fonts.iter().map(|f| f.glyph_id(c)).enumerate().find(|(_, id)| id.0 != 0).unwrap_or((0, GlyphId(0)))
    }

    /// The scale of font `font` whose letters match the main font's at `size`:
    /// `PxScale` sets the height from ascent to descent, which differs between fonts.
    fn scale(&self, font: usize, size: f32) -> PxScale {
        let em = |f: &FontVec| f.units_per_em().unwrap_or(1000.0) / f.height_unscaled();
        PxScale::from(size / em(&self.fonts[0]) * em(&self.fonts[font]))
    }

    /// Greedy word wrap, paragraph by paragraph; a single word wider than the
    /// box gets its own line.
    fn wrap(&self, text: &str, size: f32, max_width: f32) -> Vec<String> {
        let mut lines = Vec::new();
        for paragraph in text.lines() {
            let mut current = String::new();
            for word in paragraph.split_whitespace() {
                let candidate = if current.is_empty() { word.to_owned() } else { format!("{current} {word}") };
                if self.text_width(&candidate, size) <= max_width || current.is_empty() {
                    current = candidate;
                } else {
                    lines.push(std::mem::replace(&mut current, word.to_owned()));
                }
            }
            if !current.is_empty() {
                lines.push(current);
            }
        }
        lines
    }

    /// Gives the Japanese punctuation kept by the translator (【】, 「」) and
    /// full-width letters their Latin form, which spaces like the text around
    /// it; drops what no font can draw rather than drawing an empty box.
    fn drawable(&self, text: &str) -> String {
        text.chars()
            .flat_map(|c| {
                let latin: &[char] = match c {
                    '【' | '［' | '〔' => &['['],
                    '】' | '］' | '〕' => &[']'],
                    '「' | '『' => &['“'],
                    '」' | '』' => &['”'],
                    '（' => &['('],
                    '）' => &[')'],
                    '・' => &['·'],
                    '～' | '〜' => &['~'],
                    '、' => &[',', ' '],
                    '。' => &['.', ' '],
                    '：' => &[':', ' '],
                    '　' => &[' '],
                    _ => &[],
                };
                // Full-width ASCII (０-９, Ａ-Ｚ, ！-～) to its half-width form.
                let wide = ('！'..='～').contains(&c).then(|| char::from_u32(c as u32 - 0xFEE0)).flatten();
                if !latin.is_empty() {
                    latin.to_vec()
                } else if let Some(w) = wide {
                    vec![w]
                } else if self.glyph(c).1.0 != 0 {
                    vec![c]
                } else {
                    vec![]
                }
            })
            .collect()
    }

    /// The game's box colour behind the block when it is plain (most of the
    /// box, text strokes aside, within a small tolerance of its median colour),
    /// with black or white text for the best contrast; the configured colours
    /// otherwise (a picture or a gradient behind the text).
    fn colors(&self, frame: &DynamicImage, block: &Block) -> Colors {
        let backdrop = self.sample_background.then(|| plain_color(frame, block)).flatten();
        let (backdrop, text) = match backdrop {
            Some(backdrop) => {
                let l = luminance(backdrop);
                // WCAG contrast ratios against white and black.
                let white = 1.05 / (l + 0.05);
                let black = (l + 0.05) / 0.05;
                (backdrop, if white >= black { Rgba([255, 255, 255, 255]) } else { Rgba([0, 0, 0, 255]) })
            }
            None => (self.background, self.text_color),
        };
        let mut outline = backdrop;
        outline.0[3] = 255;
        Colors { backdrop, text, outline }
    }

    /// Draws one line of text, outlined: the coverage of the whole line is
    /// rasterized first, so the outline of a letter never covers its neighbour.
    fn draw_line(&self, img: &mut RgbaImage, text: &str, x: f32, baseline: f32, size: f32, colors: &Colors) {
        let glyphs: Vec<_> = self
            .layout_line(text, size)
            .into_iter()
            .filter_map(|(font, id, dx)| self.fonts[font].outline_glyph(id.with_scale_and_position(self.scale(font, size), point(x + dx, baseline))))
            .collect();
        let Some(first) = glyphs.first() else { return };

        let radius = self.outline * size;
        let margin = radius.ceil() as i32 + 1;
        let (mut min, mut max) = (first.px_bounds().min, first.px_bounds().max);
        for glyph in &glyphs {
            let b = glyph.px_bounds();
            (min.x, min.y, max.x, max.y) = (min.x.min(b.min.x), min.y.min(b.min.y), max.x.max(b.max.x), max.y.max(b.max.y));
        }
        let (left, top) = (min.x as i32 - margin, min.y as i32 - margin);
        let width = (max.x - min.x).ceil() as i32 + 2 * margin;
        let height = (max.y - min.y).ceil() as i32 + 2 * margin;
        let mut coverage = vec![0.0f32; (width * height) as usize];
        for glyph in &glyphs {
            let b = glyph.px_bounds();
            glyph.draw(|gx, gy, c| {
                let (px, py) = (b.min.x as i32 + gx as i32 - left, b.min.y as i32 + gy as i32 - top);
                if (0..width).contains(&px) && (0..height).contains(&py) {
                    let cell = &mut coverage[(py * width + px) as usize];
                    *cell = (*cell + c).min(1.0);
                }
            });
        }

        let mut paint = |cover: &[f32], color: Rgba<u8>| {
            for (i, &c) in cover.iter().enumerate() {
                let (px, py) = (left + i as i32 % width, top + i as i32 / width);
                if c > 0.0 && px >= 0 && py >= 0 && (px as u32) < img.width() && (py as u32) < img.height() {
                    let mut color = color;
                    color.0[3] = (color.0[3] as f32 * c) as u8;
                    blend(img.get_pixel_mut(px as u32, py as u32), color);
                }
            }
        };
        if radius > 0.0 {
            paint(&dilate(&coverage, width, height, radius), colors.outline);
        }
        paint(&coverage, colors.text);
    }
}

/// The shape the game is shown at. A frame that has exactly a usual shape has
/// square pixels and keeps it: 16:9 (PSP), 3:2 (Game Boy Advance), 10:9 (Game
/// Boy), 4:3. A much wider one has tall pixels, as consoles drawing 240 lines
/// over a TV screen (PS1 at 512x240, here 2048x960): it is shown at 4:3. Any
/// other is shown at the nearest of 4:3 and 16:9 (PS2 at 2560x1792: 4:3).
fn game_aspect(frame_width: u32, frame_height: u32) -> f32 {
    let frame = frame_width as f32 / frame_height.max(1) as f32;
    let distance = |aspect: f32| (frame / aspect).ln().abs();
    const SQUARE: [f32; 4] = [16.0 / 9.0, 3.0 / 2.0, 10.0 / 9.0, 4.0 / 3.0];
    if let Some(&aspect) = SQUARE.iter().find(|&&a| distance(a) < 0.02) {
        return aspect;
    }
    if frame > 2.0 || distance(4.0 / 3.0) < distance(16.0 / 9.0) { 4.0 / 3.0 } else { 16.0 / 9.0 }
}

/// Aspect of the primary screen's current mode, from `xrandr` (X11, or
/// XWayland under GNOME, KDE and most Wayland desktops).
fn primary_screen_aspect() -> Option<f32> {
    let output = std::process::Command::new("xrandr").arg("--current").output().ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    let connected: Vec<&str> = text.lines().filter(|l| l.contains(" connected")).collect();
    let line = connected.iter().find(|l| l.contains(" primary ")).or(connected.first())?;
    // "DP-8 connected primary 2560x1440+0+0 (normal ...": the size is the first
    // word shaped like WxH+X+Y.
    let geometry = line.split_whitespace().find(|w| w.contains('x') && w.contains('+'))?;
    let (w, h) = geometry.split('+').next()?.split_once('x')?;
    let (w, h) = (w.parse::<f32>().ok()?, h.parse::<f32>().ok()?);
    (w > 0.0 && h > 0.0).then_some(w / h)
}

/// Grows a coverage map by `radius` pixels (a disk, anti-aliased at its edge).
fn dilate(coverage: &[f32], width: i32, height: i32, radius: f32) -> Vec<f32> {
    let r = radius.ceil() as i32;
    let offsets: Vec<(i32, i32, f32)> = (-r..=r)
        .flat_map(|dy| (-r..=r).map(move |dx| (dx, dy)))
        .filter_map(|(dx, dy)| {
            let weight = (radius + 0.5 - ((dx * dx + dy * dy) as f32).sqrt()).min(1.0);
            (weight > 0.0).then_some((dx, dy, weight))
        })
        .collect();
    let mut out = vec![0.0f32; coverage.len()];
    for y in 0..height {
        for x in 0..width {
            let c = coverage[(y * width + x) as usize];
            if c <= 0.0 {
                continue;
            }
            for &(dx, dy, weight) in &offsets {
                let (nx, ny) = (x + dx, y + dy);
                if (0..width).contains(&nx) && (0..height).contains(&ny) {
                    let cell = &mut out[(ny * width + nx) as usize];
                    *cell = cell.max(c * weight);
                }
            }
        }
    }
    out
}

/// Median colour of the frame inside the block, when most of the block is
/// close to it: a plain box behind the text, rather than a picture.
fn plain_color(frame: &DynamicImage, block: &Block) -> Option<Rgba<u8>> {
    let (fw, fh) = (frame.width() as f32, frame.height() as f32);
    let (x0, x1) = ((block.x0 * fw) as u32, ((block.x1 * fw) as u32).min(frame.width()));
    let (y0, y1) = ((block.y0 * fh) as u32, ((block.y1 * fh) as u32).min(frame.height()));
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    // A few thousand samples are plenty, whatever the frame resolution.
    let step = (((x1 - x0) * (y1 - y0)) as f32 / 4000.0).sqrt().max(1.0) as usize;
    let pixels: Vec<[u8; 3]> = (y0..y1)
        .step_by(step)
        .flat_map(|y| (x0..x1).step_by(step).map(move |x| (x, y)))
        .map(|(x, y)| {
            let [r, g, b, _] = frame.get_pixel(x, y).0;
            [r, g, b]
        })
        .collect();
    let median: [u8; 3] = std::array::from_fn(|c| {
        let mut channel: Vec<u8> = pixels.iter().map(|p| p[c]).collect();
        channel.sort_unstable();
        channel[channel.len() / 2]
    });
    let close = pixels.iter().filter(|p| (0..3).all(|c| p[c].abs_diff(median[c]) <= PLAIN_TOLERANCE)).count();
    (close as f32 >= PLAIN_SHARE * pixels.len() as f32).then_some(Rgba([median[0], median[1], median[2], 255]))
}

/// Relative luminance (WCAG) of an sRGB colour.
fn luminance(color: Rgba<u8>) -> f32 {
    let linear = |v: u8| {
        let v = v as f32 / 255.0;
        if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
    };
    0.2126 * linear(color.0[0]) + 0.7152 * linear(color.0[1]) + 0.0722 * linear(color.0[2])
}

#[derive(Debug, Clone, Copy)]
struct Rect {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
}

impl Rect {
    fn w(&self) -> f32 {
        self.x1 - self.x0
    }
    fn h(&self) -> f32 {
        self.y1 - self.y0
    }
    fn cx(&self) -> f32 {
        (self.x0 + self.x1) / 2.0
    }
    fn cy(&self) -> f32 {
        (self.y0 + self.y1) / 2.0
    }
    fn clamp_to(self, outer: &Rect) -> Rect {
        Rect {
            x0: self.x0.clamp(outer.x0, outer.x1),
            y0: self.y0.clamp(outer.y0, outer.y1),
            x1: self.x1.clamp(outer.x0, outer.x1),
            y1: self.y1.clamp(outer.y0, outer.y1),
        }
    }
    /// Moves (without resizing) so as to stay inside `outer` as much as possible.
    fn shift_into(self, outer: &Rect) -> Rect {
        let dx = if self.x1 > outer.x1 { outer.x1 - self.x1 } else { 0.0 };
        let dx = if self.x0 + dx < outer.x0 { outer.x0 - self.x0 } else { dx };
        let dy = if self.y1 > outer.y1 { outer.y1 - self.y1 } else { 0.0 };
        let dy = if self.y0 + dy < outer.y0 { outer.y0 - self.y0 } else { dy };
        self.translate(dx, dy)
    }
    fn translate(self, dx: f32, dy: f32) -> Rect {
        Rect { x0: self.x0 + dx, y0: self.y0 + dy, x1: self.x1 + dx, y1: self.y1 + dy }
    }
    /// Overlap along each axis (negative: gap).
    fn overlap(&self, other: &Rect) -> (f32, f32) {
        (self.x1.min(other.x1) - self.x0.max(other.x0), self.y1.min(other.y1) - self.y0.max(other.y0))
    }
}

/// A block translated line for line (a list, a menu) is laid out as one item
/// per line, each on its original line; anything else stays one item. The
/// flag tells which items come from such a list.
fn split_list<'a>(block: &Block, text: &'a str) -> Vec<(Block, &'a str, bool)> {
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    if block.lines.len() < 2 || lines.len() != block.lines.len() || block.line_boxes.len() != block.lines.len() {
        return vec![(block.clone(), text, false)];
    }
    lines
        .into_iter()
        .zip(block.lines.iter().zip(&block.line_boxes))
        .map(|(translation, (original, &[x0, y0, x1, y1]))| {
            let line = Block {
                x0,
                y0,
                x1,
                y1,
                line_height: block.line_height,
                lines: vec![original.clone()],
                line_boxes: vec![[x0, y0, x1, y1]],
                item: block.item,
            };
            (line, translation, true)
        })
        .collect()
}

/// A translation ready to draw: its backdrop and wrapped lines.
struct Panel {
    /// Index of the item it translates.
    index: usize,
    rect: Rect,
    size: f32,
    lines: Vec<String>,
    centered: bool,
}

/// Room a box may grow into: up to its neighbours on the same row (left/right)
/// or in the same column (above/below), and within a sensible distance of the
/// original so the translation stays where the player expects it.
fn free_space(i: usize, anchors: &[Rect], bounds: &Rect) -> Rect {
    const GAP: f32 = 6.0;
    let a = anchors[i];
    let mut space = Rect {
        x0: a.x0 - bounds.w() * 0.12,
        y0: a.y0 - a.h() * 0.5,
        x1: a.x1 + bounds.w() * 0.2,
        y1: a.y1 + a.h().max(bounds.h() * 0.05) * 1.5,
    }
    .clamp_to(bounds);
    for (j, b) in anchors.iter().enumerate() {
        if j == i {
            continue;
        }
        let (ox, oy) = a.overlap(b);
        if oy > 0.0 && ox <= 0.0 {
            if b.cx() > a.cx() {
                space.x1 = space.x1.min(b.x0 - GAP).max(a.x1);
            } else {
                space.x0 = space.x0.max(b.x1 + GAP).min(a.x0);
            }
        } else if ox > 0.0 && oy <= 0.0 {
            if b.cy() > a.cy() {
                space.y1 = space.y1.min(b.y0 - GAP).max(a.y1);
            } else {
                space.y0 = space.y0.max(b.y1 + GAP).min(a.y0);
            }
        }
    }
    space
}

/// Groups blocks whose original glyphs have similar heights (within 20%) and
/// returns, for each block, the smallest font size fitted in its group.
fn harmonize(line_heights: &[f32], fitted: &[Panel]) -> Vec<f32> {
    let mut order: Vec<usize> = (0..line_heights.len()).collect();
    order.sort_by(|&a, &b| line_heights[a].total_cmp(&line_heights[b]));
    let mut caps = vec![f32::MAX; line_heights.len()];
    let mut start = 0;
    while start < order.len() {
        let base = line_heights[order[start]];
        let end = order[start..].iter().position(|&i| line_heights[i] > base * 1.2).map_or(order.len(), |p| start + p);
        let size = order[start..end].iter().map(|&i| fitted[i].size).fold(f32::MAX, f32::min);
        for &i in &order[start..end] {
            caps[i] = size;
        }
        start = end;
    }
    caps
}

/// Pushes overlapping panels apart along the axis where they overlap least,
/// half each way, until nothing overlaps (or the iteration budget runs out).
fn resolve_overlaps(panels: &mut [Panel], bounds: &Rect) {
    const MARGIN: f32 = 4.0;
    for _ in 0..100 {
        let mut moved = false;
        for i in 0..panels.len() {
            for j in i + 1..panels.len() {
                let (a, b) = (panels[i].rect, panels[j].rect);
                let (ox, oy) = a.overlap(&b);
                if ox <= -MARGIN || oy <= -MARGIN {
                    continue;
                }
                moved = true;
                let (dx, dy) = if ox < oy {
                    let d = (ox + MARGIN) / 2.0;
                    (if a.cx() <= b.cx() { -d } else { d }, 0.0)
                } else {
                    let d = (oy + MARGIN) / 2.0;
                    (0.0, if a.cy() <= b.cy() { -d } else { d })
                };
                panels[i].rect = a.translate(dx, dy).shift_into(bounds);
                panels[j].rect = b.translate(-dx, -dy).shift_into(bounds);
            }
        }
        if !moved {
            break;
        }
    }
}

/// Panels one above the other, of similar extent and almost touching: they
/// read as one box in the game, so they are drawn on a shared backdrop.
fn stacked_pairs(panels: &[Panel]) -> Vec<(&Panel, &Panel)> {
    let mut pairs = Vec::new();
    for (i, a) in panels.iter().enumerate() {
        for b in &panels[i + 1..] {
            let (ox, oy) = a.rect.overlap(&b.rect);
            let narrower = a.rect.w().min(b.rect.w());
            if ox >= 0.7 * narrower && -oy < 0.6 * a.size.min(b.size) {
                pairs.push((a, b));
            }
        }
    }
    pairs
}

/// Composites the overlay over the frame, as RetroArch would display it.
pub fn composite(frame: &DynamicImage, overlay: &RgbaImage, layout: Layout) -> RgbaImage {
    let game = frame.resize_exact(layout.game_width, layout.game_height, FilterType::Nearest).to_rgba8();
    let mut out = RgbaImage::from_pixel(layout.width, layout.height, Rgba([0, 0, 0, 255]));
    image::imageops::replace(&mut out, &game, layout.game_x as i64, layout.game_y as i64);
    for (dst, src) in out.pixels_mut().zip(overlay.pixels()) {
        blend(dst, *src);
    }
    out
}

/// Draws the OCR boxes over the frame, for debugging detection.
pub fn draw_boxes(frame: &DynamicImage, blocks: &[Block], width: u32) -> RgbaImage {
    let height = (frame.height() as f32 * width as f32 / frame.width() as f32).round() as u32;
    let mut out = frame.resize_exact(width, height, FilterType::Nearest).to_rgba8();
    let (w, h) = (width as f32, height as f32);
    let color = Rgba([0, 255, 0, 255]);
    for b in blocks {
        let (x0, y0) = ((b.x0 * w) as u32, (b.y0 * h) as u32);
        let (x1, y1) = (((b.x1 * w) as u32).min(width - 1), ((b.y1 * h) as u32).min(height - 1));
        for t in 0..3 {
            for x in x0..=x1 {
                out.put_pixel(x, (y0 + t).min(height - 1), color);
                out.put_pixel(x, y1.saturating_sub(t), color);
            }
            for y in y0..=y1 {
                out.put_pixel((x0 + t).min(width - 1), y, color);
                out.put_pixel(x1.saturating_sub(t), y, color);
            }
        }
    }
    out
}

fn fill_rounded(img: &mut RgbaImage, x: f32, y: f32, w: f32, h: f32, radius: f32, color: Rgba<u8>) {
    let (x0, y0) = (x.max(0.0) as u32, y.max(0.0) as u32);
    let x1 = ((x + w).ceil() as u32).min(img.width());
    let y1 = ((y + h).ceil() as u32).min(img.height());
    for py in y0..y1 {
        for px in x0..x1 {
            // Distance outside the rounded corner; anti-aliased over one pixel.
            let cx = (px as f32 + 0.5).clamp(x + radius, x + w - radius);
            let cy = (py as f32 + 0.5).clamp(y + radius, y + h - radius);
            let d = ((px as f32 + 0.5 - cx).powi(2) + (py as f32 + 0.5 - cy).powi(2)).sqrt() - radius;
            let coverage = (0.5 - d).clamp(0.0, 1.0);
            // Backdrops merge into one shape instead of darkening where they
            // overlap: the most opaque one sets the pixel, the latest on a tie.
            let alpha = (color.0[3] as f32 * coverage) as u8;
            let dst = img.get_pixel_mut(px, py);
            if alpha > 0 && alpha >= dst.0[3] {
                *dst = Rgba([color.0[0], color.0[1], color.0[2], alpha]);
            }
        }
    }
}

/// Straight-alpha "source over" blending.
fn blend(dst: &mut Rgba<u8>, src: Rgba<u8>) {
    let sa = src.0[3] as f32 / 255.0;
    if sa <= 0.0 {
        return;
    }
    let da = dst.0[3] as f32 / 255.0;
    let out_a = sa + da * (1.0 - sa);
    for i in 0..3 {
        let c = (src.0[i] as f32 * sa + dst.0[i] as f32 * da * (1.0 - sa)) / out_a;
        dst.0[i] = c.round() as u8;
    }
    dst.0[3] = (out_a * 255.0).round() as u8;
}




#[cfg(test)]
mod tests {
    use super::game_aspect;

    #[test]
    fn shows_frames_at_their_console_shape() {
        let shape = |w, h| (game_aspect(w, h) * 100.0).round() / 100.0;
        // Square pixels: the frame's own shape.
        assert_eq!(shape(480, 272), 1.78); // PSP
        assert_eq!(shape(1440, 816), 1.78); // PSP, upscaled
        assert_eq!(shape(240, 160), 1.5); // Game Boy Advance
        assert_eq!(shape(160, 144), 1.11); // Game Boy
        assert_eq!(shape(640, 480), 1.33);
        // Tall pixels: a TV screen.
        assert_eq!(shape(2048, 960), 1.33); // PS1, 512x240 upscaled
        assert_eq!(shape(256, 224), 1.33); // SNES
        assert_eq!(shape(2560, 1792), 1.33); // PS2
        assert_eq!(shape(1920, 1000), 1.78);
    }
}
