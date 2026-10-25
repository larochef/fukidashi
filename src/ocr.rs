//! Text detection and transcription with PP-OCR (detection + recognition ONNX
//! models, run on CPU), then grouping of lines into blocks.

use std::{
    collections::HashMap,
    sync::{Arc, LazyLock},
};

use anyhow::anyhow;
use image::{DynamicImage, GrayImage, imageops::FilterType};
use oar_ocr::{
    domain::tasks::TextDetectionConfig,
    prelude::{OAROCR, OAROCRBuilder},
    processors::LimitType,
};

use serde::{Deserialize, Serialize};

use crate::config::OcrConfig;

/// One block of text, with its box normalized to 0..1 of the frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    /// Height of one line of text, normalized to the frame height.
    pub line_height: f32,
    /// Original lines of the block.
    pub lines: Vec<String>,
    /// Box of each original line, normalized like the block's: lists keep
    /// each translated line at the place of its original.
    pub line_boxes: Vec<[f32; 4]>,
    /// One entry of a bulleted list (a choice): never merged with its neighbours.
    pub item: bool,
}

impl Block {
    /// The text as sent to the translator, with ⏎ between original lines.
    pub fn joined(&self) -> String {
        self.lines.join("⏎")
    }
}

pub struct Ocr {
    engine: Arc<OAROCR>,
}

impl Ocr {
    pub fn new(config: &OcrConfig) -> anyhow::Result<Self> {
        let engine = OAROCRBuilder::new(config.detection.as_str(), config.recognition.as_str(), config.dictionary.as_str())
            // The pipeline's "general" defaults, but for the box threshold.
            .text_detection_config(TextDetectionConfig {
                box_threshold: config.box_threshold,
                unclip_ratio: 2.0,
                limit_side_len: Some(960),
                limit_type: Some(LimitType::Max),
                max_side_len: Some(4000),
                ..Default::default()
            })
            .build()
            .map_err(|e| anyhow!("cannot load PP-OCR models: {e}"))?;
        Ok(Self { engine: Arc::new(engine) })
    }

    pub async fn detect(&self, config: &OcrConfig, frame: &DynamicImage) -> anyhow::Result<Vec<Block>> {
        let scaled = resize_to_width(frame, config.width);
        let blocks = recognize(self.engine.clone(), config, scaled).await?;
        let mut blocks = merge_stacked(merge_continued(blocks));
        for block in &mut blocks {
            fix_ra_read_as_five(&mut block.lines);
        }
        // A lone kana is a stray fragment (a cursor, an icon) or one key of a
        // kana keyboard: nothing to translate. A lone kanji can be a label (音).
        blocks.retain(|b| {
            let mut chars = b.joined().chars().filter(|c| !c.is_whitespace()).collect::<Vec<_>>();
            chars.dedup();
            !(chars.len() == 1 && matches!(chars[0], '\u{3041}'..='\u{30ff}'))
        });
        Ok(blocks)
    }
}

fn resize_to_width(frame: &DynamicImage, width: u32) -> DynamicImage {
    if frame.width() == width {
        return frame.clone();
    }
    let height = (frame.height() as f32 * width as f32 / frame.width() as f32).round() as u32;
    // Nearest keeps pixel fonts sharp on integer upscales; anything else needs filtering.
    let filter = if width.is_multiple_of(frame.width()) { FilterType::Nearest } else { FilterType::Triangle };
    frame.resize_exact(width, height.max(1), filter)
}

/// A recognized line (or line fragment), in pixels of the OCR input image.
#[derive(Debug, Clone)]
struct Line {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    text: String,
}

impl Line {
    fn height(&self) -> f32 {
        self.y1 - self.y0
    }
}

async fn recognize(engine: Arc<OAROCR>, config: &OcrConfig, image: DynamicImage) -> anyhow::Result<Vec<Block>> {
    let (width, height) = (image.width() as f32, image.height() as f32);
    let rgb = image.to_rgb8();
    let gray = image.to_luma8();
    let results = tokio::task::spawn_blocking(move || engine.predict(vec![rgb]))
        .await?
        .map_err(|e| anyhow!("PP-OCR failed: {e}"))?;

    let lines: Vec<Line> = results
        .first()
        .map(|r| r.text_regions.as_slice())
        .unwrap_or_default()
        .iter()
        .filter(|r| r.confidence.unwrap_or(0.0) >= config.min_confidence)
        .filter_map(|r| {
            let text = to_japanese_forms(strip_cursor(r.text.as_deref()?.trim()));
            // Only Japanese is worth translating: drops icons, digits and stray Latin.
            text.chars().any(is_japanese).then(|| Line {
                x0: r.bounding_box.x_min(),
                y0: r.bounding_box.y_min(),
                x1: r.bounding_box.x_max(),
                y1: r.bounding_box.y_max(),
                text,
            })
        })
        .collect();
    tracing::debug!(?lines, "ocr lines");

    Ok(group(join_rows(split_columns(drop_furigana(lines))))
        .into_iter()
        .flat_map(|rows| split_items(rows, &gray))
        .map(|(rows, item)| to_block(&rows, width, height, item))
        .collect())
}

/// Drops furigana: small kana right above a line at least 1.8 times as tall,
/// within its width. They repeat the reading of the kanji below and would end
/// up as a line of their own. Small kana anywhere else (a menu hint) stay.
fn drop_furigana(lines: Vec<Line>) -> Vec<Line> {
    let kana = |c: char| matches!(c, '\u{3041}'..='\u{30ff}');
    let is_furigana = |small: &Line| {
        let h = small.height();
        small.text.chars().all(kana)
            && lines.iter().any(|big| {
                let gap = big.y0 - small.y1;
                let center = (small.x0 + small.x1) / 2.0;
                big.height() >= 1.8 * h && (-0.3 * h..0.3 * h).contains(&gap) && (big.x0..big.x1).contains(&center)
            })
    };
    let keep: Vec<bool> = lines.iter().map(|l| !is_furigana(l)).collect();
    lines.into_iter().zip(keep).filter_map(|(l, k)| k.then_some(l)).collect()
}

/// Splits a column of single characters read as one vertical line: the first
/// characters of names spaced over the width of a line (相　原「…⏎咲　野「…
/// read as 相咲, 原「…, 野「…). Each character goes back to the row on its right
/// at its height, as many rows as characters.
fn split_columns(lines: Vec<Line>) -> Vec<Line> {
    let mut out = Vec::new();
    for line in &lines {
        let chars: Vec<char> = line.text.chars().collect();
        let (w, h) = (line.x1 - line.x0, line.height());
        let rows: Vec<&Line> = lines
            .iter()
            .filter(|r| {
                let cy = (r.y0 + r.y1) / 2.0;
                r.x0 > line.x1 - 0.2 * w && r.x0 - line.x1 < 2.0 * r.height() && (line.y0..line.y1).contains(&cy)
            })
            .collect();
        if chars.len() < 2 || h < 1.6 * w || rows.len() != chars.len() {
            out.push(line.clone());
            continue;
        }
        let step = h / chars.len() as f32;
        for (i, c) in chars.into_iter().enumerate() {
            let (y0, y1) = (line.y0 + i as f32 * step, line.y0 + (i + 1) as f32 * step);
            out.push(Line { x0: line.x0, y0, x1: line.x1, y1, text: c.to_string() });
        }
    }
    out
}

/// Joins fragments of the same visual line (detection often splits a line at
/// punctuation or wide gaps).
fn join_rows(mut lines: Vec<Line>) -> Vec<Line> {
    lines.sort_by(|a, b| a.x0.total_cmp(&b.x0));
    let mut rows: Vec<Line> = Vec::new();
    for line in lines {
        let h = line.height();
        // A speaker's name is spaced out, away from its 「 (柊　「…, 相　原「…), and
        // read in boxes of their own, a little off the line, sometimes
        // overlapping it: then the overlapped characters are read twice
        // (相原 + 原「…), or garbled (咲里 + 野「…).
        let is_name = |row: &Line| row.text.chars().count() <= 2 && line.text.chars().take(2).any(|c| c == '「');
        let same_row = rows.iter_mut().find(|row| {
            let name = is_name(row);
            let overlap = line.y1.min(row.y1) - line.y0.max(row.y0);
            let gap = line.x0 - row.x1;
            let ratio = h / row.height();
            // Menu entries laid out on one row are further apart than fragments of a
            // sentence, except after ？ or ！, which a full-width space follows.
            let max_gap = if row.text.ends_with(['？', '！', '?', '!']) || name { 1.2 * h } else { 0.35 * h };
            let (min_gap, ratios) = if name { (-(row.x1 - row.x0), 0.6..=1.6) } else { (-0.5 * h, 0.8..=1.25) };
            overlap > 0.6 * h.min(row.height()) && ratios.contains(&ratio) && gap < max_gap && gap > min_gap
        });
        match same_row {
            Some(row) if is_name(row) => {
                let chars = row.text.chars().count();
                let overlapped = ((row.x1 - line.x0).max(0.0) / ((row.x1 - row.x0) / chars as f32)).round() as usize;
                row.text = row.text.chars().take(chars - overlapped.min(chars)).chain(line.text.chars()).collect();
                // The line's own box: the name's may stick out above or below it.
                (row.x1, row.y0, row.y1) = (line.x1, line.y0, line.y1);
            }
            Some(row) => {
                row.text.push_str(&line.text);
                row.x1 = row.x1.max(line.x1);
                row.y0 = row.y0.min(line.y0);
                row.y1 = row.y1.max(line.y1);
            }
            None => rows.push(line),
        }
    }
    rows
}

/// Stacks rows into blocks: a row joins the block right above it when it has a
/// similar height, starts close below it and overlaps it horizontally.
fn group(mut rows: Vec<Line>) -> Vec<Vec<Line>> {
    rows.sort_by(|a, b| a.y0.total_cmp(&b.y0));
    let mut blocks: Vec<Vec<Line>> = Vec::new();
    for row in rows {
        let h = row.height();
        let target = blocks.iter_mut().rev().find(|block| {
            let last = block.last().expect("blocks are never empty");
            // Compared with the block's typical line rather than the last one.
            let ratio = h / median_height(block);
            let gap = row.y0 - last.y1;
            // Paragraph lines share their left edge, their right edge (long justified
            // lines; short menu labels can line up by chance) or their center
            // (centered text: a short entry under a paragraph is a menu, not the
            // paragraph's last line).
            let long = |l: &Line| l.x1 - l.x0 > 10.0 * l.height();
            // Long lines tolerate more height noise: they are paragraph text, where
            // a stray glyph at the edge inflates a single line's box.
            // Lines whose boxes overlap deeply are one piece of text (a slanted logo:
            // レベル⏎アップ!!), even when punctuation makes one of them taller.
            let max_ratio = if last.y1 - row.y0 > 0.25 * last.height() { 1.5 } else if long(&row) && long(last) { 1.35 } else { 1.25 };
            // Two labels (a few characters, no prose punctuation) are a list (a
            // profile card) only when their left edges line up closely: labels
            // scattered over a map or buttons stacked beside icons start a little apart.
            let label = |l: &Line| l.text.chars().count() <= 6 && !l.text.contains(['。', '、', '「', '」', '『', '』', '…']);
            let left = if label(&row) && label(last) { 0.25 * h } else { h };
            let aligned = (row.x0 - last.x0).abs() < left
                || (long(&row) && long(last) && (row.x1 - last.x1).abs() < 0.5 * h)
                || ((row.x0 + row.x1) - (last.x0 + last.x1)).abs() < 0.5 * h;
            // Lines of a paragraph nearly touch (detection boxes even overlap); a
            // title or a separate box leaves a clear gap.
            (0.8..=max_ratio).contains(&ratio)
                && row.y0 > last.y0 + 0.5 * last.height()
                && gap < 0.2 * h.max(last.height())
                && aligned
        });
        match target {
            Some(block) => block.push(row),
            None => blocks.push(vec![row]),
        }
    }
    blocks
}

/// Splits a bulleted list (the choices of a dialogue) into one block per entry:
/// read as one text, the translator writes the choices as a single sentence.
/// An entry starts at a line with a marker; the markers of two lines or more
/// line up in a column, to the pixel.
fn split_items(rows: Vec<Line>, gray: &GrayImage) -> Vec<(Vec<Line>, bool)> {
    let markers: Vec<Option<f32>> = rows.iter().map(|row| marker(gray, row)).collect();
    let xs: Vec<f32> = markers.iter().flatten().copied().collect();
    let h = median_height(&rows);
    if xs.len() < 2 || xs.iter().any(|x| (x - xs[0]).abs() > 0.1 * h) {
        return vec![(rows, false)];
    }
    tracing::debug!(?markers, first = %rows[0].text, "list markers");
    let mut items: Vec<(Vec<Line>, bool)> = Vec::new();
    for (row, marker) in rows.into_iter().zip(markers) {
        match items.last_mut() {
            Some((lines, _)) if marker.is_none() => lines.push(row),
            _ => items.push((vec![row], marker.is_some())),
        }
    }
    items
}

/// The centre of a list marker (・, ●, ▶) left of a line, which the detector
/// skips: a small blob standing out from the background, centred on the line,
/// apart from the text. The leftmost one, as an ellipsis can start the text.
fn marker(gray: &GrayImage, line: &Line) -> Option<f32> {
    let h = line.height();
    let cy = (line.y0 + line.y1) / 2.0;
    let (x0, x1) = ((line.x0 - 1.2 * h).max(0.0) as u32, (line.x0 - 0.05 * h).max(0.0) as u32);
    let (y0, y1) = ((cy - 0.45 * h).max(0.0) as u32, ((cy + 0.45 * h) as u32).min(gray.height()));
    if x1 < x0 + 3 || y1 < y0 + 3 {
        return None;
    }
    let (w, ht) = ((x1 - x0) as usize, (y1 - y0) as usize);
    let luma: Vec<u8> = (y0..y1).flat_map(|y| (x0..x1).map(move |x| gray.get_pixel(x, y).0[0])).collect();
    let mut sorted = luma.clone();
    sorted.sort_unstable();
    let background = sorted[sorted.len() / 2];
    let mut ink: Vec<bool> = luma.iter().map(|&l| l.abs_diff(background) > 64).collect();
    let mut found: Option<f32> = None;
    for start in 0..ink.len() {
        if !ink[start] {
            continue;
        }
        // Flood fill of one blob, tracking its bounding box.
        ink[start] = false;
        let mut stack = vec![start];
        let (mut bx0, mut by0, mut bx1, mut by1) = (w, ht, 0, 0);
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            (bx0, by0, bx1, by1) = (bx0.min(x), by0.min(y), bx1.max(x), by1.max(y));
            let neighbours = [(x > 0).then(|| i - 1), (x + 1 < w).then(|| i + 1), (y > 0).then(|| i - w), (y + 1 < ht).then(|| i + w)];
            for n in neighbours.into_iter().flatten() {
                if ink[n] {
                    ink[n] = false;
                    stack.push(n);
                }
            }
        }
        // A blob touching the edge is something bigger: a frame, a picture.
        let edge = bx0 == 0 || by0 == 0 || bx1 + 1 == w || by1 + 1 == ht;
        let (bw, bh) = ((bx1 - bx0 + 1) as f32, (by1 - by0 + 1) as f32);
        let centre = y0 as f32 + (by0 + by1) as f32 / 2.0;
        let small = (0.04 * h..=0.2 * h).contains(&bw) && (0.04 * h..=0.2 * h).contains(&bh);
        // Specks of a picture behind the text sit anywhere; a marker sits on the
        // line's middle.
        if !edge && small && (centre - cy).abs() < 0.08 * h {
            let x = x0 as f32 + (bx0 + bx1) as f32 / 2.0;
            found = Some(found.map_or(x, |f: f32| f.min(x)));
        }
    }
    found
}

fn median_height(rows: &[Line]) -> f32 {
    let mut heights: Vec<f32> = rows.iter().map(Line::height).collect();
    heights.sort_by(f32::total_cmp);
    heights[heights.len() / 2]
}

fn to_block(rows: &[Line], width: f32, height: f32, item: bool) -> Block {
    Block {
        x0: rows.iter().map(|r| r.x0).fold(f32::MAX, f32::min) / width,
        y0: rows.iter().map(|r| r.y0).fold(f32::MAX, f32::min) / height,
        x1: rows.iter().map(|r| r.x1).fold(0.0, f32::max) / width,
        y1: rows.iter().map(|r| r.y1).fold(0.0, f32::max) / height,
        line_height: median_height(rows) / height,
        lines: rows.iter().map(|r| r.text.clone()).collect(),
        line_boxes: rows.iter().map(|r| [r.x0 / width, r.y0 / height, r.x1 / width, r.y1 / height]).collect(),
        item,
    }
}

/// Drops the game's "next page" cursor, read as an opening quote ending the line
/// (Japanese never ends a line on one) or as one or two characters without kana
/// after a closing quote ("1", "心").
fn strip_cursor(text: &str) -> &str {
    let text = text.trim_end_matches(['「', '『']).trim_end();
    match text.rfind(['」', '』']) {
        Some(i) => {
            let end = i + '」'.len_utf8();
            let tail = &text[end..];
            let junk = !tail.is_empty() && tail.chars().count() <= 2 && !tail.chars().any(|c| matches!(c, '\u{3040}'..='\u{30ff}'));
            if junk { &text[..end] } else { text }
        }
        None => text,
    }
}

/// Japanese form of the Chinese glyphs PP-OCR reads instead (决 → 決, 带 → 帯).
static JAPANESE_FORMS: LazyLock<HashMap<char, char>> = LazyLock::new(|| {
    include_str!("../assets/simplified-to-japanese.txt")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .filter_map(|line| {
            let mut chars = line.chars();
            Some((chars.next()?, chars.next()?))
        })
        .collect()
});

fn to_japanese_forms(text: &str) -> String {
    text.chars().map(|c| JAPANESE_FORMS.get(&c).copied().unwrap_or(c)).collect()
}

/// PP-OCR reads ら as 5 in some fonts (「…怒ってるか⏎5、口きいて」). A lone 5 is
/// taken for ら when what surrounds it reads as text, not as a number: followed
/// by hiragana other than a counter (5つ) or a particle (第5の), even on the next
/// line; or followed by a particle or punctuation after kana or punctuation
/// (さ5に, ⏎5、), where a number would follow a kanji (第5、) or a digit.
fn fix_ra_read_as_five(lines: &mut [String]) {
    const PARTICLES: &str = "のはをにがとでもへやか";
    let chars: Vec<(usize, char)> = lines.iter().enumerate().flat_map(|(i, l)| l.chars().map(move |c| (i, c))).collect();
    let digit = |c: char| c.is_ascii_digit() || ('０'..='９').contains(&c);
    let hiragana = |c: char| ('ぁ'..='ゖ').contains(&c);
    let number_context = |c: char| digit(c) || c.is_ascii_alphabetic() || (is_japanese(c) && !matches!(c, '\u{3040}'..='\u{30ff}'));
    let mut fixed = vec![String::new(); lines.len()];
    for (k, &(line, c)) in chars.iter().enumerate() {
        let prev = k.checked_sub(1).map(|p| chars[p].1);
        let next = chars.get(k + 1).map(|&(_, n)| n);
        let lone_five = matches!(c, '5' | '５') && !prev.is_some_and(digit) && !next.is_some_and(digit);
        let ra = lone_five
            && match next {
                Some(n) if hiragana(n) && n != 'つ' && !PARTICLES.contains(n) => true,
                Some(n) if PARTICLES.contains(n) || "、。」』！？!?…".contains(n) => !prev.is_some_and(number_context),
                _ => false,
            };
        fixed[line].push(if ra { 'ら' } else { c });
    }
    for (line, text) in lines.iter_mut().zip(fixed) {
        *line = text;
    }
}

fn is_japanese(c: char) -> bool {
    matches!(c, '\u{3040}'..='\u{30ff}' | '\u{3400}'..='\u{4dbf}' | '\u{4e00}'..='\u{9fff}' | '\u{ff66}'..='\u{ff9d}')
}

/// A block continues in the block right below it, starting under its text,
/// when it opens a quote or a parenthesis without closing it (「…, 相原（…),
/// or when its last line is cut mid-sentence, by a comma or long: the game wraps
/// dialogue with a hanging indent that the row grouping can't follow, and the
/// translator needs whole sentences. Short labels never continue this way.
fn merge_continued(blocks: Vec<Block>) -> Vec<Block> {
    const OPENING: [char; 4] = ['「', '『', '（', '('];
    const CLOSING: [char; 4] = ['」', '』', '）', ')'];
    const SENTENCE_END: [char; 9] = ['。', '！', '？', '!', '?', '」', '』', '）', ')'];
    let mut merged: Vec<Block> = Vec::new();
    for block in blocks {
        if let Some(last) = merged.last_mut().filter(|last| !last.item && !block.item) {
            let text = last.joined();
            let open = text.matches(OPENING).count() > text.matches(CLOSING).count();
            let line = last.lines.last().map(|l| l.trim_end()).unwrap_or_default();
            // Cut after a comma, or a long line without final punctuation (a short
            // one is a label or a form field: 苗字入力ふらんそわ).
            let cut = line.ends_with(['、', ',']) || (line.chars().count() >= 12 && !line.ends_with(SENTENCE_END));
            let h = last.line_height;
            let ratio = block.line_height / h;
            // An open quote may continue after a wider gap (a block of its own).
            let max_gap = if open { last.y1 - last.y0 } else { h };
            let below = block.y0 >= last.y0 && block.y0 - last.y1 < max_gap;
            let under = block.x0 > last.x0 - h && block.x0 < last.x1;
            // The rest of a sentence reads as prose (hiragana), not as a label (決定).
            let prose = block.lines.first().is_some_and(|l| l.chars().any(|c| ('ぁ'..='ゖ').contains(&c)));
            let continues = cut && prose && under && (0.8..=1.25).contains(&ratio);
            if (open || continues) && below {
                last.x0 = last.x0.min(block.x0);
                last.x1 = last.x1.max(block.x1);
                last.y1 = last.y1.max(block.y1);
                last.lines.extend(block.lines);
                last.line_boxes.extend(block.line_boxes);
                continue;
            }
        }
        merged.push(block);
    }
    merged
}

/// Merges blocks of nearly the same width stacked with almost no gap between
/// them (as within a paragraph): they are one text box in the game, and splitting them gives
/// the translator fragments and the reader mismatched boxes.
fn merge_stacked(mut blocks: Vec<Block>) -> Vec<Block> {
    blocks.sort_by(|a, b| a.y0.total_cmp(&b.y0));
    let mut merged: Vec<Block> = Vec::new();
    for block in blocks {
        let target = merged.iter_mut().rev().find(|above| {
            let overlap = block.x1.min(above.x1) - block.x0.max(above.x0);
            let wider = (block.x1 - block.x0).max(above.x1 - above.x0);
            let line = above.line_height.min(block.line_height);
            let ratio = block.line_height / above.line_height;
            let gap = block.y0 - above.y1;
            !above.item && !block.item && overlap >= 0.75 * wider && gap < 0.2 * line && gap > -line && (0.8..=1.25).contains(&ratio)
        });
        match target {
            Some(above) => {
                above.x0 = above.x0.min(block.x0);
                above.x1 = above.x1.max(block.x1);
                above.y1 = above.y1.max(block.y1);
                above.lines.extend(block.lines);
                above.line_boxes.extend(block.line_boxes);
            }
            None => merged.push(block),
        }
    }
    merged
}

#[cfg(test)]
mod tests {
    use image::{GrayImage, Luma};

    use super::{Block, Line, drop_furigana, fix_ra_read_as_five, join_rows, merge_continued, split_columns, split_items, to_japanese_forms};

    fn fix(lines: &[&str]) -> Vec<String> {
        let mut lines: Vec<String> = lines.iter().map(|l| l.to_string()).collect();
        fix_ra_read_as_five(&mut lines);
        lines
    }

    fn line(x0: f32, y0: f32, x1: f32, y1: f32, text: &str) -> Line {
        Line { x0, y0, x1, y1, text: text.into() }
    }

    fn block(x0: f32, y0: f32, x1: f32, y1: f32, text: &str) -> Block {
        Block { x0, y0, x1, y1, line_height: y1 - y0, lines: vec![text.into()], line_boxes: vec![[x0, y0, x1, y1]], item: false }
    }

    #[test]
    fn keeps_spaced_out_speaker_names() {
        let rows = |lines: Vec<Line>| -> Vec<String> { join_rows(split_columns(lines)).into_iter().map(|l| l.text).collect() };
        // KimiKiss: 相　原「…⏎咲　野「…, whose first characters are read as one column.
        let column = vec![
            line(60.0, 434.0, 112.0, 555.0, "相咲"),
            line(124.0, 432.0, 334.0, 490.0, "原「ははは」"),
            line(125.0, 498.0, 630.0, 558.0, "野「ご飯食べて"),
        ];
        assert_eq!(rows(column), ["相原「ははは」", "咲野「ご飯食べて"]);
        // 柊　「…: a one-character name far from its line.
        let apart = vec![line(96.0, 569.0, 144.0, 617.0, "柊"), line(180.0, 566.0, 771.0, 622.0, "「まさか。」")];
        assert_eq!(rows(apart), ["柊「まさか。」"]);
        // Boxes overlapping the line: 原 read twice, 野 read as 里, 柊 boxed with the space.
        let twice = vec![line(58.0, 561.0, 157.0, 622.0, "相原"), line(123.0, 566.0, 629.0, 621.0, "原「今日も」")];
        assert_eq!(rows(twice), ["相原「今日も」"]);
        let garbled = vec![line(63.0, 436.0, 154.0, 486.0, "咲里"), line(123.0, 433.0, 436.0, 490.0, "野「お肉も、")];
        assert_eq!(rows(garbled), ["咲野「お肉も、"]);
        let wide = vec![line(92.0, 500.0, 204.0, 555.0, "柊"), line(175.0, 499.0, 816.0, 557.0, "「俺に」")];
        assert_eq!(rows(wide), ["柊「俺に」"]);
        // A name's box off the line, taller than it.
        let off = vec![line(66.0, 438.0, 109.0, 492.0, "咲"), line(124.0, 433.0, 440.0, 492.0, "野「うん」"), line(62.0, 468.0, 108.0, 553.0, "相"), line(125.0, 499.0, 634.0, 558.0, "原「ずいぶん」")];
        assert_eq!(rows(off), ["相原「ずいぶん」", "咲野「うん」"]);
        // Menu entries on one row stay apart.
        let menu = vec![line(100.0, 100.0, 148.0, 148.0, "決定"), line(184.0, 100.0, 280.0, 148.0, "戻る")];
        assert_eq!(rows(menu).len(), 2);
    }

    #[test]
    fn merges_sentences_cut_by_a_hanging_indent() {
        let texts = |blocks: Vec<Block>| -> Vec<String> { merge_continued(blocks).iter().map(Block::joined).collect() };
        // KimiKiss: the line under the speaker's name is indented past it.
        let dialogue = vec![
            block(0.06, 0.64, 0.86, 0.72, "相原（…と言っても、読みたい本がある訳じゃ"),
            block(0.20, 0.73, 0.55, 0.81, "ないんだよな…）"),
        ];
        assert_eq!(texts(dialogue), ["相原（…と言っても、読みたい本がある訳じゃ⏎ないんだよな…）"]);
        let narration = vec![block(0.1, 0.1, 0.4, 0.15, "オレが指差すと、"), block(0.1, 0.16, 0.9, 0.21, "女生徒はそれを見た。")];
        assert_eq!(texts(narration), ["オレが指差すと、⏎女生徒はそれを見た。"]);
        // Form fields and labels stay apart.
        let form = vec![block(0.1, 0.1, 0.4, 0.15, "苗字入力ふらんそわ"), block(0.1, 0.16, 0.4, 0.21, "名前入力らるしゅ")];
        assert_eq!(texts(form).len(), 2);
        let keys = vec![
            block(0.1, 0.1, 0.9, 0.15, "登録されている名前の一覧から選べます"),
            block(0.1, 0.16, 0.3, 0.21, "決定"),
        ];
        assert_eq!(texts(keys).len(), 2);
    }

    #[test]
    fn splits_bulleted_choices() {
        // Three lines on a grey box, a dot before each of the first two lines
        // of an entry; the third line continues the second entry.
        let mut image = GrayImage::from_pixel(400, 120, Luma([90]));
        for cy in [15, 45] {
            for y in cy - 1..=cy + 1 {
                for x in 19..=21 {
                    image.put_pixel(x, y, Luma([240]));
                }
            }
        }
        let rows = vec![line(30.0, 5.0, 300.0, 25.0, "そんな考え方はよくないよ"), line(30.0, 35.0, 300.0, 55.0, "死んだら、悲しむ"), line(30.0, 65.0, 200.0, 85.0, "人がいるだろ？")];
        let items: Vec<(usize, bool)> = split_items(rows.clone(), &image).iter().map(|(lines, item)| (lines.len(), *item)).collect();
        assert_eq!(items, [(1, true), (2, true)]);
        // Without markers (or with one only) the block stays whole.
        let plain = GrayImage::from_pixel(400, 120, Luma([90]));
        assert_eq!(split_items(rows, &plain).len(), 1);
    }

    #[test]
    fn drops_furigana_only() {
        let lines = vec![
            line(120.0, 88.0, 160.0, 100.0, "きのう"),
            line(100.0, 100.0, 300.0, 124.0, "昨日は雨だった"),
            // Small kana under a line, or beside it: a hint, not furigana.
            line(100.0, 126.0, 180.0, 138.0, "つぎへ"),
            line(320.0, 88.0, 380.0, 100.0, "メモ"),
        ];
        let texts: Vec<String> = drop_furigana(lines).into_iter().map(|l| l.text).collect();
        assert_eq!(texts, ["昨日は雨だった", "つぎへ", "メモ"]);
    }

    #[test]
    fn maps_chinese_glyphs_to_japanese() {
        assert_eq!(to_japanese_forms("决定"), "決定");
        assert_eq!(to_japanese_forms("携带電話"), "携帯電話");
        // Japanese kanji, even rare or old-style ones, stay as read.
        assert_eq!(to_japanese_forms("水澤摩央、危惧、蠍座"), "水澤摩央、危惧、蠍座");
    }

    #[test]
    fn reads_lone_five_as_ra_in_text() {
        assert_eq!(fix(&["「あの、怒ってる？", "5、口きいてくんないんスか？」"]), ["「あの、怒ってる？", "ら、口きいてくんないんスか？」"]);
        assert_eq!(fix(&["今日の最終組5", "しい。"]), ["今日の最終組ら", "しい。"]);
        assert_eq!(fix(&["さ5に彼5れた"]), ["さらに彼られた"]);
    }

    #[test]
    fn keeps_numbers() {
        assert_eq!(fix(&["本当は５分前の予定", "5月13日", "5つ", "第5の扉", "第5、", "レベル15に", "PS5で"]), [
            "本当は５分前の予定",
            "5月13日",
            "5つ",
            "第5の扉",
            "第5、",
            "レベル15に",
            "PS5で"
        ]);
    }
}
