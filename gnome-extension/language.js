// Languages to translate into, shared by the extension and its preferences.

import GLib from 'gi://GLib';

// The languages the app names in its prompt (language_name in translate.rs).
export const LANGUAGES = [
    ['en', 'English'],
    ['fr', 'French'],
    ['de', 'German'],
    ['es', 'Spanish'],
    ['it', 'Italian'],
    ['pt', 'Portuguese'],
    ['nl', 'Dutch'],
    ['ru', 'Russian'],
    ['pl', 'Polish'],
    ['ko', 'Korean'],
    ['zh-CN', 'Simplified Chinese'],
    ['zh-TW', 'Traditional Chinese'],
];

/**
 * The session's language as the app's codes write it: `fr` for fr_FR.UTF-8,
 * `zh-TW` for zh_TW, `en` when the locale is C.
 *
 * @returns {string}
 */
export function systemLanguage() {
    const locale = GLib.get_language_names()[0].split(/[.@]/)[0];
    const [language, region] = locale.split('_');
    if (language === 'C' || language === 'POSIX')
        return 'en';
    return language === 'zh' && region ? `zh-${region}` : language;
}

/**
 * The language of the `target-lang` setting, the system's when it is empty.
 *
 * @param {Gio.Settings} settings - the extension's settings
 * @returns {string}
 */
export function targetLanguage(settings) {
    return settings.get_string('target-lang') || systemLanguage();
}
