// Preferences: the shortcut, the app's address and the target language.

import Adw from 'gi://Adw';
import Gdk from 'gi://Gdk';
import Gio from 'gi://Gio';
import Gtk from 'gi://Gtk';

import {ExtensionPreferences} from 'resource:///org/gnome/Shell/Extensions/js/extensions/prefs.js';

import {LANGUAGES, systemLanguage} from './language.js';

export default class GameTranslatorPreferences extends ExtensionPreferences {
    fillPreferencesWindow(window) {
        const settings = this.getSettings();
        const group = new Adw.PreferencesGroup({
            description: 'Needs Fukidashi running. A key, a click or a gamepad button hides the translation.',
        });

        const shortcut = new Adw.ShortcutLabel({disabled_text: 'None', valign: Gtk.Align.CENTER});
        const showShortcut = () => (shortcut.accelerator = settings.get_strv('translate')[0] ?? '');
        settings.connect('changed::translate', showShortcut);
        showShortcut();
        const row = new Adw.ActionRow({title: 'Shortcut', subtitle: 'Translates the screen', activatable: true});
        row.add_suffix(shortcut);
        row.connect('activated', () => askShortcut(window, settings));
        group.add(row);

        // An empty code follows the system's language; a code set by hand
        // (gsettings) that is not in the list is kept as a choice.
        const system = systemLanguage();
        const systemName = LANGUAGES.find(([code]) => code === system)?.[1] ?? system;
        const languages = [['', `${systemName} (system)`], ...LANGUAGES];
        const current = settings.get_string('target-lang');
        if (!languages.some(([code]) => code === current))
            languages.push([current, current]);
        const language = new Adw.ComboRow({
            title: 'Translate into',
            model: Gtk.StringList.new(languages.map(([, name]) => name)),
            selected: languages.findIndex(([code]) => code === current),
        });
        language.connect('notify::selected', () =>
            settings.set_string('target-lang', languages[language.selected][0]));
        group.add(language);

        const server = new Adw.EntryRow({title: 'Address of Fukidashi'});
        settings.bind('server', server, 'text', Gio.SettingsBindFlags.DEFAULT);
        group.add(server);

        const page = new Adw.PreferencesPage();
        page.add(group);
        window.add(page);
    }
}

// Waits for a key with at least one modifier (or a function key alone, which
// typing never needs), then saves it as the shortcut.
function askShortcut(window, settings) {
    const dialog = new Adw.AlertDialog({
        heading: 'New Shortcut',
        body: 'Press the new key combination, or Escape to cancel.',
    });
    dialog.add_response('cancel', 'Cancel');
    const keys = new Gtk.EventControllerKey({propagation_phase: Gtk.PropagationPhase.CAPTURE});
    keys.connect('key-pressed', (_controller, keyval, _keycode, state) => {
        const modifiers = state & Gtk.accelerator_get_default_mod_mask();
        const functionKey = keyval >= Gdk.KEY_F1 && keyval <= Gdk.KEY_F35;
        if (keyval === Gdk.KEY_Escape && !modifiers) {
            dialog.close();
        } else if (Gtk.accelerator_valid(keyval, modifiers) && (modifiers || functionKey)) {
            settings.set_strv('translate', [Gtk.accelerator_name(Gdk.keyval_to_lower(keyval), modifiers)]);
            dialog.close();
        }
        return Gdk.EVENT_STOP;
    });
    dialog.add_controller(keys);
    dialog.present(window);
}
