// Translates whatever is on screen with Fukidashi, outside RetroArch.
//
// The shortcut captures the monitor in front and sends it to the app as
// RetroArch's AI Service does. The translation it returns is shown over that
// monitor until a key, a click or a gamepad button hides it. OCR, translation
// and layout are the app's.

import Clutter from 'gi://Clutter';
import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import Meta from 'gi://Meta';
import Shell from 'gi://Shell';
import Soup from 'gi://Soup?version=3.0';
import St from 'gi://St';

import {Extension} from 'resource:///org/gnome/shell/extensions/extension.js';
import * as Main from 'resource:///org/gnome/shell/ui/main.js';

import {targetLanguage} from './language.js';

// Gamepads are read with libmanette when it is installed; without it, only
// the keyboard, the mouse and touch hide the translation.
const Manette = await import('gi://Manette?version=0.2').then(m => m.default, () => null);

export default class GameTranslator extends Extension {
    enable() {
        this._settings = this.getSettings();
        // A new screen can take a while on a cold model.
        this._session = new Soup.Session({timeout: 300});
        Main.wm.addKeybinding('translate', this._settings,
            Meta.KeyBindingFlags.IGNORE_AUTOREPEAT, Shell.ActionMode.NORMAL,
            () => this._show().catch(error => Main.notifyError('Fukidashi', error.message)));
    }

    disable() {
        Main.wm.removeKeybinding('translate');
        this._hide();
        this._session.abort();
        this._session = null;
        this._settings = null;
    }

    async _show() {
        if (this._busy || this._overlay)
            return;
        this._busy = true;
        try {
            await this._translate();
        } finally {
            this._busy = false;
        }
    }

    async _translate() {
        // The monitor of the focused window: the game, even with several screens.
        const window = global.display.focus_window;
        const monitor = window?.get_monitor() ?? global.display.get_current_monitor();
        const {x, y, width, height} = global.display.get_monitor_geometry(monitor);

        const stream = Gio.MemoryOutputStream.new_resizable();
        await new Shell.Screenshot().screenshot_area(x, y, width, height, stream);
        stream.close(null);

        // The window's class names the game's memory and profile, like
        // RetroArch's label: steam_app_1234 for a Steam game, pcsx2-qt...
        const request = {
            image: GLib.base64_encode(stream.steal_as_bytes().get_data()),
            format: 'png',
            label: window?.get_wm_class() || 'screen',
            fit: 'frame',
        };
        const url = `${this._settings.get_string('server')}?target_lang=${
            encodeURIComponent(targetLanguage(this._settings))}`;
        const message = Soup.Message.new('POST', url);
        if (!message)
            throw new Error(`invalid server address ${url}`);
        message.set_request_body_from_bytes('application/json',
            new GLib.Bytes(new TextEncoder().encode(JSON.stringify(request))));
        const bytes = await this._session.send_and_read_async(message, GLib.PRIORITY_DEFAULT, null)
            .catch(error => {
                throw new Error(`cannot reach Fukidashi at ${url}: ${error.message}`);
            });
        if (message.get_status() !== Soup.Status.OK)
            throw new Error(`Fukidashi answered ${message.get_status()}`);
        const reply = JSON.parse(new TextDecoder().decode(bytes.get_data()));
        if (reply.error)
            throw new Error(reply.error);
        // Disabled while the app was busy.
        if (!this._settings)
            return;

        // St draws the overlay from a file: a new name each time, as images are
        // cached by name.
        this._file = GLib.build_filenamev([GLib.get_user_runtime_dir(),
            `fukidashi-overlay-${Date.now()}.png`]);
        GLib.file_set_contents(this._file, GLib.base64_decode(reply.image));
        this._overlay = new St.Widget({
            x, y, width, height,
            reactive: true,
            style: `background-image: url("${this._file}"); background-size: contain;`,
        });
        for (const signal of ['key-press-event', 'button-press-event', 'touch-event']) {
            this._overlay.connect(signal, () => {
                this._hide();
                return Clutter.EVENT_STOP;
            });
        }
        Main.layoutManager.addTopChrome(this._overlay);
        // Keyboard and pointer come to the overlay until it is destroyed; this
        // also shows it over a fullscreen game, which is otherwise displayed directly.
        Main.pushModal(this._overlay);
        this._watchGamepads();
    }

    _watchGamepads() {
        if (!Manette)
            return;
        this._gamepads = new Manette.Monitor();
        const watch = device => device.connectObject('button-press-event', () => this._hide(), this);
        this._gamepads.connectObject('device-connected', (_monitor, device) => watch(device), this);
        const devices = this._gamepads.iterate();
        for (let [found, device] = devices.next(); found; [found, device] = devices.next())
            watch(device);
    }

    _hide() {
        // Destroying the overlay also ends its modal grab.
        this._overlay?.destroy();
        this._overlay = null;
        if (this._gamepads) {
            const devices = this._gamepads.iterate();
            for (let [found, device] = devices.next(); found; [found, device] = devices.next())
                device.disconnectObject(this);
            this._gamepads.disconnectObject(this);
            this._gamepads = null;
        }
        if (this._file) {
            GLib.unlink(this._file);
            this._file = null;
        }
    }
}
