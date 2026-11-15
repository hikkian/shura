// Shura desk signal: READ-ONLY. It watches which window is active and whether a window is full screen, and prints one line
// to KWin's log (the user journal) whenever that changes:
//
//   shura-desk-signal {"v":1,"active":"brave-browser","fullscreen":true,"any_fullscreen":true,"fullscreen_class":"brave-browser"}
//
// Only application classes are logged, never window titles. The script does not move, resize, raise, minimize or close a window,
// does not set any property and does not talk to anything but the log. The gateway reads the newest line:
//   journalctl --user -b -t kwin_wayland -r -n 1 -g 'shura-desk-signal ' -o cat
// There is no keyboard or mouse idle time in KWin's script API, so this script does not report one.

var TAG = "shura-desk-signal ";
var last = "";
var watched = [];

function cls(w) {
    try {
        return w ? String(w.resourceClass) : "";
    } catch (e) {
        return "";
    }
}

function isFull(w) {
    try {
        return !!(w && w.fullScreen && !w.minimized);
    } catch (e) {
        return false;
    }
}

function publish() {
    try {
        var active = workspace.activeWindow;
        var fullClass = "";
        var any = false;
        var windows = workspace.windowList();
        for (var i = 0; i < windows.length; i++) {
            if (isFull(windows[i])) {
                any = true;
                if (windows[i] === active || fullClass === "") {
                    fullClass = cls(windows[i]);
                }
            }
        }
        var line = JSON.stringify({
            v: 1,
            active: cls(active),
            fullscreen: isFull(active),
            any_fullscreen: any,
            fullscreen_class: fullClass
        });
        if (line !== last) {
            last = line;
            console.info(TAG + line);
        }
    } catch (e) {
        console.info(TAG + JSON.stringify({ v: 1, error: String(e) }));
    }
}

function watch(w) {
    if (!w || watched.indexOf(w) >= 0) {
        return;
    }
    watched.push(w);
    try {
        w.fullScreenChanged.connect(publish);
        w.minimizedChanged.connect(publish);
    } catch (e) {
        // a window type without these signals is simply not watched
    }
}

workspace.windowAdded.connect(function (w) {
    watch(w);
    publish();
});
workspace.windowRemoved.connect(function (w) {
    var at = watched.indexOf(w);
    if (at >= 0) {
        watched.splice(at, 1);
    }
    publish();
});
workspace.windowActivated.connect(publish);
workspace.windowList().forEach(watch);
publish();
