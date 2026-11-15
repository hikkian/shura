// A fake KWin `workspace` for running kwin/shura-desk-signal/contents/code/main.js under node. Windows are Proxies that THROW on any
// write: the script must only read. Usage: node kwin_fake.js <main.js> <scenario.json>; prints the log lines and a final "WRITES n".
const fs = require("fs");
const vm = require("vm");

function signal() {
  const slots = [];
  return { connect: (f) => slots.push(f), emit: (...a) => slots.forEach((f) => f(...a)) };
}
let writes = 0;
const targets = new Map();
function makeWindow(cls, state) {
  const w = {
    resourceClass: cls, fullScreen: !!state.fullScreen, minimized: !!state.minimized, caption: "SECRET TITLE",
    fullScreenChanged: signal(), minimizedChanged: signal(),
  };
  const proxy = new Proxy(w, { set() { writes++; return true; } });
  targets.set(proxy, w);
  return proxy;
}
const lines = [];
const workspaceWindows = [];
const ws = {
  activeWindow: null,
  windowList: () => workspaceWindows.slice(),
  windowAdded: signal(), windowRemoved: signal(), windowActivated: signal(),
};
const scenario = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const wins = {};
function add(name, cls, state) {
  const w = makeWindow(cls, state || {});
  wins[name] = w;
  return w;
}
const ctx = { workspace: ws, console: { info: (s) => lines.push(String(s)) }, JSON, String, Array };
const code = fs.readFileSync(process.argv[2], "utf8");
for (const step of scenario) {
  if (step.op === "load") {
    for (const w of step.windows) { workspaceWindows.push(add(w.name, w.cls, w)); }
    ws.activeWindow = wins[step.active] || null;
    vm.runInNewContext(code, ctx);
  } else if (step.op === "add") {
    const w = add(step.name, step.cls, step); workspaceWindows.push(w); ws.windowAdded.emit(w);
  } else if (step.op === "activate") {
    ws.activeWindow = wins[step.name]; ws.windowActivated.emit(ws.activeWindow);
  } else if (step.op === "fullscreen") {
    targets.get(wins[step.name]).fullScreen = step.value; // KWin changing the state, not the script: bypass the write trap
  } else if (step.op === "emit") {
    wins[step.name][step.signal].emit();
  } else if (step.op === "remove") {
    const i = workspaceWindows.indexOf(wins[step.name]); if (i >= 0) workspaceWindows.splice(i, 1); ws.windowRemoved.emit(wins[step.name]);
  } else if (step.op === "break") {
    ws.windowList = () => { throw new Error("boom"); }; ws.windowActivated.emit(null);
  }
}
lines.forEach((l) => console.log(l));
console.log("WRITES " + writes);
