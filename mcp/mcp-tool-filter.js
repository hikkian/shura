#!/usr/bin/env node
// Transparent stdio proxy for an MCP server that trims the tools/list response
// to an allowlist before OpenCode ever sees it - the underlying server (e.g.
// Playwright MCP) has no built-in way to do this itself, and every extra tool
// definition costs real prompt-processing time on every fresh session or
// model switch (measured: ~5000 tokens / ~27 tools for Playwright's full set).
// Usage: node mcp-tool-filter.js <allowlist.json> -- <real command> [args...]
const { spawn } = require('child_process');
const fs = require('fs');
const readline = require('readline');

const sepIdx = process.argv.indexOf('--');
if (sepIdx === -1) {
    console.error('Usage: node mcp-tool-filter.js <allowlist.json> -- <command> [args...]');
    process.exit(1);
}
const allowlistPath = process.argv[2];
const allowlist = new Set(JSON.parse(fs.readFileSync(allowlistPath, 'utf8')));
const realCmd = process.argv[sepIdx + 1];
const realArgs = process.argv.slice(sepIdx + 2);

const child = spawn(realCmd, realArgs, { stdio: ['pipe', 'pipe', 'inherit'], shell: true });

// OpenCode -> child: pass through untouched.
process.stdin.pipe(child.stdin);

// child -> OpenCode: filter only the tools/list response's "tools" array.
const rl = readline.createInterface({ input: child.stdout, terminal: false });
rl.on('line', (line) => {
    if (!line.trim()) return;
    let msg;
    try { msg = JSON.parse(line); } catch (e) { process.stdout.write(line + '\n'); return; }
    if (msg && msg.result && Array.isArray(msg.result.tools)) {
        const before = msg.result.tools.length;
        msg.result.tools = msg.result.tools.filter(t => allowlist.has(t.name));
        process.stderr.write(`[mcp-tool-filter] tools/list: ${before} -> ${msg.result.tools.length}\n`);
    }
    process.stdout.write(JSON.stringify(msg) + '\n');
});

child.on('exit', (code) => process.exit(code === null ? 1 : code));
process.on('SIGINT', () => child.kill('SIGINT'));
process.on('SIGTERM', () => child.kill('SIGTERM'));
