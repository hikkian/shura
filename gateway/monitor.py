"""Read-only, on-demand telemetry and the self-contained monitor page.

The collector owns only bounded in-memory request/event buffers. It creates no
threads and never retains request or response text, headers, or credentials.
"""
from __future__ import annotations

from collections import deque
import copy
import http.client
import hmac
import json
import math
import os
import re
import shutil
import threading
import time
from pathlib import Path
import os

from monitor_config import ESTIMATED_WEIGHT_READ_MB_PER_TOKEN, THRESHOLDS
from monitor_sources import LinuxSystemSource, WindowsSystemSource, gpu_sources


TEXTS = {
    "page_title": "Shura Monitor", "compact_title": "Shura Monitor (compact)", "subtitle": "LOCAL INFERENCE · SYSTEM STATUS", "footer": "SHURA · LOCAL READ-ONLY MONITOR",
    "header": "SYSTEM", "updated": "Updated", "refresh": "LIVE · 1 SEC", "demo": "DEMO DATA · FICTIONAL VALUES",
    "now": "SPEED", "last_request": "RECENT DECODE", "context": "CONTEXT", "vram": "VRAM", "gpu": "GPU", "memory": "MEMORY",
    "cpu": "CPU", "reliability": "RELIABILITY", "events": "EVENTS", "requests": "REQUESTS", "model": "Model",
    "mode": "Mode", "state": "State", "model_uptime": "Model uptime", "generation": "Generation",
    "generating": "Generating", "idle": "Idle", "live_rate": "DECODE", "request_number": "Request number",
    "request_age": "Request age", "time_to_first_token": "FIRST TOKEN", "prompt_rate": "PREFILL", "decode_rate": "DECODE",
    "drafts": "MTP", "draft_fraction": "ACCEPTANCE", "finish_reason": "Finish reason", "slot_used": "USED TOKENS",
    "slot_percent": "SLOT USAGE", "kv_size": "KV size", "saved_slot": "SAVED SLOT", "prefix_reuse": "PREFIX REUSE",
    "vram_total": "VRAM TOTAL", "vram_used": "VRAM USED", "vram_free": "FREE", "model_vram": "MODEL",
    "other_vram": "DESKTOP", "reserve_margin": "RESERVE 1.5 GB", "gpu_utilization": "GPU UTIL", "temperature": "TEMP",
    "power": "POWER", "pcie": "PCIe LINK", "expert_cache": "Expert cache", "expert_hits": "Expert cache hits",
    "ram_used": "RAM USED", "ram_total": "RAM TOTAL", "ram_available": "AVAILABLE", "zram_original": "ZRAM uncompressed",
    "zram_compressed": "COMPRESSED", "zram_device_used": "DEVICE FILL", "psi_some": "PSI SOME", "psi_full": "PSI FULL",
    "weight_read": "Weight reads", "per_core_load": "CPU load by core", "cpu_frequency": "FREQUENCY", "polite_mode": "Polite mode",
    "length_count": "LENGTH CUTS", "request_errors": "REQUEST ERRORS", "server_restarts": "RESTARTS", "server_crashes": "CRASHES",
    "tool_errors": "Tool errors", "ssd_free": "SSD FREE", "nvme_written": "NVMe writes", "event_time": "Time",
    "event_type": "Event", "event_detail": "Details", "time": "Time", "endpoint": "Endpoint", "http_status": "Status",
    "prompt_tokens": "Prompt", "completion_tokens": "Output", "duration": "Duration", "no_requests": "No recent requests",
    "no_events": "No recent events", "unavailable": "—", "monitor_error": "Telemetry temporarily unavailable", "ok": "OK",
    "warn": "WARNING", "crit": "CRITICAL", "health": "HEALTH", "good": "HEALTHY", "unnamed_model": "NO MODEL",
    "unloaded": "UNLOADED", "loaded": "LOADED", "loading": "LOADING", "error": "ERROR",
    "mode_auto": "AUTO", "mode_game": "GAME", "mode_vision": "VISION", "mode_off": "OFF", "speed": "SPEED",
    "live": "LIVE", "context_tokens": "TOKENS", "ram": "RAM", "zram": "ZRAM", "gpu_temp": "GPU TEMP",
    "cpu_temp": "CPU TEMP", "core_load": "CORE LOAD", "desktop": "DESKTOP", "free": "FREE",
    "request_log": "Latest activity · up to 100", "event_log": "Gateway activity", "details": "Details", "demo_tag": "DEMO",
    "normal_scenario": "Normal", "high_scenario": "High load", "unloaded_scenario": "Unloaded",
    "approx_tip": "Approximate value. Live rate counts SSE chunks; weight reads use model size.",
    "ring_tip": "Accepted MTP drafts divided by proposed drafts.", "reserve_tip": "Configured desktop reserve: 1536 MiB.",
    "psi_tip": "Memory pressure stall information, avg10.", "saved_yes": "Saved slot available", "saved_no": "No saved slot",
    "prefix_tip": "Cached prompt tokens divided by prompt tokens.", "health_tip": "Worst status across GPU, memory, CPU, context and reliability.",
    "primary_metrics": "Primary metrics", "system_meters": "System meters", "spark_aria": "Decode rate over the last 120 seconds",
    "vram_aria": "VRAM allocated to model, desktop and free", "slots": "slots", "tokens_in": "in", "tokens_out": "out",
    "nvme_tip": "Bytes written since the monitor was first opened.", "request_ok": "Successful request", "request_error": "Request error",
    "finish_length": "Output limit reached", "offline": "Monitor endpoint is temporarily unavailable.", "offline_short": "Offline",
    "auth_prompt": "Gateway access key", "auth_rejected": "Access key rejected", "open_full": "Open full",
    "monitor_cost": "MONITOR", "monitor_cpu": "CPU", "monitor_rss": "RSS", "monitor_streams": "STREAMS",
    "online": "LIVE",
    "cpu_core": "CPU", "pcie_gen": "Gen", "pcie_width_separator": " ×", "estimate_suffix": "", "approx": "≈",
    "unit_gb_s": "GB/s", "unit_mib": "MiB", "unit_gb": "GB", "unit_celsius": "°C", "unit_watt": "W",
    "unit_mhz": "MHz", "unit_tok_s": "tok/s", "unit_percent": "%", "unit_seconds": "s", "unit_minutes": "m", "unit_minute_long": "min",
    "on": "On", "off": "Off", "yes": "Yes", "no": "No",
}

# Request failures and cut-off answers age out of the health verdict; the lifetime counters stay visible.
RELIABILITY_WINDOW_SECONDS = 3600
ACTIVE_IDLE_SECONDS = 1800   # an active request with no activity this long was leaked by its owner
MAX_STREAM_SUBSCRIBERS = 8
STREAM_WRITE_TIMEOUT_SECONDS = 15

STATUS_LINES = {
    "neutral": "Monitoring model and system health",
    "waiting_first_token": "Waiting for the first response token",
    "generating": "Generating a response",
    "polite_busy": "Slowed: CPU is busy with another app (polite mode)",
    "cold_cache": "Cache is cold after a restart",
    "low_vram": "GPU memory is low: model may be unloaded for a game",
    "game_parked": "Model is parked while a game needs resources",
    "memory_pressure": "Slowed: system memory is under pressure",
}

HEALTH_REASONS = {
    "gpu": {"warn": ("gpu_headroom_low", "GPU memory or temperature needs attention"), "crit": ("gpu_critical", "GPU memory or temperature is critical")},
    "memory": {"warn": ("memory_pressure", "Available memory or memory pressure needs attention"), "crit": ("memory_critical", "Available memory or memory pressure is critical")},
    "cpu": {"warn": ("cpu_temperature_high", "CPU temperature is high"), "crit": ("cpu_temperature_critical", "CPU temperature is critical")},
    "reliability": {"warn": ("disk_space_low", "Free disk space is low"), "crit": ("disk_space_critical", "Free disk space is critically low")},
    "context": {"warn": ("context_high", "The active context is nearly full"), "crit": ("context_critical", "The active context is almost full")},
    "reliability": {"warn": ("reliability_warning", "A response was limited or disk space is low"),
                     "crit": ("reliability_critical", "A request/server error occurred or disk space is critically low")},
}


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="dark light"><title></title>
<style>
@property --tone{syntax:'<color>';inherits:true;initial-value:#48d7a0}
:root{color-scheme:dark;--bg:#09111b;--panel:#111e2b;--panel2:#172838;--line:#263c4d;--text:#e8f2fa;--muted:#8da4b7;--green:#48d7a0;--amber:#ffbf5b;--red:#ff6477;--blue:#5bc5ff;--shadow:0 12px 30px #0003}
@media(prefers-color-scheme:light){:root{color-scheme:light;--bg:#edf3f7;--panel:#fff;--panel2:#f4f8fb;--line:#d4e0e8;--text:#122332;--muted:#60788a;--green:#078e60;--amber:#ad6a00;--red:#c72e49;--blue:#067bb8;--shadow:0 8px 22px #17364b12}}
*{box-sizing:border-box}body{margin:0;min-width:320px;background:radial-gradient(ellipse at 85% 0,#12304766,transparent 42%),var(--bg);color:var(--text);font:13px/1.35 system-ui,-apple-system,"Segoe UI",sans-serif;font-variant-numeric:tabular-nums}button,summary{font:inherit;color:inherit}.topbar{min-height:64px;padding:10px clamp(12px,2vw,26px);display:flex;align-items:center;justify-content:space-between;gap:14px;border-bottom:1px solid var(--line);background:color-mix(in srgb,var(--panel) 94%,transparent)}.brand{display:flex;align-items:center;gap:12px;min-width:0}.brand-mark{width:34px;height:34px;border-radius:11px;background:linear-gradient(145deg,#53d2ff,#277fb4);display:grid;place-items:center;color:#071725;font-weight:950;font-size:18px;box-shadow:0 0 22px #39b5ee3b}.brand-copy h1{margin:0;font-size:17px;letter-spacing:.01em}.brand-copy p{margin:2px 0 0;color:var(--muted);font-size:9px;letter-spacing:.13em;font-weight:750}.identity{display:flex;align-items:center;gap:8px 12px;margin-left:14px;flex-wrap:wrap}.model-name{font-size:13px;font-weight:800;max-width:230px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.pill,.health-pill{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);border-radius:99px;padding:5px 9px;font-size:10px;font-weight:850;letter-spacing:.06em;background:var(--panel2);white-space:nowrap}.pill::before,.health-dot,.led{content:"";width:7px;height:7px;border-radius:50%;background:var(--tone,var(--green));box-shadow:0 0 9px color-mix(in srgb,var(--tone,var(--green)) 60%,transparent)}.meta{font-size:10px;color:var(--muted);white-space:nowrap}.top-right{display:flex;align-items:center;gap:12px}.health-pill{padding:7px 11px;color:var(--tone,var(--green));border-color:color-mix(in srgb,var(--tone,var(--green)) 45%,var(--line));font-size:11px}.health-pill .health-dot{width:8px;height:8px}.demo-switch{display:flex;align-items:center;gap:4px;padding:3px;border:1px solid var(--line);border-radius:10px;background:var(--panel2)}.demo-switch[hidden]{display:none}.demo-switch button{border:0;background:transparent;color:var(--muted);border-radius:7px;padding:6px 8px;font-size:10px;font-weight:750;cursor:pointer}.demo-switch button[aria-pressed=true]{background:#24506b;color:#dff5ff}.demo-flag{position:static;flex:none;border:1px solid #ab7f2d;background:#332713;color:#ffd782;border-radius:7px;padding:4px 8px;font-size:9px;font-weight:850;letter-spacing:.08em;pointer-events:none}main{max-width:1580px;margin:auto;padding:10px clamp(10px,1.5vw,20px) 14px}.error{margin:0 0 5px;color:var(--red);font-size:11px;min-height:0}.error:empty{display:none}.status-line{margin:0 0 5px;padding:5px 9px;border-left:2px solid var(--blue);border-radius:4px;background:var(--panel);color:var(--muted);font-size:10px;min-height:24px}.dashboard{display:flex;flex-direction:column;gap:9px}.hero-grid{display:grid;grid-template-columns:1.12fr 1fr 1fr;gap:9px}.meter-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px}#system-meters>#card-reliability{grid-column:span 2}#system-meters>#card-reliability{grid-column:span 2}.card{--tone:var(--green);min-width:0;border:1px solid var(--line);border-radius:13px;padding:11px 13px;background:linear-gradient(145deg,color-mix(in srgb,var(--panel) 97%,var(--tone) 3%),var(--panel));box-shadow:var(--shadow);transition:border-color .35s ease,box-shadow .35s ease,background-color .35s ease,color .35s ease}.card[data-severity=warn]{border-color:color-mix(in srgb,var(--amber) 55%,var(--line))}.card[data-severity=crit]{border-color:color-mix(in srgb,var(--red) 72%,var(--line));animation:softPulse 2.8s ease-in-out infinite}.card-head{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:7px}.eyebrow{margin:0;color:var(--muted);font-size:9px;font-weight:850;letter-spacing:.16em}.severity{display:inline-flex;align-items:center;gap:5px;color:var(--tone);font-size:9px;font-weight:900;letter-spacing:.07em}.severity::before{content:"●";font-size:8px}.card[data-severity=ok]{--tone:var(--green)}.card[data-severity=warn]{--tone:var(--amber)}.card[data-severity=crit]{--tone:var(--red)}.speed-layout{display:grid;grid-template-columns:minmax(0,1fr) 82px;gap:8px;align-items:center}.speed-value{font-size:clamp(36px,4.2vw,54px);line-height:.98;font-weight:850;letter-spacing:-.055em;white-space:nowrap}.speed-value small{font-size:13px;letter-spacing:0;color:var(--muted);font-weight:700;margin-left:5px}.live-row{display:flex;align-items:center;gap:6px;color:var(--muted);font-size:9px;font-weight:750;letter-spacing:.1em;margin:3px 0 5px}.live-row .led{width:6px;height:6px}.spark{width:100%;height:38px;display:block;overflow:visible}.spark-fill{fill:url(#sparkFill)}.spark-line{fill:none;stroke:var(--blue);stroke-width:2;stroke-linecap:round;stroke-linejoin:round;filter:drop-shadow(0 0 4px #45c3ff70);transition:d .45s ease}.speed-foot{display:flex;gap:12px;align-items:center;margin-top:3px}.tiny-metric{display:flex;flex-direction:column;gap:1px}.tiny-metric span{color:var(--muted);font-size:8px;font-weight:800;letter-spacing:.12em}.tiny-metric b{font-size:12px;font-weight:800}.approx{display:inline-grid;place-items:center;width:15px;height:15px;border:1px solid color-mix(in srgb,var(--amber) 50%,var(--line));border-radius:50%;color:var(--amber);font-size:10px;cursor:help;margin-left:2px}.ring{--progress:0%;--tone:var(--green);width:72px;height:72px;border-radius:50%;display:grid;place-items:center;background:conic-gradient(var(--tone) var(--progress),var(--panel2) 0);position:relative;transition:background .45s ease}.ring::before{content:"";position:absolute;inset:6px;border-radius:50%;background:var(--panel)}.ring-copy{position:relative;text-align:center;line-height:1.05}.ring-copy b{display:block;font-size:17px}.ring-copy span{font-size:7px;color:var(--muted);font-weight:850;letter-spacing:.1em}.inline-title{display:flex;align-items:center;gap:5px}.hero-number{font-size:clamp(25px,3vw,38px);font-weight:850;letter-spacing:-.04em;line-height:1}.hero-number small{font-size:11px;color:var(--muted);letter-spacing:0;margin-left:4px}.context-top,.vram-top{display:flex;justify-content:space-between;align-items:flex-end;gap:8px}.context-count{font-size:clamp(21px,2.5vw,30px);font-weight:850;letter-spacing:-.045em}.context-count small{font-size:11px;color:var(--muted);font-weight:650;letter-spacing:0}.percent{font-size:14px;font-weight:850;color:var(--tone)}.track{position:relative;width:100%;height:10px;border-radius:99px;background:var(--panel2);overflow:hidden;margin:12px 0 9px}.fill{display:block;height:100%;width:0;border-radius:inherit;background:var(--tone,var(--green));transition:width .7s cubic-bezier(.2,.7,.2,1),background-color .45s ease}.context-chips{display:flex;flex-wrap:wrap;gap:6px}.chip{display:inline-flex;align-items:center;gap:5px;border:1px solid var(--line);border-radius:7px;padding:4px 7px;font-size:9px;color:var(--muted);font-weight:750}.chip strong{color:var(--text);font-size:10px}.chip .led{width:6px;height:6px}.vram-free{font-size:clamp(27px,3vw,38px);font-weight:880;letter-spacing:-.04em;line-height:1}.vram-free small{font-size:11px;color:var(--muted);letter-spacing:0}.vram-stack{height:17px;margin:13px 0 8px;position:relative;display:flex;border-radius:6px;background:var(--panel2);overflow:visible}.segment{height:100%;transition:width .65s ease}.segment.model{background:#56bfff;border-radius:6px 0 0 6px}.segment.desktop{background:#8776e8}.segment.free{background:var(--green);border-radius:0 6px 6px 0}.reserve-mark{position:absolute;z-index:2;top:-4px;bottom:-4px;width:2px;background:var(--text);box-shadow:0 0 4px var(--bg);transition:left .5s ease}.reserve-mark::after{content:"";position:absolute;top:-3px;left:-3px;width:8px;height:8px;transform:rotate(45deg);background:var(--amber);border:1px solid var(--bg)}.legend{display:flex;gap:11px;flex-wrap:wrap;color:var(--muted);font-size:8px;font-weight:800;letter-spacing:.07em}.legend span{display:flex;align-items:center;gap:4px}.swatch{width:7px;height:7px;border-radius:2px;background:var(--c)}.vram-mini{display:flex;justify-content:space-between;align-items:center;margin-top:7px;color:var(--muted);font-size:9px}.vram-mini b{color:var(--text)}.meter-card{min-height:143px}.meter-main{display:flex;align-items:flex-end;justify-content:space-between;gap:6px}.meter-value{font-size:25px;font-weight:850;letter-spacing:-.04em;line-height:1;color:var(--tone)}.meter-value small{font-size:10px;color:var(--muted);letter-spacing:0}.meter-sub{font-size:9px;color:var(--muted);text-align:right}.track.zram{height:12px}.compressed-mark{position:absolute;left:0;top:3px;bottom:3px;background:#d2e2ffb3;border-radius:99px;mix-blend-mode:screen;transition:width .65s ease}.psi-line{display:flex;gap:6px;margin-top:8px}.psi-chip{display:flex;align-items:center;gap:4px;padding:3px 5px;border:1px solid var(--line);border-radius:6px;font-size:8px;color:var(--muted);font-weight:800}.psi-chip b{color:var(--tone,var(--text))}.dial-row{display:flex;align-items:center;gap:10px}.dial{--angle:0deg;--needle-angle:-135deg;--tone:var(--green);width:76px;height:76px;flex:none;position:relative;display:grid;place-items:center}.dial-ring{position:absolute;inset:0;border-radius:50%;background:conic-gradient(from 225deg,var(--tone) 0deg var(--angle),var(--panel2) var(--angle) 270deg,transparent 270deg);mask:radial-gradient(farthest-side,transparent calc(100% - 7px),#000 calc(100% - 6px));transition:background .5s ease}.needle{position:absolute;left:calc(50% - 1px);top:7px;width:2px;height:31px;background:var(--tone);border-radius:2px;transform-origin:50% 31px;transform:rotate(var(--needle-angle));transition:transform .5s ease,background-color .4s ease}.needle::after{content:"";position:absolute;top:-2px;left:-2px;width:6px;height:6px;border-radius:50%;background:var(--tone)}.dial-center{position:relative;text-align:center}.dial-center b{display:block;font-size:17px;line-height:1;font-weight:850}.dial-center span{font-size:8px;color:var(--muted)}.meter-side{min-width:0;flex:1;display:grid;gap:7px}.bar-label{display:flex;justify-content:space-between;gap:4px;font-size:8px;color:var(--muted);font-weight:800;letter-spacing:.08em}.thin-track{height:5px;background:var(--panel2);border-radius:99px;overflow:hidden;margin-top:3px}.thin-track i{display:block;height:100%;width:0;background:var(--tone);border-radius:inherit;transition:width .65s ease,background-color .4s ease}.pcie-line{margin-top:7px;display:flex;justify-content:space-between;gap:5px;color:var(--muted);font-size:8px}.cpu-layout{display:flex;align-items:center;gap:8px}.cores{display:grid;grid-template-columns:repeat(6,minmax(5px,1fr));align-items:end;gap:3px;height:55px;flex:1}.core-bar{display:block;width:100%;height:var(--h,0%);min-height:2px;border-radius:3px 3px 1px 1px;background:var(--tone,var(--blue));transition:height .65s ease,background-color .4s ease}.core-wrap{height:100%;display:flex;align-items:flex-end;position:relative}.core-wrap::after{content:attr(data-core);position:absolute;bottom:-12px;left:50%;transform:translateX(-50%);font-size:6px;color:var(--muted)}.cpu-foot{margin-top:16px;display:flex;justify-content:space-between;color:var(--muted);font-size:8px}.reliability{padding:8px 11px;min-height:143px}.reliability .card-head{margin-bottom:6px}.reliability-grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:6px}.reliability-item{display:flex;align-items:center;gap:7px;min-width:0;padding:5px 7px;border:1px solid var(--line);border-radius:8px;background:var(--panel2);transition:border-color .35s ease,color .35s ease}.rel-icon{width:21px;height:21px;flex:none;display:grid;place-items:center;border-radius:6px;background:color-mix(in srgb,var(--tone) 15%,transparent);color:var(--tone);font-size:11px;font-weight:900}.rel-copy{min-width:0}.rel-copy span{display:block;color:var(--muted);font-size:7px;font-weight:850;letter-spacing:.08em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.rel-copy b{font-size:12px;color:var(--text)}.feeds{display:grid;grid-template-columns:1.45fr 1fr;gap:9px}.feeds.single{grid-template-columns:1fr}.feed-card{min-width:0;padding:0 11px 7px;max-height:168px;overflow:hidden}.feed-card>summary{list-style:none;cursor:pointer;display:flex;align-items:center;justify-content:space-between;gap:8px;padding:9px 0 7px;color:var(--muted);font-size:9px;letter-spacing:.13em;font-weight:900}.feed-card>summary::-webkit-details-marker{display:none}.feed-card>summary::after{content:"＋";color:var(--blue);font-size:12px}.feed-card[open]>summary::after{content:"−"}.feed-card:only-child{grid-column:1/-1}.feed-body{max-height:124px;overflow:auto}.request-item{display:grid;grid-template-columns:8px 76px minmax(120px,1fr) 75px minmax(100px,.9fr);align-items:center;gap:7px;padding:5px 2px;border-top:1px solid var(--line);font-size:9px}.request-item:first-child{border-top:0}.request-status{width:7px;height:7px;border-radius:50%;background:var(--tone,var(--green));box-shadow:0 0 7px color-mix(in srgb,var(--tone) 60%,transparent)}.request-model{font-weight:800;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.request-endpoint{color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.request-tokens{color:var(--muted);font-size:8px;white-space:nowrap}.request-rate{display:grid;grid-template-columns:minmax(30px,1fr) 34px;gap:5px;align-items:center}.request-rate .thin-track{margin:0}.request-rate b{text-align:right;font-size:8px}.event-item{display:grid;grid-template-columns:58px 11px minmax(0,1fr);gap:7px;align-items:center;padding:5px 2px;border-top:1px solid var(--line);font-size:9px}.event-time{color:var(--muted);font-size:8px}.event-dot{width:7px;height:7px;border-radius:50%;background:var(--tone,var(--blue))}.event-copy{min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.event-copy b{font-weight:850;margin-right:6px}.empty{padding:15px 4px;text-align:center;color:var(--muted);font-size:10px}.details{margin-top:7px;border-top:1px solid var(--line);padding-top:4px}.details summary{cursor:pointer;color:var(--muted);font-size:8px;font-weight:800;letter-spacing:.08em}.details-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:4px 10px;margin-top:6px}.detail-row{display:flex;justify-content:space-between;gap:5px;color:var(--muted);font-size:8px}.detail-row[hidden]{display:none}.detail-row b{color:var(--text);font-weight:750}.foot{padding:2px 4px;text-align:right;color:var(--muted);font-size:8px;letter-spacing:.08em}
@keyframes softPulse{0%,100%{box-shadow:0 0 0 0 color-mix(in srgb,var(--red) 8%,transparent)}50%{box-shadow:0 0 18px 1px color-mix(in srgb,var(--red) 18%,transparent)}}
@media(max-width:1099px){.hero-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.speed-card{grid-column:span 2}.meter-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.reliability-grid{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media(max-width:699px){#system-meters>#card-reliability{grid-column:1/-1}.topbar{align-items:flex-start;flex-wrap:wrap}.brand{width:100%}.identity{margin-left:auto}.top-right{width:100%;justify-content:space-between}.demo-switch button{padding:6px}.hero-grid,.meter-grid,.feeds{grid-template-columns:minmax(0,1fr)}.speed-card{grid-column:auto}.reliability-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.speed-layout{grid-template-columns:minmax(0,1fr) 74px}.speed-value{font-size:42px}.ring{width:66px;height:66px}.request-item{grid-template-columns:8px 66px minmax(70px,1fr) 82px}.request-item .hide-phone{display:none}.feed-card{max-height:220px}.feed-body{max-height:175px}.brand-mark{width:30px;height:30px}.model-name{max-width:140px}}

body.page-hidden .card[data-severity=crit]{animation:none}
body.is-offline .health-pill{--tone:var(--muted)!important;color:var(--muted);border-color:var(--line)}body.is-offline.compact .identity{display:none}body.is-offline.compact #card-speed,body.is-offline.compact #card-context,body.is-offline.compact #card-vram,body.is-offline.compact #card-ram,body.is-offline.compact #card-zram,body.is-offline.compact #card-gpu-temp,body.is-offline.compact #card-cpu{display:none!important}body.is-offline .health-dot{background:var(--muted);box-shadow:none}body.is-offline .card{filter:grayscale(1);opacity:.68}
.compact-full{display:none}
body.compact{width:min(360px,100vw);min-width:320px;min-height:100vh;overflow:hidden}body.compact .topbar{min-height:34px;padding:3px 7px;gap:3px}body.compact .brand{gap:0}body.compact .brand-mark,body.compact .brand-copy,body.compact #mode,body.compact #uptime{display:none}body.compact .identity{margin:0;gap:4px}body.compact .model-name{max-width:105px;font-size:10px}body.compact .pill{font-size:8px;padding:3px 5px}body.compact .top-right{gap:3px}body.compact .health-pill{font-size:8px;padding:4px 6px}
body.compact main{padding:3px;max-width:none}body.compact .dashboard{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:4px}body.compact .hero-grid,body.compact .meter-grid{display:contents}body.compact .feeds,body.compact #card-reliability,body.compact .foot,body.compact .demo-switch,body.compact #demo-flag{display:none!important}body.compact .card{padding:5px 7px;border-radius:8px;box-shadow:none}body.compact .card-head{margin-bottom:3px}body.compact .eyebrow{font-size:7px;letter-spacing:.12em}body.compact .severity{font-size:7px;gap:3px}body.compact .severity::before{font-size:6px}
body.compact #card-speed{grid-column:1/-1;height:65px;overflow:hidden}body.compact #card-speed .card-head,body.compact #card-speed .ring-column,body.compact #card-speed .speed-foot{display:none}body.compact #card-speed .speed-layout{display:block}body.compact #card-speed .live-row{font-size:7px;margin:0 0 1px;gap:4px}body.compact #card-speed .live-row .led{width:5px;height:5px}body.compact #card-speed .speed-value{font-size:27px;line-height:1}body.compact #card-speed .speed-value small{font-size:9px;margin-left:3px}body.compact #card-speed .approx{width:12px;height:12px;font-size:8px}body.compact #card-speed .spark{height:19px;margin-top:1px}
body.compact #card-context{grid-column:1/-1;height:48px;overflow:hidden}body.compact #card-context .card-head{margin-bottom:1px}body.compact #card-context .context-top{align-items:center}body.compact #card-context .context-count{font-size:15px}body.compact #card-context .context-count small{font-size:8px}body.compact #card-context .percent{font-size:11px}body.compact #card-context .track{height:5px;margin:4px 0 0}body.compact #card-context .context-chips,body.compact #card-context .details{display:none}
body.compact #card-vram,body.compact #card-ram,body.compact #card-zram{grid-column:span 2;height:49px;min-height:0;overflow:hidden;padding:4px 6px}body.compact #card-vram .card-head,body.compact #card-ram .card-head,body.compact #card-zram .card-head{margin-bottom:1px}body.compact #card-vram .vram-top{align-items:center}body.compact #card-vram .vram-free{font-size:14px}body.compact #card-vram .vram-free small{font-size:8px}body.compact #card-vram #vram-exact,body.compact #card-vram .legend,body.compact #card-vram .vram-mini,body.compact #card-vram .details{display:none}body.compact #card-vram .vram-stack{height:5px;margin:5px 0 0}body.compact #card-ram .meter-main{align-items:center}body.compact #card-ram .meter-value{font-size:15px}body.compact #card-ram .meter-value small{font-size:8px}body.compact #card-ram .meter-sub,body.compact #card-ram .bar-label,body.compact #card-ram .details{display:none}body.compact #card-ram .track,body.compact #card-zram .track{height:5px;margin:5px 0 0}body.compact #card-zram .meter-value{font-size:15px}body.compact #card-zram .meter-value small{font-size:8px}body.compact #card-zram .meter-sub{font-size:7px}body.compact #card-zram .bar-label,body.compact #card-zram .psi-line,body.compact #card-zram .details{display:none}
body.compact #card-gpu-temp,body.compact #card-cpu{grid-column:span 3;height:58px;min-height:0;overflow:hidden;padding:4px 7px}body.compact #card-gpu-temp .card-head,body.compact #card-cpu .card-head{margin-bottom:1px}body.compact #card-gpu-temp .dial-row{gap:5px}body.compact #card-gpu-temp .meter-side,body.compact #card-gpu-temp .pcie-line,body.compact #card-cpu .meter-side,body.compact #card-cpu .cpu-foot{display:none}body.compact .dial{width:40px;height:40px}body.compact .dial-ring{mask:radial-gradient(farthest-side,transparent calc(100% - 5px),#000 calc(100% - 4px))}body.compact .needle{top:4px;height:16px;transform-origin:50% 16px}body.compact .dial-center b{font-size:11px}body.compact .dial-center span{font-size:6px}
body.compact .compact-full{grid-column:1/-1;display:block;text-align:center;height:14px;color:var(--blue);font-size:9px;line-height:14px;text-decoration:none;font-weight:800}body.compact .compact-full:hover{text-decoration:underline}.monitor-self{font-size:8px;color:var(--muted);text-align:right}body.compact .monitor-self{height:10px;overflow:hidden;text-align:center;font-size:6px;line-height:10px}
body.compact #error{display:none}body.compact .status-line{min-height:18px;margin:0;padding:2px 6px;font-size:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}body.compact .topbar{height:32px;overflow:hidden}body.compact .dashboard{gap:3px}body.compact .card[data-severity=crit]{animation:softPulse 2.8s ease-in-out infinite}body.page-hidden .card[data-severity=crit]{animation:none}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{scroll-behavior:auto!important;animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}}

@media(min-width:1100px){main{padding-top:5px;padding-bottom:0}.dashboard{gap:6px}.meter-card .track{margin:8px 0 6px}}
[hidden]{display:none!important}
</style></head><body>
<header class="topbar"><div class="brand"><div class="brand-mark" aria-hidden="true">S</div><div class="brand-copy"><h1 id="title"></h1><p id="subtitle"></p></div><div class="identity"><strong class="model-name" id="model"></strong><span class="pill" id="state"></span><span class="pill" id="mode"></span><span class="meta" id="uptime"></span></div></div>
<div class="top-right"><div id="demo-flag" class="demo-flag" hidden></div><div class="demo-switch" id="demo-switch" hidden><button type="button" data-scenario="normal" aria-pressed="true"></button><button type="button" data-scenario="high" aria-pressed="false"></button><button type="button" data-scenario="unloaded" aria-pressed="false"></button></div><div class="health-pill" id="health" title=""><span class="health-dot"></span><span id="health-word"></span></div></div></header>
<main><p id="error" class="error" role="status"></p><p id="status-line" class="status-line"></p><div class="dashboard">
<section class="hero-grid" id="hero-section" aria-label="">
 <article class="card speed-card" id="card-speed"><div class="card-head"><h2 class="eyebrow" data-text="now"></h2><span class="severity" id="speed-health"></span></div><div class="speed-layout"><div><div class="live-row"><i class="led"></i><span data-text="live"></span><span id="generation"></span><span id="request-number" class="meta"></span></div><div class="speed-value"><span id="decode-live">0.0</span><small data-unit="unit_tok_s"></small><span class="approx" id="speed-approx" title="" aria-label="">≈</span></div><svg class="spark" id="sparkline" viewBox="0 0 300 48" preserveAspectRatio="none" role="img" aria-label=""><defs><linearGradient id="sparkFill" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#59c8ff" stop-opacity=".34"/><stop offset="1" stop-color="#59c8ff" stop-opacity="0"/></linearGradient></defs><path id="spark-area" class="spark-fill"></path><path id="spark-line" class="spark-line"></path></svg><div class="speed-foot"><div class="tiny-metric" id="prefill-field"><span data-text="prompt_rate"></span><b id="prefill"></b></div><div class="tiny-metric" id="ttft-field"><span data-text="time_to_first_token"></span><b id="ttft"></b></div><div class="tiny-metric" id="decode-field"><span data-text="decode_rate"></span><b id="decode-last"></b></div></div></div><div class="ring-column" id="ring-column"><div class="ring" id="mtp-ring" title=""><div class="ring-copy"><b id="mtp-percent">—</b><span data-text="drafts"></span></div></div><span class="ring-caption" data-text="draft_fraction"></span></div></div></article>
 <article class="card" id="card-context"><div class="card-head"><h2 class="eyebrow" data-text="context"></h2><span class="severity" id="context-health"></span></div><div class="context-top"><div class="context-count"><span id="ctx-used">0</span><small id="ctx-limit-field"> / <span id="ctx-limit">—</span> <span data-text="context_tokens"></span></small></div><b class="percent" id="ctx-pct">—</b></div><div class="track"><i class="fill" id="ctx-fill"></i></div><div class="context-chips"><span class="chip" id="saved-chip"><i class="led"></i><span data-text="saved_slot"></span><strong id="saved-value"></strong></span><span class="chip" id="prefix-chip"><span data-text="prefix_reuse"></span><strong id="prefix-value"></strong></span></div><details class="details"><summary data-text="details"></summary><div class="details-grid"><div class="detail-row" id="kv-detail"><span data-text="kv_size"></span><b id="kv-size"></b></div><div class="detail-row" id="ctx-age-detail"><span data-text="request_age"></span><b id="ctx-age"></b></div></div></details></article>
 <article class="card" id="card-vram"><div class="card-head"><h2 class="eyebrow" data-text="vram"></h2><span class="severity" id="vram-health"></span></div><div class="vram-top"><div class="vram-free"><span id="vram-free-gb">—</span><small data-unit="unit_gb"></small></div><div class="tiny-metric" id="vram-exact"><span data-text="vram_free"></span><b id="vram-free-mib"></b></div></div><div class="vram-stack" id="vram-stack" role="img" aria-label=""><i class="segment model" id="vram-model"></i><i class="segment desktop" id="vram-desktop"></i><i class="segment free" id="vram-free-seg"></i><i class="reserve-mark" id="reserve-mark" title=""></i></div><div class="legend"><span><i class="swatch" style="--c:#56bfff"></i><span data-text="model_vram"></span></span><span><i class="swatch" style="--c:#8776e8"></i><span data-text="other_vram"></span></span><span><i class="swatch" style="--c:var(--green)"></i><span data-text="free"></span></span></div><div class="vram-mini" id="vram-margin-row"><span data-text="reserve_margin"></span><b id="reserve-margin"></b></div><details class="details"><summary data-text="details"></summary><div class="details-grid"><div class="detail-row" id="vram-total-detail"><span data-text="vram_total"></span><b id="vram-total"></b></div><div class="detail-row" id="vram-used-detail"><span data-text="vram_used"></span><b id="vram-used"></b></div><div class="detail-row" id="gpu-util-detail-row"><span data-text="gpu_utilization"></span><b id="gpu-util-detail"></b></div></div></details></article>
</section>
<section class="meter-grid" id="system-meters" aria-label="">
 <article class="card meter-card" id="card-ram"><div class="card-head"><h2 class="eyebrow" data-text="ram"></h2><span class="severity" id="ram-health"></span></div><div class="meter-main"><div class="meter-value"><span id="ram-avail">—</span><small data-unit="unit_gb"></small></div><span class="meter-sub" data-text="ram_available"></span></div><div class="track"><i class="fill" id="ram-fill"></i></div><div class="bar-label"><span id="ram-used-label"></span><span id="ram-total-label"></span></div><details class="details"><summary data-text="details"></summary><div class="details-grid"><div class="detail-row" id="ram-used-detail"><span data-text="ram_used"></span><b id="ram-used"></b></div><div class="detail-row" id="ram-total-detail"><span data-text="ram_total"></span><b id="ram-total"></b></div></div></details></article>
 <article class="card meter-card" id="card-zram"><div class="card-head"><h2 class="eyebrow" data-text="zram"></h2><span class="severity" id="zram-health"></span></div><div class="meter-main"><div class="meter-value"><span id="zram-percent">—</span><small data-unit="unit_percent"></small></div><span class="meter-sub" id="zram-device-label"></span></div><div class="track zram"><i class="fill" id="zram-fill"></i><i class="compressed-mark" id="zram-compressed-mark"></i></div><div class="bar-label" id="zram-compressed-line"><span data-text="zram_compressed"></span><b id="zram-compressed"></b></div><div class="psi-line"><span class="psi-chip" id="psi-some-chip" title=""><span data-text="psi_some"></span><b id="psi-some"></b></span><span class="psi-chip" id="psi-full-chip" title=""><span data-text="psi_full"></span><b id="psi-full"></b></span></div><details class="details"><summary data-text="details"></summary><div class="details-grid"><div class="detail-row" id="zram-original-detail"><span data-text="zram_original"></span><b id="zram-original"></b></div><div class="detail-row" id="weight-read-detail"><span data-text="weight_read"></span><b id="weight-read"></b><span class="approx" title="" aria-label="">≈</span></div></div></details></article>
 <article class="card meter-card" id="card-gpu-temp"><div class="card-head"><h2 class="eyebrow" data-text="gpu"></h2><span class="severity" id="gpu-temp-health"></span></div><div class="dial-row"><div class="dial" id="gpu-dial"><i class="dial-ring"></i><i class="needle"></i><div class="dial-center"><b id="gpu-temp">—</b><span data-unit="unit_celsius"></span></div></div><div class="meter-side"><div id="gpu-util-block"><div class="bar-label"><span data-text="gpu_utilization"></span><b id="gpu-util"></b></div><div class="thin-track"><i id="gpu-util-fill"></i></div></div><div id="gpu-power-block"><div class="bar-label"><span data-text="power"></span><b id="gpu-power"></b></div><div class="thin-track"><i id="gpu-power-fill"></i></div></div></div></div><div class="pcie-line" id="pcie-block"><span data-text="pcie"></span><b id="pcie"></b><span id="gpu-cache"></span></div></article>
 <article class="card meter-card" id="card-cpu"><div class="card-head"><h2 class="eyebrow" data-text="cpu"></h2><span class="severity" id="cpu-health"></span></div><div class="cpu-layout"><div class="dial" id="cpu-dial"><i class="dial-ring"></i><i class="needle"></i><div class="dial-center"><b id="cpu-temp">—</b><span data-unit="unit_celsius"></span></div></div><div class="meter-side"><div class="bar-label"><span data-text="core_load"></span><b id="cpu-frequency"></b></div><div id="cores" class="cores"></div></div></div><div class="cpu-foot" id="polite-field"><span data-text="polite_mode"></span><span id="polite"></span></div></article>
<section class="card reliability" id="card-reliability"><div class="card-head"><h2 class="eyebrow" data-text="reliability"></h2><span class="severity" id="reliability-health"></span></div><div class="reliability-grid">
 <div class="reliability-item" id="rel-length"><i class="rel-icon">↧</i><div class="rel-copy"><span data-text="length_count"></span><b id="length-count"></b></div></div>
 <div class="reliability-item" id="rel-errors"><i class="rel-icon">!</i><div class="rel-copy"><span data-text="request_errors"></span><b id="request-errors"></b></div></div>
 <div class="reliability-item" id="rel-restarts"><i class="rel-icon">↻</i><div class="rel-copy"><span data-text="server_restarts"></span><b id="server-restarts"></b></div></div>
 <div class="reliability-item" id="rel-crashes"><i class="rel-icon">×</i><div class="rel-copy"><span data-text="server_crashes"></span><b id="server-crashes"></b></div></div>
 <div class="reliability-item" id="rel-ssd"><i class="rel-icon">▱</i><div class="rel-copy"><span data-text="ssd_free"></span><b id="ssd-free"></b></div></div>
 <div class="reliability-item" id="rel-nvme"><i class="rel-icon">↗</i><div class="rel-copy"><span data-text="nvme_written"></span><b id="nvme-written"></b></div></div>
</div><div class="detail-row" id="tool-errors-field" hidden><span data-text="tool_errors"></span><b id="tool-errors"></b></div></section>
</section>
<section class="feeds" id="feeds"><details class="card feed-card" id="requests-details" open><summary><span data-text="requests"></span><span class="meta" data-text="request_log"></span></summary><div class="feed-body" id="requests-body"></div><div id="requests-empty" class="empty" hidden></div></details><details class="card feed-card" id="events-details" open><summary><span data-text="events"></span><span class="meta" data-text="event_log"></span></summary><div class="feed-body" id="events-body"></div><div id="events-empty" class="empty" hidden></div></details></section>
</div><div id="monitor-self" class="monitor-self"></div><a class="compact-full" id="compact-full" href="/monitor" target="_blank" rel="noopener" data-text="open_full"></a><div class="foot"><span id="updated"></span> · <span id="footer"></span></div></main>
<script>
const T=__TEXTS__,DEMO=__DEMO__,LIMITS=__LIMITS__;const $=id=>document.getElementById(id),NA=T.unavailable;
const compactQuery=new URLSearchParams(location.search).get('view')==='compact';let compact=compactQuery||innerWidth<420;document.body.classList.toggle('compact',compact);document.title=compact?T.compact_title:T.page_title;$('title').textContent=document.title;$('subtitle').textContent=T.subtitle;$('footer').textContent=T.footer;$('hero-section').setAttribute('aria-label',T.primary_metrics);$('system-meters').setAttribute('aria-label',T.system_meters);$('sparkline').setAttribute('aria-label',T.spark_aria);$('vram-stack').setAttribute('aria-label',T.vram_aria);
document.querySelectorAll('[data-text]').forEach(e=>{e.textContent=T[e.dataset.text]||e.dataset.text});document.querySelectorAll('[data-unit]').forEach(e=>{e.textContent=T[e.dataset.unit]||''});
const demoSwitch=$('demo-switch'),scenarioButtons=[...document.querySelectorAll('[data-scenario]')],speedHistory=[];let currentScenario=new URLSearchParams(location.search).get('scenario')||'normal',lastState=null;
function seedDemoHistory(){if(!DEMO?.scenarios)return;const base=Number(DEMO.scenarios[currentScenario]?.state?.now?.live_tokens_per_second)||0;speedHistory.length=0;for(let i=0;i<120;i++)speedHistory.push(Math.max(0,base*(.9+.07*Math.sin(i*.14)+.035*Math.sin(i*.37))));drawSpark()}
if(DEMO&&DEMO.scenarios){demoSwitch.hidden=false;$('demo-flag').hidden=false;$('demo-flag').textContent=T.demo_tag;$('demo-flag').title=T.demo;scenarioButtons.forEach(b=>b.textContent=T[b.dataset.scenario+'_scenario']);seedDemoHistory();scenarioButtons.forEach(b=>b.addEventListener('click',()=>{currentScenario=b.dataset.scenario;scenarioButtons.forEach(x=>x.setAttribute('aria-pressed',String(x===b)));seedDemoHistory();refresh()}))}
function finite(v){return v!==null&&v!==undefined&&v!==''&&Number.isFinite(Number(v))}
function number(v,d=0){return finite(v)?Number(v).toFixed(d):''}
function format(v,d=0,unit=''){return finite(v)?`${Number(v).toFixed(d)}${unit?' '+unit:''}`:''}
function pct(v){return finite(v)?`${Number(v).toFixed(0)}%`:''}
function duration(v){if(!finite(v))return'';return Number(v)<60?`${Number(v).toFixed(1)} ${T.unit_seconds}`:`${Math.floor(v/60)} ${T.unit_minute_long}`}
function clamp(v,min=0,max=100){return Math.max(min,Math.min(max,finite(v)?Number(v):min))}
function lerpColor(a,b,t){const pa=parseInt(a.slice(1),16),pb=parseInt(b.slice(1),16),f=clamp(t,0,1);return'#'+[16,8,0].map(s=>Math.round(((pa>>s)&255)*(1-f)+((pb>>s)&255)*f).toString(16).padStart(2,'0')).join('')}
function lowBad(v,crit,warn,max){if(!finite(v))return'var(--muted)';v=Number(v);if(v<=crit)return'#ff6477';if(v<warn)return lerpColor('#ff6477','#ffbf5b',(v-crit)/(warn-crit));return'#48d7a0'}
const word={ok:T.ok,warn:T.warn,crit:T.crit};function statusTone(s){return s==='crit'?'var(--red)':s==='warn'?'var(--amber)':'var(--green)'}
function healthTag(id,state){const el=$(id);el.textContent=word[state]||T.warn;el.parentElement?.style.setProperty('--tone',state==='crit'?'var(--red)':state==='warn'?'var(--amber)':'var(--green)')}
function setCard(id,state){const el=$(id);if(el)el.dataset.severity=state||'warn'}
function setText(id,text,wrap=null){const el=$(id);if(!el)return;const show=text!==null&&text!==undefined&&text!=='';el.textContent=show?String(text):'';if(wrap){const box=$(wrap);if(box)box.hidden=!show}return show}
function setFill(id,value,tone,max=100){const el=$(id);if(!el)return;el.style.width=`${clamp(Number(value)/max*100)}%`;el.style.backgroundColor=tone}
function setDial(id,value,state){const el=$(id);if(!finite(value)){el.hidden=true;return state}el.hidden=false;const x=clamp(value,0,100),angle=x*2.7,tone=statusTone(state);el.style.setProperty('--angle',angle+'deg');el.style.setProperty('--needle-angle',(angle-135)+'deg');el.style.setProperty('--tone',tone);return state}
function drawSpark(){const values=speedHistory;if(!values.length){$('spark-line').setAttribute('d','');$('spark-area').setAttribute('d','');return}const w=300,h=44,max=Math.max(1,...values),min=Math.min(0,...values),span=Math.max(1,max-min),coords=values.map((v,i)=>[values.length===1?w:(i/(values.length-1))*w,h-3-((v-min)/span)*(h-8)]),line=coords.map((p,i)=>`${i?'L':'M'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' '),area=`${line} L${w},${h} L0,${h} Z`;$('spark-line').setAttribute('d',line);$('spark-area').setAttribute('d',area)}
function addSpeed(v){if(!finite(v))return;const next=Number(v),changed=speedHistory.length===0||speedHistory[speedHistory.length-1]!==next;speedHistory.push(next);if(speedHistory.length>120)speedHistory.shift();if(changed)drawSpark()}
function renderState(d){lastState=d;const h=d.header||{},n=d.now||{},l=d.last_request||{},c=d.context||{},g=d.gpu||{},m=d.memory||{},cpu=d.cpu||{},r=d.reliability||{},bs=d.block_status||{},caps=d.capabilities||{};const serverState=k=>['ok','warn','crit'].includes(bs[k])?bs[k]:'warn',hasGpu=Boolean(caps.gpu_nvidia||caps.gpu_amd||caps.gpu_intel),hasCpu=finite(cpu.temperature_c)||finite(cpu.frequency_mhz)||(cpu.cores||[]).some(x=>finite(x.load_percent));$('card-vram').hidden=!hasGpu;$('card-gpu-temp').hidden=!hasGpu;$('card-ram').hidden=!caps.system_linux;$('card-zram').hidden=!caps.zram;$('card-cpu').hidden=!caps.system_linux||!hasCpu;$('card-context').hidden=!caps.slots||!finite(c.used_percent);$('psi-some-chip').hidden=!caps.psi;$('psi-full-chip').hidden=!caps.psi;$('rel-nvme').hidden=!caps.nvme;$('status-line').textContent=d.status_line?.text||'';$('status-line').hidden=!d.status_line?.text;const self=d.monitor_self||{};setText('monitor-self',[T.monitor_cost,finite(self.cpu_seconds)?T.monitor_cpu+' '+Number(self.cpu_seconds).toFixed(1)+'s':'',finite(self.rss_mib)?T.monitor_rss+' '+Number(self.rss_mib).toFixed(0)+' MiB':'',finite(self.stream_subscribers)?T.monitor_streams+' '+self.stream_subscribers:''].filter(Boolean).join(' · '));
 const stateKey=String(h.state||'UNLOADED').toLowerCase();setText('model',h.model||T.unnamed_model);setText('state',T[stateKey]||String(h.state||'UNLOADED'));$('state').style.setProperty('--tone',h.state_color==='crit'?'var(--red)':h.state_color==='warn'?'var(--amber)':'var(--green)');$('state').dataset.state=h.state_color||'ok';setText('mode',T['mode_'+String(h.mode||'AUTO').toLowerCase()]||String(h.mode||'AUTO'));$('mode').style.setProperty('--tone','var(--blue)');setText('uptime',duration(h.model_uptime_seconds));
 const generating=Boolean(n.generating);setText('generation',generating?T.generating:T.idle);setText('request-number',finite(n.request_number)?`#${n.request_number}`:'');const live=finite(n.live_tokens_per_second)?Number(n.live_tokens_per_second):0;setText('decode-live',live.toFixed(1));$('speed-approx').title=T.approx_tip;$('speed-approx').setAttribute('aria-label',T.approx_tip);$('speed-approx').hidden=!n.rate_is_estimate;addSpeed(live);setCard('card-speed','ok');healthTag('speed-health','ok');
 setText('prefill',format(l.prompt_per_second,0,T.unit_tok_s),'prefill-field');setText('ttft',duration(l.ttft_seconds),'ttft-field');setText('decode-last',format(l.decode_per_second,1,T.unit_tok_s),'decode-field');$('prefill-field').hidden=!caps.timings||!finite(l.prompt_per_second);$('ttft-field').hidden=!caps.timings||!finite(l.ttft_seconds);$('decode-field').hidden=!caps.timings||!finite(l.decode_per_second);const mp=l.draft_acceptance_percent;const ring=$('mtp-ring');if(caps.timings&&finite(mp)){$("ring-column").hidden=false;ring.hidden=false;ring.style.setProperty('--progress',`${clamp(mp)}%`);ring.style.setProperty('--tone',lowBad(mp,40,75,100));setText('mtp-percent',`${Number(mp).toFixed(0)}%`);ring.title=T.ring_tip}else{$("ring-column").hidden=true;ring.hidden=true;}
 setText('ctx-used',finite(c.used_tokens)?Number(c.used_tokens).toLocaleString('en-US'):'0');setText('ctx-limit',finite(c.limit_tokens)?Number(c.limit_tokens).toLocaleString('en-US'):'','ctx-limit-field');const ctxPercent=finite(c.used_percent)?Number(c.used_percent):null,ctxState=serverState('context'),ctxColor=statusTone(ctxState);setText('ctx-pct',pct(c.used_percent));$('ctx-fill').style.setProperty('--tone',ctxColor);setFill('ctx-fill',c.used_percent,ctxColor);setText('saved-value',c.saved_slot===true?T.yes:c.saved_slot===false?T.no:'','saved-chip');$('saved-chip').title=c.saved_slot===true?T.saved_yes:c.saved_slot===false?T.saved_no:'';setText('prefix-value',finite(c.prefix_reuse_percent)?pct(c.prefix_reuse_percent):'','prefix-chip');setText('kv-size',finite(c.kv_bytes)?format(c.kv_bytes/1048576,0,T.unit_mib):'','kv-detail');setText('ctx-age',duration(n.request_age_seconds),'ctx-age-detail');setCard('card-context',ctxState);healthTag('context-health',ctxState);
 const total=Number(g.total_mib)||0,free=finite(g.free_mib)?Number(g.free_mib):null,model=finite(g.model_process_mib)?Number(g.model_process_mib):null,desktop=finite(g.other_processes_mib)?Number(g.other_processes_mib):null,vramState=serverState('gpu'),vramTone=statusTone(vramState);setText('vram-free-gb',free===null?'—':(free/1024).toFixed(2));setText('vram-free-mib',free===null?'':format(free,0,T.unit_mib),'vram-exact');const sum=(model||0)+(desktop||0)+(free||0),den=total||sum||1;$('vram-model').style.width=`${clamp((model||0)/den*100)}%`;$('vram-desktop').style.width=`${clamp((desktop||0)/den*100)}%`;$('vram-free-seg').style.width=`${clamp((free||0)/den*100)}%`;$('reserve-mark').style.left=`${clamp((den-LIMITS.vram_reserve_mib)/den*100)}%`;$('reserve-mark').title=T.reserve_tip;setText('reserve-margin',finite(g.margin_over_reserve_mib)?`${Number(g.margin_over_reserve_mib)>=0?'+':''}${format(g.margin_over_reserve_mib,0,T.unit_mib)}`:'','vram-margin-row');$('reserve-margin').style.color=vramTone;setText('vram-total',format(g.total_mib,0,T.unit_mib),'vram-total-detail');setText('vram-used',format(g.used_mib,0,T.unit_mib),'vram-used-detail');setText('gpu-util-detail',format(g.utilization_percent,0,T.unit_percent),'gpu-util-detail-row');setCard('card-vram',vramState);healthTag('vram-health',vramState);$('card-vram').hidden=!hasGpu||free===null;$('card-vram').style.setProperty('--tone',vramTone);$('vram-free-gb').style.color=vramTone;
 const ramGiB=finite(m.ram_available_mib)?Number(m.ram_available_mib)/1024:null,ramState=serverState('memory'),ramTone=statusTone(ramState);setText('ram-avail',ramGiB===null?'—':ramGiB.toFixed(1));$('ram-avail').style.color=ramTone;const ramUsed=finite(m.ram_used_mib)?Number(m.ram_used_mib):null,ramTotal=finite(m.ram_total_mib)?Number(m.ram_total_mib):null;setFill('ram-fill',ramUsed||0,ramTone,ramTotal||1);setText('ram-used-label',format(ramUsed,0,T.unit_mib));setText('ram-total-label',format(ramTotal,0,T.unit_mib));setText('ram-used',format(ramUsed,0,T.unit_mib),'ram-used-detail');setText('ram-total',format(ramTotal,0,T.unit_mib),'ram-total-detail');setCard('card-ram',ramState);healthTag('ram-health',ramState);$('card-ram').hidden=!caps.system_linux||ramGiB===null;
 const z=finite(m.zram_used_percent)?Number(m.zram_used_percent):null,zState=serverState('memory'),memState=zState,zTone=statusTone(memState);setText('zram-percent',z===null?'—':z.toFixed(0));$('zram-percent').style.color=zTone;setFill('zram-fill',z||0,zTone);const zSize=finite(m.zram_disk_mib)?Number(m.zram_disk_mib):0,comp=finite(m.zram_compressed_mib)?Number(m.zram_compressed_mib):null;setText('zram-device-label',zSize&&z!==null?`${format(m.zram_device_mib??(zSize*z/100),0,T.unit_mib)} / ${format(zSize,0,T.unit_mib)}`:'');$('zram-compressed-mark').style.width=`${clamp((comp||0)/zSize*100)}%`;setText('zram-compressed',format(comp,0,T.unit_mib),'zram-compressed-line');setText('zram-original',format(m.zram_original_mib,0,T.unit_mib),'zram-original-detail');setText('psi-some',finite(m.psi_some_avg10)?`${Number(m.psi_some_avg10).toFixed(1)}%`:'','psi-some-chip');setText('psi-full',finite(m.psi_full_avg10)?`${Number(m.psi_full_avg10).toFixed(1)}%`:'','psi-full-chip');document.querySelectorAll('.psi-chip').forEach(e=>e.title=T.psi_tip);$('psi-some').style.color=statusTone(memState);$('psi-full').style.color=statusTone(memState);setText('weight-read',finite(m.weight_read_gb_per_second)?format(m.weight_read_gb_per_second,1,T.unit_gb_s):'','weight-read-detail');$('weight-read-detail').title=T.approx_tip;$('weight-read-detail').querySelector('.approx').title=T.approx_tip;$('weight-read-detail').querySelector('.approx').setAttribute('aria-label',T.approx_tip);setCard('card-zram',memState);healthTag('zram-health',memState);$('card-zram').hidden=!caps.zram||z===null;$('card-zram').style.setProperty('--tone',memState==='crit'?'var(--red)':memState==='warn'?'var(--amber)':zTone);
 const gpuState=serverState('gpu'),gpuTone=statusTone(gpuState);setText('gpu-temp',format(g.temperature_c,0));$('gpu-temp').style.color=gpuTone;setDial('gpu-dial',g.temperature_c,gpuState);setFill('gpu-util-fill',g.utilization_percent,'var(--blue)');setText('gpu-util',format(g.utilization_percent,0,T.unit_percent),'gpu-util-block');setFill('gpu-power-fill',g.power_w,'var(--blue)',350);setText('gpu-power',format(g.power_w,0,T.unit_watt),'gpu-power-block');const pcieText=finite(g.pcie_generation)&&finite(g.pcie_width)?`${T.pcie_gen} ${g.pcie_generation}${T.pcie_width_separator}${g.pcie_width}`:'';setText('pcie',pcieText,'pcie-block');const cacheText=caps.expert_cache&&finite(g.expert_cache_slots)?`${g.expert_cache_slots} ${T.slots}`+(finite(g.expert_cache_hit_percent)?` · ${Number(g.expert_cache_hit_percent).toFixed(0)}%`: ''):'';setText('gpu-cache',cacheText);$('pcie-block').hidden=!pcieText&&!cacheText;setCard('card-gpu-temp',gpuState);healthTag('gpu-temp-health',gpuState);$('card-gpu-temp').style.setProperty('--tone',gpuTone);
 const cpuState=serverState('cpu'),cpuTone=statusTone(cpuState);setDial('cpu-dial',cpu.temperature_c,cpuState);setText('cpu-temp',format(cpu.temperature_c,0));$('cpu-temp').style.color=cpuTone;setText('cpu-frequency',format(cpu.frequency_mhz,0,T.unit_mhz));setText('polite',cpu.polite_mode===null?'':(cpu.polite_mode?T.on:T.off),'polite-field');const cores=$('cores');cores.replaceChildren();(cpu.cores||[]).forEach(x=>{if(!finite(x.load_percent))return;const wrap=document.createElement('span');wrap.className='core-wrap';wrap.dataset.core=x.id;const bar=document.createElement('i'),load=clamp(x.load_percent),coreTone=load>LIMITS.cpu_core_crit_percent?'#ff6477':load>LIMITS.cpu_core_warn_percent?'#ffbf5b':'#48d7a0';bar.className='core-bar';bar.style.setProperty('--h',`${load}%`);bar.style.setProperty('--tone',coreTone);wrap.appendChild(bar);cores.appendChild(wrap)});setCard('card-cpu',cpuState);healthTag('cpu-health',cpuState);$('card-cpu').style.setProperty('--tone',cpuTone);
 const zerosBad=(key)=>finite(r[key])&&Number(r[key])>0?'crit':'ok';[['rel-length','length-count',r.finish_length_count,zerosBad('finish_length_count')],['rel-errors','request-errors',r.request_errors,zerosBad('request_errors')],['rel-restarts','server-restarts',r.server_restarts,zerosBad('server_restarts')],['rel-crashes','server-crashes',r.server_crashes,zerosBad('server_crashes')]].forEach(([box,id,value,state])=>{setText(id,finite(value)?Number(value).toLocaleString('en-US'):'',null);$(box).hidden=!finite(value);$(box).style.setProperty('--tone',state==='crit'?'var(--red)':'var(--green)')});const ssdState=serverState('reliability');setText('ssd-free',format(r.ssd_free_gb,0,T.unit_gb),'rel-ssd');$('rel-ssd').style.setProperty('--tone',ssdState==='crit'?'var(--red)':'var(--green)');setText('nvme-written',finite(r.nvme_written_bytes)?format(Number(r.nvme_written_bytes)/1073741824,1,T.unit_gb):'','rel-nvme');$('rel-nvme').title=T.nvme_tip;if(r.tool_errors!==null&&r.tool_errors!==undefined){$('tool-errors-field').hidden=false;setText('tool-errors',r.tool_errors)}else $('tool-errors-field').hidden=true;const relState=serverState('reliability');setCard('card-reliability',relState);healthTag('reliability-health',relState);
 const worst=['ok','warn','crit'].includes(d.health?.level)?d.health.level:'warn';$('health').style.setProperty('--tone',statusTone(worst));$('health').title=d.health?.hint||T.health_tip;setText('health-word',worst==='crit'?T.crit:worst==='warn'?T.warn:T.good);
 const ev=$('events-body');ev.replaceChildren();const events=d.events||[];events.slice(0,8).forEach(x=>{const row=document.createElement('div');row.className='event-item';const time=document.createElement('span');time.className='event-time';time.textContent=finite(x.at)?new Date(x.at*1000).toLocaleTimeString('en',{hour:'2-digit',minute:'2-digit'}):'';const dot=document.createElement('i');dot.className='event-dot';const copy=document.createElement('span');copy.className='event-copy';const title=document.createElement('b');title.textContent=x.reason||'';const detail=document.createElement('span');detail.textContent=x.detail||'';copy.append(title,detail);row.append(time,dot,copy);ev.appendChild(row)});$('events-empty').hidden=events.length>0;$('events-empty').textContent=T.no_events;$('events-details').hidden=false;
}
function renderRequests(items){const body=$('requests-body');body.replaceChildren();(items||[]).slice(0,100).forEach(x=>{const row=document.createElement('div'),status=Number(x.status)>=400?'crit':'ok';row.className='request-item';row.style.setProperty('--tone',status==='crit'?'var(--red)':'var(--green)');const dot=document.createElement('i');dot.className='request-status';dot.title=status==='crit'?T.request_error:T.request_ok;const model=document.createElement('b');model.className='request-model';model.textContent=x.model||'';const endpoint=document.createElement('span');endpoint.className='request-endpoint';endpoint.textContent=x.endpoint||'';endpoint.title=x.endpoint||'';const tokens=document.createElement('span');tokens.className='request-tokens hide-phone';const ins=finite(x.prompt_tokens)?Number(x.prompt_tokens):null,outs=finite(x.completion_tokens)?Number(x.completion_tokens):null;tokens.textContent=ins===null&&outs===null?'':`${ins===null?'':ins+' '+T.tokens_in}${ins!==null&&outs!==null?' / ':''}${outs===null?'':outs+' '+T.tokens_out}`;const rate=document.createElement('span');rate.className='request-rate';const track=document.createElement('i');track.className='thin-track';const fill=document.createElement('i');fill.style.display='block';fill.style.height='100%';fill.style.borderRadius='99px';fill.style.width=`${clamp(Number(x.decode_per_second)/80*100)}%`;fill.style.background='var(--blue)';track.appendChild(fill);const rateText=document.createElement('b');rateText.textContent=finite(x.decode_per_second)?`${Number(x.decode_per_second).toFixed(0)}`:'';if(!finite(x.decode_per_second))track.hidden=true;rate.append(track,rateText);if(finite(x.at))row.title=new Date(x.at*1000).toLocaleString('en');row.append(dot,model,endpoint,tokens,rate);body.appendChild(row)});$('requests-empty').hidden=(items||[]).length>0;$('requests-empty').textContent=T.no_requests;$('requests-details').hidden=false;$('feeds').classList.toggle('single',$('requests-details').hidden!==$('events-details').hidden)}
function drawDemo(){if(!DEMO)return;const pack=DEMO.scenarios[currentScenario]||DEMO.scenarios.normal;renderState(pack.state);renderRequests(pack.requests)}
let authPrompt=null,authRejected=false,authDismissed=false,timer=null;const AUTH_KEY='shura-monitor-key';
function storedKey(){try{return sessionStorage.getItem(AUTH_KEY)||''}catch(_){return''}}
async function askKey(){if(authRejected||authDismissed)return null;if(authPrompt)return authPrompt;authPrompt=(async()=>{const value=window.prompt(T.auth_prompt);if(!value){authDismissed=true;return null}try{sessionStorage.setItem(AUTH_KEY,value)}catch(_){}return value})();try{return await authPrompt}finally{authPrompt=null}}
async function apiFetch(path,options={}){let key=storedKey(),headers=key?{Authorization:`Bearer ${key}`}:{},res=await fetch(path,{...options,cache:'no-store',headers});if(res.status===401&&!key){key=await askKey();if(key===null)throw new Error('auth-cancelled');res=await fetch(path,{...options,cache:'no-store',headers:{Authorization:`Bearer ${key}`}})}if(res.status===401){authRejected=true;try{sessionStorage.removeItem(AUTH_KEY)}catch(_){}throw new Error('auth-rejected')}if(!res.ok)throw new Error(T.offline);return res}
let streamController=null,streamActive=false,streamFailed=false,requestTimer=null,reconnectTimer=null,fallbackTimer=null;
function showOffline(e){document.body.classList.add('is-offline');$('error').textContent=e?.message==='auth-rejected'?T.auth_rejected:T.offline_short;$('health-word').textContent=T.offline_short;$('health').title=T.offline;$('updated').textContent=T.offline_short}
async function fetchRequests(){if(compact||document.hidden||DEMO)return;try{const res=await apiFetch('/monitor/api/requests');renderRequests(await res.json())}catch(_){}}
async function pollState(){try{if(DEMO&&DEMO.scenarios){drawDemo();document.body.classList.remove('is-offline');return}const res=await apiFetch('/monitor/api/state');renderState(await res.json());document.body.classList.remove('is-offline');$('error').textContent='';$('updated').textContent=T.updated+' '+new Date().toLocaleTimeString('en')}catch(e){showOffline(e)}}
async function readStream(){if(DEMO||document.hidden||streamActive)return;streamController=new AbortController();streamActive=true;try{const res=await apiFetch('/monitor/api/stream',{signal:streamController.signal});if(!res.body)throw new Error('stream unavailable');streamFailed=false;if(fallbackTimer){clearInterval(fallbackTimer);fallbackTimer=null}const reader=res.body.getReader(),decoder=new TextDecoder();let pending='';while(true){const part=await reader.read();if(part.done){streamFailed=true;break}pending+=decoder.decode(part.value,{stream:true});const frames=pending.split(/\r?\n\r?\n/);pending=frames.pop()||'';for(const frame of frames){const data=frame.split(/\r?\n/).filter(x=>x.startsWith('data:')).map(x=>x.slice(5).trim()).join('\n');if(!data)continue;try{renderState(JSON.parse(data));document.body.classList.remove('is-offline');$('error').textContent='';$('updated').textContent=T.updated+' '+new Date().toLocaleTimeString('en')}catch(_){}}}}catch(e){if(e.name!=='AbortError'){streamFailed=true;showOffline(e)}}finally{streamActive=false;streamController=null;schedule();if(!document.hidden&&!DEMO){clearTimeout(reconnectTimer);reconnectTimer=setTimeout(()=>{readStream();pollState()},1500)}}}
function schedule(){if(fallbackTimer)clearInterval(fallbackTimer);if(requestTimer)clearInterval(requestTimer);const interval=document.hidden?5000:1000;if(document.hidden||streamFailed||DEMO)fallbackTimer=setInterval(pollState,interval);requestTimer=setInterval(fetchRequests,5000)}
window.addEventListener('resize',()=>{if(!compactQuery){const next=innerWidth<420;if(next!==compact){compact=next;document.body.classList.toggle('compact',compact);document.title=compact?T.compact_title:T.page_title;schedule();pollState();fetchRequests()}}});document.addEventListener('visibilitychange',()=>{document.body.classList.toggle('page-hidden',document.hidden);if(document.hidden){streamController?.abort();clearTimeout(reconnectTimer)}else if(!DEMO){readStream();pollState()}schedule()});
if(DEMO&&DEMO.scenarios){drawDemo()}else{pollState();readStream()}schedule();
</script></body></html>'''


def render_page(demo_data=None):
    """Render one self-contained page. demo_data is only used for the preview file."""
    text_json = json.dumps(TEXTS, separators=(",", ":")).replace("</", "<\\/")
    demo_json = "null" if demo_data is None else json.dumps(demo_data, separators=(",", ":")).replace("</", "<\\/")
    limits_json = json.dumps(THRESHOLDS, separators=(",", ":"))
    return PAGE.replace("__TEXTS__", text_json).replace("__DEMO__", demo_json).replace("__LIMITS__", limits_json)


def model_uptime_seconds(pid, proc_root="/proc", now_wall=None):
    if not pid:
        return None
    try:
        raw = (Path(proc_root) / str(pid) / "stat").read_text()
        # comm may contain spaces or parentheses; fields after the final ')' start at field 3.
        rest = raw.rsplit(")", 1)[1].split()
        start_ticks = int(rest[19])  # field 22 (starttime), zero-based after field 2
        btime = next(int(x.split()[1]) for x in (Path(proc_root) / "stat").read_text().splitlines() if x.startswith("btime "))
        hz = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        started = btime + start_ticks / hz
        return max(0.0, (time.time() if now_wall is None else now_wall) - started)
    except (OSError, ValueError, StopIteration, KeyError, IndexError):
        return None


def _read_meminfo(proc_root):
    values = {}
    try:
        for line in (Path(proc_root) / "meminfo").read_text().splitlines():
            m = re.match(r"(MemTotal|MemAvailable):\s+(\d+)\s+kB", line)
            if m:
                values[m.group(1)] = int(m.group(2))
    except OSError:
        pass
    return values.get("MemTotal"), values.get("MemAvailable")


def _read_psi(proc_root):
    result = {}
    try:
        for line in (Path(proc_root) / "pressure/memory").read_text().splitlines():
            fields = line.split()
            if fields and fields[0] in ("some", "full"):
                raw = next((v.split("=", 1)[1] for v in fields[1:] if v.startswith("avg10=")), None)
                result[fields[0]] = float(raw) if raw is not None else None
    except (OSError, ValueError):
        pass
    return result


def _read_zram(sys_root):
    original = compressed = used = capacity = 0
    found = False
    for root in Path(sys_root).glob("block/zram*"):
        try:
            fields = (root / "mm_stat").read_text().split()
            if len(fields) < 3:
                continue
            device_original = int(fields[0])
            device_compressed = int(fields[1])
            device_used = int(fields[2])
            device_capacity = int((root / "disksize").read_text().strip())
            if min(device_original, device_compressed, device_used) < 0 or device_capacity <= 0:
                continue
        except (OSError, ValueError):
            continue
        # Commit a device to the aggregate only after every value for it was read and validated.
        original += device_original
        compressed += device_compressed
        used += device_used
        capacity += device_capacity
        found = True
    if not found:
        return {"original_bytes": None, "compressed_bytes": None, "used_bytes": None, "disk_bytes": None, "used_percent": None}
    return {"original_bytes": original, "compressed_bytes": compressed, "used_bytes": used,
            "disk_bytes": capacity,
            # A zram device is full when the uncompressed data reaches disksize, not its compressed footprint.
            "used_percent": 100 * original / capacity if capacity else None}


def _read_cpu(proc_root, sys_root, previous):
    current, cores = {}, []
    try:
        for line in (Path(proc_root) / "stat").read_text().splitlines():
            match = re.match(r"cpu(\d+)\s+(.+)", line)
            if not match:
                continue
            ident = int(match.group(1)); values = [int(x) for x in match.group(2).split()]
            current[ident] = values
            old = previous.get(ident) if previous else None
            load = None
            if old and len(old) == len(values):
                total = sum(values) - sum(old)
                idle = values[3] + values[4] - old[3] - old[4] if len(values) > 4 else values[3] - old[3]
                if total > 0:
                    load = max(0.0, min(100.0, 100.0 * (total - idle) / total))
            cores.append({"id": ident, "load_percent": load})
    except (OSError, ValueError, IndexError):
        pass
    frequencies = []
    for path in Path(sys_root).glob("devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"):
        try:
            frequencies.append(int(path.read_text().strip()) / 1000)
        except (OSError, ValueError):
            pass
    return current, cores, sum(frequencies) / len(frequencies) if frequencies else None


def _cpu_temperature(sys_root):
    values = []
    recognized = {"k10temp", "coretemp", "zenpower", "cpu_thermal", "acpitz", "peci_cputemp"}
    for hwmon in Path(sys_root).glob("class/hwmon/hwmon*"):
        try:
            name = (hwmon / "name").read_text().strip()
            if name not in recognized:
                continue
            found = []
            for inp in hwmon.glob("temp*_input"):
                try:
                    label = (hwmon / inp.name.replace("_input", "_label")).read_text().strip().lower()
                    if label in ("tctl", "tdie", "package", "package id 0", "tctl/tdie", "cpu", "die"):
                        found.append(float(inp.read_text()) / 1000)
                except OSError:
                    pass
            if not found:
                found = [float(inp.read_text()) / 1000 for inp in hwmon.glob("temp*_input")]
            values.extend(found)
        except (OSError, ValueError):
            pass
    return max(values) if values else None


def _nvme_written(sys_root):
    total = 0; found = False
    for path in Path(sys_root).glob("block/nvme*/stat"):
        try:
            fields = path.read_text().split()
            if len(fields) > 6:
                total += int(fields[6]) * 512; found = True
        except (OSError, ValueError):
            pass
    return total if found else None


def _float_or_none(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _int_or_none(value):
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def _bytes_to_mib(value):
    value = _float_or_none(value)
    return value / 1048576 if value is not None else None


def _kib_to_mib(value):
    value = _float_or_none(value)
    return value / 1024 if value is not None else None


def _ratio_percent(numerator, denominator):
    n = _float_or_none(numerator); d = _float_or_none(denominator)
    return (100.0 * n / d) if n is not None and d else None


def _safe_text(value):
    if value is None:
        return None
    text = str(value)
    text = re.sub(r"(?i)(authorization|api[_-]?key|token)\s*[:=]\s*[^\s,;]+", r"\1=[redacted]", text)
    text = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/-]+=*", "Bearer [redacted]", text)
    return text[:240]


def _safe_endpoint(value):
    path = str(value).split("?", 1)[0]
    return path[:120]


def _safe_event_detail(value):
    return _safe_text(value)


def _weight_estimate(model):
    model = str(model or "").lower()
    if "occamy" in model:
        return ESTIMATED_WEIGHT_READ_MB_PER_TOKEN["occamy"]
    if "tiel" in model or "coder" in model:
        return ESTIMATED_WEIGHT_READ_MB_PER_TOKEN["tiel-coder"]
    return None


def _counter_sum(counters, keys):
    if not isinstance(counters, dict):
        return 0
    return sum(_int_or_none(counters.get(k)) or 0 for k in keys)


def _severity(value, crit, warn, greater=False):
    value = _float_or_none(value)
    if value is None:
        return "warn"
    if greater:
        return "crit" if value > crit else "warn" if value > warn else "ok"
    return "crit" if value < crit else "warn" if value < warn else "ok"


def _combine(*states):
    return "crit" if "crit" in states else "warn" if "warn" in states else "ok"


def _health(blocks, capabilities, data):
    available = {
        "gpu": any(capabilities.get(k) for k in ("gpu_nvidia", "gpu_amd", "gpu_intel")),
        "memory": any(data.get(k) is not None for k in ("ram_available_mib", "zram_used_percent", "psi_full_avg10")),
        "cpu": data.get("cpu_temperature_c") is not None,
        "reliability": data.get("reliability_available", False),
        "context": data.get("context_percent") is not None,
    }
    ranks = {"ok": 0, "warn": 1, "crit": 2}
    level = max((blocks.get(name, "ok") for name, yes in available.items() if yes), key=lambda x: ranks.get(x, 0), default="ok")
    reasons = []
    for block, state in blocks.items():
        if not available.get(block) or state == "ok":
            continue
        if block == "reliability":
            if data.get("finish_length_count", 0):
                reasons.append({"code": "length_cuts", "text": "A response reached its output limit"})
            if data.get("request_errors", 0):
                reasons.append({"code": "request_errors", "text": "One or more requests failed"})
            if data.get("server_restarts", 0):
                reasons.append({"code": "server_restarts", "text": "The model server restarted"})
            if data.get("server_crashes", 0):
                reasons.append({"code": "server_crashes", "text": "The model server crashed"})
            if data.get("ssd_free_gb") is not None and data["ssd_free_gb"] < THRESHOLDS["ssd_crit_free_gb"]:
                reasons.append({"code": "disk_space_low", "text": "Free disk space is low"})
            continue
        code, text = HEALTH_REASONS.get(block, {}).get(state, (None, None))
        if code:
            reasons.append({"code": code, "text": text})
    hint = {"ok": "No action needed.", "warn": "Check the items marked WARNING.",
            "crit": "Reduce load or free the affected resource."}[level]
    return {"level": level, "reasons": reasons, "hint": hint}


def _status_line(status, generating, live_rate, free_vram, polite, last_request, events):
    game = status.get("game_mode") if isinstance(status.get("game_mode"), dict) else {}
    policy = polite if isinstance(polite, dict) else {}
    sample = policy.get("sample") if isinstance(policy.get("sample"), dict) else {}
    busy_limit = _float_or_none(policy.get("other_busy_cpu_threshold"))
    if game.get("active") and not status.get("llama_pid"):
        code = "game_parked"
    elif status.get("memory_pressure"):
        code = "memory_pressure"
    elif status.get("vram_guard_enabled") and free_vram is not None and free_vram < THRESHOLDS["vram_warn_free_mib"]:
        code = "low_vram"
    elif policy.get("enabled") and busy_limit is not None and (sample.get("other_busy_cpus") or 0) >= busy_limit:
        code = "polite_busy"
    elif generating and not live_rate:
        code = "waiting_first_token"
    elif generating:
        code = "generating"
    elif (_counter_sum(status.get("event_counters", {}), ("server_restarted", "server_restart")) > 0 or
          _counter_sum(status.get("event_counters", {}), ("server_started", "model_loaded")) > 1) and any(
             isinstance(item, dict) and item.get("reason") in ("server_restarted", "server_restart", "server_started", "model_loaded")
             and time.time() - (_float_or_none(item.get("at")) or 0) < 30 for item in events):
        code = "cold_cache"
    else:
        code = "neutral"
    return {"code": code, "text": STATUS_LINES[code]}


def _status_name(status):
    return {"READY": "LOADED", "PARKED_BY_GAME": "UNLOADED"}.get(status or "UNLOADED", status or "UNLOADED")


def _state_color(status):
    return "ok" if status in ("LOADED", "UNLOADED") else "warn" if status in ("LOADING", "MEMORY_PRESSURE") else "crit"


def _mode(snapshot):
    game = snapshot.get("game_mode") if isinstance(snapshot.get("game_mode"), dict) else {}
    return "OFF" if snapshot.get("override") == "OFF" else "GAME" if game.get("active") else "VISION" if snapshot.get("vision") else "AUTO"


def _read_saved_slot(slot_dir, model, vision):
    if not model:
        return False
    if not slot_dir:
        return None
    try:
        path = Path(slot_dir) / f"{model}-{'vision' if vision else 'text'}.bin"
        return path.is_file() and not path.is_symlink()
    except OSError:
        return None


def _slot_context(slots, limit):
    if slots is None:
        return None, limit if limit else None, None, None
    if not isinstance(slots, list) or not slots:
        return None, limit if limit else None, None, None
    # Report the busiest slot if multiple slots are configured.
    valid = [s for s in slots if isinstance(s, dict)]
    if not valid:
        return None, limit if limit else None, None, None
    slot = max(valid, key=lambda s: _int_or_none(s.get("n_past", s.get("n_prompt_tokens", 0))) or 0)
    used = _int_or_none(slot.get("n_past"))
    if used is None:
        used = _int_or_none(slot.get("n_prompt_tokens"))
    if used is None:
        used = _int_or_none(slot.get("n_tokens"))
    actual_limit = _int_or_none(slot.get("n_ctx")) or _int_or_none(limit)
    if used is None:
        return None, actual_limit, None, _int_or_none(slot.get("kv_size_bytes"))
    used = max(0, used)
    return used, actual_limit, (100.0 * used / actual_limit if actual_limit else None), _int_or_none(slot.get("kv_size_bytes"))


def _empty_state():
    return {"schema": 1, "capabilities": {name: False for name in (
                "gpu_nvidia", "gpu_amd", "gpu_intel", "system_linux", "zram", "psi", "cpu_temp", "nvme", "slots", "timings", "expert_cache")},
            "health": {"level": "ok", "reasons": [], "hint": "Telemetry is temporarily unavailable."},
            "status_line": {"code": "neutral", "text": STATUS_LINES["neutral"]},
            "monitor_self": {"cpu_seconds": None, "rss_mib": None, "stream_subscribers": 0},
            "header": {"model": None, "mode": "AUTO", "state": "UNLOADED", "state_color": "ok", "model_uptime_seconds": None},
            "now": {"generating": False, "live_tokens_per_second": 0.0, "rate_is_estimate": True, "request_number": None, "request_age_seconds": None},
            "last_request": {k: None for k in ("ttft_seconds", "prompt_per_second", "decode_per_second", "drafts_proposed", "drafts_accepted", "draft_acceptance_percent", "finish_reason", "prefix_reuse_percent")},
            "context": {"used_tokens": 0, "limit_tokens": None, "used_percent": None, "kv_bytes": None, "saved_slot": None, "prefix_reuse_percent": None},
            "gpu": {}, "memory": {}, "cpu": {}, "reliability": {}, "events": [],
            "block_status": {k: "warn" for k in ("gpu", "memory", "cpu", "reliability", "context")}}


class Monitor:
    def __init__(self, proc_root="/proc", sys_root="/sys", home="/home", wall_clock=time.time, monotonic=time.monotonic):
        self.proc_root=Path(proc_root);self.sys_root=Path(sys_root);self.home=Path(home)
        self.wall_clock=wall_clock;self.monotonic=monotonic
        self._lock=threading.RLock();self._request_lock=threading.RLock();self._cache=None;self._cache_at=None;self._cpu_previous=None
        self._nvme_base=None;self._nvme_delta=None;self._seq=0
        self._active={};self._requests=deque(maxlen=100);self._events=deque(maxlen=30)
        self._subscribers = 0
        self._sub_lock = threading.Lock()  # counting streams must not wait behind a slow sample
        self.system_source = (WindowsSystemSource() if os.name == "nt" else
                              LinuxSystemSource(self.proc_root, self.sys_root, self.home, self.wall_clock))
        self.request_errors=0;self.finish_length=0
        self._error_times=deque(maxlen=200);self._length_times=deque(maxlen=200)

    def get_or_sample(self, factory, cache_seconds=1.0):
        """Cache the complete on-demand sample, including bounded backend queries, for one second."""
        try:
            now = self.monotonic()
            with self._lock:
                if self._cache is not None and self._cache_at is not None and now - self._cache_at < cache_seconds:
                    cached = copy.deepcopy(self._cache)
                    if isinstance(cached.get("monitor_self"), dict):
                        cached["monitor_self"]["stream_subscribers"] = self._subscribers
                    return cached
                return factory()
        except Exception:
            return _empty_state()

    def begin_request(self, model, endpoint, method="POST"):
        try:
            if not str(endpoint).startswith(("/v1/", "/completion", "/infill")):
                return None
            with self._request_lock:
                self._seq+=1;ident=self._seq
                self._prune_active()
                if len(self._active) >= 64:
                    return None
                self._active[ident]={"id":ident,"last_mono":self.monotonic(),"model":_safe_text(model),"endpoint":_safe_endpoint(endpoint),"stream":False,"started_at":self.wall_clock(),"started_mono":self.monotonic(),"pending":b"","chunks":deque(maxlen=2000),"ttft":None,"prompt_tokens":None,"completion_tokens":None,"cached_tokens":None,"prompt_per_second":None,"decode_per_second":None,"draft_n":None,"draft_n_accepted":None,"finish_reason":None,"status":None}
            return ident
        except Exception:
            return None

    def _prune_active(self):
        """Forget requests whose owner never finished them (caller holds _request_lock)."""
        now=self.monotonic()
        for ident in [i for i,r in self._active.items() if now-r["last_mono"]>ACTIVE_IDLE_SECONDS]:
            del self._active[ident]

    def set_request_model(self, ident, model):
        try:
            with self._request_lock:
                if ident in self._active:
                    self._active[ident]["model"] = _safe_text(model)
        except Exception:
            pass

    def _parse_payload(self, request, payload):
        try:obj=json.loads(payload)
        except (TypeError,ValueError):return
        if not isinstance(obj,dict):return
        usage=obj.get("usage")
        if isinstance(usage,dict):
            request["prompt_tokens"]=usage.get("prompt_tokens",request["prompt_tokens"])
            request["completion_tokens"]=usage.get("completion_tokens",request["completion_tokens"])
            details=usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
            if isinstance(details,dict):request["cached_tokens"]=details.get("cached_tokens",request["cached_tokens"])
        timing=obj.get("timings")
        if isinstance(timing,dict):
            request["prompt_per_second"]=timing.get("prompt_per_second",request["prompt_per_second"])
            request["decode_per_second"]=timing.get("predicted_per_second",request["decode_per_second"])
            request["draft_n"]=timing.get("draft_n",request["draft_n"])
            request["draft_n_accepted"]=timing.get("draft_n_accepted",request["draft_n_accepted"])
        for choice in obj.get("choices",[]) if isinstance(obj.get("choices",[]),list) else []:
            if not isinstance(choice,dict):continue
            if choice.get("finish_reason") is not None:request["finish_reason"]=_safe_text(choice.get("finish_reason"))
            generated=False
            for item in (choice.get("delta"),choice.get("message")):
                if isinstance(item,dict):
                    generated=generated or bool(item.get("content")) or bool(item.get("tool_calls"))
            if generated and request.get("stream"):
                now=self.monotonic();request["chunks"].append(now)
                if request["ttft"] is None:request["ttft"]=max(0.0,now-request["started_mono"])

    def observe_sse(self, ident, chunk):
        try:
            with self._request_lock:
                req=self._active.get(ident)
                if req is None:return
                req["last_mono"]=self.monotonic()
                req["stream"] = True
                data=(req["pending"]+bytes(chunk)).replace(b"\r\n",b"\n")
                if len(data)>131072:data=data[-65536:]
                frames=data.split(b"\n\n");req["pending"]=frames.pop()[-65536:]
                for frame in frames:
                    lines=[line[5:].lstrip() for line in frame.split(b"\n") if line.startswith(b"data:")]
                    if not lines:continue
                    payload=b"\n".join(lines)
                    if payload!=b"[DONE]":self._parse_payload(req,payload)
        except Exception:
            return

    def observe_json(self, ident, payload):
        try:
            with self._request_lock:
                req=self._active.get(ident)
                if req is not None:
                    req["last_mono"]=self.monotonic();self._parse_payload(req,payload)
        except Exception:
            return

    def set_status(self, ident, status):
        try:
            with self._request_lock:
                if ident in self._active:self._active[ident]["status"]=int(status)
        except Exception:pass

    def finish_request(self, ident, status=None, error=False, expected=False):
        """`expected` marks a deliberate refusal (game pause, AI off); 4xx is the client's fault.
        Neither is a reliability failure of the gateway or the model server."""
        try:
            if ident is None:return
            with self._request_lock:
                req=self._active.pop(ident,None)
                if req is None:return
                code=_int_or_none(status if status is not None else req.get("status")) or 0
                if (error or code>=400 or code==0) and not expected and not 400<=code<500:
                    self.request_errors+=1;self._error_times.append(self.monotonic())
                if req.get("finish_reason")=="length":
                    self.finish_length+=1;self._length_times.append(self.monotonic())
                duration=max(0.0,self.monotonic()-req["started_mono"])
                self._requests.append({"at":self.wall_clock(),"model":req["model"],"endpoint":req["endpoint"],"status":code or None,
                    "prompt_tokens":_int_or_none(req.get("prompt_tokens")),"completion_tokens":_int_or_none(req.get("completion_tokens")),
                    "ttft_seconds":_float_or_none(req.get("ttft")),"prompt_per_second":_float_or_none(req.get("prompt_per_second")),
                    "decode_per_second":_float_or_none(req.get("decode_per_second")),"drafts_proposed":_int_or_none(req.get("draft_n")),
                    "drafts_accepted":_int_or_none(req.get("draft_n_accepted")),"draft_acceptance_percent":_ratio_percent(req.get("draft_n_accepted"),req.get("draft_n")),
                    "finish_reason":req.get("finish_reason"),"duration_seconds":round(duration,3),"cached_tokens":_int_or_none(req.get("cached_tokens")),
                    "prefix_reuse_percent":_ratio_percent(req.get("cached_tokens"),req.get("prompt_tokens"))})
        except Exception:
            return

    def requests(self):
        try:
            with self._request_lock:return copy.deepcopy(list(reversed(self._requests)))
        except Exception:return []

    def add_event(self, item):
        try:
            if not isinstance(item,dict):return
            with self._request_lock:self._events.append({"reason":_safe_text(item.get("reason")),"detail":_safe_event_detail(item.get("detail")),"at":_float_or_none(item.get("at")) or self.wall_clock()})
        except Exception:pass

    def events(self, source=()):
        try:
            if source:
                items = list(source)[-30:][::-1]
                return [{"reason": _safe_text(x.get("reason")), "detail": _safe_event_detail(x.get("detail")),
                         "at": _float_or_none(x.get("at"))} for x in items if isinstance(x, dict)]
            with self._request_lock:return copy.deepcopy(list(reversed(self._events)))
        except Exception:return []

    def _system_sample(self):
        readers = {"meminfo": _read_meminfo, "psi": _read_psi, "zram": _read_zram,
                   "cpu": _read_cpu, "cpu_temp": _cpu_temperature, "nvme": _nvme_written}
        if isinstance(self.system_source, LinuxSystemSource):
            return self.system_source.sample(readers)
        return {"memory": {key: None for key in ("ram_total_mib", "ram_available_mib", "ram_used_mib",
                    "zram_original_mib", "zram_compressed_mib", "zram_used_percent", "zram_disk_mib",
                    "psi_some_avg10", "psi_full_avg10")},
                "cpu": {"temperature_c": None, "frequency_mhz": None, "cores": []},
                "ssd_free_gb": None, "nvme_written_bytes": None, "sampled_at": self.wall_clock()}

    def snapshot(self, *, status, state, model_config, gpu, slots=None, expert_cache=None, polite=None, gpu_capabilities=None, cache_seconds=1):
        try:
            now_mono=self.monotonic()
            with self._lock:
                if self._cache is not None and self._cache_at is not None and now_mono-self._cache_at<cache_seconds:
                    cached = copy.deepcopy(self._cache)
                    if isinstance(cached.get("monitor_self"), dict):
                        cached["monitor_self"]["stream_subscribers"] = self._subscribers
                    return cached
                system=self._system_sample();cfg=model_config or {};model=status.get("model") or status.get("default_model") or ""
                pid=status.get("llama_pid");slot_limit=_int_or_none(cfg.get("ctxSize"));slot=slots[0] if isinstance(slots,list) and slots and isinstance(slots[0],dict) else None
                used,limit,pct,kv_bytes=_slot_context(slots,slot_limit)
                if not pid and _status_name(status.get("status")) == "UNLOADED":
                    used=0;pct=0.0 if limit else None
                with self._request_lock:
                    self._prune_active()
                    active=max(self._active.values(),key=lambda r:r["started_mono"],default=None)
                    last=copy.deepcopy(self._requests[-1]) if self._requests else {}
                    request_errors=self.request_errors;finish_length=self.finish_length
                    recent_errors=sum(now_mono-t<=RELIABILITY_WINDOW_SECONDS for t in self._error_times)
                    recent_length=sum(now_mono-t<=RELIABILITY_WINDOW_SECONDS for t in self._length_times)
                timings_available = any(last.get(k) is not None for k in ("prompt_per_second", "decode_per_second"))
                if active:
                    timings_available = timings_available or any(active.get(k) is not None for k in ("prompt_per_second", "decode_per_second"))
                if active:
                    chunks=[x for x in active["chunks"] if now_mono-x<=2]
                    live=len(chunks)/2.0;age=max(0,now_mono-active["started_mono"]);rid=active["id"];generating=True
                else:live=0.0;age=None;rid=None;generating=False
                free=_float_or_none(gpu.get("free_mib"));used_vram=_float_or_none(gpu.get("used_mib"));proc_mib=_float_or_none(gpu.get("model_process_mib"))
                other=max(0,used_vram-proc_mib) if used_vram is not None and proc_mib is not None else used_vram if not pid else None
                av=system["memory"]["ram_available_mib"];av_gib=av/1024 if av is not None else None
                psi=system["memory"]["psi_full_avg10"];zram=system["memory"]["zram_used_percent"];disk=system["ssd_free_gb"]
                event_counts = status.get("event_counters", {})
                gpu_checks=[]
                if free is not None:gpu_checks.append(_severity(free,THRESHOLDS["vram_crit_free_mib"],THRESHOLDS["vram_warn_free_mib"]))
                if _float_or_none(gpu.get("temperature_c")) is not None:gpu_checks.append(_severity(gpu.get("temperature_c"),THRESHOLDS["gpu_temp_crit_c"],THRESHOLDS["gpu_temp_warn_c"],True))
                gpu_color=_combine(*gpu_checks) if gpu_checks else "warn"
                mem_checks=[]
                if av_gib is not None:mem_checks.append(_severity(av_gib,THRESHOLDS["ram_crit_available_gib"],THRESHOLDS["ram_warn_available_gib"]))
                if psi is not None:mem_checks.append(_severity(psi,THRESHOLDS["psi_full_crit_avg10"],THRESHOLDS["psi_full_warn_avg10"],True))
                if zram is not None:mem_checks.append(_severity(zram,THRESHOLDS["zram_crit_percent"],THRESHOLDS["zram_warn_percent"],True))
                mem_color=_combine(*mem_checks) if mem_checks else "warn"
                cpu_color=_severity(system["cpu"]["temperature_c"],THRESHOLDS["cpu_temp_crit_c"],THRESHOLDS["cpu_temp_warn_c"],True)
                reliability_checks=[_severity(disk,THRESHOLDS["ssd_crit_free_gb"],THRESHOLDS["ssd_crit_free_gb"])] if disk is not None else []
                if recent_length:reliability_checks.append("warn")
                if recent_errors:reliability_checks.append("crit")
                if _counter_sum(event_counts,("server_restarted","server_restart")):reliability_checks.append("crit")
                if _counter_sum(event_counts,("server_crashed",)) or status.get("crash_count"):reliability_checks.append("crit")
                reliability_color=_combine(*reliability_checks) if reliability_checks else "ok"
                slots_cache=_int_or_none(expert_cache.get("slots")) if isinstance(expert_cache,dict) else None
                cache_pct=expert_cache.get("hit_percent") if isinstance(expert_cache,dict) else None
                weight_mb=_weight_estimate(model)
                weight_rate=live*weight_mb/1000 if weight_mb is not None and generating else (0.0 if weight_mb is not None else None)
                state_name=_status_name(status.get("status"));slot_saved=status.get("saved_slot")
                if slot_saved is None:slot_saved=_read_saved_slot(status.get("slot_dir"),model,status.get("vision"))
                server_starts = _counter_sum(event_counts, ("server_started", "model_loaded"))
                server_crashes = _counter_sum(event_counts,("server_crashed",)) or (_int_or_none(status.get("crash_count")) or 0)
                # Idle unloads, model/vision switches and gear changes also start the server again;
                # only a start that follows a crash is a restart.
                server_restarts = min(max(0, server_starts - 1), server_crashes) + _counter_sum(event_counts,("server_restarted","server_restart"))
                block_status={"gpu":gpu_color,"memory":mem_color,"cpu":cpu_color,
                    "reliability":reliability_color,"context":"ok" if pct is not None and pct < THRESHOLDS["context_warn_percent"] else
                    "crit" if pct is not None and pct > THRESHOLDS["context_crit_percent"] else
                    "warn" if pct is not None else "warn"}
                body={"schema":1,"header":{"model":model or None,"mode":_mode(status),"state":state_name,"state_color":_state_color(state_name),
                                "model_uptime_seconds":model_uptime_seconds(pid,self.proc_root,self.wall_clock()) if pid else None},
                    "now":{"generating":generating,"live_tokens_per_second":live,"rate_is_estimate":True,"request_number":rid,"request_age_seconds":age},
                    "last_request":{k:last.get(k) for k in ("ttft_seconds","prompt_per_second","decode_per_second","drafts_proposed","drafts_accepted","draft_acceptance_percent","finish_reason","prefix_reuse_percent")},
                    "context":{"used_tokens":used,"limit_tokens":limit,"used_percent":pct,"kv_bytes":kv_bytes,"saved_slot":slot_saved,
                               "prefix_reuse_percent":last.get("prefix_reuse_percent")},
                    "gpu":{"total_mib":_float_or_none(gpu.get("total_mib")),"used_mib":used_vram,"free_mib":free,"model_process_mib":proc_mib,
                            "other_processes_mib":other,"margin_over_reserve_mib":free-THRESHOLDS["vram_reserve_mib"] if free is not None else None,
                            "utilization_percent":gpu.get("utilization_percent"),"temperature_c":_float_or_none(gpu.get("temperature_c")),
                            "power_w":_float_or_none(gpu.get("power_w")),"pcie_generation":_int_or_none(gpu.get("pcie_generation")),
                            "pcie_width":_int_or_none(gpu.get("pcie_width")),"expert_cache_slots":slots_cache,"expert_cache_hit_percent":cache_pct},
                    "memory":{**system["memory"],"weight_read_gb_per_second":weight_rate},
                    "cpu":{**system["cpu"],"polite_mode":polite.get("enabled") if isinstance(polite,dict) else None},
                    "reliability":{"finish_length_count":finish_length,"request_errors":request_errors,
                        "server_restarts":server_restarts,
                        "server_crashes":server_crashes,
                        "tool_errors":None,"ssd_free_gb":disk,
                        "nvme_written_bytes":system["nvme_written_bytes"]},
                    "events":self.events(status.get("events",())),"block_status":block_status,"sampled_at":system["sampled_at"]}
                capabilities = dict(gpu_capabilities or {})
                sys_caps = self.system_source.capabilities() if isinstance(self.system_source, LinuxSystemSource) else {}
                capabilities.update(sys_caps)
                capabilities.update({"slots": (isinstance(slots, list) and bool(slots)) or bool(slot_limit),
                    "timings": timings_available,
                    "expert_cache": isinstance(expert_cache, dict) and any(expert_cache.get(k) is not None for k in ("slots", "hit_percent"))})
                body["capabilities"] = {name: bool(capabilities.get(name, False)) for name in (
                    "gpu_nvidia", "gpu_amd", "gpu_intel", "system_linux", "zram", "psi",
                    "cpu_temp", "nvme", "slots", "timings", "expert_cache")}
                body["health"] = _health(block_status, body["capabilities"], {
                    "ram_available_mib": av, "zram_used_percent": zram, "psi_full_avg10": psi,
                    "cpu_temperature_c": system["cpu"].get("temperature_c"), "ssd_free_gb": disk,
                    "reliability_available": True,
                    "finish_length_count": recent_length, "request_errors": recent_errors,
                    "server_restarts": body["reliability"]["server_restarts"], "server_crashes": body["reliability"]["server_crashes"],
                    "context_percent": pct})
                body["status_line"] = _status_line(status, generating, live, free, polite, last, self.events(status.get("events", ())))
                body["monitor_self"] = self.self_metrics()
                self._cache=copy.deepcopy(body);self._cache_at=now_mono
                return body
        except Exception:
            return _empty_state()

    def self_metrics(self):
        rss_mib = None
        try:
            match = re.search(r"(?m)^VmRSS:\s+(\d+)\s+kB", (self.proc_root / str(os.getpid()) / "status").read_text())
            rss_mib = int(match.group(1)) / 1024 if match else None
        except OSError:
            pass
        return {"cpu_seconds": _float_or_none(time.process_time()), "rss_mib": rss_mib,
                "stream_subscribers": self.stream_subscribers}

    @property
    def stream_subscribers(self):
        with self._sub_lock:
            return self._subscribers

    def stream(self, state_provider, send, sleep=time.sleep):
        """Serve events only while this call has an attached downstream subscriber."""
        with self._sub_lock:
            if self._subscribers >= MAX_STREAM_SUBSCRIBERS:
                return
            self._subscribers += 1
        try:
            while True:
                try:
                    state = state_provider()
                except Exception:  # noqa: BLE001
                    state = _empty_state()  # headers are out: never start a second HTTP response
                send(state)
                sleep(1)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError, ValueError):
            return
        finally:
            with self._sub_lock:
                self._subscribers = max(0, self._subscribers - 1)


def monitor_route(handler, path, state_provider, requests_provider, authorized=True, monitor_instance=None):
    if path not in ("/monitor","/monitor/","/monitor/api/state","/monitor/api/requests","/monitor/api/stream"):
        return False
    if handler.command!="GET":handler.send_json(405,{"error":"GET only"});return True
    if not authorized and path not in ("/monitor", "/monitor/"):
        handler.send_response(401);handler.send_header("WWW-Authenticate","Bearer");handler.send_header("Content-Length","0");handler.end_headers();return True
    streaming = False
    try:
        if path in ("/monitor","/monitor/"):
            body=render_page().encode();handler.send_response(200);handler.send_header("Content-Type","text/html; charset=utf-8");handler.send_header("Content-Length",str(len(body)));handler.send_header("Cache-Control","no-store");handler.end_headers();handler.wfile.write(body)
        elif path=="/monitor/api/requests":
            body=json.dumps(requests_provider(),separators=(",",":")).encode();handler.send_response(200);handler.send_header("Content-Type","application/json; charset=utf-8");handler.send_header("Content-Length",str(len(body)));handler.send_header("Cache-Control","no-store");handler.end_headers();handler.wfile.write(body)
        elif path == "/monitor/api/stream":
            monitor = monitor_instance
            if monitor is None:
                handler.send_json(503, {"error": "stream unavailable"})
                return True
            if monitor.stream_subscribers >= MAX_STREAM_SUBSCRIBERS:
                handler.send_json(503, {"error": "too many monitor streams"})
                return True
            streaming = True  # from here on an exception must not produce a second HTTP response
            try:
                # A client that stops reading must not hold this thread forever.
                handler.connection.settimeout(STREAM_WRITE_TIMEOUT_SECONDS)
            except (AttributeError, OSError):
                pass
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
            handler.send_header("Cache-Control", "no-cache, no-transform")
            handler.send_header("Connection", "keep-alive")
            handler.send_header("X-Accel-Buffering", "no")
            handler.end_headers()
            def send(state):
                raw = json.dumps(state, separators=(",", ":"), ensure_ascii=False)
                handler.wfile.write((f"data: {raw}\n\n").encode("utf-8"))
                handler.wfile.flush()
            monitor.stream(state_provider, send)
        else:handler.send_json(200,state_provider())
    except Exception:
        if not streaming:
            try:handler.send_json(200,_empty_state())
            except Exception:pass
    return True


def monitor_authorized(headers, config):
    """The monitor shares an optional gateway key; it never embeds a key in HTML."""
    expected = config.get("gatewayApiKey") or config.get("monitorApiKey") or config.get("apiKey")
    if not expected:
        return True
    supplied = headers.get("Authorization", "") or ""
    # compare_digest raises TypeError for non-ASCII str; compare the exact bytes instead.
    return hmac.compare_digest(supplied.encode("utf-8", "surrogatepass"),
                               f"Bearer {expected}".encode("utf-8", "surrogatepass"))


def read_expert_cache_metrics(host, port, api_key=None, timeout=0.35):
    """Parse only explicit expert-cache metrics; absent server metrics mean n/a."""
    conn = None
    try:
        conn = http.client.HTTPConnection(host, int(port), timeout=timeout)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        conn.request("GET", "/metrics", headers=headers)
        reply = conn.getresponse()
        if reply.status != 200:
            return None
        text = reply.read(131072).decode("utf-8", errors="replace")
        hits = misses = slots = direct_rate = None
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            name = parts[0].split("{", 1)[0].lower()
            if not ("expert" in name or "moe" in name or "expert" in parts[0].lower()):
                continue
            try:
                value = float(parts[-1])
            except ValueError:
                continue
            if "hit_rate" in name or "hit_ratio" in name:
                direct_rate = value * 100 if value <= 1 else value
            elif "hit" in name:
                hits = (hits or 0.0) + value
            elif "miss" in name:
                misses = (misses or 0.0) + value
            elif "slot" in name:
                slots = int(value)
        rate = direct_rate
        if rate is None and hits is not None and misses is not None and hits + misses > 0:
            rate = 100 * hits / (hits + misses)
        if rate is None and slots is None:
            return None
        return {"slots": slots, "hit_percent": rate}
    except Exception:
        return None
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def gateway_snapshot(monitor, status_provider, model_config_provider, gpu_provider,
                     slots_provider, expert_metrics_provider, polite_provider=None):
    """Lazily gather a complete snapshot, and cache all I/O (including /slots) for 1 s."""
    def build():
        status = status_provider()
        model_id = status.get("model") or status.get("default_model")
        try:
            model_cfg = model_config_provider(model_id, bool(status.get("vision"))) if model_id else {}
        except Exception:
            model_cfg = {}
        try:
            gpu = gpu_provider(status)
        except Exception:
            gpu = {}
        try:
            slots = slots_provider(status)
        except Exception:
            slots = None
        try:
            expert = expert_metrics_provider(status)
        except Exception:
            expert = None
        try:
            polite = polite_provider() if polite_provider else None
        except Exception:
            polite = None
        try:
            nvidia_provider = (lambda: gpu) if gpu.get("source") == "gpu_nvidia" else None
            adapters = gpu_sources(monitor.sys_root, nvidia_provider)
            gpu_caps = {key: bool(adapters[key].available()) for key in ("gpu_nvidia", "gpu_amd", "gpu_intel")}
        except Exception:
            gpu_caps = {"gpu_nvidia": False, "gpu_amd": False, "gpu_intel": False}
        return monitor.snapshot(status=status, state=None, model_config=model_cfg, gpu=gpu,
                               slots=slots, expert_cache=expert, polite=polite, gpu_capabilities=gpu_caps)
    return monitor.get_or_sample(build)


def demo_data():
    """Fictional preview fixtures only; live context limits come from /slots or gateway config."""
    now = time.time()
    cores = [{"id": i, "load_percent": (i * 13 + 17) % 75} for i in range(12)]
    events = [
        {"at": now - 92, "reason": "server_started", "detail": "tiel-coder · text"},
        {"at": now - 38, "reason": "slot_restored", "detail": "Saved context restored"},
    ]
    requests = [
        {"at": now - 9, "model": "tiel-coder", "endpoint": "/v1/chat/completions", "status": 200, "prompt_tokens": 1824, "completion_tokens": 94, "ttft_seconds": .38, "prompt_per_second": 930, "decode_per_second": 51.4, "draft_acceptance_percent": 86, "finish_reason": "stop", "duration_seconds": 2.7},
        {"at": now - 24, "model": "tiel-coder", "endpoint": "/v1/chat/completions", "status": 200, "prompt_tokens": 756, "completion_tokens": 122, "ttft_seconds": .31, "prompt_per_second": 1010, "decode_per_second": 48.7, "draft_acceptance_percent": 82, "finish_reason": "stop", "duration_seconds": 3.0},
        {"at": now - 47, "model": "tiel-coder", "endpoint": "/v1/completions", "status": 200, "prompt_tokens": 1120, "completion_tokens": 77, "ttft_seconds": .29, "prompt_per_second": 970, "decode_per_second": 52.1, "draft_acceptance_percent": 88, "finish_reason": "stop", "duration_seconds": 2.1},
    ]
    normal = {
        "header": {"model": "tiel-coder", "mode": "AUTO", "state": "LOADED", "state_color": "ok", "model_uptime_seconds": 8123},
        "now": {"generating": True, "live_tokens_per_second": 49.8, "rate_is_estimate": True, "request_number": 147, "request_age_seconds": 2.6},
        "last_request": {"ttft_seconds": .38, "prompt_per_second": 930, "decode_per_second": 51.4, "drafts_proposed": 120, "drafts_accepted": 103, "draft_acceptance_percent": 85.8, "finish_reason": "stop", "prefix_reuse_percent": 73},
        "context": {"used_tokens": 24318, "limit_tokens": 200000, "used_percent": 12.16, "kv_bytes": 2147483648, "saved_slot": True, "prefix_reuse_percent": 73},
        "gpu": {"total_mib": 12282, "used_mib": 9882, "free_mib": 2400, "model_process_mib": 9400, "other_processes_mib": 482, "margin_over_reserve_mib": 864, "utilization_percent": 67, "temperature_c": 69, "power_w": 221, "pcie_generation": 4, "pcie_width": 16, "expert_cache_slots": 26, "expert_cache_hit_percent": 74},
        "memory": {"ram_total_mib": 32768, "ram_used_mib": 20480, "ram_available_mib": 12288, "zram_original_mib": 8200, "zram_compressed_mib": 4100, "zram_used_percent": 50, "zram_device_mib": 8192, "zram_disk_mib": 16384, "psi_some_avg10": 1.2, "psi_full_avg10": .4, "weight_read_gb_per_second": 20.34},
        "cpu": {"temperature_c": 58, "frequency_mhz": 3850, "polite_mode": False, "cores": cores},
        "reliability": {"finish_length_count": 0, "request_errors": 0, "server_restarts": 0, "server_crashes": 0, "tool_errors": None, "ssd_free_gb": 421, "nvme_written_bytes": 123456789},
        "events": events, "block_status": {"gpu": "ok", "memory": "ok", "cpu": "ok", "reliability": "ok", "context": "ok"},
    }
    high = copy.deepcopy(normal)
    high["header"].update({"state": "LOADED", "state_color": "ok"})
    high["now"].update({"live_tokens_per_second": 31.2, "request_number": 148})
    high["context"].update({"used_tokens": 168000, "used_percent": 84, "prefix_reuse_percent": 61, "kv_bytes": 8589934592})
    high["gpu"].update({"used_mib": 11982, "free_mib": 300, "model_process_mib": 10000, "other_processes_mib": 1982, "margin_over_reserve_mib": -1236, "utilization_percent": 96, "temperature_c": 84, "power_w": 310})
    high["memory"].update({"ram_available_mib": 2200, "ram_used_mib": 30568, "zram_used_percent": 90, "zram_device_mib": 14746, "zram_compressed_mib": 11000, "psi_some_avg10": 18, "psi_full_avg10": 12})
    high["cpu"].update({"temperature_c": 87, "cores": [{"id": i, "load_percent": 78 + (i % 5) * 4} for i in range(12)]})
    high["reliability"].update({"finish_length_count": 2, "request_errors": 1, "server_restarts": 1})
    high["block_status"].update({"gpu": "crit", "memory": "crit", "cpu": "warn", "reliability": "ok"})
    high["events"] = events + [{"at": now - 6, "reason": "memory_pressure", "detail": "VRAM reserve crossed"}]
    unloaded = copy.deepcopy(normal)
    unloaded["header"].update({"state": "UNLOADED", "state_color": "ok", "model_uptime_seconds": None})
    unloaded["now"].update({"generating": False, "live_tokens_per_second": 0, "request_number": None, "request_age_seconds": None})
    unloaded["last_request"] = {k: None for k in normal["last_request"]}
    unloaded["context"].update({"used_tokens": 0, "used_percent": 0, "kv_bytes": None, "saved_slot": False, "prefix_reuse_percent": None})
    unloaded["gpu"].update({"used_mib": 720, "free_mib": 11562, "model_process_mib": 0, "other_processes_mib": 720, "margin_over_reserve_mib": 10026, "utilization_percent": 3, "temperature_c": 43, "power_w": 34, "expert_cache_slots": None, "expert_cache_hit_percent": None})
    unloaded["block_status"].update({"gpu": "ok", "memory": "ok", "cpu": "ok", "context": "ok"})
    all_caps = {name: True for name in ("gpu_nvidia", "system_linux", "zram", "psi", "cpu_temp", "nvme", "slots", "timings", "expert_cache")}
    all_caps.update({"gpu_amd": False, "gpu_intel": False})
    for state, level, line in ((normal, "ok", "Generating at full speed"), (high, "crit", "GPU memory is low: model may be unloaded for a game"), (unloaded, "ok", "Monitoring model and system health")):
        state.update({"schema": 1, "capabilities": dict(all_caps),
                      "health": {"level": level, "reasons": [], "hint": "No action needed." if level == "ok" else "Reduce load or free the affected resource."},
                      "status_line": {"code": "demo", "text": line},
                      "monitor_self": {"cpu_seconds": 0.1, "rss_mib": 12.0, "stream_subscribers": 0}})
    return {"scenarios": {"normal": {"state": normal, "requests": requests}, "high": {"state": high, "requests": requests}, "unloaded": {"state": unloaded, "requests": []}}}
