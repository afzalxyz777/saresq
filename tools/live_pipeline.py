"""The whole payload, live in a browser: thermal, RGB, gate, detector, crops.

    ~/saresq-venv/bin/python3 tools/live_pipeline.py
    then open  http://<pi-ip>:8091

Runs on the Pi. This is the demonstration piece -- it shows the pipeline doing
what the deck says it does, on the hardware, in front of whoever is watching:

    MLX90640  ->  gate (z >= z_t)  ->  crop the RGB frame there  ->  detector

Each stage is on screen at once, so the thing a judge asks about ("what does
the thermal actually give you?") is answerable by pointing rather than
explaining.

Design notes worth knowing before changing anything:

* The two sensors run in their own threads at their own rates. Thermal reads
  block for ~40 ms; RGB capture shells out to rpicam-still and takes ~900 ms.
  Coupling them would drop the thermal view to about 1 fps for no reason.
* Detection runs in a third thread on the newest RGB frame, not per request,
  so browser viewers never wait on a 350 ms inference.
* Gate maths comes from configs/pipeline.yaml, never from constants typed
  here -- the thresholds on screen are the ones that would fly.
* The RGB detector is a SEPARATE model from the thermal one. The shipped
  yolov8n_p3_*.tflite is trained on thermal and finds nothing in visible
  light; that is a domain gap, not a fault. Point --rgb-model at a COCO
  export for the visible branch.
"""
from __future__ import annotations

import argparse
import collections
import functools
import http.server
import json
import pathlib
import socketserver
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
SAVE = REPO / "results" / "demo"


def orient(a: np.ndarray, mode: str) -> np.ndarray:
    """Put the thermal array the same way up as the RGB frame.

    There is no canonical 'up' for the MLX90640 -- row 0 lands wherever the
    board was mounted. What matters is that thermal and RGB agree, because the
    gate maps a blob's (row, col) onto the RGB frame to choose a crop. If the
    two disagree, every crop comes from the wrong corner and the pipeline looks
    broken while each half works perfectly.

    Find the right value once by touching a finger to a known corner and seeing
    where the hot spot lands.
    """
    if "h" in mode:
        a = np.fliplr(a)
    if "v" in mode:
        a = np.flipud(a)
    if "r" in mode:
        a = np.rot90(a, 2)
    return a

@functools.lru_cache(maxsize=4)
def app_icon(size: int = 512) -> bytes:
    """PNG of the thermal hot-spot marker, for the phone's home screen.

    Drawn rather than shipped as a file: it is the same INFERNO map and the
    same white peak ring that colorize() puts on the live view, so the icon on
    the home screen and the thing on screen are recognisably one instrument.
    """
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    c = (size - 1) / 2
    r = np.hypot(x - c, y - c) / (size * 0.44)
    field = (np.clip(1.0 - r, 0, 1) ** 1.7 * 255).astype(np.uint8)
    img = cv2.applyColorMap(field, cv2.COLORMAP_INFERNO)
    cv2.circle(img, (int(c), int(c)), int(size * 0.29), (255, 255, 255),
               max(2, size // 72), cv2.LINE_AA)
    return cv2.imencode(".png", img)[1].tobytes()


MANIFEST = json.dumps({
    "name": "SaResQ Payload",
    "short_name": "SaResQ",
    "start_url": "/",
    "display": "standalone",
    "background_color": "#0b0d11",
    "theme_color": "#0b0d11",
    "icons": [
        {"src": "/icon-180.png", "sizes": "180x180", "type": "image/png"},
        {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png",
         "purpose": "any maskable"},
    ],
}).encode()

# 180 is what iOS asks for as the apple-touch-icon; 512 is the Android/PWA
# size. Both are rendered on first request and cached by app_icon().
ICONS = {"/icon-180.png": 180, "/icon-512.png": 512}


# No webfont is linked on purpose. At the venue the Pi's only network is a
# phone hotspot that may have no data at all, and a page that waits on
# fonts.googleapis.com in front of judges is a page that flashes unstyled
# text. System faces render instantly with no network.
PAGE = """<!doctype html><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name=theme-color content="#0b0d11">
<meta name=apple-mobile-web-app-capable content=yes>
<meta name=apple-mobile-web-app-status-bar-style content=black-translucent>
<meta name=apple-mobile-web-app-title content=SaResQ>
<link rel=manifest href=/manifest.webmanifest>
<link rel=apple-touch-icon href=/icon-180.png>
<title>SaResQ Payload</title>
<style>
 :root{
   color-scheme:dark;
   --bg:#0a0c10; --card:#13171e; --sink:#0e1116; --rule:#242a34;
   --ink:#eceae6; --soft:#8d96a3; --dim:#5d6672;
   --ok:#37c07a; --heat:#ff8a3d; --warn:#e8b23a;
   --ui:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
   --mono:ui-monospace,SFMono-Regular,Menlo,monospace;
 }
 *{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
 /* .card sets display:flex, which outranks the browser's own [hidden] rule.
    Without this the tabs switch nothing and every panel stays on screen. */
 [hidden]{display:none!important}
 body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 var(--ui);
      padding:0 0 calc(72px + env(safe-area-inset-bottom)) 0;
      -webkit-text-size-adjust:100%}

 /* --- top bar: sticky, because on a phone you scroll away from it ------ */
 header{position:sticky;top:0;z-index:20;display:flex;align-items:center;
        gap:10px;padding:calc(10px + env(safe-area-inset-top)) 14px 10px;
        background:rgba(10,12,16,.92);backdrop-filter:blur(12px);
        border-bottom:1px solid var(--rule)}
 .brand{font-weight:650;letter-spacing:.02em;font-size:15px;
        display:flex;align-items:center;gap:8px}
 .dot{width:8px;height:8px;border-radius:50%;background:var(--dim);flex:none}
 .dot.live{background:var(--ok);box-shadow:0 0 0 3px rgba(55,192,122,.16)}
 .dot.hot{background:var(--heat);box-shadow:0 0 0 3px rgba(232,138,48,.18)}
 .where{color:var(--dim);font:11px/1 var(--mono);letter-spacing:.08em;
        text-transform:uppercase}
 header .sp{flex:1}

 /* --- the verdict: the one thing a judge reads from two metres away --- */
 #verdict{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;
          margin:12px 14px;padding:14px 16px;border-radius:10px;
          border:1px solid var(--rule);background:var(--card);
          border-left:4px solid var(--dim)}
 #verdict b{font:650 26px/1.1 var(--ui);letter-spacing:-.01em}
 #verdict span{color:var(--soft);font:12px/1.4 var(--mono)}
 #verdict.found{border-left-color:var(--ok);background:#0e1c15}
 #verdict.found b{color:var(--ok)}
 #verdict.heat{border-left-color:var(--heat);background:#1d130c}
 #verdict.heat b{color:var(--heat)}

 /* ---- position strip: sits under the verdict, above the panels ---- */
 #pos{display:flex;align-items:center;gap:10px;flex-wrap:wrap;
      margin:0 14px 12px;padding:11px 16px;border-radius:10px;
      border:1px solid var(--rule);background:var(--card)}
 #pos b{font:600 15px/1.2 var(--mono);letter-spacing:.01em;
        font-variant-numeric:tabular-nums;color:var(--ink)}
 #pos span:not(.dot){color:var(--soft);font:11.5px/1.4 var(--mono)}
 #pos.fix{border-color:#1f6642;background:#0e1c15}
 #pos.fix b{color:var(--ok)}
 #pos.nofix b{color:var(--dim);font-size:13px}

 /* ---- scene strip: what KIND of disaster, from the whole frame ---- */
 #scene{display:flex;align-items:center;gap:10px;flex-wrap:wrap;
        margin:0 14px 12px;padding:11px 16px;border-radius:10px;
        border:1px solid var(--rule);background:var(--card)}
 #scene b{font:600 15px/1.2 var(--mono);letter-spacing:.06em;color:var(--ink)}
 #scene span:not(.dot){color:var(--soft);font:11.5px/1.4 var(--mono)}
 #scene.alert{border-color:#7a3d13;background:#1d130c}
 #scene.alert b{color:var(--heat)}
 #scene.calm b{color:var(--ok)}
 #scene.off b{color:var(--dim);font-size:13px}

 .grid{display:grid;gap:12px;padding:0 14px;max-width:1500px;margin:0 auto}
 .card{background:var(--card);border:1px solid var(--rule);border-radius:10px;
       overflow:hidden;display:flex;flex-direction:column}
 .card h2{font:600 10.5px/1 var(--mono);letter-spacing:.12em;
          text-transform:uppercase;color:var(--soft);margin:0;
          padding:11px 13px;border-bottom:1px solid var(--rule);
          display:flex;justify-content:space-between;align-items:center;gap:8px}
 .card canvas.view{width:100%;display:block;aspect-ratio:4/3;background:#000}
 .foot{padding:9px 13px;font:12px/1.45 var(--mono);color:var(--soft);
       border-top:1px solid var(--rule);min-height:36px;
       font-variant-numeric:tabular-nums}
 .foot b{color:var(--ink);font-weight:600}
 .foot.stale b{color:var(--warn)}

 .pill{font:11px/1 var(--mono);padding:4px 8px;border-radius:20px;
       border:1px solid var(--rule);color:var(--dim);letter-spacing:.04em;
       white-space:nowrap}
 .pill.on{background:#0f2a1d;border-color:#1f6642;color:var(--ok)}
 .pill.hot{background:#2a1710;border-color:#6d3a1c;color:var(--heat)}

 #crops{display:grid;gap:8px;padding:12px 13px;min-height:70px;
        grid-template-columns:repeat(auto-fill,minmax(92px,1fr))}
 #crops figure{margin:0}
 #crops img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:6px;
            border:1px solid var(--rule);background:var(--sink);display:block}
 #crops figcaption{font:11px/1.3 var(--mono);color:var(--soft);padding-top:4px;
                   text-align:center;font-variant-numeric:tabular-nums}
 #crops .none{grid-column:1/-1;color:var(--dim);font:12px/1.5 var(--mono)}

 #sys{padding:14px;text-align:center;color:var(--dim);
      font:11.5px/1.5 var(--mono);font-variant-numeric:tabular-nums}

 /* --- bottom tabs: thumb reach, 56px targets ------------------------- */
 /* ---- review: one card per captured event ---- */
 #events{display:flex;flex-direction:column;gap:12px;padding:12px 13px;
         max-height:min(70vh,720px);overflow-y:auto}
 /* flex:none matters: #events is a flex container with a max-height, so
    without it every card shrinks to fit and the crop strip is clipped away. */
 .ev{border:1px solid var(--rule);border-radius:9px;background:var(--sunk);
     overflow:hidden;flex:none}
 .ev .hd{display:flex;align-items:center;gap:8px;padding:9px 11px;
         border-bottom:1px solid var(--rule);flex-wrap:wrap}
 .ev .when{font:600 13px/1 var(--mono);color:var(--ink);
           font-variant-numeric:tabular-nums}
 .ev .sp{flex:1}
 .ev .pair{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--rule)}
 .ev .pair figure{margin:0;background:#000;position:relative}
 .ev .pair img{width:100%;aspect-ratio:4/3;object-fit:cover;display:block}
 .ev .pair figcaption{position:absolute;left:6px;top:6px;
      font:600 9.5px/1 var(--mono);letter-spacing:.1em;text-transform:uppercase;
      color:#fff;background:rgba(0,0,0,.62);padding:4px 6px;border-radius:4px}
 .ev .strip{display:flex;gap:6px;padding:9px 11px;overflow-x:auto}
 .ev .strip img{width:62px;height:62px;flex:none;object-fit:cover;
                border-radius:5px;border:1px solid var(--rule);background:#000}
 .ev .mt{padding:0 11px 6px;font:11.5px/1.5 var(--mono);color:var(--soft);
         font-variant-numeric:tabular-nums}
 .ev .pin{padding-bottom:11px;color:var(--ok);font-weight:500}
 .ev .pin.nofix{color:var(--dim);font-weight:400}
 #events .none{color:var(--dim);font:12px/1.6 var(--mono);padding:6px 0}

 nav{position:fixed;left:0;right:0;bottom:0;z-index:20;display:grid;
     grid-template-columns:repeat(5,1fr);
     background:rgba(10,12,16,.94);backdrop-filter:blur(12px);
     border-top:1px solid var(--rule);
     padding-bottom:env(safe-area-inset-bottom)}
 .tab{appearance:none;background:none;border:0;color:var(--dim);cursor:pointer;
      min-height:56px;padding:8px 4px;font:600 10.5px/1.3 var(--mono);
      letter-spacing:.08em;text-transform:uppercase;
      display:flex;flex-direction:column;align-items:center;justify-content:center;
      gap:4px;border-top:2px solid transparent;margin-top:-1px}
 .tab i{font-style:normal;font-size:16px;line-height:1}
 .tab[aria-selected=true]{color:var(--ink);border-top-color:var(--heat)}
 .tab:active{background:#171c24}

 button.act{background:var(--sink);border:1px solid var(--rule);color:var(--ink);
            border-radius:8px;padding:0 16px;min-height:44px;min-width:44px;
            font:600 12px/1 var(--mono);letter-spacing:.06em;cursor:pointer}
 button.act:active{background:#1b212a}
 button.act:disabled{color:var(--dim)}
 :focus-visible{outline:2px solid var(--heat);outline-offset:2px}

 #toast{position:fixed;left:50%;transform:translateX(-50%);
        bottom:calc(84px + env(safe-area-inset-bottom));z-index:30;
        background:#1b212a;border:1px solid var(--rule);border-radius:20px;
        padding:9px 16px;font:12px/1 var(--mono);color:var(--ink);
        opacity:0;transition:opacity .2s;pointer-events:none}
 #toast.show{opacity:1}

 /* --- wide screens: show the whole pipeline at once ------------------- */
 @media (min-width:760px){
   body{padding-bottom:0}
   nav{display:none}
   .grid{grid-template-columns:repeat(2,1fr)}
   #verdict{margin:14px auto;max-width:1472px}
   #pos{margin:0 auto 14px;max-width:1472px}
   /* Review is a different kind of content from a live panel -- it is a log.
      Give it the full width under the pipeline rather than a quarter column. */
   #card-review{grid-column:1/-1}
   #events{flex-direction:row;flex-wrap:wrap;max-height:none;overflow:visible}
   .ev{width:340px}
 }
 @media (min-width:1240px){ .grid{grid-template-columns:repeat(4,1fr)} }
 @media (prefers-reduced-motion:reduce){ *{transition:none!important} }
</style>

<header>
  <span class=brand><span class=dot id=dot></span>SaResQ</span>
  <span class=where>payload &middot; live</span>
  <span class=sp></span>
  <button class=act id=save>SAVE</button>
</header>

<div id=verdict><b id=v-main>&mdash;</b><span id=v-sub>connecting to payload&hellip;</span></div>

<div id=pos>
  <span class=dot id=gdot></span>
  <b id=g-coord>&mdash;</b>
  <span id=g-state>waiting for GPS&hellip;</span>
</div>

<div id=scene>
  <span class=dot id=sdot></span>
  <b id=s-main>&mdash;</b>
  <span id=s-sub>scene classifier starting&hellip;</span>
</div>

<div class=grid>
  <div class=card id=card-thermal>
    <h2>Thermal 32&times;24 <span id=p-gate class=pill>gate</span></h2>
    <canvas class=view id=c-thermal width=512 height=384></canvas>
    <div class=foot id=f-thermal>&hellip;</div>
  </div>
  <div class=card id=card-rgb>
    <h2>Camera <span id=p-rgb class=pill>rgb</span></h2>
    <canvas class=view id=c-rgb width=640 height=480></canvas>
    <div class=foot id=f-rgb>&hellip;</div>
  </div>
  <div class=card id=card-detect>
    <h2>Detector <span id=p-det class=pill>model</span></h2>
    <canvas class=view id=c-detect width=640 height=480></canvas>
    <div class=foot id=f-det>&hellip;</div>
  </div>
  <div class=card id=card-crops>
    <h2>Gate crops</h2>
    <div id=crops></div>
    <div class=foot>Regions the thermal gate handed to the detector.</div>
  </div>
  <div class=card id=card-review>
    <h2>Review <span id=p-rev class=pill>0 events</span></h2>
    <div id=events></div>
    <div class=foot>Captured automatically on every confirmed detection.</div>
  </div>
</div>

<div id=sys>&hellip;</div>
<div id=toast></div>

<nav>
  <button class=tab data-v=detect  aria-selected=true><i>&#9673;</i>Detect</button>
  <button class=tab data-v=thermal><i>&#9788;</i>Thermal</button>
  <button class=tab data-v=rgb><i>&#9635;</i>Camera</button>
  <button class=tab data-v=crops><i>&#9638;</i>Crops</button>
  <button class=tab data-v=review><i>&#9634;</i>Review</button>
</nav>

<script>
const VIEWS = ['detect','thermal','rgb','crops','review'];
const wide  = matchMedia('(min-width:760px)');
let active  = 'detect';
const f = (x, n) => (x ?? 0).toFixed(n);

function sync(){
  const all = wide.matches;
  for (const v of VIEWS)
    document.getElementById('card-' + v).hidden = !(all || v === active);
  for (const b of document.querySelectorAll('.tab'))
    b.setAttribute('aria-selected', String(!all && b.dataset.v === active));
}

/* Video is fetched frame by frame and drawn to a canvas, NOT streamed into an
   <img> as MJPEG. Chromium repaints a multipart/x-mixed-replace image while it
   is still arriving, so every panel showed the top of the incoming frame above
   a band of flat grey -- it looked exactly like a broken camera. Drawing only
   from an Image that has finished decoding makes a partial frame impossible.
   Fetching one frame at a time also gives natural backpressure (a slow phone
   simply asks for fewer) and holds no connection open, which matters against
   Safari's per-host cap. The .mjpg endpoints still exist for IP-camera apps. */
function stream(view, path, gap){
  const card = document.getElementById('card-' + view);
  const cvs  = document.getElementById('c-' + view);
  const ctx  = cvs.getContext('2d');
  const tick = () => {
    if (card.hidden || document.hidden){ setTimeout(tick, 500); return; }
    const im = new Image();
    im.onload = () => {
      if (cvs.width !== im.width){ cvs.width = im.width; cvs.height = im.height; }
      ctx.drawImage(im, 0, 0);
      setTimeout(tick, gap);
    };
    im.onerror = () => setTimeout(tick, 1000);
    im.src = path + '?t=' + Date.now();
  };
  tick();
}
for (const b of document.querySelectorAll('.tab'))
  b.addEventListener('click', () => { active = b.dataset.v; sync(); });
wide.addEventListener('change', sync);
sync();

/* The thermal sensor only produces ~4 full frames a second, so asking faster
   just burns CPU re-encoding the same array. */
stream('detect',  '/detect.jpg',  80);
stream('rgb',     '/rgb.jpg',     80);
stream('thermal', '/thermal.jpg', 160);

let toastT;
const toast = m => {
  const el = document.getElementById('toast');
  el.textContent = m; el.classList.add('show');
  clearTimeout(toastT); toastT = setTimeout(() => el.classList.remove('show'), 1800);
};
document.getElementById('save').addEventListener('click', async e => {
  e.target.disabled = true;
  try { await fetch('/save'); toast('frame set saved'); }
  catch { toast('save failed'); }
  e.target.disabled = false;
});

/* The review log. Event images are immutable and served with a long
   max-age, so the list is only rebuilt when the newest id actually changes --
   otherwise every poll would tear down and refetch every thumbnail. */
let lastEvent = -1, evT = null;
async function events(){
  clearTimeout(evT);
  const card = document.getElementById('card-review');
  try{
    if (!card.hidden && !document.hidden){
      const evs = await (await fetch('/events')).json();
      document.getElementById('p-rev').textContent =
        evs.length + ' event' + (evs.length === 1 ? '' : 's');
      const newest = evs.length ? evs[0].id : 0;
      if (newest !== lastEvent){
        lastEvent = newest;
        document.getElementById('events').innerHTML = evs.length ? evs.map(e => {
          /* ?s= is the run token. Event ids restart at 1 each time the service
             does, and these images are cached hard, so without it the browser
             re-serves a previous run's picture for this run's event. */
          const q = `?s=${e.s || 0}`;
          const crops = Array.from({length: e.ncrops}, (_, i) =>
            `<img src="/event/${e.id}/crop${i}${q}" alt="crop ${i + 1}">`).join('');
          return `<article class=ev>
            <div class=hd>
              <span class=when>${e.t}</span>
              <span class="pill ${e.n ? 'hot' : ''}">${
                e.n ? e.n + ' person' + (e.n > 1 ? 's' : '') : 'saved'}</span>
              <span class=sp></span>
              <span class=pill>+${f(e.z,1)}&sigma;</span>
            </div>
            <div class=pair>
              <figure><img src="/event/${e.id}/thermal${q}" alt="thermal" loading=lazy>
                <figcaption>thermal</figcaption></figure>
              <figure><img src="/event/${e.id}/detect${q}" alt="detections" loading=lazy>
                <figcaption>detector</figcaption></figure>
            </div>
            ${crops ? `<div class=strip>${crops}</div>` : ''}
            <div class=mt>peak ${f(e.tmax,1)}&deg;C &middot; ${
              e.n ? `confidence ${f(e.conf,2)}` : 'manual capture'} &middot; ${e.why}</div>
            <div class="mt pin${e.lat == null ? ' nofix' : ''}">${
              e.lat == null
                ? `&#9678; no GPS fix at capture &middot; ${e.sats || 0} satellites`
                : `&#9679; ${e.lat.toFixed(6)}, ${e.lon.toFixed(6)}`}</div>
          </article>`;
        }).join('') : '<span class=none>No events yet. The payload captures one '+
          'automatically each time the detector confirms a person &mdash; or press '+
          'SAVE to capture the current moment.</span>';
      }
    }
  }catch(e){}
  evT = setTimeout(events, card.hidden ? 3000 : 1500);
}
events();

/* One timer handle, so waking the tab reschedules the loop instead of
   starting a second one alongside it. */
let pollT = null;
async function poll(){
  clearTimeout(pollT);
  let next = 800;
  try{
    const s = await (await fetch('/stats')).json();
    const t = s.thermal || {}, d = s.detect || {}, g = s.gate || {};
    document.getElementById('dot').className = 'dot live';

    /* Verdict: a detection outranks a gate hit, because the gate firing on a
       radiator is not a find. Both are shown so the two stages stay legible
       as two stages. */
    const n = d.n || 0, ver = document.getElementById('verdict');
    ver.className = n ? 'found' : (g.fired ? 'heat' : '');
    document.getElementById('v-main').textContent =
      n ? `${n} PERSON${n > 1 ? 'S' : ''}` : (g.fired ? 'HEAT SOURCE' : 'CLEAR');
    document.getElementById('v-sub').textContent =
      n ? `detector confirmed · ${f(d.ms,0)} ms`
        : (g.fired ? `gate fired at +${f(g.z_max,1)}σ · no person confirmed`
                   : `peak +${f(g.z_max,1)}σ of ${f(g.z_t,1)} needed`);

    /* Scene class. This answers a different question from the verdict: the
       verdict says whether anyone is there, this says what kind of place it
       is. They are deliberately separate -- a flooded street with nobody in
       it is still worth routing a boat to. */
    const hz = s.hazard || {}, sc = document.getElementById('scene');
    const SCENE = {collapsed_building:'COLLAPSED BUILDING', fire:'FIRE',
                   flooded_areas:'FLOODING', traffic_incident:'TRAFFIC INCIDENT',
                   normal:'NORMAL SCENE'};
    document.getElementById('sdot').className =
      'dot' + (hz.ok ? (hz.top === 'normal' ? ' live' : ' hot') : '');
    if (hz.ok) {
      sc.className = hz.top === 'normal' ? 'calm' : 'alert';
      document.getElementById('s-main').textContent = SCENE[hz.top] || hz.top;
      document.getElementById('s-sub').textContent =
        `${f(hz.p * 100, 0)}% \u00b7 ${f(hz.ms, 0)} ms \u00b7 MobileNetV2/AIDER, 95.2% val acc`;
    } else {
      sc.className = 'off';
      document.getElementById('s-main').textContent = 'SCENE \u2014';
      document.getElementById('s-sub').textContent = hz.why || 'starting\u2026';
    }

    document.getElementById('f-thermal').innerHTML =
      `<b>${f(t.min,1)}&ndash;${f(t.max,1)}&deg;C</b> &middot; spread ${f(t.spread,1)}&deg;C`+
      ` &middot; peak <b>+${f(g.z_max,1)}&sigma;</b> of ${f(g.z_t,1)}`+
      ` &middot; ${f(t.fps,1)} fps`;
    const gp = document.getElementById('p-gate');
    gp.textContent = g.fired ? `fired · ${g.n_blobs}` : 'quiet';
    gp.className = 'pill' + (g.fired ? ' on' : '');

    const fr = document.getElementById('f-rgb'), age = s.rgb_age ?? 0;
    fr.innerHTML = `${s.rgb_w||0}&times;${s.rgb_h||0} &middot; <b>${f(age,1)}s</b> old`+
      (s.rotated ? ' &middot; rotated 180&deg;' : '');
    fr.className = 'foot' + (age > 3 ? ' stale' : '');
    document.getElementById('p-rgb').textContent = age > 3 ? 'stalled' : 'streaming';

    document.getElementById('f-det').innerHTML =
      `<b>${n}</b> detection(s) &middot; ${f(d.ms,0)} ms &middot; conf &ge; ${f(d.conf,2)}`+
      `<br>${d.model || ''}`;
    const dp = document.getElementById('p-det');
    dp.textContent = n ? `${n} found` : 'clear';
    dp.className = 'pill' + (n ? ' hot' : '');

    const crops = s.crops || [];
    document.getElementById('crops').innerHTML = crops.length
      ? crops.map(c => `<figure><img src="/crop/${c.i}?t=${s.seq}" alt="crop">`+
          `<figcaption>+${f(c.z,1)}&sigma;<br>${f(c.T,1)}&deg;C</figcaption></figure>`).join('')
      : '<span class=none>no gate hits &mdash; nothing warm enough in frame</span>';

    /* Position. A receiver that is talking but has no fix is NOT a fault --
       it is what every GPS does indoors, and saying so is better than a blank
       panel that looks broken. */
    const p = s.gps || {}, box = document.getElementById('pos');
    const hasFix = p.quality > 0 && p.lat != null;
    box.className = hasFix ? 'fix' : (p.live ? 'nofix' : '');
    document.getElementById('gdot').className = 'dot' + (hasFix ? ' live' : '');
    document.getElementById('g-coord').textContent = hasFix
      ? `${p.lat.toFixed(6)}, ${p.lon.toFixed(6)}`
      : (p.live ? 'no fix yet' : 'no GPS data');
    document.getElementById('g-state').innerHTML = hasFix
      ? `${p.state} · ${p.sats} satellites` +
        (p.hdop != null ? ` · HDOP ${f(p.hdop,1)}` : '') +
        (p.alt != null ? ` · ${f(p.alt,0)} m` : '')
      : (p.live
          ? `receiver alive · ${p.n} sentences · ${p.sats || 0} satellites ` +
            `&mdash; a GPS needs sky view, so indoors there is nothing to report`
          : 'no sentences on the serial port');

    document.getElementById('sys').textContent =
      `cpu ${s.temp || '?'}  ·  ${s.up || ''}`;
  }catch(e){
    document.getElementById('dot').className = 'dot';
    document.getElementById('v-sub').textContent = 'lost the payload — retrying';
    next = 1500;
  }
  /* A backgrounded tab still costs the Pi a request every 800 ms for numbers
     nobody can see. */
  pollT = setTimeout(poll, document.hidden ? 4000 : next);
}
poll();
document.addEventListener('visibilitychange', () => { if (!document.hidden) poll(); });
</script>
"""


class Thermal(threading.Thread):
    daemon = True

    def __init__(self, hz: int = 8, mode: str = "h"):
        super().__init__()
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.fps = 0.0
        self.hz = hz
        self.mode = mode

    def run(self) -> None:
        import adafruit_mlx90640
        import board
        import busio
        i2c = busio.I2C(board.SCL, board.SDA)
        mlx = adafruit_mlx90640.MLX90640(i2c)
        mlx.refresh_rate = {2: adafruit_mlx90640.RefreshRate.REFRESH_2_HZ,
                            4: adafruit_mlx90640.RefreshRate.REFRESH_4_HZ,
                            8: adafruit_mlx90640.RefreshRate.REFRESH_8_HZ}[self.hz]
        buf = [0.0] * 768
        last = time.time()
        while True:
            try:
                mlx.getFrame(buf)
            except (ValueError, RuntimeError, OSError):
                time.sleep(0.05)
                continue
            a = np.array(buf, dtype=np.float32).reshape(24, 32)
            if not np.isfinite(a).all():
                continue
            now = time.time()
            dt = now - last
            last = now
            with self.lock:
                self.frame = orient(a, self.mode)
                if dt > 0:
                    self.fps = 0.8 * self.fps + 0.2 / dt

    def read(self):
        with self.lock:
            return (None, 0.0) if self.frame is None else (self.frame.copy(), self.fps)


def _nmea_ok(line: str) -> bool:
    """True if the sentence carries a correct XOR checksum.

    Worth doing rather than trusting the line: a UART at the wrong baud, or a
    half-read line, produces text that still looks like NMEA. The checksum is
    the only thing that says the bytes arrived intact.
    """
    if not line.startswith("$") or "*" not in line:
        return False
    body, _, cks = line[1:].partition("*")
    x = 0
    for b in body.encode():
        x ^= b
    try:
        return int(cks[:2], 16) == x
    except ValueError:
        return False


def _nmea_deg(raw: str, hemi: str) -> float | None:
    """NMEA ddmm.mmmm + hemisphere -> signed decimal degrees.

    The format is NOT decimal degrees, and treating it as such is the classic
    GPS bug: 2234.5678 is 22 degrees 34.5678 minutes, i.e. 22.5761, not 2234.57.
    """
    if not raw or "." not in raw:
        return None
    try:
        head, mins = raw.split(".")[0], float(raw[len(raw.split(".")[0]) - 2:])
        deg = float(head[:-2] or 0) + mins / 60.0
    except (ValueError, IndexError):
        return None
    return -deg if hemi in ("S", "W") else deg


class Gps(threading.Thread):
    """Keep the newest GPS fix, without ever blocking anything else.

    If the port is missing, busy or silent, this thread reports 'no fix'
    forever and every other panel carries on. A demo must not lose its thermal
    view because a satellite receiver is unhappy.

    Note it holds /dev/serial0 while running, so tools/sensor_selftest.py
    cannot read the GPS at the same time as the demo. Stop one to run the other.
    """

    daemon = True
    QUALITY = {0: "no fix", 1: "GPS fix", 2: "DGPS fix", 4: "RTK", 5: "RTK float"}

    def __init__(self, port: str, baud: int = 9600):
        super().__init__()
        self.port, self.baud = port, baud
        self.lock = threading.Lock()
        self.n = 0                      # valid sentences seen, ever
        self.stamp = 0.0                # when the last valid sentence arrived
        self.fix = {"quality": 0, "sats": 0, "lat": None, "lon": None,
                    "alt": None, "hdop": None, "in_view": 0}

    def run(self) -> None:
        import serial
        while True:
            try:
                with serial.Serial(self.port, self.baud, timeout=2) as ser:
                    while True:
                        raw = ser.readline().decode("ascii", "ignore").strip()
                        if not _nmea_ok(raw):
                            continue
                        self._consume(raw.split("*")[0].split(","))
            except Exception:
                # missing port, permissions, unplugged mid-run -- all the same
                # from here: wait, then try again.
                time.sleep(3)

    def _consume(self, f: list[str]) -> None:
        kind = f[0][3:] if len(f[0]) >= 6 else ""
        with self.lock:
            self.n += 1
            self.stamp = time.time()
            if kind == "GGA" and len(f) > 9:
                self.fix.update(
                    quality=int(f[6] or 0), sats=int(f[7] or 0),
                    lat=_nmea_deg(f[2], f[3]), lon=_nmea_deg(f[4], f[5]),
                    hdop=float(f[8]) if f[8] else None,
                    alt=float(f[9]) if f[9] else None)
            elif kind == "GSV" and len(f) > 3:
                self.fix["in_view"] = int(f[3] or 0)

    def read(self) -> dict:
        with self.lock:
            d = dict(self.fix)
            d["n"] = self.n
            d["age"] = (time.time() - self.stamp) if self.stamp else None
            d["state"] = self.QUALITY.get(d["quality"], "unknown")
            d["live"] = bool(self.stamp and time.time() - self.stamp < 5)
            return d


def jpeg_frame(buf: bytes) -> tuple[bytes | None, bytes]:
    """Pull one COMPLETE JPEG out of an MJPEG byte stream.

    Scanning naively for FFD8 ... FFD9 is wrong, and wrong in a way that looks
    like a hardware fault: the bytes FF D9 occur happily inside a quantization
    or Huffman table, and cutting the frame there yields a JPEG that decodes
    to a few rows of picture above a flat grey field.

    Marker segments carry their own length, so walk them properly. Only once
    the entropy-coded scan has begun (SOS) is it safe to hunt for EOI, because
    there every literal 0xFF is stuffed as FF 00 and a bare FF D9 really is
    the end of the frame.

    Returns (frame, rest); frame is None when more bytes are needed.
    """
    i = buf.find(b"\xff\xd8")
    if i < 0:
        return None, buf[-1:]          # a trailing FF may begin the next SOI
    n = len(buf)
    p = i + 2
    while p + 1 < n:
        if buf[p] != 0xFF:             # desynchronised: resync on the next SOI
            j = buf.find(b"\xff\xd8", i + 2)
            return None, (buf[j:] if j > 0 else b"")
        m = buf[p + 1]
        if m == 0xFF:                              # fill byte
            p += 1
            continue
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:  # standalone markers
            p += 2
            continue
        if p + 3 >= n:
            return None, buf
        seg = int.from_bytes(buf[p + 2:p + 4], "big")
        if m == 0xDA:                              # start of scan
            q = p + 2 + seg
            while True:
                k = buf.find(b"\xff", q)
                if k < 0 or k + 1 >= n:
                    return None, buf
                if buf[k + 1] == 0xD9:
                    return buf[i:k + 2], buf[k + 2:]
                q = k + 2                          # FF00 stuffing, or a RSTn
        p += 2 + seg
    return None, buf


class Camera(threading.Thread):
    """Continuous MJPEG video off the camera, not a sequence of stills.

    rpicam-still costs ~900 ms per frame because it reconfigures the sensor
    every time -- fine for a single capture, a slideshow for a demo. rpicam-vid
    holds the pipeline open and streams, so this runs at the requested frame
    rate instead.

    Frames are recovered from the stream by scanning for JPEG start- and
    end-of-image markers; MJPEG has no other container, which is exactly why
    it is easy to parse and why every IP-camera app can read it.

    Falls back to rpicam-still if rpicam-vid is unavailable, so the demo
    degrades to slow rather than to nothing.
    """

    daemon = True

    def __init__(self, w: int, h: int, rotate: int, fps: int = 15):
        super().__init__()
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.stamp = 0.0
        self.fps_measured = 0.0
        self.w, self.h, self.rotate, self.fps = w, h, rotate, fps
        self.tmp = pathlib.Path("/tmp/_live_rgb.jpg")

    # -- video path -------------------------------------------------------
    def _vid_cmd(self) -> list[str]:
        cmd = ["rpicam-vid", "-n", "-t", "0", "--codec", "mjpeg",
               "--width", str(self.w), "--height", str(self.h),
               "--framerate", str(self.fps), "-o", "-"]
        if self.rotate:
            cmd += ["--rotation", str(self.rotate)]
        return cmd

    def _stream(self) -> bool:
        """True if the stream ran at all; False to fall back to stills."""
        try:
            proc = subprocess.Popen(self._vid_cmd(), stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, bufsize=0)
        except FileNotFoundError:
            return False
        buf = b""
        last = time.time()
        got_any = False
        try:
            while True:
                chunk = proc.stdout.read(65536)
                if not chunk:
                    break
                buf += chunk
                while True:
                    jpg, buf = jpeg_frame(buf)
                    if jpg is None:
                        break
                    img = cv2.imdecode(np.frombuffer(jpg, np.uint8),
                                       cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    now = time.time()
                    dt = now - last
                    last = now
                    got_any = True
                    with self.lock:
                        self.frame, self.stamp = img, now
                        if dt > 0:
                            self.fps_measured = (0.85 * self.fps_measured
                                                 + 0.15 / dt)
                # a runaway buffer means we are not finding markers; reset
                if len(buf) > 4_000_000:
                    buf = b""
        finally:
            proc.kill()
        return got_any

    # -- stills fallback --------------------------------------------------
    def _still(self) -> None:
        cmd = ["rpicam-still", "-n", "-o", str(self.tmp), "--width", str(self.w),
               "--height", str(self.h), "-t", "600", "--immediate"]
        if self.rotate:
            cmd += ["--rotation", str(self.rotate)]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=30)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            time.sleep(2)
            return
        if r.returncode == 0:
            img = cv2.imread(str(self.tmp))
            if img is not None:
                with self.lock:
                    self.frame, self.stamp = img, time.time()
                    self.fps_measured = 1.0

    def run(self) -> None:
        use_video = True
        while True:
            if use_video:
                if not self._stream():
                    print("rpicam-vid unavailable; falling back to stills",
                          flush=True)
                    use_video = False
                else:
                    # stream ended unexpectedly -- the camera was grabbed by
                    # something else, or it stalled. Retry rather than give up.
                    time.sleep(1)
            else:
                self._still()
                time.sleep(0.1)

    def read(self):
        with self.lock:
            return (None, 0.0) if self.frame is None else (self.frame.copy(),
                                                           self.stamp)


def gate_stats(thermal: np.ndarray, cfg: dict) -> dict:
    g = cfg.get("gate", {})
    z_t = float(g.get("z_t", 2.5))
    sigma = max(float(thermal.std()), float(g.get("sigma_min_k", 0.15)))
    z = (thermal - thermal.mean()) / sigma
    mask = (z >= z_t).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) > int(g.get("max_blob_px", 30)):
            continue
        blobs.append({"row": float(cents[i][1]), "col": float(cents[i][0]),
                      "z": float(z[labels == i].max()),
                      "T": float(thermal[labels == i].max()),
                      "px": int(stats[i, cv2.CC_STAT_AREA])})
    blobs.sort(key=lambda b: -b["z"])
    return {"z_max": float(z.max()), "z_t": z_t, "fired": bool(blobs),
            "n_blobs": len(blobs), "blobs": blobs[:6]}


def colorize(thermal: np.ndarray, size=(512, 384), span_min: float = 6.0
             ) -> np.ndarray:
    """Render the 32x24 array for the screen.

    Two things here are deliberately display-only -- the gate and the detector
    are always handed the raw array, so nothing below changes a number the
    pipeline acts on or the page reports.

    * A 3x3 median. At 32x24 a single noisy or dead pixel is a large bright
      square on screen. A person covers many pixels, so none of the signal
      this payload exists to find is lost.
    * A floor under the colour span. Stretching min..max over a scene that
      only varies by 3 C maps the MLX90640's own noise floor across the whole
      palette, which is why an empty room rendered as a chessboard. Below
      span_min the scale is widened about the scene midpoint instead, so a
      uniform room looks uniform and a person still lights up.
    """
    disp = cv2.medianBlur(thermal, 3)
    lo, hi = (float(v) for v in np.percentile(disp, (1, 99)))
    if hi - lo < span_min:
        # Extend the TOP of the scale, not both ends. A uniform room then sits
        # at the dark floor of the palette and anything body-warm climbs out of
        # it; centring the span instead washes an empty room out to mid-orange
        # and leaves a person nowhere brighter to go.
        hi = lo + span_min
    norm = np.clip((disp - lo) / max(hi - lo, 1e-6), 0, 1)
    img = cv2.applyColorMap(
        cv2.resize((norm * 255).astype(np.uint8), size,
                   interpolation=cv2.INTER_CUBIC), cv2.COLORMAP_INFERNO)
    hy, hx = np.unravel_index(int(disp.argmax()), disp.shape)
    sx, sy = size[0] / 32, size[1] / 24
    cv2.circle(img, (int((hx + .5) * sx), int((hy + .5) * sy)), 12,
               (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"{float(disp.max()):.1f}C", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


CALIB = REPO / "saresq" / "calib" / "thermal_to_rgb.json"


@functools.lru_cache(maxsize=1)
def load_affine():
    """The thermal->RGB affine, or None if this payload has not been calibrated.

    Cached: it is a 2x3 matrix read from disk, and the detect loop would
    otherwise stat the file thirty times a second for a number that changes
    when someone runs the calibration, not while flying.
    """
    try:
        m = json.loads(CALIB.read_text())
        return (np.asarray(m["matrix"], dtype=np.float32),
                float(m.get("residual_thermal_px", 0.0)))
    except (OSError, ValueError, KeyError):
        return None


def draw_thermal_contours(vis, thermal, affine, levels=(2.0, 3.0, 4.0)):
    """Outline the heat, in the RGB frame, where the calibration says it is.

    CONTOURS RATHER THAN A BLEND. An alpha-blended heat map over the detector
    view hides the thing the detector is drawing boxes on, and at 32x24 a blend
    is 99.96% interpolation painted over real pixels. Outlines add a layer
    without taking one away, and they make the registration itself checkable at
    a glance: if the line does not sit on the warm object, the calibration is
    wrong and you can see it immediately.

    The z-map is warped at QUARTER resolution and the contour points scaled up.
    Warping 768 thermal pixels into 2 million RGB ones costs real milliseconds
    on a Pi 4 for detail that does not exist in the source; a quarter-size warp
    is ~8x cheaper and, from a 32x24 array, loses nothing.
    """
    if affine is None or thermal is None:
        return 0
    H, W = vis.shape[:2]
    q = 4
    m = affine.copy()
    m[0, :] /= q
    m[1, :] /= q                      # same mapping, quarter-scale destination

    med = float(np.median(thermal))
    mad = float(np.median(np.abs(thermal - med)))
    z = (thermal - med) / max(1.4826 * mad, 0.15)

    warped = cv2.warpAffine(z.astype(np.float32), m, (W // q, H // q),
                            flags=cv2.INTER_LINEAR, borderValue=0.0)
    drawn = 0
    for lv, col, th in zip(levels, ((120, 90, 220), (70, 170, 255), (90, 240, 255)), (1, 2, 2)):
        mask = (warped >= lv).astype(np.uint8)
        if not mask.any():
            continue
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < 2:
                continue
            cv2.polylines(vis, [c * q], True, col, th, cv2.LINE_AA)
            drawn += 1
    return drawn


class App:
    """Shared state: newest of everything, plus the detector thread."""

    def __init__(self, cfg: dict, args):
        self.cfg = cfg
        self.args = args
        self.thermal = Thermal(hz=args.hz, mode=args.tflip)
        self.camera = Camera(args.width, args.height, args.rotate, args.fps)
        self.gps = Gps(args.gps_port, args.gps_baud)
        self.lock = threading.Lock()
        self.det_img: np.ndarray | None = None
        self.det_info = {"n": 0, "ms": 0.0, "conf": args.conf,
                         "model": pathlib.Path(args.rgb_model).name}
        self.crops: list[np.ndarray] = []
        self.crop_meta: list[dict] = []
        self.gate = {"z_max": 0.0, "z_t": 2.5, "fired": False, "n_blobs": 0,
                     "blobs": []}
        # Scene class, filled by hazard_loop. Starts "not ok" with a reason, so
        # the page says WHY there is no scene rather than showing a blank strip.
        self.hazard = {"ok": False, "why": "starting\u2026"}
        self.seq = 0
        self.save_next = False
        # Captured events live in RAM, not on the card. A ring buffer of ~24
        # costs a couple of MB of the 730 MB free, writes nothing to a microSD
        # that has already been reflashed once this week, and needs no cleanup
        # logic that could fail mid-demo. The Save button still writes evidence
        # to results/demo/ when someone deliberately wants a copy.
        self.events: collections.deque = collections.deque(maxlen=24)
        self.event_seq = 0
        self.last_auto = 0.0
        # Event ids restart at 1 on every service restart, and event images are
        # served with a long max-age because a captured frame never changes.
        # Those two facts together served a PREVIOUS run's picture for this
        # run's event #1 -- stale crops in the Review tab after any restart.
        # This token makes each run's URLs distinct, so the cache can stay.
        self.session = f"{int(time.time()):x}"

    def start(self):
        self.thermal.start()
        self.camera.start()
        self.gps.start()
        threading.Thread(target=self.detect_loop, daemon=True).start()
        threading.Thread(target=self.hazard_loop, daemon=True).start()

    def detect_loop(self):
        from saresq.detect.tflite_detector import CropDetector
        det = CropDetector(self.args.rgb_model, conf=self.args.conf,
                           iou=float(self.cfg.get("detector", {}).get("nms_iou", 0.5)),
                           num_threads=int(self.cfg.get("detector", {})
                                           .get("threads", 4)))
        print(f"detector: {pathlib.Path(self.args.rgb_model).name} "
              f"({det.imgsz}px, boxes {det.box_units})", flush=True)
        crop_px = int(self.cfg.get("detector", {}).get("crop_px", 160))
        while True:
            rgb, stamp = self.camera.read()
            th, _ = self.thermal.read()
            if rgb is None:
                time.sleep(0.3)
                continue
            g = gate_stats(th, self.cfg) if th is not None else self.gate

            t0 = time.time()
            dets = det.detect(rgb)
            ms = (time.time() - t0) * 1000
            # A COCO model reports all 80 classes; on a desk that means chairs
            # and laptops. Class 0 is person, which is the only one this
            # payload is looking for.
            if self.args.classes:
                dets = [d for d in dets if int(d.cls) in self.args.classes]

            vis = rgb.copy()
            # Heat outlines go on FIRST, so detection boxes and scores stay the
            # topmost thing on the frame.
            cal = None if self.args.no_contours else load_affine()
            n_cont = draw_thermal_contours(vis, th, cal[0] if cal else None)
            for d in dets:
                x0, y0, x1, y1 = (int(v) for v in d.xyxy)
                s = float(d.score)
                cv2.rectangle(vis, (x0, y0), (x1, y1), (80, 235, 130), 3)
                cv2.putText(vis, f"{s:.2f}", (x0, max(y0 - 8, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 235, 130), 2,
                            cv2.LINE_AA)

            # crop the RGB where the gate fired. Without a thermal->RGB
            # homography this is a linear FOV map, which is close enough to
            # show the idea but is NOT the calibrated mapping in
            # saresq/calib/thermal_to_rgb.json.
            crops, meta = [], []
            H, W = rgb.shape[:2]
            for i, b in enumerate(g.get("blobs", [])):
                cx = int((b["col"] + 0.5) / 32 * W)
                cy = int((b["row"] + 0.5) / 24 * H)
                x0 = max(0, min(cx - crop_px // 2, W - crop_px))
                y0 = max(0, min(cy - crop_px // 2, H - crop_px))
                crops.append(rgb[y0:y0 + crop_px, x0:x0 + crop_px].copy())
                meta.append({"i": i, "z": b["z"], "T": b["T"]})
                cv2.rectangle(vis, (x0, y0), (x0 + crop_px, y0 + crop_px),
                              (70, 170, 255), 2)
                cv2.putText(vis, f"+{b['z']:.1f}s", (x0, y0 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 170, 255), 2,
                            cv2.LINE_AA)

            with self.lock:
                self.det_img = vis
                self.det_info = {"n": len(dets), "ms": ms, "conf": self.args.conf,
                                 "model": pathlib.Path(self.args.rgb_model).name,
                                 "contours": n_cont,
                                 "calib": (round(cal[1], 3) if cal else None)}
                self.crops, self.crop_meta, self.gate = crops, meta, g
                self.seq += 1
                manual, self.save_next = self.save_next, False
                if manual:
                    self._save(rgb, vis, th)

            # Capture evidence whenever the detector actually confirms someone,
            # rate-limited so a person standing still does not fill the buffer
            # with 40 near-identical frames of themselves. Encoding happens
            # outside the lock: it costs ~40 ms and no viewer should wait on it.
            now = time.time()
            if manual or (dets and now - self.last_auto >= self.args.event_gap):
                self.last_auto = now
                self._capture(rgb, vis, th, g, dets, crops,
                              "saved" if manual else "detection")
            time.sleep(0.05)

    def hazard_loop(self):
        """Classify the WHOLE scene -- flood / fire / collapse -- at ~1 Hz.

        Its own thread, and that is not incidental. The classifier costs 33 ms
        on a 640x480 frame (results/pi_benchmark.json) against a 250 ms frame
        budget of which the detector already spends 169 ms. Running it inline
        would put the pipeline at 202 ms and leave nothing for a slow frame;
        at 1 Hz in a separate thread it is a ~3% duty cycle and the detector
        never waits on it.

        1 Hz rather than per-frame because AIDER is a dataset of whole aerial
        scenes and scene context does not change between consecutive frames of
        a survey pass -- the rate lives in configs/pipeline.yaml for that
        reason, not as a constant here.

        Every failure path here is caught and reported into self.hazard rather
        than raised. This is the piece most likely to be missing on a fresh
        card (a 2.7 MB model file that nothing else needs), and a demo that
        dies at boot because a nice-to-have banner could not load would be a
        far worse outcome than a strip that says "model not found".
        """
        hz_cfg = self.cfg.get("hazard", {})

        def fail(why: str):
            with self.lock:
                self.hazard = {"ok": False, "why": why}
            print(f"hazard: {why}", flush=True)

        if self.args.no_hazard:
            return fail("disabled (--no-hazard)")

        path = pathlib.Path(self.args.hazard_model)
        if not path.is_absolute():
            path = REPO / path
        if not path.exists():
            return fail(f"model not found: {path.name}")

        try:
            # Imported here, not at module scope: the demo must still start on
            # a card where saresq/detect/hazard.py or LiteRT is unavailable.
            from saresq.detect.hazard import HazardClassifier
            # Deliberately NOT passing classes=cfg["hazard"]["classes"]. The
            # model carries no class names, so a wrong list would silently
            # relabel every prediction and still pass the arity check. The
            # ordering in saresq.detect.hazard.CLASSES is the one that matches
            # train_hazard.py's class_names=, which IS the output order.
            clf = HazardClassifier(str(path), num_threads=2)
        except Exception as e:                      # noqa: BLE001 - see docstring
            return fail(f"{type(e).__name__}: {e}")

        period = 1.0 / max(float(hz_cfg.get("rate_hz", 1.0)), 0.05)
        print(f"hazard: {path.name} at {1 / period:.1f} Hz", flush=True)

        while True:
            rgb, _ = self.camera.read()
            if rgb is None:
                time.sleep(0.5)
                continue
            t0 = time.time()
            try:
                probs = clf.predict(rgb)
            except Exception as e:                  # noqa: BLE001
                fail(f"predict failed: {type(e).__name__}: {e}")
                time.sleep(5.0)
                continue
            ms = (time.time() - t0) * 1000
            top = max(probs, key=probs.get)
            with self.lock:
                self.hazard = {"ok": True, "top": top, "p": float(probs[top]),
                               "ms": ms, "probs": {k: round(v, 4) for k, v in probs.items()}}
            time.sleep(period)

    def _capture(self, rgb, vis, th, g, dets, crops, why: str) -> None:
        """Freeze one moment as JPEGs so the Review tab can show it later."""
        def enc(img, q=80):
            ok, b = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
            return b.tobytes() if ok else b""

        # Stamp the position the payload was at when this was seen. NOT the
        # projected ground point of the detection -- that needs height above
        # ground, which a bench demo does not have and must not invent.
        fx = self.gps.read()
        self.event_seq += 1
        ev = {
            "id": self.event_seq,
            "t": time.strftime("%H:%M:%S"),
            "why": why,
            "n": len(dets),
            "conf": max((float(d.score) for d in dets), default=0.0),
            "z": float(g.get("z_max", 0.0)),
            "tmax": float(th.max()) if th is not None else 0.0,
            "lat": fx["lat"] if fx["quality"] else None,
            "lon": fx["lon"] if fx["quality"] else None,
            "sats": fx["sats"],
            # The scene the payload was looking at when this was frozen. An
            # event reviewed an hour later is far more readable as "person,
            # flooded area" than "person"; and it is the field an operator
            # triages by when the queue is long.
            "scene": (self.hazard.get("top") if self.hazard.get("ok") else None),
            "scene_p": (round(self.hazard.get("p", 0.0), 3)
                        if self.hazard.get("ok") else None),
            "s": self.session,
            "img": {
                "thermal": enc(colorize(th)) if th is not None else b"",
                "rgb": enc(cv2.resize(rgb, (640, 480))),
                "detect": enc(cv2.resize(vis, (640, 480))),
            },
        }
        for i, c in enumerate(crops[:6]):
            ev["img"][f"crop{i}"] = enc(c, 88)
        ev["ncrops"] = min(len(crops), 6)
        # The RAW thermal array, not the colour-mapped picture. uint16
        # centi-kelvin little-endian is exactly what saresq/store/media.py
        # stores: 1,536 bytes for a 32x24 frame, lossless to 0.01 K. The
        # palette is a display choice and must not be what gets archived --
        # an operator re-examining a find a week later needs the temperatures,
        # not a screenshot of them.
        ev["raw"] = (b"" if th is None else
                     (np.clip(th + 273.15, 0, 655.35) * 100.0)
                     .astype("<u2").tobytes())
        self.events.append(ev)          # deque append is atomic; no lock needed

    def _save(self, rgb, vis, th):
        SAVE.mkdir(parents=True, exist_ok=True)
        s = time.strftime("%H%M%S")
        cv2.imwrite(str(SAVE / f"live_rgb_{s}.jpg"), rgb)
        cv2.imwrite(str(SAVE / f"live_detect_{s}.jpg"), vis)
        if th is not None:
            cv2.imwrite(str(SAVE / f"live_thermal_{s}.png"), colorize(th))
            np.save(SAVE / f"live_thermal_{s}.npy", th)
        for i, c in enumerate(self.crops):
            cv2.imwrite(str(SAVE / f"live_crop_{s}_{i}.jpg"), c)
        print(f"saved frame set {s}", flush=True)


def _frame_thermal(app: App):
    t, _ = app.thermal.read()
    return None if t is None else colorize(t)


def _frame_rgb(app: App):
    f, _ = app.camera.read()
    return None if f is None else cv2.resize(f, (640, 480))


def _frame_detect(app: App):
    with app.lock:
        f = None if app.det_img is None else app.det_img.copy()
    return None if f is None else cv2.resize(f, (640, 480))


FRAMES = {"/thermal.jpg": _frame_thermal, "/rgb.jpg": _frame_rgb,
          "/detect.jpg": _frame_detect}


def make_handler(app: App):
    def jpg(img, q=85):
        ok, b = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        return b.tobytes() if ok else None

    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, ctype, body: bytes, cache: str = "no-store"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)

        def _mjpeg(self, produce):
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=f")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    img = produce()
                    if img is None:
                        time.sleep(0.15)
                        continue
                    b = jpg(img)
                    if b:
                        self.wfile.write(
                            b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: "
                            + str(len(b)).encode() + b"\r\n\r\n" + b + b"\r\n")
                    time.sleep(1 / 10)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):                                       # noqa: N802
            p = self.path.split("?")[0]
            if p == "/":
                self._send(200, "text/html; charset=utf-8", PAGE.encode())
            elif p == "/manifest.webmanifest":
                self._send(200, "application/manifest+json", MANIFEST)
            elif p in ICONS:
                self._send(200, "image/png", app_icon(ICONS[p]))
            elif p in FRAMES:
                # One complete JPEG per request -- what the page uses. The
                # .mjpg variants below are kept for IP-camera apps.
                img = FRAMES[p](app)
                if img is None:
                    self._send(503, "text/plain", b"no frame yet")
                else:
                    self._send(200, "image/jpeg", jpg(img, 80))
            elif p == "/thermal.mjpg":
                self._mjpeg(lambda: (lambda t: colorize(t) if t is not None else None)
                            (app.thermal.read()[0]))
            elif p == "/rgb.mjpg":
                def rgb_small():
                    f, _ = app.camera.read()
                    return None if f is None else cv2.resize(f, (640, 480))
                self._mjpeg(rgb_small)
            elif p == "/detect.mjpg":
                def det_small():
                    with app.lock:
                        f = None if app.det_img is None else app.det_img.copy()
                    return None if f is None else cv2.resize(f, (640, 480))
                self._mjpeg(det_small)
            elif p.startswith("/crop/"):
                try:
                    i = int(p.rsplit("/", 1)[1])
                    with app.lock:
                        c = app.crops[i].copy()
                    self._send(200, "image/jpeg", jpg(c, 90))
                except (ValueError, IndexError):
                    self._send(404, "text/plain", b"no crop")
            elif p == "/events":
                # Metadata only -- the pictures are fetched per event, and only
                # for the ones actually scrolled into view.
                evs = list(app.events)
                self._send(200, "application/json", json.dumps([
                    {k: e[k] for k in
                     ("id", "t", "why", "n", "conf", "z", "tmax", "ncrops",
                      "lat", "lon", "sats", "scene", "scene_p", "s")}
                    for e in reversed(evs)]).encode())
            elif p.startswith("/event/"):
                try:
                    _, _, eid, kind = p.split("/", 3)
                    ev = next(e for e in app.events if e["id"] == int(eid))
                    blob = ev["raw"] if kind == "raw" else ev["img"][kind]
                except (ValueError, StopIteration, KeyError):
                    self._send(404, "text/plain", b"no such event")
                else:
                    # An event is frozen the moment it is captured, so unlike
                    # every live endpoint here it is safe to cache hard.
                    self._send(200,
                               "application/octet-stream" if kind == "raw" else "image/jpeg",
                               blob, "public, max-age=86400")
            elif p == "/save":
                with app.lock:
                    app.save_next = True
                self._send(200, "text/plain", b"ok")
            elif p == "/stats":
                th, fps = app.thermal.read()
                _, stamp = app.camera.read()
                with app.lock:
                    info, g, meta, seq = (dict(app.det_info), dict(app.gate),
                                          list(app.crop_meta), app.seq)
                    hz = dict(app.hazard)
                t = ({"min": float(th.min()), "max": float(th.max()),
                      "spread": float(th.max() - th.min()), "fps": fps}
                     if th is not None else {})
                try:
                    temp = subprocess.run(["vcgencmd", "measure_temp"],
                                          capture_output=True, timeout=2
                                          ).stdout.decode().strip().split("=")[-1]
                except Exception:
                    temp = "?"
                self._send(200, "application/json", json.dumps({
                    "thermal": t, "gate": g, "detect": info, "crops": meta,
                    "gps": app.gps.read(), "hazard": hz,
                    "seq": seq, "rotated": bool(app.args.rotate),
                    "rgb_w": app.args.width, "rgb_h": app.args.height,
                    "rgb_age": max(0.0, time.time() - stamp) if stamp else 0.0,
                    "temp": temp,
                    "up": time.strftime("%H:%M:%S"),
                }).encode())
            else:
                self._send(404, "text/plain", b"no")
    return H


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--hz", type=int, default=8, choices=[2, 4, 8])
    ap.add_argument("--width", type=int, default=1640)
    ap.add_argument("--height", type=int, default=1232)
    ap.add_argument("--fps", type=int, default=15,
                    help="camera frame rate for the video stream")
    ap.add_argument("--rotate", type=int, default=180, choices=[0, 90, 180, 270],
                    help="the module is mounted inverted; 180 corrects it")
    ap.add_argument("--tflip", default="h",
                    help="thermal orientation: any of h (mirror), v (flip), "
                         "r (rotate 180), or '' for none. Must end up matching "
                         "the RGB frame or the gate crops the wrong corner.")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--event-gap", type=float, default=6.0,
                    help="minimum seconds between auto-captured review events")
    ap.add_argument("--gps-port", default="/dev/serial0",
                    help="NMEA serial port; the demo holds it while running, so "
                         "sensor_selftest.py cannot read the GPS at the same time")
    ap.add_argument("--gps-baud", type=int, default=9600)
    ap.add_argument("--classes", default="0",
                    help="comma-separated class ids to keep; 0 is person in "
                         "COCO. Empty string keeps every class.")
    ap.add_argument("--no-contours", action="store_true",
                    help="do not outline the thermal field on the detector view")
    ap.add_argument("--hazard-model",
                    default=str(REPO / "models" / "mobilenetv2_aider_224_int8.tflite"),
                    help="AIDER scene classifier (MobileNetV2, int8)")
    ap.add_argument("--no-hazard", action="store_true",
                    help="skip scene classification entirely")
    ap.add_argument("--rgb-model",
                    default=str(REPO / "models" / "yolov8n_coco_640_w8a32.tflite"),
                    help="VISIBLE-light detector. The thermal-trained "
                         "yolov8n_p3_* finds nothing in RGB.")
    args = ap.parse_args()
    args.classes = {int(c) for c in args.classes.split(",") if c.strip()}

    import yaml
    cfg = yaml.safe_load((REPO / "configs" / "pipeline.yaml").read_text())
    if not pathlib.Path(args.rgb_model).exists():
        print(f"missing RGB model: {args.rgb_model}")
        return 1

    app = App(cfg, args)
    app.start()
    with Server(("0.0.0.0", args.port), make_handler(app)) as srv:
        print(f"open http://<pi-ip>:{args.port}", flush=True)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
