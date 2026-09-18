/* Raster basemap tiles, drawn UNDER the vector layers mapcore already renders.
 *
 * mapcore.js was already doing correct Web Mercator on a 256 px tile grid
 * (wx/wy/lonAt/latAt), so tiles drop straight in -- nothing about the existing
 * projection, flood raster, road classification or target overlay changes.
 *
 * WHY THIS EXISTS
 * basemap.js is ~96 KB of geometry baked for roughly 11 km around one point in
 * Kolkata. That is the right thing to fly with, because a disaster area has no
 * network and a payload that needs one is a payload that fails on the day. It
 * is the wrong thing to plan with: a coordinator wants to see the whole city,
 * and the next deployment is somewhere else entirely.
 *
 * So: tiles when the network is there, baked geometry when it is not, and the
 * UI says which. The baked layers keep drawing on top either way, because the
 * flood model and the road passability classification are ours and exist at no
 * zoom in anyone's tile server.
 *
 * DARK TILES ON PURPOSE. A standard OSM raster is near-white and would put a
 * bright rectangle behind a console designed to be read in a dim operations
 * room -- and would bury the orange/red hazard colours the whole map exists to
 * show. CARTO's dark basemap sits behind them instead of fighting them.
 */
window.TileLayer = (function () {
  var TILE = 256;

  /* Keyless providers only. CARTO's dark basemap was the obvious aesthetic
   * match and was tried first -- it now returns HTTP 200 with the words
   * "API KEY REQUIRED" painted across every tile, which is worse than an
   * error because the map looks like it is working. Each of these was fetched
   * and checked before being listed here.
   *
   * Esri's tile path is {z}/{y}/{x} -- row before column -- where OSM's is
   * {z}/{x}/{y}. Getting that backwards yields valid-looking tiles of entirely
   * the wrong place, so the order is part of each source rather than assumed.
   */
  var SOURCES = {
    sat: {
      url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      sub: null, max: 19, r: false,
      attr: "Imagery \u00a9 Esri, Maxar, Earthstar Geographics"
    },
    dark: {
      url: "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
      sub: null, max: 16, r: false,
      attr: "\u00a9 Esri \u00b7 \u00a9 OpenStreetMap contributors"
    },
    street: {
      url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
      sub: null, max: 19, r: false,
      attr: "\u00a9 OpenStreetMap contributors"
    }
  };

  var cache = new Map();          // key -> {img, ok}
  var inflight = 0;
  var okCount = 0, failCount = 0;
  var MAX_CACHE = 600;
  var MAX_INFLIGHT = 16;

  function key(s, z, x, y) { return s + "/" + z + "/" + x + "/" + y; }

  function url(src, z, x, y) {
    var r = src.r && (window.devicePixelRatio || 1) > 1.4 ? "@2x" : "";
    var u = src.url.replace("{z}", z).replace("{x}", x).replace("{y}", y).replace("{r}", r);
    if (src.sub) u = u.replace("{s}", src.sub[(x + y) % src.sub.length]);
    return u;
  }

  function get(name, z, x, y, onload) {
    var k = key(name, z, x, y);
    var hit = cache.get(k);
    if (hit) { cache.delete(k); cache.set(k, hit); return hit; }   // LRU touch
    if (inflight >= MAX_INFLIGHT) return null;

    var rec = { img: new Image(), ok: false };
    // Without this the canvas is tainted and any later getImageData throws --
    // mapcore builds its flood raster on a separate canvas today, but tainting
    // this one would be a trap laid for whoever adds a pixel read later.
    rec.img.crossOrigin = "anonymous";
    inflight++;
    rec.img.onload = function () {
      rec.ok = true; inflight--; okCount++;
      if (onload) onload();
    };
    rec.img.onerror = function () {
      inflight--; failCount++;
      cache.delete(k);            // let it retry on a later pan; do not cache a failure
    };
    rec.img.src = url(SOURCES[name], z, x, y);
    cache.set(k, rec);
    while (cache.size > MAX_CACHE) cache.delete(cache.keys().next().value);
    return rec;
  }

  /* An ancestor of this tile that IS loaded, plus the sub-rectangle of it that
   * covers the child. Standard pyramid behaviour, and the reason a real map
   * goes blurry-then-sharp while a naive one goes black-then-sharp: the dark
   * rectangles people see on a slow network are simply tiles that had not
   * arrived, drawn as bare background.
   */
  function ancestor(name, z, x, y) {
    for (var up = 1; up <= 5 && z - up >= 0; up++) {
      var f = Math.pow(2, up);
      var px = Math.floor(x / f), py = Math.floor(y / f);
      var rec = cache.get(key(name, z - up, px, py));
      if (rec && rec.ok) {
        var span = TILE / f;
        return { img: rec.img, sx: (x - px * f) * span, sy: (y - py * f) * span, ss: span };
      }
    }
    return null;
  }

  /* Draw the tile pyramid for the current camera.
   * cam: {lat, lon, z} (z fractional), vp: {w, h}. wx/wy are mapcore's. */
  function draw(ctx, cam, vp, wx, wy, opts) {
    opts = opts || {};
    var name = opts.source || "dark";
    var src = SOURCES[name];
    if (!src) return { drawn: 0, want: 0 };

    // Tiles exist only at integer zoom; scale the chosen level to the camera's
    // fractional zoom rather than snapping the camera, which would make every
    // pinch jump.
    var zi = Math.max(0, Math.min(src.max, Math.round(cam.z)));
    var scale = Math.pow(2, cam.z - zi);
    var span = TILE * scale;
    var n = Math.pow(2, zi);

    // Camera position in this level's pixel space, then the tile window.
    var cxp = wx(cam.lon, zi), cyp = wy(cam.lat, zi);
    var left = cxp - (vp.w / 2) / scale, top = cyp - (vp.h / 2) / scale;
    var x0 = Math.floor(left / TILE), y0 = Math.floor(top / TILE);
    var x1 = Math.floor((left + vp.w / scale) / TILE);
    var y1 = Math.floor((top + vp.h / scale) / TILE);

    // Warm the parent level first, so the fallback above has something to
    // upscale on the very first paint rather than only after a zoom out.
    if (zi > 0) {
      var pz = zi - 1, pcx = wx(cam.lon, pz), pcy = wy(cam.lat, pz);
      var pl = pcx - (vp.w / 2) / (scale * 2), pt = pcy - (vp.h / 2) / (scale * 2);
      for (var py2 = Math.floor(pt / TILE); py2 <= Math.floor((pt + vp.h / (scale * 2)) / TILE); py2++) {
        for (var px2 = Math.floor(pl / TILE); px2 <= Math.floor((pl + vp.w / (scale * 2)) / TILE); px2++) {
          var pn = Math.pow(2, pz);
          if (py2 < 0 || py2 >= pn) continue;
          get(name, pz, ((px2 % pn) + pn) % pn, py2, opts.onTile);
        }
      }
    }

    var drawn = 0, want = 0, redraw = opts.onTile;
    ctx.save();
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    if (opts.alpha != null) ctx.globalAlpha = opts.alpha;
    for (var ty = y0; ty <= y1; ty++) {
      if (ty < 0 || ty >= n) continue;
      for (var tx = x0; tx <= x1; tx++) {
        want++;
        var wrapped = ((tx % n) + n) % n;       // wrap at the antimeridian
        var rec = get(name, zi, wrapped, ty, redraw);
        var sx = (tx * TILE - cxp) * scale + vp.w / 2;
        var sy = (ty * TILE - cyp) * scale + vp.h / 2;
        if (!rec || !rec.ok) {
          // Not here yet: upscale whatever ancestor we already hold rather than
          // leaving a black rectangle. It sharpens when the real tile lands.
          var a = ancestor(name, zi, wrapped, ty);
          if (a) {
            ctx.drawImage(a.img, a.sx, a.sy, a.ss, a.ss, sx, sy, span + 1, span + 1);
            drawn++;
          }
          continue;
        }
        // +1 px: adjacent tiles otherwise show hairline seams at fractional
        // zoom, because each edge rounds independently.
        ctx.drawImage(rec.img, sx, sy, span + 1, span + 1);
        drawn++;
      }
    }
    ctx.restore();
    // `up` is how far past this source's own maximum zoom the camera has
    // gone. It matters: at a bench scale the camera sits six levels above the
    // dark basemap's z16, every tile is magnified 64x, and the result is a
    // uniform grey rectangle that looks exactly like a broken map. Reporting
    // the factor lets the UI say "upscaled" instead of leaving the operator to
    // guess whether the tiles failed.
    return { drawn: drawn, want: want, z: zi,
             up: Math.pow(2, Math.max(0, cam.z - zi)) };
  }

  return {
    draw: draw,
    sources: SOURCES,
    attribution: function (n) { return (SOURCES[n] || SOURCES.dark).attr; },
    /* Healthy only once something has actually loaded. "navigator.onLine" is
     * not a usable signal here: it reports whether an interface exists, not
     * whether the tile host is reachable, and it is true on a phone hotspot
     * with no data and on a venue wifi behind a captive portal. */
    health: function () {
      return { ok: okCount, fail: failCount, inflight: inflight,
               live: okCount > 0 && failCount < okCount + 8 };
    },
    clear: function () { cache.clear(); okCount = failCount = 0; }
  };
})();
