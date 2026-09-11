# Putting SaResQ on a phone

The ground station installs as an app on Android and iOS with no app store, no
developer account and no second codebase. It is a Progressive Web App: the same
Flask console, plus a manifest and a service worker, which the phone installs to
the home screen and runs full-screen with no browser chrome.

This was the right call for this project rather than React Native or Flutter,
for three reasons worth being able to say out loud:

* **No store, no fee, no review.** A Play Console listing is $25 and a review
  queue; an Apple Developer Program membership is $99/year. A judge or a
  volunteer coordinator installs this in fifteen seconds from a link.
* **One codebase.** The radar scope is 700 lines of canvas and SVG. Rewriting it
  natively twice, twelve days before submission, with the detector still
  untrained, would be the wrong thing to spend the time on.
* **Offline is the point.** The service worker caches the app shell *including*
  the 97 KB of baked-in OSM geometry and SRTM terrain, so the map works in a
  field tent with no signal. That is the same offline-first argument the rest of
  the system makes.

## Run the server so a phone can install it

A service worker requires a **secure context**. `localhost` counts;
`http://192.168.1.5:5050` does not. Reached over plain HTTP from a phone, the
console still works, but there is no offline cache and Android shows no install
prompt. So start it with TLS:

```bash
python -m saresq.dashboard.app \
    --db saresq.db --media-dir results/media \
    --https --port 5050
```

On first run this writes a self-signed certificate to `.certs/` covering
`localhost`, `127.0.0.1` and the machine's current LAN address, and prints the
URL to open. It uses the `openssl` binary rather than Flask's `ssl_context="adhoc"`,
which needs the `cryptography` package — one fewer dependency on the Pi image,
and the certificate persists so a phone only has to trust it once.

> If the laptop moves to a different network its LAN IP changes and the
> certificate no longer covers it. Delete `.certs/` and restart.

## Android (Chrome / Edge)

1. Connect the phone to the same Wi-Fi as the ground station.
2. Open the `https://<lan-ip>:5050/radar` address the server printed.
3. Chrome warns about the self-signed certificate → **Advanced** → **Proceed**.
4. Tap **Install app** in the console's top bar, or Chrome's own ⋮ → *Install app*.

The app lands on the home screen with the SaResQ diamond, opens straight to the
radar scope, and gets long-press shortcuts to Radar, Review and Map.

## iPhone / iPad (Safari)

Apple does not implement `beforeinstallprompt`, so there is no install button
and no way to trigger one — it has to be done by hand. Tapping **Install app**
in the top bar shows these steps rather than pretending otherwise.

**Step 1 — trust the certificate.** iOS will happily add an untrusted site to
the home screen, but it will *not* register a service worker on one, so the app
would install with no offline cache — the one capability that matters in a
field tent. Two minutes, once per phone:

1. In **Safari**, open `https://<lan-ip>:5050/cert`. Tap **Allow** when it
   offers to download a configuration profile.
2. **Settings → Profile Downloaded → Install** (top of the Settings list).
   Enter the passcode, tap **Install** again.
3. **Settings → General → About → Certificate Trust Settings** →
   turn **on** the switch for *SaResQ Ground Station*.

The certificate is generated with everything iOS 13+ requires — an explicit
`serverAuth` extended key usage, the machine's LAN IP in the subject
alternative name, and 820 days of validity (Apple rejects anything over 825).
`app.py` regenerates it automatically if the machine's address changes, because
a certificate that no longer covers the URL is rejected outright rather than
merely warned about.

**Step 2 — install the app.**

1. Open `https://<lan-ip>:5050/radar` in **Safari** — Chrome and Firefox on iOS
   cannot install web apps, only Safari can.
2. Tap **Share** (the square with the up-arrow) → scroll → **Add to Home
   Screen** → **Add**.

It lands on the home screen as *SaResQ*, opens full-screen with no browser
chrome, and starts on the radar scope.

Skipping step 1 still works — the app installs and runs fine while the ground
station is reachable — you just get no offline cache.

## What works offline, and what deliberately does not

The service worker (`saresq/dashboard/static/sw.js`, served from `/sw.js` so its
scope covers the whole site) splits traffic three ways:

| Request | Strategy | Why |
|---|---|---|
| Pages | network first, cached shell as fallback | a running ground station always wins |
| `/static/*` | cache first | `basemap.js` is 97 KB and changes about never |
| `/media/*`, `/thumbs/*` | cache first, separate cache | evidence blobs are content-addressed and immutable |
| **`/api/*`** | **network only, never cached** | see below |

**Live data is never served from cache.** A shell that loads offline is useful.
A two-hour-old aircraft position presented as current is worse than a blank
screen, because the operator cannot tell the difference. When the ground station
is unreachable the radar page says *"ground station unreachable"* and shows
nothing rather than something stale.

## If a real store listing is ever needed

For the December Grand Finale, if a signed `.apk` is wanted:

* **Android** — wrap the same PWA with [Bubblewrap](https://github.com/GoogleChromeLabs/bubblewrap)
  as a Trusted Web Activity, or Capacitor if native plugins are needed. No
  rewrite; it packages this URL.
* **iOS** — Capacitor plus Xcode plus a paid developer account. Only worth it if
  something native is genuinely required (background location, BLE to the
  airframe). Nothing in the current design is.

## Files

| Path | What |
|---|---|
| `saresq/dashboard/static/manifest.webmanifest` | name, icons, `start_url: /radar`, shortcuts |
| `saresq/dashboard/static/sw.js` | caching policy above |
| `saresq/dashboard/static/icons/` | 180/192/512 plus a maskable 512 |
| `tools/make_icons.py` | regenerates those icons |
| `saresq/dashboard/templates/base.html` | manifest link, iOS meta, install button |
