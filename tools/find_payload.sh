#!/bin/bash
# Locate the SaResQ payload on whatever network it landed on, and
# optionally bring the ground station up against it.
#
#   tools/find_payload.sh          # just find it
#   tools/find_payload.sh --serve  # find it, then start the ground station
#
# Searches the iPhone-hotspot /28 first (it is 14 addresses, so it is
# instant), then the Mac's current /24.

PORT=8091
found=""

probe() {  # probe <subnet-prefix> <first> <last>
  local pfx=$1 a=$2 b=$3
  for i in $(seq "$a" "$b"); do
    ( nc -z -G 1 "$pfx.$i" "$PORT" 2>/dev/null && echo "$pfx.$i" ) &
  done
  wait
}

echo "== looking for the payload on :$PORT =="

# 1. The payload pins 172.20.10.14 on the iPhone hotspot -- try it alone first.
if nc -z -G 1 172.20.10.14 "$PORT" 2>/dev/null; then
  found=172.20.10.14
  echo "   found at its pinned hotspot address: $found"
fi

# 2. Otherwise the hotspot /28 is only 14 addresses, so sweeping it is instant.
if [ -z "$found" ]; then
  hit=$(probe 172.20.10 1 14 | head -1)
  [ -n "$hit" ] && found=$hit && echo "   found on iPhone hotspot (DHCP): $found"
fi

# 3. Otherwise sweep whatever subnet this Mac is on
if [ -z "$found" ]; then
  myip=$(ipconfig getifaddr en0 2>/dev/null)
  if [ -n "$myip" ]; then
    pfx=${myip%.*}
    echo "   not on hotspot; sweeping $pfx.0/24 ..."
    hit=$(probe "$pfx" 1 254 | head -1)
    [ -n "$hit" ] && found=$hit && echo "   found on local network: $found"
  fi
fi

if [ -z "$found" ]; then
  echo "   NOT FOUND. Checks, in order:"
  echo "     - is the Pi powered on and past boot (~40s)?"
  echo "     - is this Mac on the same network as the Pi?"
  echo "     - hotspot on but Mac on the router instead? turn the hotspot off"
  echo "     - hotspot renamed? the Pi profile expects SSID 'Afzal Amanullah'"
  exit 1
fi

echo "   payload:       http://$found:$PORT"
echo "   ssh:           ssh -o HostName=$found saresq"

[ "$1" = "--serve" ] || exit 0

# Clear a stale ground station first -- one pointed at a dead IP is the
# single most common cause of "the dashboard shows old data".
stale=$(lsof -nP -iTCP:5050 -sTCP:LISTEN -t 2>/dev/null)
if [ -n "$stale" ]; then
  echo "== stopping stale ground station (pid $stale) =="
  kill $stale 2>/dev/null; sleep 2
fi

# The app lives in .venv -- a bare python3 will not have flask.
cd "$(dirname "$0")/.." || exit 1
[ -x .venv/bin/python ] || { echo "ERROR: .venv/bin/python missing"; exit 1; }
echo "== starting ground station against $found =="
.venv/bin/python -m saresq.dashboard.app --payload "$found" --port 5050
