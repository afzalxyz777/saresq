#!/usr/bin/env bash
# Pull the frame sets saved by the demo's "Save frame set" button.
#
#     bash tools/calib_pull.sh            # uses $PI (default: the saresq alias)
#     PI=192.168.1.2 bash tools/calib_pull.sh
#
# mDNS lags for minutes after the Pi changes network, so an IP override is the
# documented escape hatch rather than a workaround -- see saresq_pi_hardware.
set -euo pipefail
PI=${PI:-saresq}
SSH_OPTS=${SSH_OPTS:--o StrictHostKeyChecking=no -o UserKnownHostsFile=$HOME/.ssh/known_hosts_saresq -i $HOME/.ssh/saresq_pi}
[[ "$PI" == *@* ]] || PI="afzalamanullah@$PI"

mkdir -p results/calib
scp $SSH_OPTS -q "$PI:~/saresq/results/demo/live_*" results/calib/
echo "frame sets now local:"
ls -1 results/calib/live_thermal_*.npy | sed 's|^|   |'
echo
echo "next:  .venv/bin/python tools/calib_pick.py results/calib/live_thermal_*.npy --sign cold"
