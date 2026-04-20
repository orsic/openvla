#!/bin/bash
set -e

echo "Initialising firewall..."
/usr/local/bin/init-firewall.sh

exec "$@"
