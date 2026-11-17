#!/usr/bin/env bash
# Snapshot shared storage. Output contains private data; keep it encrypted.
set -euo pipefail
umask 077
output="${1:?Usage: scripts/backup.sh /secure/path/backup.dump}"
if [ -e "$output" ]; then
  echo "Refusing to overwrite an existing backup" >&2
  exit 1
fi
# noclobber protects against a race between the check and opening the file.
set -o noclobber
docker compose exec -T postgres pg_dump -U postgres -d docuintel -Fc > "$output"
echo "Backup written. Test restore into a separate database before relying on it."
