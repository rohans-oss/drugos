#!/bin/bash
# Persistent npm install runner — retries up to 10 times until success.

cd /home/z/my-project

LOG=/tmp/npm-install-loop.log
echo "=== Starting install loop at $(date) ===" > $LOG

for attempt in 1 2 3 4 5 6 7 8 9 10; do
  echo "--- Attempt $attempt at $(date) ---" >> $LOG
  npm install --no-audit --no-fund --omit=optional --prefer-offline --legacy-peer-deps --loglevel=error >> $LOG 2>&1
  rc=$?
  echo "Attempt $attempt exit code: $rc" >> $LOG
  if [ $rc -eq 0 ]; then
    echo "SUCCESS on attempt $attempt" >> $LOG
    break
  fi
  # Brief pause before retry
  sleep 5
done

echo "=== Install loop finished at $(date) ===" >> $LOG
ls node_modules 2>/dev/null | wc -l >> $LOG
