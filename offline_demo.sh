#!/bin/sh
# Offline demonstration: run with Wi-Fi OFF and Ollama running (`ollama serve` in another tab).
# Everything printed is also saved to evidence/offline/offline-run-<time>.txt.
cd "$(dirname "$0")" || exit 1
mkdir -p evidence/offline
LOG="evidence/offline/offline-run-$(date +%Y%m%d-%H%M%S).txt"

run() {
  echo ""
  echo "\$ $*"
  "$@"
}

{
  echo "Offline run started $(date)"
  run ./wiki status
  run ./wiki --help
  run ./wiki ingest "vault/raw/Aug 2025 - Resume Only Workshop.pdf" --force
  run ./wiki ingest
  run ./wiki search "cold outreach message" --save
  run ./wiki test
  run ./wiki modecheck
  echo ""
  echo "Offline run finished $(date)"
} 2>&1 | tee "$LOG"

echo ""
echo "Saved: $LOG"
echo "Now start an interactive chat for your own 3+ turns:  ./wiki chat   (type /exit to save the transcript)"
