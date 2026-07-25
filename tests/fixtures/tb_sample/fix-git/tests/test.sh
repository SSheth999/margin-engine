#!/bin/bash
# LOCAL-RUNNABLE test fixture (simplified). The real upstream test.sh installs uv and
# runs pytest inside the E2B sandbox against absolute paths (/tests, /logs/verifier);
# this fixture uses $PWD-relative paths so it also runs under LocalExecEnv (no chroot),
# exercising our verify plumbing (mkdir logs, run test.sh, write reward.txt) offline.
mkdir -p logs/verifier
if [ -f solution.txt ] && grep -q "recovered" solution.txt; then
  echo 1 > logs/verifier/reward.txt
else
  echo 0 > logs/verifier/reward.txt
fi
