#!/bin/bash

if [ "$1" ]; then
  THREADS="$1"
else
  THREADS="10"
fi

rm -rf jmeter-results.txt jmeter-report

echo "Running jmeter with $THREADS threads"

jmeter -n -t jmeter-asr.jmx -Jthreads="$THREADS" -Jrampup=10 -Jiterations=10 -Jfile=3sec.flac -Jprotocol=https -Jhost=ainfer.tovera.io \
  -l jmeter-results.txt -e -o jmeter-report
