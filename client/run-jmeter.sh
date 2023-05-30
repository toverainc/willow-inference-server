#!/usr/bin/env bash

if [ "$1" ]; then
  THREADS="$1"
else
  THREADS="10"
fi

if [ -d "/opt/jmeter/bin" ]; then
  export PATH="/opt/jmeter/bin":"$PATH"
fi

rm -rf jmeter-results.txt jmeter-report

echo "Running jmeter with $THREADS threads"

jmeter -n -t jmeter-asr.jmx -Jthreads="$THREADS" -Jrampup=10 -Jiterations=10000 -Jfile=3sec.flac -Jprotocol=http -Jhost=10.201.0.229 -Jport=19000 \
  -e -l jmeter-results.txt -o jmeter-report
