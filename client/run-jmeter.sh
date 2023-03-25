#!/bin/bash

rm -rf jmeter-results.txt jmeter-report

jmeter -n -t jmeter-asr-3sec.jmx -l jmeter-results.txt -e -o jmeter-report
