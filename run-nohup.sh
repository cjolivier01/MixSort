#!/bin/bash
LOG_FILE="$(basename $1 .sh).log"
nohup $@ > "${LOG_FILE}" 2>&1 &
echo "Tailing log file ${LOG_FILE} ..."
tail -f "${LOG_FILE}"
