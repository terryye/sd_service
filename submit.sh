#!/bin/bash
# Run one illustration batch on Explorer from this Mac.
# Usage: ./submit.sh words.csv [generate.py options, e.g. --variants 4]
# Uploads the CSV, submits a GPU job, waits for it, and downloads the images to ./images/<csv name>/.

set -euo pipefail
CSV=$1; shift
U=ye.te
XFER=$U@xfer.discovery.neu.edu
LOGIN=$U@login.explorer.northeastern.edu
SVC=/home/$U/sd_service
NAME=$(basename "$CSV" .csv)
HERE=$(cd "$(dirname "$0")" && pwd)
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=60"

rsync -a "$HERE/generate.py" "$HERE/postprocess.py" "$HERE/sd_job.sh" "$XFER:$SVC/"
$SSH "$XFER" "mkdir -p $SVC/batches $SVC/out $SVC/logs"
rsync -a "$CSV" "$XFER:$SVC/batches/$NAME.csv"

# sbatch sometimes reports "Unexpected message received" yet the job is still created,
# possibly minutes later, so a failed submit cannot be trusted. Every submission of this
# batch shares one job name: --dependency=singleton stops copies running at once, and
# sd_job.sh exits early in any copy that is not the first, so retrying is harmless.
JOBNAME="sd_${NAME}_$(date +%s)"
for attempt in 1 2 3; do
    if OUT=$($SSH "$LOGIN" "sbatch --parsable --job-name=$JOBNAME --dependency=singleton $SVC/sd_job.sh $SVC/batches/$NAME.csv $SVC/out/$NAME $*" 2>&1); then
        break
    fi
    echo "sbatch attempt $attempt: $OUT" >&2
    sleep 60
    if $SSH "$LOGIN" "sacct -n -X -u $U -S now-7days --name=$JOBNAME -o JobID" 2>/dev/null | grep -q '[0-9]'; then break; fi
done

# The job doing the work is the lowest ID with this name. Track it through sacct,
# which (unlike squeue) keeps reporting a job after it leaves the queue.
JOB= STATE=
while :; do
    if ROWS=$($SSH "$LOGIN" "sacct -n -X -P -u $U -S now-7days --name=$JOBNAME -o JobID,State" 2>/dev/null) && [ -n "$ROWS" ]; then
        read -r JOB STATE < <(echo "$ROWS" | sort -n | head -1 | tr '|' ' ')
        case "$STATE" in PENDING|RUNNING|REQUEUED|CONFIGURING|COMPLETING|SUSPENDED|RESIZING) ;; *) break ;; esac
    elif [ -z "$JOB" ] && [ $((SECONDS)) -gt 900 ]; then
        echo "no job named $JOBNAME appeared; submission failed" >&2; exit 1
    fi
    sleep 30
done
echo "job $JOB: $STATE (log: $SVC/logs/$JOBNAME.$JOB.out)"
$SSH "$LOGIN" "tail -5 $SVC/logs/$JOBNAME.$JOB.out"

mkdir -p "$HERE/images/$NAME"
rsync -a "$XFER:$SVC/out/$NAME/" "$HERE/images/$NAME/"
echo "images in $HERE/images/$NAME/"
[ "$STATE" = COMPLETED ]
