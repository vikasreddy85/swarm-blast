#!/usr/bin/env bash
# Runs every experiment back to back with no babysitting.
#
#   ./run_queue.sh             full queue
#   ./run_queue.sh --stub      tiny offline pass over every job (no API, no cost) to check the pipeline
#   ./run_queue.sh --dry-run   print the job list and exit
#
# Resumable: a finished job leaves <OUT>/<job>/.done and is skipped next time (FORCE=1 reruns everything).
# One failed job does not stop the queue; failures are listed at the end. LLM calls are cached, so a rerun is cheap.
#
# Environment knobs (all optional):
#   MODELS="openai/gpt-4o-mini anthropic/claude-3.5-haiku"   OpenRouter slugs, run in order
#   VIGS="0 1 2 3"        vigilance levels for the LLM experiments
#   BUDGET=25             hard USD cap across everything in cache/ (see README section 7)
#   SKIP="e0 e4"          experiments to leave out
#   PRICE_IN / PRICE_OUT  fallback USD per million tokens if you switch to a model with very different pricing
#   OUT=queue_out         output root
#   HYDRA_EXTRA="reasoning={enabled:false}"   extra Hydra overrides passed to every LLM job (space separated)
set -u
cd "$(dirname "$0")"

STUB=0; DRY=0
for a in "$@"; do
  case "$a" in --stub) STUB=1 ;; --dry-run) DRY=1 ;; *) echo "unknown option $a"; exit 2 ;; esac
done

OUT=${OUT:-queue_out}
MODELS=${MODELS:-"openai/gpt-4o-mini"}
VIGS=${VIGS:-"0 1 2 3"}
BUDGET=${BUDGET:-25}
SKIP=${SKIP:-""}
FORCE=${FORCE:-0}
[ "$STUB" = 1 ] && OUT="${OUT}_stub"

# add the e3l / e7 / e8 config blocks once
if [ "$DRY" = 0 ] && ! grep -q '^adapt:' conf/config.yaml; then
  echo >> conf/config.yaml
  cat conf/queue_block.yaml >> conf/config.yaml
  echo "[setup] appended conf/queue_block.yaml to conf/config.yaml"
fi

SMALL=()
if [ "$STUB" = 1 ]; then
  SMALL=(stub=true stub_p_copy=0.7 llm.n=8 llm.ticks=6 llm.neg=6 llm.ref_runs=3 llm.seeds=2 adapt.seeds=2 llm.dms_samples=3
         sweep.seeds=2 sweep.neg=4 "sweep.mu_core=[0,0.4]" ttd.seeds=2 ttd.neg=5 "ttd.mu_core=[0,0.4]" hgt.seeds=4)
fi

HAVE_KEY=1
if [ "$STUB" = 0 ] && [ "$DRY" = 0 ] && [ -z "${OPENROUTER_API_KEY:-}" ]; then HAVE_KEY=0; fi

FAILED=(); N=0
skipped() { for s in $SKIP; do [ "$s" = "$1" ] && return 0; done; return 1; }

run() {  # run <job-name> <experiment> [hydra overrides...]
  local name=$1 exp=$2; shift 2
  skipped "$exp" && return 0
  N=$((N+1))
  local d="$PWD/$OUT/$name"
  if [ "$DRY" = 1 ]; then echo "  $name: python run.py experiment=$exp $*"; return 0; fi
  if [ -f "$d/.done" ] && [ "$FORCE" != 1 ]; then echo "[skip] $name"; return 0; fi
  mkdir -p "$PWD/$OUT"
  echo "[run ] $name  $(date +%H:%M:%S)"
  if python run.py "experiment=$exp" "$@" "hydra.run.dir=$d" "hydra.job.chdir=true" > "$PWD/$OUT/$name.log" 2>&1; then
    touch "$d/.done"; echo "[ ok ] $name  $(date +%H:%M:%S)"
  else
    FAILED+=("$name"); echo "[FAIL] $name  (tail of $OUT/$name.log follows)"; tail -n 5 "$PWD/$OUT/$name.log" | sed 's/^/        /'
  fi
}

# ---- 1. mock experiments: free, no API key ------------------------------------------------------------
run e8_hgt_mock     e8 ${SMALL[@]+"${SMALL[@]}"}
run e7_ttd_mock     e7 ${SMALL[@]+"${SMALL[@]}"}
run e0_mock         e0 ${SMALL[@]+"${SMALL[@]}"}
run e4_mock         e4 ${SMALL[@]+"${SMALL[@]}"}
run e6_mock         e6 ${SMALL[@]+"${SMALL[@]}"}

# ---- 2. LLM experiments ---------------------------------------------------------------------------------
if [ "$HAVE_KEY" = 0 ]; then
  echo "[warn] OPENROUTER_API_KEY not set: skipping all LLM jobs (export it, or use --stub)"
else
  EXTRA=()
  [ -n "${PRICE_IN:-}" ]  && EXTRA+=("price_in=$PRICE_IN")
  [ -n "${PRICE_OUT:-}" ] && EXTRA+=("price_out=$PRICE_OUT")
  [ -n "${HYDRA_EXTRA:-}" ] && EXTRA+=($HYDRA_EXTRA)
  # most important first, so a budget stop costs the least: e3l (adaptive adversary + time to detect + horizontal transfer on logs), then e5, then e1
  for exp in e3l e5 e1; do
    for m in $MODELS; do
      for v in $VIGS; do
        if [ "$STUB" = 1 ]; then
          run "${exp}_v${v}_stub" "$exp" "llm.vigilance=$v" ${SMALL[@]+"${SMALL[@]}"}
        else
          run "${exp}_v${v}_${m//\//_}" "$exp" "model=$m" "llm.vigilance=$v" "budget_usd=$BUDGET" llm.neg=16 llm.ref_runs=4 ${EXTRA[@]+"${EXTRA[@]}"}
        fi
      done
      [ "$STUB" = 1 ] && break
    done
  done
fi

[ "$DRY" = 1 ] && { echo "$N jobs"; exit 0; }

# ---- 3. combine everything ------------------------------------------------------------------------------
python aggregate.py "$OUT" > "$OUT/aggregate.log" 2>&1 && echo "[ ok ] aggregate -> $OUT/summary/" || echo "[FAIL] aggregate (see $OUT/aggregate.log)"

echo
echo "finished $N jobs, ${#FAILED[@]} failed"
[ "${#FAILED[@]}" -gt 0 ] && printf '  failed: %s\n' "${FAILED[@]}"
exit 0