export PYTHONPATH="../..${PYTHONPATH:+:$PYTHONPATH}"

PREDICTION_ROOT="../../../buffer/analysis/Ambiguity/consistency"

python -m criteria.ambiguity.run consistency \
  --prediction "eagle25=${PREDICTION_ROOT}/eagle25.jsonl" \
  --prediction "internvl35=${PREDICTION_ROOT}/internvl35.jsonl" \
  --prediction "qwen3_vl=${PREDICTION_ROOT}/qwen3vl.jsonl" \
  --prediction "videoautor1=${PREDICTION_ROOT}/videoauto.jsonl" \
  --prediction "videor1=${PREDICTION_ROOT}/videor1.json"
