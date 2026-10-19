export PYTHONPATH="../..${PYTHONPATH:+:$PYTHONPATH}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}" python -m criteria.visual.run_stt "$@"
