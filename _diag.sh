#!/bin/bash
source /opt/arena_ws/source
cd /opt/arena_ws/data/_fixwave/roots/social/arena_evaluation
python -m pytest tests/unit/test_proxemics_extended.py -x -q --tb=long -k "time_is_attributed" 2>&1 | head -200
