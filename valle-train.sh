#!/bin/bash
set -e 
export DATA="/app/deps/vall-e/data/test"

python -m vall_e.emb.qnt "$DATA"

python -m vall_e.emb.g2p "$DATA"

#python -m vall_e.train yaml=config/your_data/ar_or_nar.yml