./phase1.sh \
    --game='nchain' \
    --high_score_weight=10.0 \
    --horiz_weight=0.3 \
    --vert_weight=0.1 \
    --batch_size 1 \
    --explore_steps=100 \
    --max_hours=1 \
    --max_compute_steps=1000000 \
    --remember_rooms \
    --explorer='repeated'
    --log_path='log/Simple_environment'

