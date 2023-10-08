#!/bin/bash
cd ../with_ram/

DATASET_NAME="peersum_all"
PLM_MODEL_PATH="allenai/led-large-16384"
MAX_LENGTH_INPUT=4096
MAX_LENGTH_TGT=512
MIN_LENGTH_TGT=32
GENERATION_LOSS=1
ACCEPTANCE_LOSS=0
RATING_LOSS=0
CONFIDENCE_LOSS=0
DOCUMENT_TYPE_LOSS=0

python mtsum_ram_from_led.py  \
                --batch_size 1 \
                --devices 1  \
                --accelerator gpu \
                --speed_strategy no_ddp \
                --mode test \
                --model_path ../../result/MTSum_meta_ram_from_led_zeroshot_${GENERATION_LOSS}_${ACCEPTANCE_LOSS}_${RATING_LOSS}_${CONFIDENCE_LOSS}_${DOCUMENT_TYPE_LOSS}_${DATASET_NAME}_${MAX_LENGTH_INPUT}_${MAX_LENGTH_TGT}/ \
                --data_path ../../crawling_data/data/ \
                --dataset_name ${DATASET_NAME} \
                --pretrained_model ${PLM_MODEL_PATH} \
                --num_workers 4 \
                --optimizer adamw \
                --scheduler_type cosine_schedule_with_warmup \
                --beam_size 5 \
                --length_penalty 1.0 \
                --repetition_penalty 1.0 \
                --test_imediate \
                --max_length_tgt ${MAX_LENGTH_TGT} \
                --min_length_tgt ${MIN_LENGTH_TGT} \
                --max_length_input ${MAX_LENGTH_INPUT} \
                --total_steps 50000 \
                --accum_data_per_step 16 \
                --early_stopping_patience 5 \
                --val_check_interval 50 \
                --num_train_data -1 \
                --num_val_data 128 \
                --num_test_data -1 \
                --lr 5e-5 \
                --apply_triblck \
                --label_smoothing 0.1 \
                --generation_loss_weight ${GENERATION_LOSS} \
                --acceptance_loss_weight ${ACCEPTANCE_LOSS} \
                --rating_loss_weight ${RATING_LOSS} \
                --confidence_loss_weight ${CONFIDENCE_LOSS} \
                --document_type_loss_weight ${DOCUMENT_TYPE_LOSS} \
                --warmup_steps 1000