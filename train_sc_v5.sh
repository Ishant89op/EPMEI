#!/bin/bash
# V5: Multi-Perspective Interactive Span-Pair Scorer with Residual Fusion (A4-v5)
#     + Enhanced GCN with Xavier/GELU/Residual scale (A5)
#     + R-Drop symmetric KL consistency regularization (alpha=0.5)
#     + Label-Smoothed Focal Loss (epsilon=0.05)
#     + Pair dropout=0.15, weight_decay=0.01, new_param_lr=1e-4
GPU_ID=0
dataset="14lap"
for seed in 41 42 43 44 45; do 
CUDA_VISIBLE_DEVICES=$GPU_ID python3 run_sc.py --model_type bertsub \
    --model_name_or_path bert-base-uncased --do_lower_case \
    --data_dir data/ASTE-Data-V2-EMNLP2020_pro/$dataset/ \
    --learning_rate 4e-5 --num_train_epochs 10 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16 --gradient_accumulation_steps 1 \
    --max_seq_length 256 --max_pair_length 16 --save_steps 50 \
    --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --eval_logsoftmax \
    --seed $seed \
    --test_file data/ner_predictions/$dataset/ent_pred_test_pro.json \
    --use_ner_results \
    --use_typemarker --gcn --n_gcn 2 \
    --use_pair_scorer --pair_hidden_dim 256 --pair_dropout 0.15 \
    --use_enhanced_gcn \
    --weight_decay 0.01 \
    --new_param_lr 1e-4 \
    --rdrop_alpha 0.5 \
    --label_smoothing 0.05 \
    --output_dir result/sc_v5/$dataset/result$seed --overwrite_output_dir --focalloss
done
