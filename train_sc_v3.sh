#!/bin/bash
# V3: A4 (Span-Pair Interactive Scorer) + A5 (Enhanced GCN)
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
    --use_pair_scorer --pair_hidden_dim 256 --pair_dropout 0.2 \
    --use_enhanced_gcn \
    --output_dir result/sc_v3/$dataset/result$seed --overwrite_output_dir --focalloss
done
