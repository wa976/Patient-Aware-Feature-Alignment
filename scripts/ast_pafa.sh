MODEL="ast"
SEED="1 2 3 4 5"

for s in $SEED
do
    for m in $MODEL
    do
        TAG="seed${s}_best"
        CUDA_VISIBLE_DEVICES=0 python main.py --tag $TAG \
                                        --dataset icbhi \
                                        --seed $s \
                                        --class_split lungsound \
                                        --n_cls 4 \
                                        --epochs 50 \
                                        --batch_size 32 \
                                        --desired_length 5 \
                                        --optimizer adam \
                                        --learning_rate 5e-5 \
                                        --weight_decay 1e-6 \
                                        --cosine \
                                        --model $m \
                                        --test_fold official \
                                        --pad_types repeat \
                                        --resz 1 \
                                        --n_mels 128 \
                                        --ma_update \
                                        --ma_beta 0.5 \
                                        --from_sl_official \
                                        --audioset_pretrained \
                                        --method pafa \
                                        --w_ce 1.0 \
                                        --w_pafa 0.5 \
                                        --lambda_pcsl 10.0\
                                        --lambda_gpal 0.01 \
                                        --norm_type ln \
                                        --output_dim 768 \
                                        --nospec

    done
done






# for s in $SEED
# do
#     for m in $MODEL
#     do
#         TAG="seed${s}_best_param"
#         CUDA_VISIBLE_DEVICES=0 python main.py --tag $TAG \
#                                         --dataset icbhi \
#                                         --seed $s \
#                                         --class_split lungsound \
#                                         --n_cls 4 \
#                                         --epochs 50 \
#                                         --batch_size 32 \
#                                         --desired_length 5 \
#                                         --optimizer adam \
#                                         --learning_rate 5e-5 \
#                                         --weight_decay 1e-6 \
#                                         --cosine \
#                                         --model $m \
#                                         --test_fold official \
#                                         --pad_types repeat \
#                                         --resz 1 \
#                                         --n_mels 128 \
#                                         --ma_update \
#                                         --ma_beta 0.5 \
#                                         --from_sl_official \
#                                         --audioset_pretrained \
#                                         --method pccl \
#                                         --w_class 0.6 \
#                                         --w_same_patient 0.2 \
#                                         --w_diff_patient 0.2 \
#                                         --norm_type ln \
#                                         --output_dim 128 \


#                                         # only for evaluation, add the following arguments
#                                         # --eval \
#                                         # --pretrained \
#                                         # --pretrained_ckpt ./save/icbhi_ast_ce_bs8_lr5e-5_ep50_seed1/best.pth

#     done
# done