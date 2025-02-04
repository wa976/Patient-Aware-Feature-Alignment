MODEL="beats"
SEED="1 2 3 4 5"

for s in $SEED
do
    for m in $MODEL
    do
        TAG="seed${s}_best"
        CUDA_VISIBLE_DEVICES=0 python ./main.py --tag $TAG \
                                        --dataset icbhi \
                                        --seed $s \
                                        --class_split lungsound \
                                        --n_cls 4 \
                                        --epochs 100 \
                                        --batch_size 32 \
                                        --optimizer adam \
                                        --learning_rate 5e-5 \
                                        --weight_decay 1e-6 \
                                        --desired_length 5 \
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
                                        --method pvil \
                                        --w_ce 1.0 \
                                        --w_pvil 0.5 \
                                        --norm_type ln \
                                        --proj_type end2end \
                                        --output_dim 768 \
                                        --nospec \
                                        --lambda_pvr 10.0\
                                        --lambda_pi 0.01 \


                                        # only for evaluation, add the following arguments
                                        # --eval \
                                        # --pretrained \
                                        # --pretrained_ckpt ./save/icbhi_ast_ce_bs8_lr5e-5_ep50_seed1/best.pth

    done
done