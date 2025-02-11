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
                                        --nospec \
                                        --eval \
                                        --pretrained \
                                        --pretrained_ckpt ./save/icbhi_beats_pafa_seed${s}_best/best.pth

    done
done