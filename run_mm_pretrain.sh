python -m torch.distributed.launch --nproc_per_node=8 --master_port $(expr $RANDOM + 1000) --use_env Pretrain_mm.py\
 --config ./configs/Pretrain_mm.yaml \
 --text_encoder  bert-base-chinese \
 --output_dir output/Pretrain_mm \
 --checkpoint output/Pretrain_gis_encoder/checkpoint_29.pth \

