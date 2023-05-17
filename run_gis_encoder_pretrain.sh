python -m torch.distributed.launch --nproc_per_node=8 --master_port $(expr $RANDOM + 1000) --use_env Pretrain_gis.py \
 --config ./configs/Pretrain_gis.yaml \
 --output_dir output/Pretrain_gis_encoder \
 --text_encoder resources/nlp_structbert_backbone_tiny_std \
