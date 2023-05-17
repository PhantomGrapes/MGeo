python -m torch.distributed.launch --nproc_per_node=4 --master_port $(expr $RANDOM + 1000) --use_env Rerank.py \
  --config ./configs/Rerank.yaml \
  --text_encoder bert-base-chinese \
  --checkpoint output/Pretrain_mm/checkpoint_09.pth \
  --output_dir output/Rerank \
