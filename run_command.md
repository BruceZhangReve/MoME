# Use train.py
CUDA_VISIBLE_DEVICES=1 python train.py --instructor_query 3 --n_experts 4 --topk 2 --modulation --router_modulation --output_dir output/FFLr --task finance_forecast --in_len 134 --out_len 33 --dataset_path ./data/processed/finance/long/train --epoch 8 --hidden_dim 32 --patch_len 4 --ts_encoder MoMe --use_bfloat16

CUDA_VISIBLE_DEVICES=2 python train.py --instructor_query 3 --n_experts 4 --topk 2 --modulation --router_modulation --output_dir output/FFSr --task finance_forecast --in_len 312 --out_len 78 --dataset_path ./data/processed/finance/short/train --epoch 6 --hidden_dim 32 --patch_len 4 --ts_encoder MoMe --use_bfloat16

CUDA_VISIBLE_DEVICES=1 python evaluate.py --task finance_forecast --in_len 134  --out_len 33 --dataset_path ./data/processed/finance/long/test --checkpoint_path output/FFLr/ts_encoder_epoch7.pt --output_dir ./output/FFLr --hidden_dim 32 --patch_len 4 --n_experts 4 --topk 2 --ts_encoder MoMe --modulation --router_modulation --instructor_query 3 --use_bfloat16 --eval_mode full_test

CUDA_VISIBLE_DEVICES=2 python train.py --instructor_query 3 --n_experts 4 --topk 2 --modulation --router_modulation --output_dir output/socialgood_momer  --task socialgood_forecast --in_len 14 --out_len 3 --dataset_path ./data/processed/TimeMMD/SocialGood/train --epoch 8 --hidden_dim 32 --patch_len 4 --ts_encoder MoMe --use_bfloat16


CUDA_VISIBLE_DEVICES=2 python train.py --instructor_query 3 --n_experts 4 --topk 2 --modulation --router_modulation --output_dir output/FFSr --task finance_forecast --in_len 312 --out_len 78 --dataset_path ./data/processed/finance/short/train --epoch 6 --hidden_dim 32 --patch_len 4 --ts_encoder MoMe --use_bfloat16

# Use evaluate.py

Test the the performance on the test set:
```
CUDA_VISIBLE_DEVICES=0 python evaluate.py --task socialgood_forecast --in_len 14  --out_len 3 --dataset_path ./data/processed/TimeMMD/SocialGood/test --checkpoint_path output/socialgood_momer/ts_encoder_epoch7.pt --output_dir ./output/Expert_Selection/socialgood_momer --hidden_dim 32 --patch_len 4 --n_experts 4 --topk 2 --ts_encoder MoMe --modulation --router_modulation --instructor_query 3 --use_bfloat16 --return_expert_selection --eval_mode full_test
```

Test the the performance on a single sample:
```
CUDA_VISIBLE_DEVICES=0 python evaluate.py --task socialgood_forecast --in_len 14  --out_len 3 --dataset_path ./data/processed/TimeMMD/SocialGood/test --checkpoint_path output/socialgood_momer/ts_encoder_epoch7.pt --output_dir ./output/Expert_Selection/socialgood_momer --hidden_dim 32 --patch_len 4 --n_experts 4 --topk 2 --ts_encoder MoMe --modulation --router_modulation --instructor_query 3 --use_bfloat16 --return_expert_selection --eval_mode random_sample --sample_seed 42
```

Test the expert assignment situation:
```
CUDA_VISIBLE_DEVICES=0 python evaluate.py --task socialgood_forecast --in_len 14  --out_len 3 --dataset_path ./data/processed/TimeMMD/SocialGood/test --checkpoint_path output/socialgood_momer/ts_encoder_epoch7.pt --output_dir ./output/Expert_Selection/socialgood_momer --hidden_dim 32 --patch_len 4 --n_experts 4 --topk 2 --ts_encoder MoMe --modulation --router_modulation --instructor_query 3 --use_bfloat16 --return_expert_selection --eval_mode expert_selection
```