CUDA_VISIBLE_DEVICES=6 python train_distill_bert.py --output_dir ../checkpoints_mnli/bert_focal1_lr5_epoch3_seed111 --do_train --do_eval --mode focal_loss --seed 111 --which_bias hans --num_train_epochs 3 --focal_loss_gamma 1.0

CUDA_VISIBLE_DEVICES=5 python train_distill_bert.py --output_dir ../checkpoints_mnli/bert_focal2_lr5_epoch3_seed111 --do_train --do_eval --mode focal_loss --seed 111 --which_bias hans --num_train_epochs 3 --focal_loss_gamma 2.0
