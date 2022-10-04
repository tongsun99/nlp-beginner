python train.py --num_train_epochs 1 \
--seed 0 \
--batch_size 16 \
--model_weights epoch_1_dev_acc_71.9648_model_weights.bin \
--log_dir log/seed0_batch16_epoch1_eval.log \
--model_name bert-base-chinese \
--do_eval
