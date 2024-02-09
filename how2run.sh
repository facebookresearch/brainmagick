dora run dset.selections=[brennan2019] dset.n_recordings=4
dora grid nmi.main_table --dry_run --init
#then you have to find the signature from this above command
dora run -f 6e3bf7d7 optim.batch_size=16
dora run -f c5455d58 optim.batch_size=16
dora run -f 87a001d2 optim.batch_size=16
dora run -f 13767159 optim.batch_size=16
dora run -f c512a1a6 optim.batch_size=16
dora run -f ac6cdb20 optim.batch_size=16
dora grid nmi.main_table --dry_run --init
python -m scripts.run_eval_probs grid_name="nmi.main_table"
