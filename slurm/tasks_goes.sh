# GOES era (2018-01 .. 2019-06 train, 2019-H2 val, 2020-21 test). A, B and C on the
# SAME window so the comparison isolates the inputs. Swin uses the SSIM default.
#   DATA_CFG=configs/data_vista_goes.yaml bash slurm/backfill_chain.sh slurm/tasks_goes.sh
R=runs/data_store_goes
TASKS=(
  "gxA|xgbx --track A|$R/xgb_A/meta.json|00:40:00|3"
  "gxB|xgbx --track B|$R/xgb_B/meta.json|00:40:00|3"
  "gxC|xgbx --track C|$R/xgb_C/meta.json|00:40:00|3"
  "gsA|swinx --track A|$R/swin_A/meta.json|00:35:00|15"
  "gsB|swinx --track B|$R/swin_B/meta.json|00:35:00|15"
  "gsC|swinx --track C|$R/swin_C/meta.json|00:35:00|15"
)
