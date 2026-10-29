# Station-anchored targets on the 5 km store as short backfill jobs, one per
# fold (time unknown at 5 km; 1 h limit); done when the target variable exists in the store.
#   DATA_CFG=configs/data_vista_5km_main.yaml NO_EVAL=1 \
#     bash slurm/backfill_chain.sh slurm/tasks_anchor_5km.sh
Z=/scratch/11755/gurup12/downscaling-radiation/data_store_5km/radiation.zarr
A="TX_Austin_33_NW TX_Monahans_6_ENE TX_Palestine_6_WNW TX_Port_Aransas_32_NNE"
B="TX_Bronte_11_NNE TX_Muleshoe_19_S TX_Panther_Junction_2_N"
S="--start 2018-01-01"
TASKS=(
  "an5a|anchor build-target --exclude $A --fold a --fit-net uscrn $S|$Z/nsrdb_ghi_corr_a/zarr.json|01:00:00|3"
  "an5b|anchor build-target --exclude $B --fold b --fit-net uscrn $S|$Z/nsrdb_ghi_corr_b/zarr.json|01:00:00|3"
  "an52a|anchor build-target --exclude $(cat slurm/fold2_a.txt) --fold 2a $S|$Z/nsrdb_ghi_corr_2a/zarr.json|01:00:00|3"
  "an52b|anchor build-target --exclude $(cat slurm/fold2_b.txt) --fold 2b $S|$Z/nsrdb_ghi_corr_2b/zarr.json|01:00:00|3"
)
