
#  Shell script to find available ports
# comm -23 \
# <(seq "$FROM" "$TO") \
# <(ss -tan | awk '{print $4}' | cut -d':' -f2 | grep '[0-9]\{1,5\}' | sort -n | uniq) \
# | shuf | head -n "$HOWMANY"

# get one port between 49152 65535
# MASTER_PORT="$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)"

# MASTER_PORT="29517"
# MASTER_ADDR="localhost"

torchrun --standalone --nnodes=1 --nproc_per_node=8  train3_ddp_newdata.py
# torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:9400 train3_ddp.py
