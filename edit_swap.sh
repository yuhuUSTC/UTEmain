CUDA_VISIBLE_DEVICES='0' torchrun --nproc_per_node 1 --master_port 15605   scripts/edit_swap.py  \
                    --prompt data/imnetr-fake-ti2i.yaml \
                    --outdir outputs/target  \
