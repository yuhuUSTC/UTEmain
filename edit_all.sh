CUDA_VISIBLE_DEVICES='0' torchrun --nproc_per_node 1 --master_port 15605   scripts/edit_all.py  \
                    --c1 "A photo of dog" \
                    --c2 "A photo of cat" \
                    --mode "swap" \
                    --seed 42 \
                    --outdir /mnt/workspace/workgroup/yuhu/code/TextEmbedding/outputs
