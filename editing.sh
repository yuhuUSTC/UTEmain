CUDA_VISIBLE_DEVICES='0' torchrun --nproc_per_node 1 --master_port 15605   scripts/edit.py  \
                    --prompt /mnt/workspace/workgroup/yuhu/code/TextEmbedding/data/imnetr-fake-ti2i.yaml \
                    --outdir /mnt/workspace/workgroup/yuhu/code/TextEmbedding/outputs/target  \
