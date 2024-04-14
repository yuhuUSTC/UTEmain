export PYTHONPATH=$PYTHONPATH:/mnt/workspace/workgroup/yuhu/code/TextEmbedding
CUDA_VISIBLE_DEVICES='2' torchrun --nproc_per_node 1 --master_port 15625   /mnt/workspace/workgroup/yuhu/code/TextEmbedding/scripts/disentangle.py  \
                    --c1 "A photo of dog" \
                    --c2 "A photo of cat" \
                    --mode "swap" \
                    --seed 42 \
                    --outdir /mnt/workspace/workgroup/yuhu/code/TextEmbedding/outputs
