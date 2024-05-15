CUDA_VISIBLE_DEVICES='0' HF_ENDPOINT=https://hf-mirror.com torchrun --nproc_per_node 1 --master_port 15605 ClIPDINO.py \
                        --textpath_edit data/imnetr-fake-ti2i.yaml \
                        --imgpath_source Samples/Original       \
                        --imgpath_edit Samples/P2P   \
