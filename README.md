# Uncovering the Text Embedding in Text-to-Image Diffusion Models
### [Project Page](https://yuhuustc.github.io/UTE/)


This is the official implementation of the paper "Uncovering the Text Embedding in Text-to-Image Diffusion Models".



## Requirements
Our code is based on <a href="https://github.com/CompVis/stable-diffusion">stable-diffusion</a>. This project requires one GPU with 48GB memory. Please first clone the repository and build the environment:
```bash
git clone https://github.com/wuqiuche/DiffusionDisentanglement
cd DiffusionDisentanglement
conda env create -f environment.yaml
conda activate ldm
```

You will also need to download the pretrained stable-diffusion model:
```bash
mkdir models/ldm/stable-diffusion-v1
wget -O models/ldm/stable-diffusion-v1/model.ckpt https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
```

## Methods
# Controllable Image Editing via Text Embedding Manipulation.
![](./data/pipeline1.png)

# Semantic Directions in SVD of Text Embedding.
![](./data/pipeline2.png)


## Diverse operations on the text embedding via changing --mode
```bash
/bin/bash edit_all.sh
```

## Codes for swap editing on the generated mageNet-R-TI2I dataset.
```bash
/bin/bash edit_swap.sh
```

## Calculating the scores on mageNet-R-TI2I dataset.
```bash
/bin/bash score.sh
```

### [Editing samples on the mageNet-R-TI2I dataset](https://drive.google.com/drive/folders/1kFQDcPLX7gb9EAFcikVpXfn65vaY5iZx?usp=drive_link)




## Results
Object Replace, Action Edit, Fader Control, Style Transfer, and Semantic Directions.
![](./data/result.png)

## Parent Repository
This code is adopted from <a href="">https://github.com/CompVis/stable-diffusion</a> and <a href="">https://github.com/UCSB-NLP-Chang/DiffusionDisentanglement</a>.
