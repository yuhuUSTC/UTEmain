import argparse, os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, CLIPModel
from transformers import AutoImageProcessor, AutoModel
import torch.nn as nn

             
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--imgpath_source",
        type=str,
        nargs="?",
        default="",
    )
    parser.add_argument(
        "--imgpath_edit",
        type=str,
        nargs="?",
        default="",
    )
    parser.add_argument(
        "--textpath_edit",
        type=str,
        nargs="?",
        default="",
    )
    opt = parser.parse_args()




    def readprompt(prompt):
        with open(prompt, 'r') as file:
            content = file.readlines()

        source_prompts = []
        target_prompts = []
        for line in content:
            if 'source_prompt:' in line:
                source_prompt = line.split('source_prompt:')[1].strip()
                source_prompts.append(source_prompt)
            if line.strip().startswith("-"):
                target_prompts.append(line.strip()[2:])  # 提取prompt并添加到prompts列表中（去掉"- "）

        return source_prompts, target_prompts

    def readimage(folder_path):
        def numerical_sort(file_path):
            """
            排序函数，根据文件名中的数字进行排序。
            """
            parts = os.path.split(file_path)
            file_name = parts[-1]
            # 假设文件名是由数字和扩展名组成（例如 "123.png"）
            # 使用 os.path.splitext 分离文件名和扩展，然后转换为整数进行排序
            base_name = os.path.splitext(file_name)[0]
            return int(base_name)

        supported_formats = ('.jpg', '.jpeg', '.png', '.gif')

        image_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(supported_formats):
                    full_path = os.path.join(root, file)
                    image_files.append(full_path)

        image_files.sort(key=numerical_sort)
        return image_files



    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    
    dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

    _, prompts = readprompt(opt.textpath_edit)
    img_sources = readimage(opt.imgpath_source)
    img_edits = readimage(opt.imgpath_edit)
    
    # print(prompts)
    # print(img_sources)
    # print(img_edits)
    
    clip_score = 0
    dino_score = 0
    for i in range(len(img_sources)):
        img_source = img_sources[i]

        img_source = Image.open(img_source)
        with torch.no_grad():
            img_source = dino_processor(images=img_source, return_tensors="pt").to(device)
            img_source = dino_model(**img_source)
            img_source_features = img_source.last_hidden_state
            img_source_features = img_source_features.mean(dim=1)
        
        for j in range(5):
            index = i * 5 + j

            prompt = prompts[index]
            with torch.no_grad():
                prompt = clip_tokenizer([prompt], padding=True, return_tensors="pt").to(device)
                prompt_features = clip_model.get_text_features(**prompt)
            
            img_edit = img_edits[index]
            img_edit = Image.open(img_edit)
            with torch.no_grad():
                img_edit = clip_processor(images=img_edit, return_tensors="pt").to(device)
                img_edit_features = clip_model.get_image_features(**img_edit)

            cos = nn.CosineSimilarity(dim=0)
            sim = cos(img_edit_features[0],prompt_features[0]).item()
            sim = (sim+1)/2
            clip_score += sim


            img_edit = img_edits[index]
            img_edit = Image.open(img_edit)
            with torch.no_grad():
                img_edit = dino_processor(images=img_edit, return_tensors="pt").to(device)
                img_edit = dino_model(**img_edit)
                img_edit_features = img_edit.last_hidden_state
                img_edit_features = img_edit_features.mean(dim=1)

            cos = nn.CosineSimilarity(dim=0)
            sim = cos(img_source_features[0],img_edit_features[0]).item()
            sim = (sim+1)/2
            dino_score += sim

    
    average_clip_score = clip_score / len(img_edits)
    print('CLIP Similarity:', average_clip_score)
    average_dino_score = dino_score / len(img_edits)
    print('DINOV2 Similarity:', average_dino_score)

