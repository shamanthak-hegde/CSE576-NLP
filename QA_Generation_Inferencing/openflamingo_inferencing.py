from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
import os
from PIL import Image
import requests
import pandas as pd
from pycocotools.coco import COCO
import json
import albumentations as A
import skimage.io as io
import numpy as np

fla_model, fla_processor, fla_tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
    tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
    cross_attn_every_n_layers=1,
    # cache_dir="PATH/TO/CACHE/DIR"  # Defaults to ~/.cache
)
fla_tokenizer.padding_side = "left" # For generation padding tokens should be on the left
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
fla_model.load_state_dict(torch.load(checkpoint_path), strict=False)

fla_model.to(device)

annFile = "/scratch/atherala/captions_val2014.json"
coco=COCO(annFile)
json_fp = open(r"/scratch/atherala/captions_val2014.json")
coco_captions = json.load(json_fp)
excel_path = r"/scratch/atherala/questions_df_v3.xlsx"
questions_df = pd.read_excel(excel_path)
ofa_answers_list = []
ofa_blur_answers_list = []
ofa_solarise_answers_list = []
ofa_equalize_answers_list = []
ofa_emboss_answers_list = []

def pred_openflamingo(img, prompt):

    
    vision_x = [ fla_processor(img).unsqueeze(0)]
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)
    
    lang_x = fla_tokenizer(
        [f"<image>Question: {prompt} Answer: "],
        return_tensors="pt",
    )
    
    fla_generated_text = fla_model.generate(
        vision_x=vision_x.to(device),
        lang_x=lang_x["input_ids"].to(device),
        attention_mask=lang_x["attention_mask"].to(device),
        max_new_tokens=20,
        num_beams=3,
    )
    fla_generated_text = fla_tokenizer.decode(fla_generated_text[0])
    # get the answer without the examples
    fla_generated_text = fla_generated_text.split('Question: ')[-1]
    if len(fla_generated_text.split('Answer:')) > 1:
        fla_generated_text = fla_generated_text.split('Answer:')[-1].replace('<|endofchunk|>', '').replace('.', '')  
        
    return fla_generated_text

for row in questions_df.iterrows():
    try:
        question = row[1]["Mistral_Questions"]
        image_id = row[1]["Image_Ids"]
        print("working in Image Id: ",image_id)
        img = coco.loadImgs(int(image_id))
        image = Image.open(requests.get(img[0]['coco_url'], stream=True).raw)
        # image = io.imread(img[0]['coco_url'],stream = True).raw
        ibp_prompt = question

        #original Image
        original_ans = pred_openflamingo(image,ibp_prompt)

        #medium blurred image
        blur = A.ReplayCompose([A.Blur(blur_limit=[20]*2, always_apply=True)])
        img_blur = blur(image=np.array(image))['image']
        blurred_ans = pred_openflamingo(Image.fromarray(img_blur),ibp_prompt)

        #equalized image
        equalize = A.augmentations.transforms.Equalize (mode='cv', by_channels=True, mask=None, mask_params=(), always_apply=False, p=0.5)
        img_equalize = equalize(image=np.array(image))["image"]
        equalized_ans = pred_openflamingo(Image.fromarray(img_equalize),ibp_prompt)

        #solarized image
        solarize =A.augmentations.transforms.Solarize (threshold=128, always_apply=False, p=0.5)
        img_solarize = solarize(image=np.array(image))["image"]
        solarized_ans = pred_openflamingo(Image.fromarray(img_solarize),ibp_prompt)

        #embossed image
        emboss = A.augmentations.transforms.Emboss (alpha=(0.2, 0.5), strength=(0.2, 0.7), always_apply=False, p=0.5)
        img_emboss = emboss(image=np.array(image))["image"]
        embossed_ans = pred_openflamingo(Image.fromarray(img_emboss),ibp_prompt)


        ofa_answers_list.append(original_ans)
        ofa_blur_answers_list.append(blurred_ans)
        ofa_solarise_answers_list.append(solarized_ans)
        ofa_equalize_answers_list.append(equalized_ans)
        ofa_emboss_answers_list.append(embossed_ans)
    
    except KeyError:
        error_msg = "error in image"
        ofa_answers_list.append(error_msg)
        ofa_blur_answers_list.append(error_msg)
        ofa_solarise_answers_list.append(error_msg)
        ofa_equalize_answers_list.append(error_msg)
        ofa_emboss_answers_list.append(error_msg)




questions_df["OFA_Answers"] = ofa_answers_list
questions_df["Blurred_Image_OFA_Answers"] = ofa_blur_answers_list
questions_df["Embossed_Image_OFA_Answers"] = ofa_emboss_answers_list
questions_df["Solarized_Image_OFA_Answers"] = ofa_solarise_answers_list
questions_df["Equalized_Image_OFA_Answers"] = ofa_equalize_answers_list
questions_df.to_csv("ofa_results_final.csv")