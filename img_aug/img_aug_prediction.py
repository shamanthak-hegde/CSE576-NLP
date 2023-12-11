

from pycocotools.coco import COCO
import numpy as np
import pandas as pd
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from PIL import Image
import requests
import albumentations as A
import torch
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
import argparse
import os



#### Prediction Functions
def pred_instructblip(img, prompt):
    inputs = ib_processor(images=img, text=prompt, return_tensors='pt').to(device)
    outputs = ib_model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
    )
    generated_text = ib_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    return generated_text


def pred_openflamingo(img, prompt):
    # OpenFlamingi example Q/As
    demo_image_one = coco.loadImgs([qa_df.loc[0, 'img_id']])[0]
    demo_image_one = Image.open(requests.get(demo_image_one['coco_url'], stream=True).raw)
    demo_question_one = qa_df.loc[0, 'question']
    demo_answer_one = qa_df.loc[0, 'answer']
    
    demo_image_two = coco.loadImgs([qa_df.loc[1, 'img_id']])[0]
    demo_image_two = Image.open(requests.get(demo_image_two['coco_url'], stream=True).raw)
    demo_question_two = qa_df.loc[1, 'question']
    demo_answer_two = qa_df.loc[1, 'answer']
    
    vision_x = [fla_processor(demo_image_one).unsqueeze(0), fla_processor(demo_image_two).unsqueeze(0), fla_processor(img).unsqueeze(0)]
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)
    
    lang_x = fla_tokenizer(
        [f"<image>Question: {demo_question_one}? Answer: {demo_answer_one}<|endofchunk|><image>Question: {demo_question_two}? Answer: {demo_answer_two}<|endofchunk|><image>Question: {prompt} Answer: "],
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


def get_cocoqa():
    # coco qa dataset
    qa_df = pd.concat([
        pd.DataFrame({
            'img_id': pd.read_fwf('data/coco_qa/train/img_ids.txt', header=None)[0],
            'question': pd.read_fwf('data/coco_qa/train/questions.txt', header=None)[0],
            'answer': pd.read_fwf('data/coco_qa/train/answers.txt', header=None)[0],
            'type': pd.read_fwf('data/coco_qa/train/types.txt', header=None)[0],     # 0 -> object, 1 -> number, 2 -> color, 3 -> location
        }),
        pd.DataFrame({
            'img_id': pd.read_fwf('data/coco_qa/test/img_ids.txt', header=None)[0],
            'question': pd.read_fwf('data/coco_qa/test/questions.txt', header=None)[0],
            'answer': pd.read_fwf('data/coco_qa/test/answers.txt', header=None)[0],
            'type': pd.read_fwf('data/coco_qa/test/types.txt', header=None)[0],     
        })
    ])

    # get coco 2014 val data from coco qa
    qa_df = qa_df[qa_df['img_id'].isin(coco.getImgIds())]
    return qa_df




parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str)
args = parser.parse_args()
model_name = args.model_name

#### set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# coco 2014 val dataset
annFile = 'data/coco/annotations/instances_val2014.json'
coco=COCO(annFile)



#### Init Model
if model_name == 'ib':
    #### load coco_qa
    # get the unpredicted samples if some have already been predicted
    if os.path.exists('output/image_aug_analysis_cocoqa_val2014_ib_only.csv'):
        qa_df = pd.read_csv('output/image_aug_analysis_cocoqa_val2014_ib_only.csv')
        qa_df_unpredicted = qa_df[qa_df['ib_original'].isna()]          # get unpredicted rows
    else:   # if nothing predicted yet get all samples
        qa_df = get_cocoqa()

    #### Init InstructBlip
    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
    import torch
    from PIL import Image
    import requests
    ib_model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
    ib_processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
    ib_model.to(device)

    #### prediction loop
    for i, row in qa_df_unpredicted.iterrows():
        try:
            img = coco.loadImgs([row['img_id']])[0]
            img = io.imread(img['coco_url'])
            prompt = row['question'] + '?'
            
            ## Original Image
            qa_df.at[i, 'ib_original'] = str(pred_instructblip(img, prompt))
        
            ## Blurred small
            blur1 = A.ReplayCompose([A.Blur(blur_limit=[15]*2, always_apply=True)])
            img_blur1 = blur1(image=img)['image']
            qa_df.at[i, 'ib_blurSm'] = str(pred_instructblip(img_blur1, prompt))
            
            ## Blurred medium
            blur2 = A.ReplayCompose([A.Blur(blur_limit=[20]*2, always_apply=True)])
            img_blur2 = blur2(image=img)['image']
            qa_df.at[i, 'ib_blurMd'] = str(pred_instructblip(img_blur2, prompt))
        
            ## Blurred strong
            blur3= A.ReplayCompose([A.Blur(blur_limit=[22]*2, always_apply=True)])
            img_blur3 = blur3(image=img)['image']
            qa_df.at[i, 'ib_blurLg'] = str(pred_instructblip(img_blur3, prompt))
        
            ## Channell dropout
            chnl_drop = A.ReplayCompose([A.ChannelDropout(always_apply=True)])
            img_chnl_drop = chnl_drop(image=img)['image']
            qa_df.at[i, 'ib_chnlDrp'] = str(pred_instructblip(img_chnl_drop, prompt))
            
            ## Solarize
            img_solar = A.augmentations.functional.solarize (img, threshold=128)
            qa_df.at[i, 'ib_solar'] = str(pred_instructblip(img_solar, prompt))
        
            ## Elastic transform - distorion
            elastic = A.ReplayCompose([A.ElasticTransform(alpha=5, sigma=50, always_apply=True)])
            img_elastic = elastic(image=img)['image']
            qa_df.at[i, 'ib_elastic'] = str(pred_instructblip(img_elastic, prompt))
        
            if i % 101 == 0:
                print(f'Progress: {i} / {len(qa_df)}')
                qa_df.to_csv('output/image_aug_analysis_cocoqa_val2014_ib_only.csv', index=None)
        except Exception as e:
            print(e)

    ### save all results
    qa_df.to_csv('output/image_aug_analysis_cocoqa_val2014_ib_only.csv', index=None)

else:
    #### load coco_qa
    # get the unpredicted samples if some have already been predicted
    if os.path.exists('output/image_aug_analysis_cocoqa_val2014_fla_only.csv'):
        qa_df = pd.read_csv('output/image_aug_analysis_cocoqa_val2014_fla_only.csv')
        qa_df_unpredicted = qa_df[qa_df['fla_original'].isna()]          # get unpredicted rows
    else: # if nothing predicted yet get all samples
        qa_df = get_cocoqa()

    from open_flamingo import create_model_and_transforms
    from huggingface_hub import hf_hub_download
    import torch
    fla_model, fla_processor, fla_tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        # lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
        # tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
        lang_encoder_path="anas-awadalla/mpt-7b",
        tokenizer_path="anas-awadalla/mpt-7b",
        cross_attn_every_n_layers=1,
        # cache_dir="PATH/TO/CACHE/DIR"  # Defaults to ~/.cache
    )
    fla_tokenizer.padding_side = "left" # For generation padding tokens should be on the left
    # checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
    checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B-vitl-mpt7b", "checkpoint.pt")
    fla_model.load_state_dict(torch.load(checkpoint_path), strict=False)
    fla_model.to(device)


    for i, row in qa_df_unpredicted.iterrows():
        try:
            img = coco.loadImgs([row['img_id']])[0]
            img = io.imread(img['coco_url'])
            prompt = row['question'] + '?'
            
            ## Original Image
            qa_df.at[i, 'fla_original'] = str(pred_openflamingo(Image.fromarray(img), prompt))
        
            ## Blurred small
            blur1 = A.ReplayCompose([A.Blur(blur_limit=[15]*2, always_apply=True)])
            img_blur1 = blur1(image=img)['image']
            qa_df.at[i, 'fla_blurSm'] = str(pred_openflamingo(Image.fromarray(img_blur1), prompt))
            
            ## Blurred medium
            blur2 = A.ReplayCompose([A.Blur(blur_limit=[20]*2, always_apply=True)])
            img_blur2 = blur2(image=img)['image']
            qa_df.at[i, 'fla_blurMd'] = str(pred_openflamingo(Image.fromarray(img_blur2), prompt))
        
            ## Blurred strong
            blur3= A.ReplayCompose([A.Blur(blur_limit=[22]*2, always_apply=True)])
            img_blur3 = blur3(image=img)['image']
            qa_df.at[i, 'fla_blurLg'] = str(pred_openflamingo(Image.fromarray(img_blur3), prompt))
        
            ## Channell dropout
            chnl_drop = A.ReplayCompose([A.ChannelDropout(always_apply=True)])
            img_chnl_drop = chnl_drop(image=img)['image']
            qa_df.at[i, 'fla_chnlDrp'] = str(pred_openflamingo(Image.fromarray(img_chnl_drop), prompt))
            
            ## Solarize
            img_solar = A.augmentations.functional.solarize (img, threshold=128)
            qa_df.at[i, 'fla_solar'] = str(pred_openflamingo(Image.fromarray(img_solar), prompt))
        
            ## Elastic transform - distorion
            elastic = A.ReplayCompose([A.ElasticTransform(alpha=5, sigma=50, always_apply=True)])
            img_elastic = elastic(image=img)['image']
            qa_df.at[i, 'fla_elastic'] = str(pred_openflamingo(Image.fromarray(img_elastic), prompt))
        
            if i % 101 == 0:
                print(f'Progress: {i} / {len(qa_df)}')
                qa_df.to_csv('output/image_aug_analysis_cocoqa_val2014_fla_only.csv', index=None)
        except Exception as e:
            print(e)

    ### save all results
    qa_df.to_csv('output/image_aug_analysis_cocoqa_val2014_fla_only.csv', index=None)
            





