from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import json
from tqdm import tqdm
import os

if __name__=="__main__":
    os.system("nvidia-smi")
    model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = "/home/shegde23/val2014/"
    model.to(device)
    os.system("nvidia-smi")
    f = json.load(open('vqa.json'))
    for i in tqdm(range(len(f))):
        image = Image.open(data_dir+f[i]['file_name']).convert("RGB")
        answers = []
        for j in range(len(f[i]['questions'])):
            prompt = f[i]['questions'][j]
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
            outputs = model.generate(
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
            generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            answers.append(generated_text)
        f[i]['pred_answers'] = answers
        if(i==(5000-1) or i==(10000-1) or i==(15000-1)):
            x=json.dumps(f)
            with open('pred_vqa_intmd.json','w') as w:
                w.write(x)
    x = json.dumps(f)
    with open('pred_vqa1.json','w') as w:
            w.write(x)
