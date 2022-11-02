import cv2
import uuid
import torch
import random
import argparse
import numpy as np

from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict
from torchvision import transforms as T
from torchvision.transforms.functional import crop
from diffusers import StableDiffusionInpaintPipeline

from inpainting import setup_log_file, nearest_multiple, apply_on_src_image, generate_random_prompt, get_regexps_by_group, img_name_match_to_pattern
from eg_data_tools.annotation_processing.coco_utils.coco_read_write import open_coco
from eg_data_tools.data.group_regexps import REGEXP_SELECTORS


def get_crop_coords(
    x1: int, y1: int, 
    x2: int, y2: int, 
    crop_size: int, 
    img_height: int, 
    img_width: int
):
    mid_x = int((x1 + x2) / 2)
    mid_y = int((y1 + y2) / 2)
    
    crop_x1 = max(0, mid_x - crop_size // 2)
    if crop_x1 == 0:
        crop_x2 = crop_size
    else:
        crop_x2 = min(mid_x + crop_size // 2, img_width)
    if crop_x2 == img_width:
        crop_x1 = img_width - crop_size
    
    
    crop_y1 = max(0, mid_y - crop_size // 2)
    if crop_y1 == 0:
        crop_y2 = crop_size
    else:
        crop_y2 = min(mid_y + crop_size // 2, img_height)
    if crop_y2 == img_height:
        crop_y1 = img_height - crop_size
    
    return crop_x1, crop_y1, crop_x2, crop_y2

def get_mask_with_bbox(x1, y1, x2, y2, img_height, img_width):
    mask = np.zeros((img_height, img_width, 1), dtype=np.uint8)
    mask = cv2.rectangle(mask, (x1, y1), (x2, y2), (1), -1)
    return mask


def run_sd_inpainting(
    src_images_dir: str,
    coco_ann_path: str,
    inpainted_images_dir: str,
    prompts_file_path: str,
    logs_file_path: str,
    crop_size: int,
    generate_prompt: bool = True,
    base_prompt: str = 'person, man',
    num_inference_steps: int = 60,
    device_id: int = 0,
    multiple_coef: int = 8,
    regexpx_group: str = None
):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token="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"
    )
    
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else 'cpu')
    pipe.to(device)
    
    crop_size = nearest_multiple(crop_size, multiple_coef)
    src_images_dir = Path(src_images_dir)
    prompts_file_path = Path(prompts_file_path)
    logs_file_path = Path(logs_file_path)
    inpainted_images_dir = Path(inpainted_images_dir)
    inpainted_images_dir.mkdir(exist_ok=True)
    
    prompts = [prompt.strip() for prompt in prompts_file_path.read_text().splitlines()]
    setup_log_file(logs_file_path)
    limages = open_coco(coco_ann_path)
    
    if regexpx_group is not None:
        patterns = get_regexps_by_group(REGEXP_SELECTORS.ALL_REGEXPS, regexpx_group)
        img_names = []
        for img_path in src_images_dir.iterdir():
            if img_name_match_to_pattern(img_path.name, patterns):
                img_names.append(img_path.name)
                continue
    else:
        img_names = [img_path.name for img_path in src_images_dir.iterdir()]
        
    for limage in random.choices(limages, k=len(limages)):
        if limage.name not in img_names:
            continue
        
        img_path = src_images_dir / limage.name
        img_uid = str(uuid.uuid4())[:8]
        generated_img_basename = f"{img_path.stem}_{img_uid}"
        if generate_prompt:
            prompt = generate_random_prompt(base_prompt, prompts)
        else:
            prompt = random.choice(prompts)
        
        image = Image.open(img_path)
        for bbox in limage.bbox_list:
            x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
            
            crop_x1, crop_y1, crop_x2, crop_y2 = get_crop_coords(
                x1, 
                y1, 
                x2, 
                y2, 
                crop_size, 
                image.height, 
                image.width
            )
            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1
            
            mask = get_mask_with_bbox(x1, y1, x2, y2, image.height, image.width)
            mask = Image.fromarray(np.uint8(mask[:, :, 0] * 255) , 'L')
            mask = crop(mask, crop_y1, crop_x1, crop_h, crop_w)
            img = crop(image, crop_y1, crop_x1, crop_h, crop_w)
            
            with torch.autocast(device_type=device.type):
                img = pipe(
                    prompt=prompt, 
                    image=img, 
                    mask_image=mask, 
                    height=crop_size, 
                    width=crop_size,
                    num_inference_steps=num_inference_steps,
                ).images[0]
            
            image, mask = apply_on_src_image(image, img, mask, crop_x1, crop_y1, crop_w, crop_h)

        image.save(inpainted_images_dir / f'{generated_img_basename}.jpg')
        
        with logs_file_path.open('a', encoding='utf-8') as f:
            f.write(f'{generated_img_basename}.jpg, "{prompt}"\n')

def parce_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()
    args.add_argument('--device_id', type=int, default=2)
    args.add_argument('--src_images_dir', type=str, required=True)
    args.add_argument('--coco_ann_path', type=str, required=True)
    args.add_argument('--generated_images_dir', type=str, required=True)
    args.add_argument('--prompts_file_path', type=str, required=True)
    args.add_argument('--logs_file_path', type=str, required=True)
    args.add_argument('--crop_size', type=int, default=640)
    args.add_argument('--num_infer_steps', type=int, default=60)
    args.add_argument('--generate_prompt', type=bool, default=False)
    args.add_argument('--base_prompt', type=str, default="person, man")
    args.add_argument('--regexpx_group', type=str, default=None, required=False)
    return args.parse_args()


if __name__ == "__main__":
    args = parce_args()
    
    # run_sd_inpainting(
    #     device_id=0,
    #     src_images_dir='/media/data2/vv/dvc_datasets/dataset_ppe/gerdau_hardhat_coco/images',
    #     coco_ann_path='/media/data2/vv/tasks/2022_10_31_inpaint_heads_gerdau/coco_ann.json',
    #     inpainted_images_dir='/media/data2/vv/tasks/2022_10_31_inpaint_heads_gerdau/generated_images',
    #     prompts_file_path='/media/data2/au/stable_diff_seah/filtered_prompts.txt',
    #     logs_file_path='/media/data2/vv/tasks/2022_10_31_inpaint_heads_gerdau/logs.csv',
    #     crop_size=512,
    # )
    run_sd_inpainting(
        src_images_dir=args.src_images_dir,
        coco_ann_path=args.coco_ann_path,
        inpainted_images_dir=args.generated_images_dir,
        prompts_file_path=args.prompts_file_path,
        logs_file_path=args.logs_file_path,
        crop_size=args.crop_size,
        generate_prompt=args.generate_prompt,
        base_prompt=args.base_prompt,
        num_inference_steps=args.num_infer_steps,
        device_id=args.device_id,
        regexpx_group=args.regexpx_group,
    )










