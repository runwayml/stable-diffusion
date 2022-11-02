import re
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

from eg_data_tools.data.group_regexps import REGEXP_SELECTORS


def get_regexps_by_group(group_regexps: Dict[str, List[str]], group_name: str) -> List[str]:
    patterns = []
    for group, regexp in group_regexps.items():
        if group_name in group:
            patterns.extend(regexp)
    return patterns

def img_name_match_to_pattern(img_name: str, patterns: List[str]) -> bool:
    for pattern in patterns:
        pattern = re.compile(pattern)
        if pattern.match(img_name):
            return True
    return False

def setup_log_file(file_path: Path):
    if file_path.exists():
        open(str(file_path), "w").close()
    file_path.write_text("FileName, Propmt\n", encoding="utf-8")

def nearest_multiple(x, m):
    return int(np.ceil(x / m) * m)

def add_random_polygon_on_mask(size: int, mask_rect_coef: int = 2) -> np.ndarray:
    empty_mask = np.zeros((size, size, 1), dtype=np.uint8)

    mask_size = int(size / random.randint(2, mask_rect_coef))
    height = mask_size
    width = mask_size
    start_x = random.randint(0, int(size - width))
    start_y = random.randint(0, int(size - height))

    end_x = int(start_x + width)
    end_y = int(start_y + height)

    mask = cv2.rectangle(empty_mask, (start_x, start_y), (end_x, end_y), (1), -1)
    return mask

def apply_on_src_image(
    src_image: Image, 
    generated_img: Image, 
    generated_mask: Image, 
    x: int, 
    y: int, 
    w: int, 
    h: int
) -> Tuple[np.ndarray, np.ndarray]:
    src_image = np.array(src_image)
    generated_img = np.array(generated_img)
    generated_mask = np.array(generated_mask)
    mask = np.zeros((src_image.shape[0], src_image.shape[1]), dtype=np.uint8)
    
    src_image[y:y+h, x:x+w] = generated_img
    mask[y:y+h, x:x+w] = generated_mask
    
    src_image = Image.fromarray(src_image)
    mask = Image.fromarray(mask, 'L')
    return src_image, mask
    
def generate_random_prompt(
    base_words: str,
    prompt_parts: List[str]
):
    prompt_list = [base_words]
    for _ in range(random.randint(2, 15)):
        part = random.choice(prompt_parts)
        if part in prompt_list:
            continue
        prompt_list.append(part)
    
    return ", ".join(prompt_list)


def run_sd_inference(
    device_id: int,
    src_images_dir: str,
    masks_dir: str,
    generated_images_dir: str,
    prompts_file_path: str,
    logs_file_path: str,
    crop_size: int = 640,
    num_inference_steps: int = 60,
    generate_prompt: bool = False,
    base_prompt: str = "person, man",
    max_generated_objects_per_image: int = 1,
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
    
    src_images_dir = Path(src_images_dir)
    masks_dir = Path(masks_dir)
    generated_images_dir = Path(generated_images_dir)
    prompts_file_path = Path(prompts_file_path)
    logs_file_path = Path(logs_file_path)
    masks_dir.mkdir(exist_ok=True)
    generated_images_dir.mkdir(exist_ok=True)
    
    transform = T.RandomCrop(crop_size)
    
    prompts = [prompt.strip() for prompt in prompts_file_path.read_text().splitlines()]
    setup_log_file(logs_file_path)

    if regexpx_group is not None:
        patterns = get_regexps_by_group(REGEXP_SELECTORS.ALL_REGEXPS, regexpx_group)
        img_pathes = []
        for img_path in src_images_dir.iterdir():
            if img_name_match_to_pattern(img_path.name, patterns):
                img_pathes.append(img_path)
                continue
    else:
        img_pathes = [img_path for img_path in src_images_dir.iterdir()]
        
    for img_path in random.choices(img_pathes, k=len(img_pathes)):
        img_uid = str(uuid.uuid4())[:8]
        generated_img_basename = f"{img_path.stem}_{img_uid}"
        
        src_image = Image.open(img_path)
        if src_image.height < crop_size or src_image.width < crop_size:
            continue
        height, width = (nearest_multiple(src_image.height, multiple_coef), nearest_multiple(src_image.width, multiple_coef))
        src_image = src_image.resize((width, height))
        
        if generate_prompt:
            prompt = generate_random_prompt(base_prompt, prompts)
        else:
            prompt = random.choice(prompts)
            
        for num_polygons in range(random.randint(1, max_generated_objects_per_image)):
            y1, x1, h, w = transform.get_params(src_image, (crop_size, crop_size))

            img = crop(src_image, y1, x1, h, w)
            mask = add_random_polygon_on_mask(crop_size, 2) # для огня троечку сюда
            mask = Image.fromarray(np.uint8(mask[:, :, 0] * 255) , 'L')
            
            with torch.autocast(device_type=device.type):
                img = pipe(
                    prompt=prompt, 
                    image=img, 
                    mask_image=mask, 
                    height=crop_size, 
                    width=crop_size,
                    num_inference_steps=num_inference_steps,
                ).images[0]
            
            src_image, mask = apply_on_src_image(src_image, img, mask, x1, y1, w, h)
            
            
            mask.save(masks_dir / f"{generated_img_basename}_{num_polygons}.png")
        src_image.save(generated_images_dir / f"{generated_img_basename}.jpg")
        
        with logs_file_path.open('a', encoding='utf-8') as f:
            f.write(f'{generated_img_basename}.jpg, "{prompt}"\n')
    

def parce_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()
    args.add_argument('--device_id', type=int, default=2)
    args.add_argument('--src_images_dir', type=str, required=True)
    args.add_argument('--masks_dir', type=str, required=True)
    args.add_argument('--generated_images_dir', type=str, required=True)
    args.add_argument('--prompts_file_path', type=str, required=True)
    args.add_argument('--logs_file_path', type=str, required=True)
    args.add_argument('--crop_size', type=int, default=640)
    args.add_argument('--num_infer_steps', type=int, default=60)
    args.add_argument('--generate_prompt', type=bool, default=False)
    args.add_argument('--base_prompt', type=str, default="person, man")
    args.add_argument('--max_gen_obj_per_img', type=int, default=1)
    args.add_argument('--regexpx_group', type=str, default=None, required=False)
    return args.parse_args()



if __name__ == "__main__":

    args = parce_args()
    
    run_sd_inference(
        device_id=args.device_id,
        src_images_dir=args.src_images_dir,
        masks_dir=args.masks_dir,
        generated_images_dir=args.generated_images_dir,
        prompts_file_path=args.prompts_file_path,
        logs_file_path=args.logs_file_path,
        crop_size=args.crop_size,
        num_inference_steps=args.num_infer_steps,
        generate_prompt=args.generate_prompt,
        base_prompt=args.base_prompt,
        max_generated_objects_per_image=args.max_gen_obj_per_img,
        regexpx_group=args.regexpx_group
    )
