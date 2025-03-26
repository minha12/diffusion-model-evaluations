from diffusers import (
    StableDiffusion3ControlNetPipeline, 
    SD3ControlNetModel,
    StableDiffusionControlNetPipeline, 
    ControlNetModel
)
from diffusers.utils import load_image
import torch
from accelerate import Accelerator
import os
from glob import glob
from tqdm import tqdm
import fire

def inference(
    condition_dir,
    prompts_file,
    output_dir="./outputs",
    batch_size=2,
    seed=0,
    steps=20,
    resolution=1024,
    base_model_path="/home/ubuntu/models/stabilityai/stable-diffusion-3-medium-diffusers",
    controlnet_path="/home/ubuntu/models/trained-model/sd3-controlnet",
    model_type="sd3"
):
    """
    Process a batch of images using StableDiffusion ControlNet.
    
    Args:
        condition_dir: Directory containing condition images
        prompts_file: Text file with prompts (one per line)
        output_dir: Directory to save generated images
        batch_size: Batch size for processing
        seed: Random seed for generation
        steps: Number of inference steps
        resolution: Image resolution
        base_model_path: Path to the base model
        controlnet_path: Path to the ControlNet model
        model_type: 'sd2' or 'sd3'
        checkpoint_path: Optional path to specific checkpoint file
    """
    # Initialize accelerator with mixed precision
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    
    if accelerator.is_main_process:
        print(f"Using device: {device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load condition images
    image_paths = sorted(glob(os.path.join(condition_dir, "*.png")) + 
                         glob(os.path.join(condition_dir, "*.jpg")) + 
                         glob(os.path.join(condition_dir, "*.jpeg")))
    
    if not image_paths:
        raise ValueError(f"No images found in {condition_dir}")
    
    # Load prompts
    with open(prompts_file, "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    
    # Check if the number of prompts matches the number of images
    if len(prompts) != len(image_paths):
        raise ValueError(f"Number of prompts ({len(prompts)}) doesn't match number of images ({len(image_paths)})")
    
    if accelerator.is_main_process:
        print(f"Found {len(image_paths)} images and {len(prompts)} prompts")
        print(f"Loading {model_type} model...")
    
    # Load appropriate model based on model_type
    if model_type.lower() == "sd3":
        # SD3 ControlNet pipeline
        controlnet = SD3ControlNetModel.from_pretrained(
            controlnet_path, 
            torch_dtype=torch.float16,
            local_files_only=True,
            trust_remote_code=True
        ).to(device)
        
        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            base_model_path, 
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to(device)
        
    elif model_type.lower() == "sd2":
        # SD2.1 ControlNet pipeline
        
        controlnet = ControlNetModel.from_pretrained(
            controlnet_path,
            local_files_only=True,
            torch_dtype=torch.float16
        ).to(device)
        
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model_path,
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to(device)
        
        # Enable memory optimization for SD2.1
        #pipe.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'sd2' or 'sd3'.")
    
    # Prepare pipeline with accelerator
    pipe = accelerator.prepare(pipe)
    
    # Create generator on the same device
    generator = torch.Generator(device=device).manual_seed(seed)
    
    if accelerator.is_main_process:
        print(f"Models loaded successfully on {device}")
    
    # Get process info for data partitioning
    process_idx = accelerator.process_index
    num_processes = accelerator.num_processes

    # Partition image paths for this process
    per_process_images = len(image_paths) // num_processes
    start_img_idx = process_idx * per_process_images
    end_img_idx = start_img_idx + per_process_images if process_idx < num_processes - 1 else len(image_paths)

    # Get image paths for this process only
    process_image_paths = image_paths[start_img_idx:end_img_idx]
    process_prompts = prompts[start_img_idx:end_img_idx]

    # Process only this GPU's portion of images
    num_batches = (len(process_image_paths) + batch_size - 1) // batch_size

    if accelerator.is_main_process:
        print(f"Process {process_idx}: Processing {len(process_image_paths)} of {len(image_paths)} total images")
        pbar = tqdm(total=len(image_paths), desc="Processing images")

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(process_image_paths))
        
        batch_images = []
        batch_prompts = []
        batch_paths = []
        
        # Prepare current batch
        for idx in range(start_idx, end_idx):
            try:
                control_image = load_image(process_image_paths[idx]).resize((resolution, resolution))
                batch_images.append(control_image)
                batch_prompts.append(process_prompts[idx])
                batch_paths.append(process_image_paths[idx])
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"Error processing {process_image_paths[idx]}: {e}")
                continue
        
        if not batch_images:
            continue
        
        # Generate images
        with torch.no_grad():
            if model_type.lower() == "sd3":
                outputs = pipe(
                    prompt=batch_prompts,
                    num_inference_steps=steps,
                    generator=generator,
                    control_image=batch_images
                ).images
            elif model_type.lower() == "sd2":  # SD2.1 pipeline has slightly different parameter names
                outputs = pipe(
                    prompt=batch_prompts,
                    num_inference_steps=steps,
                    generator=generator,
                    image=batch_images,
                    guidance_scale=7.5  # Add guidance scale for SD2.1
                ).images
            else:
                raise ValueError(f"Unknown model_type: {model_type}. Must be 'sd2' or 'sd3'.")
        
        # Save output images
        for j, output_image in enumerate(outputs):
            image_path = batch_paths[j]
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{image_name}.png")
            output_image.save(output_path)
            
            if accelerator.is_main_process:
                pbar.update(1)
    
    if accelerator.is_main_process:
        pbar.close()
        print(f"All processing complete. Results saved to {output_dir}")

if __name__ == "__main__":
    fire.Fire(inference)