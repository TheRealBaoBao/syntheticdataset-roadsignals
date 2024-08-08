### Install requirements ###
!pip install --upgrade diffusers[torch]
!pip install transformers

### Create image generation pipeline ###

from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")

### Generate images ###

import random
import os

import matplotlib.pyplot as plt

# Define directories for different categories of road signs in various environments
sign_categories = [
    'regulatory_signs', 
    'warning_signs', 
    'guide_signs', 
    'informational_signs', 
    'construction_signs'
]

environments = ['city', 'suburb', 'rural', 'forest', 'desert', 'mountain']

# Define the base directory on your local device where images will be saved
base_dir = 'C:/Users/YourUsername/Documents/road_symbols'  # Replace with your desired path

# Create directories for each category and environment
for category in sign_categories:
    for environment in environments:
        os.makedirs(f'{base_dir}/{category}/{environment}', exist_ok=True)

# Define the prompts for each type of sign
road_sign_prompts = {
    'stop_sign': 'a red octagonal stop sign with white text "STOP"',
    'yield_sign': 'a white triangular yield sign with red border and the word "YIELD"',
    'speed_limit_sign': 'a white rectangular speed limit sign with black text "SPEED LIMIT 55"',
    'pedestrian_crossing': 'a yellow diamond-shaped pedestrian crossing sign with black figure',
    'sharp_turn': 'a yellow diamond-shaped sign with a black arrow indicating a sharp turn',
    'highway_exit': 'a green rectangular highway exit sign with white text and exit number',
    'mile_marker': 'a green rectangular mile marker sign with white text',
    'rest_area': 'a blue rectangular rest area sign with white text',
    'road_work_ahead': 'an orange diamond-shaped road work ahead sign with black text',
    'detour': 'an orange rectangular detour sign with black arrow'
    # Add more prompts as needed for other signs
}

# Loop to generate images for each sign type in different environments
for sign_name, sign_prompt in road_sign_prompts.items():
    for environment in environments:
        for j in range(50):  # Number of images per sign type per environment
            prompt = '{} in a {}, realistic environment, photorealistic, hyperrealistic, high detail, digital art, 8k resolution'.format(sign_prompt, environment)
            negative_prompt = '3d, cartoon, anime, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ' + \
                              '((grayscale)) Low Quality, Worst Quality, plastic, fake, disfigured, deformed, blurry, bad anatomy, blurred, watermark, grainy, signature'
            
            img = pipeline(prompt, negative_prompt=negative_prompt).images[0]

            category = sign_name.split('_')[0] + '_signs'
            img.save(f'{base_dir}/{category}/{environment}/{sign_name}_{str(j).zfill(4)}.png')
