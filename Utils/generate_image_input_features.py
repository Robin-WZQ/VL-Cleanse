from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import LMSDiscreteScheduler
import torch
from PIL import Image
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

def generate_images_with_feature(dir_, backdoor_text_encoder,feature, save_path, device='cuda:0', guidance_scale = 7.5, image_size=512, ddim_steps=100,seed=42, batch_size=1):
        
    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained(dir_, subfolder="vae")
    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained(dir_, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(backdoor_text_encoder)
    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(dir_, subfolder="unet")
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    torch_device = device

    height = image_size
    width = image_size                   # default width of Stable Diffusion

    num_inference_steps = ddim_steps           # Number of denoising steps

    generator = torch.manual_seed(seed)        # Seed generator to create the inital latent noise

    text_embeddings = torch.load(feature).to(torch_device)
    
    text_embeddings = text_embeddings.expand(batch_size, -1, -1)  
    max_length = text_embeddings.shape[1]

    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latents = torch.randn(
        (batch_size, unet.config.in_channels, height // 8, width // 8),
        generator=generator,
    )
    latents = latents.to(torch_device)

    scheduler.set_timesteps(num_inference_steps)

    latents = latents * scheduler.init_noise_sigma

    from tqdm.auto import tqdm

    scheduler.set_timesteps(num_inference_steps)

    for t in tqdm(scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for num, im in enumerate(pil_images):
        im.save(f"{save_path}/Reversed_Backdoor_Image_{num}.png")


if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'Generate Images')
    parser.add_argument('--feature_path', default='/mnt/sdb1/wangzhongqi/project/backdoor_detection/VL-Cleanse/Results/InsertSent/CLIP-L-336-Text/poisoned_model_1/Backdoor_Text_Feature.pt')   
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--model', default="/mnt/sdb1/wangzhongqi/Models/stable-diffusion-v1-4")
    parser.add_argument('--backdoor_text_encoder', default="/mnt/sdb1/wangzhongqi/project/backdoor_detection/VL-Cleanse/Models/Backdoor_models/CLIP-L-336/InsertSent/poisoned_model_1")
    parser.add_argument('--output_folder', default="/mnt/sdb1/wangzhongqi/project/backdoor_detection/VL-Cleanse/Results/InsertSent/CLIP-L-336-Text/poisoned_model_1/Images")
    parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:7')
    parser.add_argument('--guidance_scale', help='guidance to run eval', type=float, required=False, default=7.5)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--ddim_steps', help='ddim steps', type=int, required=False, default=100)
    parser.add_argument('--batch_size', help='batch size used to generate images', type=int, required=False, default=10)
    args = parser.parse_args()
    
    model = args.model
    feature = args.feature_path
    save_path = args.output_folder
    device = args.device
    guidance_scale = args.guidance_scale
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    backdoor_text_encoder = args.backdoor_text_encoder
    
    generate_images_with_feature(model, backdoor_text_encoder, feature, save_path, device=device,guidance_scale = guidance_scale, image_size=image_size, ddim_steps=ddim_steps,seed=args.seed, batch_size=args.batch_size)
