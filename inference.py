def inference(model, identity_image, smpl_params, camera_params, scene_frames, occlusion_frames, device, num_inference_steps=50):
    model.eval()
    with torch.no_grad():
        # Start from random noise
        latents = torch.randn(1, 4, 16, 64, 64).to(device)  # Adjust shape based on your video dimensions
        
        # Set up the noise scheduler
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        scheduler.set_timesteps(num_inference_steps)
        
        for t in scheduler.timesteps:
            # Expand the latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            
            # Predict the noise residual
            noise_pred = model(latent_model_input, t, identity_image, smpl_params, camera_params, scene_frames, occlusion_frames)
            
            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
            
            # Compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    # Decode the latents to pixel space
    video = model.decoder.decode(latents)
    
    return video
