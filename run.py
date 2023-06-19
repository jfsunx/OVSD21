import gradio as gr
from pathlib import Path
from openvino.runtime import Core
from transformers import CLIPTokenizer
from pipelinex import OVStableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.configuration_utils import FrozenDict

conf = FrozenDict([('num_train_timesteps', 1000),
                   ('beta_start', 0.00085),
                   ('beta_end', 0.012),
                   ('beta_schedule', 'scaled_linear'),
                   ('trained_betas', None),
                   ('skip_prk_steps', True),
                   ('set_alpha_to_one', False),
                   ('prediction_type', 'epsilon'),
                   ('steps_offset', 1),
                   ('_class_name', 'PNDMScheduler'),
                   ('_diffusers_version', '0.10.0.dev0'),
                   ('clip_sample', False)])

# Configure Inference Pipeline

sd2_1_model_dir = Path("sd2.1")
TEXT_ENCODER_OV_PATH = sd2_1_model_dir / 'text_encoder.xml'
UNET_OV_PATH = sd2_1_model_dir / 'unet.xml'
VAE_DECODER_OV_PATH = sd2_1_model_dir / 'vae_decoder.xml'
VAE_ENCODER_OV_PATH = sd2_1_model_dir / 'vae_encoder.xml'

# First, you should create instances of OpenVINO Model.
core = Core()
text_enc = core.compile_model(TEXT_ENCODER_OV_PATH, "GPU")
unet_model = core.compile_model(UNET_OV_PATH, 'GPU')
vae_decoder = core.compile_model(VAE_DECODER_OV_PATH, 'GPU')
vae_encoder = core.compile_model(VAE_ENCODER_OV_PATH, 'GPU')

# Model tokenizer and scheduler are also important parts of the pipeline.
# Let us define them and put all components together.
scheduler = LMSDiscreteScheduler.from_config(conf)
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')


ov_pipe = OVStableDiffusionPipeline(
    tokenizer=tokenizer,
    text_encoder=text_enc,
    unet=unet_model,
    vae_encoder=vae_encoder,
    vae_decoder=vae_decoder,
    scheduler=scheduler
)

# Run Text-to-Image generation
'''
import ipywidgets as widgets

text_prompt = widgets.Textarea(value="valley in the Alps at sunset, epic vista, beautiful landscape, 4k, 8k", description='positive prompt', layout=widgets.Layout(width="auto"))
negative_prompt = widgets.Textarea(value="frames, borderline, text, charachter, duplicate, error, out of frame, watermark, low quality, ugly, deformed, blur", description='negative prompt', layout=widgets.Layout(width="auto"))
num_steps = widgets.IntSlider(min=1, max=50, value=25, description='steps:')
seed = widgets.IntSlider(min=0, max=10000000, description='seed: ', value=4200)
widgets.VBox([text_prompt, negative_prompt, seed, num_steps])

print('Pipeline settings')
print(f'Input text: {text_prompt.value}')
print(f'Seed: {seed.value}')
print(f'Number of steps: {num_steps.value}')
'''

'''
text_prompt = "valley in the Alps at sunset, epic vista, beautiful landscape, 4k, 8k"
negative_prompt = "frames, borderline, text, charachter, duplicate, error, out of frame, watermark, low quality, ugly, deformed, blur"

#result = ov_pipe(text_prompt.value, negative_prompt=negative_prompt.value, num_inference_steps=num_steps.value, seed=seed.value)
result = ov_pipe(text_prompt, negative_prompt=negative_prompt, num_inference_steps=int(30), seed=int(1024))

final_image = result['sample'][0]
final_image.save('result.png')
'''


def run_gr(text_prompt, negative_prompt, num_inference_steps, seed):
    result = ov_pipe(text_prompt, negative_prompt=negative_prompt, num_inference_steps=int(num_inference_steps),
                     seed=int(seed))
    final_image = result['sample'][0]
    return final_image


demo = gr.Interface(
    fn=run_gr,
    inputs=["text", "text", gr.Slider(0, 50, value=25, step=1, label="num_inference_steps"), "number"],
    outputs=["image"],
)
demo.launch()
