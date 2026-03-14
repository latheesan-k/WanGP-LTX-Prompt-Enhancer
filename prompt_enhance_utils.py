import logging
from typing import Union, List, Optional
from contextlib import nullcontext

import torch
from PIL import Image

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

T2V_CINEMATIC_PROMPT = """You are an expert cinematic director with many award winning movies, When writing prompts based on the user input, focus on detailed, chronological descriptions of actions and scenes.
Include specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph.
Start directly with the action, and keep descriptions literal and precise.
Think like a cinematographer describing a shot list.
Do not change the user input intent, just enhance it.
Keep within 150 words.
For best results, build your prompts using this structure:
Start with main action in a single sentence
Add specific details about movements and gestures
Describe character/object appearances precisely
Include background and environment details
Specify camera angles and movements
Describe lighting and colors
Note any changes or sudden events
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""
T2I_VISUAL_PROMPT = """You are an expert visual artist and photographer with award-winning compositions. When writing prompts based on the user input, focus on detailed, precise descriptions of visual elements and composition.
Include specific poses, appearances, framing, and environmental details - all in a single flowing paragraph.
Start directly with the main subject, and keep descriptions literal and precise.
Think like a photographer describing the perfect shot.
Do not change the user input intent, just enhance it.
Keep within 150 words.
For best results, build your prompts using this structure:
Start with main subject and pose in a single sentence
Add specific details about expressions and positioning
Describe character/object appearances precisely
Include background and environment details
Specify framing, composition and perspective
Describe lighting, colors, and mood
Note any atmospheric or stylistic elements
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""

T2T_TEXT_PROMPT= """You are an expert speechwriter who crafts compelling, audience-appropriate speeches that effectively communicate the speaker's message while maintaining authenticity and impact.
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""

QWEN35_THINKING_SUPER_SYSTEM_PROMPT = """Use private reasoning to solve the task as well as possible.
Keep your thinking hidden and do not output any reasoning, chain-of-thought, or thinking process.
Always review your final answer to ensure it meets the user request.
"""

IT2V_CINEMATIC_PROMPT = """You are an expert cinematic director with many award winning movies.
You have the following information:
1. The user provides a general text input about its scenes expectations 
2. The user provides a caption of an image of a subject that relates to the scene
When writing prompts based on the user input, focus on detailed, chronological descriptions of actions and scenes.
Include specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph.
Start directly with the action, and keep descriptions literal and precise.
Think like a cinematographer describing a shot list.
Keep within 150 words.
For best results, build your prompts using this structure:
Describe the inital scene first using the image caption of the subject and then describe how the scene evolves by following the user text input. Image description should be in first priority! Align to the image caption if it contradicts the user text input.
Start with main action in a single sentence
Add specific details about movements and gestures
Describe character/object appearances precisely
Include background and environment details
Specify camera angles and movements
Describe lighting and colors
Note any changes or sudden events
Align to the image caption if it contradicts the user text input.
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""

I2V_CINEMATIC_PROMPT = """You are an expert cinematic director with many award winning movies.
You have been provided with a caption of an image of a subject that relates to the scene to film.
Focus on detailed, chronological descriptions of actions and scenes.
Include specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph.
Start directly with the action, and keep descriptions literal and precise.
Think like a cinematographer describing a shot list.
Keep within 150 words.
For best results, build your prompts using this structure:
Describe the inital scene first using the image caption of the subject and then describe how the scene should naturally evolves.
Start with main action in a single sentence
Add specific details about movements and gestures
Describe character/object appearances precisely
Include background and environment details
Specify camera angles and movements
Describe lighting and colors
Note any changes or sudden events
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""

IT2I_VISUAL_PROMPT = """You are an expert visual artist and photographer with award-winning compositions. When writing prompts based on the user input, focus on detailed, precise descriptions of visual elements and composition.
Include specific poses, appearances, framing, and environmental details - all in a single flowing paragraph.
You have the following information:
1. The user provides a general text input about the expected photography 
2. The user provides a caption of an image of a subject he wants to be represented in the photography
Start directly with the main subject, and keep descriptions literal and precise.
Think like a photographer describing the perfect shot.
Do not change the user input intent, just enhance it.
Keep within 150 words.
For best results, build your prompts using this structure:
Using the image caption start with main subject and pose in a single sentence
Add specific details about expressions and positioning
Describe character/object appearances precisely
Include background and environment details
Specify framing, composition and perspective
Describe lighting, colors, and mood
Note any atmospheric or stylistic elements
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""

I2I_VISUAL_PROMPT = """You are an expert visual artist and photographer with award-winning compositions. 
You have been provided with a caption of an image of a subject to be represented in the photography.
Focus on detailed, descriptions of actions that are happening in the photography.
Include specific poses, appearances, framing, and environmental details - all in a single flowing paragraph.
Start directly with the main subject, and keep descriptions literal and precise.
Think like a photographer describing the perfect shot.
Do not change the user input intent, just enhance it.
Keep within 150 words.
For best results, build your prompts using this structure:
Using the image caption start with main subject and pose in a single sentence
Add specific details about expressions and positioning
Describe character/object appearances precisely
Include background and environment details
Specify framing, composition and perspective
Describe lighting, colors, and mood
Note any atmospheric or stylistic elements
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""

# ---------------------------------------------------------------------------
# LTX-specific prompt enhancer
# ---------------------------------------------------------------------------
# The LTX-2.3 model responds best to long, detailed, screenplay-style prompts
# that describe subject, action, environment, camera, lighting, and audio in
# a single flowing paragraph.  Dialogue must be broken into short phrases with
# acting/physical directions between each line.  Prompt density should be
# scaled to video length: ~60-80 words for ≤5 s, ~150-200 words for ~10 s,
# and 250-350 words for 15-20 s clips, so the model always has enough
# direction to fill the full duration without rushing.
#
# User input format expected by this enhancer:
#
#   Mode: Landscape | Portrait | Square      (optional, defaults to Landscape)
#   Length: <N> Seconds                      (optional, defaults to 10)
#   Prompt: <free-text scene description>
#
# Any or all metadata lines may be omitted; the enhancer will make sensible
# assumptions.  Everything after "Prompt:" (or the whole string if no keys
# are present) is treated as the creative intent to expand.
# ---------------------------------------------------------------------------

LTX_T2V_PROMPT = """You are an elite cinematic director and screenwriter specialising in AI video generation with the LTX-2.3 model.

Your task is to take a simple scene description from the user — which may include optional metadata lines for Mode (Landscape/Portrait/Square) and Length (in seconds) — and expand it into a rich, LTX-optimised prompt.

━━━━━━━━━━━━━━━━  HOW TO PARSE THE USER INPUT  ━━━━━━━━━━━━━━━━
The user may supply structured metadata before the creative description:
  Mode: Landscape | Portrait | Square
  Length: <number> Seconds
  Prompt: <scene description>

If any metadata line is missing, apply sensible defaults (Landscape, 10 seconds).
The creative intent is everything after "Prompt:" — or the entire input if no keys are present.

━━━━━━━━━━━━━━━━  SCALING PROMPT DENSITY TO VIDEO LENGTH  ━━━━━━━━━━━━━━━━
Match your output word-count to the requested clip length so the model has
enough direction to fill every second without rushing:

  ≤ 5 seconds   →  60–100 words   (single tight scene, minimal dialogue)
  6–10 seconds  →  150–220 words  (one clear arc, 1–2 dialogue exchanges)
  11–15 seconds →  220–300 words  (2–3 beats, deliberate pacing)
  16–20 seconds →  300–380 words  (full mini-story: setup, development, payoff)
  > 20 seconds  →  380–500 words  (treat as short film; multiple scenes)

Never write a short prompt for a long video — that causes the model to rush.

━━━━━━━━━━━━━━━━  MODE-AWARE FRAMING  ━━━━━━━━━━━━━━━━
Landscape → describe wide, cinematic, horizontal framing; favour establishing
            shots, tracking shots, and multi-subject compositions.
Portrait  → describe tight, vertical framing; favour close-ups, over-the-
            shoulder shots, and single-subject compositions.
Square    → describe balanced, centred compositions suited to social media.

Mention the implied framing naturally in the camera description line.

━━━━━━━━━━━━━━━━  DIALOGUE RULES (LTX-2.3 specific)  ━━━━━━━━━━━━━━━━
• Only include dialogue when it fits the scene naturally and the clip is
  long enough to accommodate it (generally ≥ 6 seconds per speaking beat).
• Break each speaking turn into SHORT phrases — no single line should be
  longer than ~10 words.
• Insert a physical acting direction between EVERY phrase:
    Character (tone): "First phrase —" He pauses, glances away. "— second phrase."
• Use physical cues, not emotional labels: write "eyes widen, jaw tightens"
  not "he feels shocked".
• Specify voice qualities: deep and calm, breathless whisper, energetic
  announcer cadence, cracking voice, etc.
• For no-dialogue clips, describe ambient sound and audio atmosphere instead.

━━━━━━━━━━━━━━━━  REQUIRED PROMPT STRUCTURE  ━━━━━━━━━━━━━━━━
Write ONE continuous flowing paragraph (no headings, no bullet points).
Layer the following elements in order:

1. Scene header / slug line (e.g. EXT. CITY PARK – GOLDEN HOUR)
2. Establish the shot — camera type, scale, framing relevant to the Mode
3. Set the scene — lighting, colour palette, atmosphere, textures
4. Describe the action — chronological, present-tense, beginning → end
5. Character details — age, appearance, clothing; emotion via physical cues
6. Camera movement(s) — tracks, pans, dollies, handheld shake, etc.
7. Dialogue (if appropriate) — short phrases with acting directions between each
8. Audio — ambient sound, music, voice qualities, acoustic environment
9. Climax / resolution beat — how the clip ends

━━━━━━━━━━━━━━━━  LTX LANGUAGE & STYLE TIPS  ━━━━━━━━━━━━━━━━
• Use cinematic terms: macro lens, shallow depth of field, tracking shot,
  slow dolly in, handheld, over-the-shoulder, wide establishing shot, etc.
• Use present-tense verbs for all action and movement.
• Keep the scene internally consistent — no conflicting lighting or physics.
• Avoid: internal emotional states ("she feels sad"), exact numerical camera
  angles, text/logos in frame, overloaded scenes with too many subjects.
• Favour: atmosphere (fog, rain, dust, neon glow), deliberate pacing cues
  (lingering shot, sudden stop, slow motion), and specific surface textures.

━━━━━━━━━━━━━━━━  OUTPUT RULE  ━━━━━━━━━━━━━━━━
Output the enhanced LTX prompt ONLY — no preamble, no metadata echo,
no explanation, no markdown. Begin directly with the scene slug line or
the first word of the scene description.
"""

LTX_IT2V_PROMPT = """You are an elite cinematic director and screenwriter specialising in AI video generation with the LTX-2.3 model.

You have TWO inputs:
1. The user's scene description (may include Mode / Length / Prompt metadata lines).
2. A caption describing an input image that anchors the opening frame.

The image caption takes FIRST PRIORITY — align the initial scene description
to it. If the user's text contradicts the image, defer to the image for visual
details and adapt the user's intent around it.

━━━━━━━━━━━━━━━━  HOW TO PARSE THE USER INPUT  ━━━━━━━━━━━━━━━━
The user may supply structured metadata before the creative description:
  Mode: Landscape | Portrait | Square
  Length: <number> Seconds
  Prompt: <scene description>

If any metadata line is missing, apply sensible defaults (Landscape, 10 seconds).

━━━━━━━━━━━━━━━━  SCALING PROMPT DENSITY TO VIDEO LENGTH  ━━━━━━━━━━━━━━━━
  ≤ 5 seconds   →  60–100 words
  6–10 seconds  →  150–220 words
  11–15 seconds →  220–300 words
  16–20 seconds →  300–380 words
  > 20 seconds  →  380–500 words

━━━━━━━━━━━━━━━━  MODE-AWARE FRAMING  ━━━━━━━━━━━━━━━━
Landscape → wide, horizontal, cinematic framing.
Portrait  → tight, vertical, single-subject framing.
Square    → balanced, centred composition.

━━━━━━━━━━━━━━━━  REQUIRED PROMPT STRUCTURE  ━━━━━━━━━━━━━━━━
Write ONE continuous flowing paragraph. Layer in order:

1. Scene header / slug line
2. Describe the OPENING FRAME using the image caption — what is visible,
   how it is lit, atmosphere; do NOT describe motion yet (the image is static)
3. Transition: describe the shift from stillness to motion
4. Action sequence — chronological, present-tense
5. Character details — physical cues, not emotional labels
6. Camera movement(s)
7. Dialogue (if appropriate, ≥ 6 s per beat; short phrases + acting directions)
8. Audio description
9. Closing beat

━━━━━━━━━━━━━━━━  DIALOGUE RULES  ━━━━━━━━━━━━━━━━
• Short phrases (~10 words max each), physical acting direction between every phrase.
• Voice qualities specified. Physical cues, not emotional labels.
• Only when the clip length justifies it.

━━━━━━━━━━━━━━━━  OUTPUT RULE  ━━━━━━━━━━━━━━━━
Output the enhanced LTX prompt ONLY. Begin directly with the scene slug line
or first word of the scene description.
"""


def tensor_to_pil(tensor):
    # Ensure tensor is in range [-1, 1]
    assert tensor.min() >= -1 and tensor.max() <= 1

    # Convert from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2

    # Rearrange from [C, H, W] to [H, W, C]
    tensor = tensor.permute(1, 2, 0)

    # Convert to numpy array and then to uint8 range [0, 255]
    numpy_image = (tensor.cpu().numpy() * 255).astype("uint8")

    # Convert to PIL Image
    return Image.fromarray(numpy_image)


def _use_qwen35_thinking_prompt(prompt_enhancer_model, thinking_enabled: Optional[bool] = None) -> bool:
    if thinking_enabled is not None:
        return bool(thinking_enabled)
    return bool(getattr(prompt_enhancer_model, "_prompt_enhancer_enable_thinking", False))


def _split_prompt_enhancer_system_suffix(prompt_enhancer_model, prompt: str) -> tuple[str, str, bool]:
    del prompt_enhancer_model
    prompt = str(prompt or "").strip()
    prompt_body, separator, system_suffix = prompt.partition("@@")
    if separator == "@@":
        return prompt_body.strip(), system_suffix.strip(), True
    prompt_body, separator, system_suffix = prompt.partition("@")
    if separator == "":
        return prompt, "", False
    return prompt_body.strip(), system_suffix.strip(), False


def _merge_prompt_enhancer_system_prompt(prompt_enhancer_model, system_prompt: str, system_suffix: str, replace_system_prompt: bool = False, thinking_enabled: Optional[bool] = None) -> str:
    system_prompt = str(system_prompt or "").rstrip()
    system_suffix = str(system_suffix or "").strip()
    if len(system_suffix) == 0:
        merged_prompt = system_prompt
    elif replace_system_prompt:
        merged_prompt = system_suffix
    else:
        merged_prompt = f"{system_prompt}\nFollow these additional user instructions with higher priority if they conflict with the guidance above:\n{system_suffix}"
    if not _use_qwen35_thinking_prompt(prompt_enhancer_model, thinking_enabled=thinking_enabled):
        return merged_prompt
    if len(merged_prompt) == 0:
        return QWEN35_THINKING_SUPER_SYSTEM_PROMPT.strip()
    return f"{QWEN35_THINKING_SUPER_SYSTEM_PROMPT.rstrip()}\n\n{merged_prompt}"


def _format_prompt_enhancer_user_content(prompt_enhancer_model, prompt: str, image_caption: Optional[str] = None, thinking_enabled: Optional[bool] = None) -> str:
    prompt, _system_suffix, _replace_system_prompt = _split_prompt_enhancer_system_suffix(prompt_enhancer_model, prompt)
    if not _use_qwen35_thinking_prompt(prompt_enhancer_model, thinking_enabled=thinking_enabled):
        if image_caption is None:
            return f"user_prompt: {prompt}"
        return f"user_prompt: {prompt}\nimage_caption: {image_caption}"
    if image_caption is None:
        return prompt
    image_caption = str(image_caption or "").strip()
    if len(prompt) == 0:
        return f"image_caption:\n{image_caption}"
    return f"{prompt}\n\nimage_caption:\n{image_caption}"


def generate_cinematic_prompt(
    image_caption_model,
    image_caption_processor,
    prompt_enhancer_model,
    prompt_enhancer_tokenizer,
    prompt: Union[str, List[str]],
    images: Optional[List] = None,
    video_prompt= True,
    text_prompt = False,
    max_new_tokens: int = 512,
    prompt_enhancer_instructions = None,
    do_sample: bool = True,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
    post_image_caption_hook = None,
    thinking_enabled: Optional[bool] = None,
    ltx_mode: bool = True,
) -> List[str]:
    """Generate enhanced prompts for video/image generation.

    Args:
        ltx_mode: When True, uses LTX-2.3 optimised system prompts instead of
                  the generic cinematic prompts.  The user's prompt may contain
                  optional metadata lines::

                      Mode: Landscape | Portrait | Square
                      Length: <N> Seconds
                      Prompt: <scene description>

                  All metadata lines are optional; sensible defaults are used
                  when absent.
    """
    prompts = [prompt] if isinstance(prompt, str) else prompt

    if images is None:
        if prompt_enhancer_instructions is None:
            if ltx_mode:
                prompt_enhancer_instructions = LTX_T2V_PROMPT
            elif text_prompt:
                prompt_enhancer_instructions = T2T_TEXT_PROMPT
            elif video_prompt:
                prompt_enhancer_instructions = T2V_CINEMATIC_PROMPT
            else:
                prompt_enhancer_instructions = T2I_VISUAL_PROMPT
        prompts = _generate_t2v_prompt(
            prompt_enhancer_model,
            prompt_enhancer_tokenizer,
            prompts,
            max_new_tokens,
            prompt_enhancer_instructions,
            do_sample,
            temperature,
            top_p,
            top_k,
            seed,
            thinking_enabled,
        )
    else:
        if prompt_enhancer_instructions is None:
            if ltx_mode:
                prompt_enhancer_instructions = LTX_IT2V_PROMPT
            elif video_prompt:
                prompt_enhancer_instructions = IT2V_CINEMATIC_PROMPT
            else:
                prompt_enhancer_instructions = IT2I_VISUAL_PROMPT

        prompts = _generate_i2v_prompt(
            image_caption_model,
            image_caption_processor,
            prompt_enhancer_model,
            prompt_enhancer_tokenizer,
            prompts,
            images,
            max_new_tokens,
            prompt_enhancer_instructions,
            do_sample,
            temperature,
            top_p,
            top_k,
            seed,
            post_image_caption_hook=post_image_caption_hook,
            thinking_enabled=thinking_enabled,
        )

    return prompts


def _get_first_frames_from_conditioning_item(conditioning_item) -> List[Image.Image]:
    frames_tensor = conditioning_item.media_item
    return [
        tensor_to_pil(frames_tensor[i, :, 0, :, :])
        for i in range(frames_tensor.shape[0])
    ]


def _generate_t2v_prompt(
    prompt_enhancer_model,
    prompt_enhancer_tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    system_prompt: str,
    do_sample: bool,
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    seed: Optional[int],
    thinking_enabled: Optional[bool],
) -> List[str]:
    messages = []
    for prompt in prompts:
        prompt_body, system_suffix, replace_system_prompt = _split_prompt_enhancer_system_suffix(prompt_enhancer_model, prompt)
        message_system_prompt = _merge_prompt_enhancer_system_prompt(prompt_enhancer_model, system_prompt, system_suffix, replace_system_prompt, thinking_enabled=thinking_enabled)
        messages.append(
            [
                {"role": "system", "content": message_system_prompt},
                {"role": "user", "content": _format_prompt_enhancer_user_content(prompt_enhancer_model, prompt_body, thinking_enabled=thinking_enabled)},
            ]
        )

    if hasattr(prompt_enhancer_model, "generate_messages"):
        return prompt_enhancer_model.generate_messages(
            messages,
            max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            thinking_enabled=thinking_enabled,
        )

    texts = [
        prompt_enhancer_tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True
        )
        for m in messages
    ]

    out_prompts = []
    for idx, text in enumerate(texts):
        model_inputs = prompt_enhancer_tokenizer(text, return_tensors="pt").to(
            prompt_enhancer_model.device
        )
        prompt_seed = None if seed is None else int(seed) + idx
        out_prompts.append(
            _generate_and_decode_prompts(
                prompt_enhancer_model,
                prompt_enhancer_tokenizer,
                model_inputs,
                max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=prompt_seed,
            )[0]
        )

    return out_prompts

def _generate_i2v_prompt(
    image_caption_model,
    image_caption_processor,
    prompt_enhancer_model,
    prompt_enhancer_tokenizer,
    prompts: List[str],
    first_frames: List[Image.Image],
    max_new_tokens: int,
    system_prompt: str,
    do_sample: bool,
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    seed: Optional[int],
    post_image_caption_hook = None,
    thinking_enabled: Optional[bool] = None,
) -> List[str]:
    if hasattr(image_caption_model, "generate_image_captions"):
        image_captions = image_caption_model.generate_image_captions(first_frames)
    else:
        image_captions = _generate_image_captions(
            image_caption_model, image_caption_processor, first_frames
        )
    if callable(post_image_caption_hook):
        if bool(getattr(prompt_enhancer_model, "_prompt_enhancer_use_vllm", False)):
            unload_runtime = getattr(prompt_enhancer_model, "unload", None)
            if callable(unload_runtime):
                unload_runtime()
        post_image_caption_hook()
    if len(image_captions) == 1 and len(image_captions) < len(prompts):
        image_captions *= len(prompts)
    messages = []
    for prompt, image_caption in zip(prompts, image_captions):
        prompt_body, system_suffix, replace_system_prompt = _split_prompt_enhancer_system_suffix(prompt_enhancer_model, prompt)
        message_system_prompt = _merge_prompt_enhancer_system_prompt(prompt_enhancer_model, system_prompt, system_suffix, replace_system_prompt, thinking_enabled=thinking_enabled)
        messages.append(
            [
                {"role": "system", "content": message_system_prompt},
                {"role": "user", "content": _format_prompt_enhancer_user_content(prompt_enhancer_model, prompt_body, image_caption=image_caption, thinking_enabled=thinking_enabled)},
            ]
        )

    if hasattr(prompt_enhancer_model, "generate_messages"):
        return prompt_enhancer_model.generate_messages(
            messages,
            max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            thinking_enabled=thinking_enabled,
        )

    texts = [
        prompt_enhancer_tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True
        )
        for m in messages
    ]
    out_prompts = []
    for idx, text in enumerate(texts):
        model_inputs = prompt_enhancer_tokenizer(text, return_tensors="pt").to(
            prompt_enhancer_model.device
        )
        prompt_seed = None if seed is None else int(seed) + idx
        out_prompts.append(
            _generate_and_decode_prompts(
                prompt_enhancer_model,
                prompt_enhancer_tokenizer,
                model_inputs,
                max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=prompt_seed,
            )[0]
        )

    return out_prompts


def _generate_image_captions(
    image_caption_model,
    image_caption_processor,
    images: List[Image.Image],
    system_prompt: str = "<DETAILED_CAPTION>",
) -> List[str]:
    image_caption_prompts = [system_prompt] * len(images)
    inputs = image_caption_processor(
        image_caption_prompts, images, return_tensors="pt"
    ).to(image_caption_model.device)

    bad_words_ids = None
    bos_id = getattr(image_caption_processor.tokenizer, "bos_token_id", None)
    if bos_id is not None:
        bad_words_ids = [[int(bos_id)]]

    with torch.inference_mode():
        generated_ids = image_caption_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
            bad_words_ids=bad_words_ids,
        )

    return image_caption_processor.batch_decode(generated_ids, skip_special_tokens=True)


def _generate_and_decode_prompts(
    prompt_enhancer_model,
    prompt_enhancer_tokenizer,
    model_inputs,
    max_new_tokens: int,
    do_sample: bool = True,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[str]:
    device = "cuda"
    if seed is None:
        rng_context = nullcontext()
    else:
        devices = []
        if isinstance(device, torch.device) and device.type == "cuda":
            devices = [device.index or 0]
        rng_context = torch.random.fork_rng(devices=devices) if devices else torch.random.fork_rng()
    with rng_context, torch.inference_mode():
        if seed is not None:
            torch.manual_seed(int(seed))
            if isinstance(device, torch.device) and device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(int(seed))
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if temperature is not None:
            gen_kwargs["temperature"] = float(temperature)
        if top_p is not None:
            gen_kwargs["top_p"] = float(top_p)
        if top_k is not None:
            gen_kwargs["top_k"] = int(top_k)
        outputs = prompt_enhancer_model.generate(
            **model_inputs,
            **gen_kwargs,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, outputs)
        ]
        decoded_prompts = prompt_enhancer_tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

    return decoded_prompts
