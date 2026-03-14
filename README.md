# WanGP-LTX-Prompt-Enhancer
LTX specific prompt enhancer that follows official prompting guide (with a little bit of extra).

## How to install

1. Make a backup of this file `shared/prompt_enhancer/prompt_enhance_utils.py`
2. Replace this file with my custom [prompt_enhance_utils.py](prompt_enhance_utils.py)
3. Start WanGP and make sure you are using Qwen3.5VL 9B/4B (either should work)

## How to use

1. Set `How to Process each Line of the Text Prompt` -> `All the Lines are Part of the Same Prompt` (on Generate > Advanced)
2. Use a simple prompt like this and click `Enhance Prompt` button

```
Mode: Landscape
Length: 20 Seconds
Prompt: A news reporter in the middle of town reporting about discovering a fountain of oil
```

## Example enhanced prompts

```
EXT. CITY PARK – GOLDEN HOUR
Wide establishing shot through tree branches, shallow depth of field. The park is unusually quiet. In the distance, a massive crater has been torn open in the ground — molten metal and rock still smoking.
A FIRE TRUCK pulls up beside it, ladder extended. Firefighters in full gear begin climbing. One drops a thermal imaging camera onto the rim. The screen shows a pulsing red core deep underground.
CUT TO: Close-up of the firefighter's face, soot streaked across his helmet. He radios in, voice strained.
FIREMAN (into radio): "We've got a breach. It's not just a fire — something's melting from below."
He looks at his partner. They both watch as the crater expands, swallowing the edge of the parking lot.
CUT TO: Underwater POV. The ocean floor is littered with twisted steel. A submersible drone descends through the silt, its lights cutting through the gloom. It approaches a massive, jagged hole in the seabed. Dark energy pulses outward — visible distortion in the water.
BACK TO SHOT: The fire truck's radio crackles.
DISPATCH (staticy): "All units, we're reading seismic activity off the coast. This isn't natural."
The lead firefighter pulls his mask off, eyes wide. He points toward the crater.
FIREMAN: "It's getting deeper. We need to evacuate the block."
The ground shudders. Debris rains down from the crater's rim. The firefighters scramble back as the hole opens wider, revealing a swirling vortex of black smoke and steam.
HOLD ON THE VORTEX as it roars upward, merging with the clouds. Cut to black.
```

## How to uninstall / revert

* Restore backup file OR
* Set `ltx_mode: bool = True,` to `False` on line #376
