import os
import json
import time
from typing import List, Dict

# Import your pipeline functions
from brainrot_lipsync_helper import gen_video_with_audio

from moviepy.editor import VideoFileClip, concatenate_videoclips

# Import Google Gen AI SDK
import base64
import os
from google import genai
from google.genai import types
import re

# 1. Configure Gemini API client
def configure_gemini(api_key: str):
    """
    Configure the Google Gen AI client with the given API key.
    """
    # Option A: via environment variable
    # os.environ["GOOGLE_API_KEY"] = api_key
    # Create client
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return client


def extract_json_array(raw_text: str) -> str:
    """
    Extract the JSON array substring from raw_text, which may be wrapped
    in Markdown fences (```json ...```) or have extra whitespace/text.
    Returns the JSON array string (e.g. "[{...}, {...}, ...]") or raises.
    """
    # 1. Try to match triple-backtick fenced JSON blocks:
    #    ```json
    #    [ ... ]
    #    ```
    fence_pattern = re.compile(r"```(?:json)?\s*(\[\s*[\s\S]*?\])\s*```", re.IGNORECASE)
    m = fence_pattern.search(raw_text)
    if m:
        return m.group(1)
    # 2. If no fenced block, try to find the first '[' and the matching closing ']'.
    #    This is a bit trickier if nested, but since we expect a top-level array,
    #    we can locate the first '[' and the last ']' in the text.
    start = raw_text.find('[')
    end = raw_text.rfind(']')
    if start != -1 and end != -1 and end > start:
        return raw_text[start:end+1]
    # 3. If still not found, error
    raise ValueError("Could not extract JSON array from Gemini output")

# 2. Generate structured conversation via Gemini
def generate_conversation(client, topic: str, model_name: str="gemini-2.5-flash") -> List[Dict[str, str]]:
    """
    Ask Gemini to produce a JSON array of utterances between Sydney Sweeney and Cristiano Ronaldo.
    Each item is {"speaker": "sydney" or "ronaldo", "text": "..."}.
    """
    system_instruction = (
        "You are crafting a Gen Z-oriented, quirky dialogue between two personalities: "
        "Sydney Sweeney and Cristiano Ronaldo. "
        "The conversation should revolve around the given topic, be engaging, youth-focused, "
        "and each turn should be concise enough for a short video cli of around 40-60 seconds (e.g., ~1-2 sentences). "
        "Output EXACTLY a JSON array (no additional commentary) where each element is an object "
        "with keys:\n"
        '  "speaker": either "sydney" or "ronaldo",\n'
        '  "text": the utterance content.\n'
        "Ensure the JSON is valid and directly parseable."
        "Remember this text is going into a lip sync model and an audio generation model. Cool Abbreviations and Exclamations or custom expressions that are not part of the english language those models are trained on will not come out well in the final product. Avoid those."
        "Avoid excliamations, the audio model goes really sharp and jerky when there are moments of exclaimation so dont add any '!' marks in thr script."
        "An example of conversation is when ronaldo asks what is a matrix in mathematics"
        "Sydney replies saying Matrices are pretty straightforward Ronaldo, dont worry too much goat. Just look into my eyes and let me help you understand it. It is just a way of representing data that has both spatial and inherent meaning. The number 1, \"alone\"  has only an inherent value of 1 but when you write it inside the square bracket is now having a position information of 0 and inherent information of the value of 1. Just keep this analogy in mind and you will be able to apply this anywhere."
        "You see how this very technical + educational and still is trying to be a little cool and quirky, keep that tone, maybe take it up by a few notches."
    )
    user_prompt = f"Topic: {topic}"
    # Combine into one prompt:
    full_prompt = system_instruction + "\n\n" + user_prompt
    response = client.models.generate_content(
        model=model_name,
        contents=full_prompt
    )
    text = response.text
    print(text)
    conversation = extract_json_array(text)
    print(conversation)
    try:
        convo = json.loads(conversation)
        if not isinstance(convo, list):
            raise ValueError("Expected a list")
        for turn in convo:
            if not isinstance(turn, dict) or "speaker" not in turn or "text" not in turn:
                raise ValueError("Each item must have 'speaker' and 'text'")
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON from Gemini output: {e}\nRaw output:\n{text}")
    return convo

# 3. Synthesize media for each utterance
def synthesize_clips(conversation: List[Dict[str, str]], output_dir: str) -> List[str]:
    """
    For each turn in conversation, call gen_video_with_audio to produce a video file.
    Returns list of file paths in order.
    Assumes `gen_video_with_audio(character, text)` writes out a video file and returns its path,
    or you adapt this function to capture the output path.
    """
    os.makedirs(output_dir, exist_ok=True)
    clip_paths = []
    for idx, turn in enumerate(conversation):
        speaker = turn["speaker"].lower()
        text = turn["text"]
        # Call your pipeline's function:
        # We assume gen_video_with_audio returns the path to the generated video file for this utterance.
        # If it does not return a path, you may modify gen_video_with_audio to return it,
        # or determine the naming convention (e.g., f"{speaker}_{idx}.mp4").
        print(f"Synthesizing for {speaker}, turn {idx}: \"{text}\"")
        video_path = gen_video_with_audio(speaker, text)
        # Optionally: move/rename into our output_dir with index prefix
        base_name = os.path.basename(video_path)
        new_name = f"{idx:03d}_{speaker}_{base_name}"
        dest_path = os.path.join(output_dir, new_name)
        os.replace(video_path, dest_path)
        clip_paths.append(dest_path)
        # Small sleep/delay if needed between calls
        time.sleep(0.5)
    return clip_paths

# 4. Stitch video clips in sequence
def stitch_clips(clip_paths: List[str], final_output_path: str):
    """
    Concatenate the list of video clip file paths into one final video.
    """
    clips = []
    for path in clip_paths:
        clip = VideoFileClip(path)
        clips.append(clip)
    if not clips:
        raise RuntimeError("No clips to stitch.")
    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(final_output_path, codec="libx264", audio_codec="aac")
    # Close clips to release resources
    for clip in clips:
        clip.close()
    final_clip.close()

# 5. Main orchestration
def create_brain_rot_video(
    topic: str,
    api_key: str,
    temp_dir: str = "tmp_conversation_clips",
    final_output: str = "sydney_ronaldo_convo.mp4",
    model_name: str = "gemini-2.5-flash"
):
    """
    Full pipeline: configure Gemini, get conversation, synthesize clips, stitch them.
    """
    client = configure_gemini(api_key)
    print(f"Generating conversation on topic: {topic}")
    conversation = generate_conversation(client, topic, model_name=model_name)
    print(f"Received {len(conversation)} turns.")
    clip_paths = synthesize_clips(conversation, temp_dir)
    print("Stitching clips...")
    stitch_clips(clip_paths, final_output)
    print(f"Final video saved to {final_output}")

# Example usage
# if __name__ == "__main__":
#     import argparse
#     import time

#     parser = argparse.ArgumentParser(description="Generate a Sydney Sweeney & Ronaldo convo video.")
#     parser.add_argument("--topic", type=str, default="Vectors", help="Topic for the conversation.")
#     parser.add_argument("--api_key", type=str, default="", help="Gemini API key.")
#     parser.add_argument("--output", type=str, default="sydney_ronaldo_convo.mp4", help="Final output video path.")
#     parser.add_argument("--temp_dir", type=str, default="tmp_conversation_clips", help="Directory for intermediate clips.")
#     parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="Gemini model to use.")
#     args = parser.parse_args()

#     start_time = time.time()
#     create_brain_rot_video(
#         topic="Dot product of 2 vectors",
#         api_key=args.api_key,
#         temp_dir="tmp_conversation_clips_1",
#         final_output="sydney_ronaldo_convo_1.mp4",
#         model_name=args.model
#     )
#     print(f"Time taken to generate skit = {time.time()-start_time}")

#     start_time = time.time()
#     create_brain_rot_video(
#         topic="Co-efficient of friction (mu)",
#         api_key=args.api_key,
#         temp_dir="tmp_conversation_clips_2",
#         final_output="sydney_ronaldo_convo_2.mp4",
#         model_name=args.model
#     )

    # print(f"Time taken to generate skit = {time.time()-start_time}")
