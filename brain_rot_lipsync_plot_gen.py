import os
import json
import time
import base64
import re
import contextlib
import io
from typing import List, Dict

from moviepy.editor import VideoFileClip, ImageClip, concatenate_videoclips, clips_array, CompositeVideoClip, TextClip
from moviepy.video.fx.loop import loop
from moviepy.video.fx.margin import margin
from moviepy.video.tools.subtitles import SubtitlesClip
import matplotlib.pyplot as plt
from google import genai

from brainrot_lipsync_helper import gen_video_with_audio

# 1. Configure Gemini API client
def configure_gemini():
    # os.environ["GOOGLE_API_KEY"] = api_key
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return client


def extract_json_array(raw_text: str) -> str:
    fence_pattern = re.compile(r"```(?:json)?\s*(\[\s*[\s\S]*?\])\s*```", re.IGNORECASE)
    m = fence_pattern.search(raw_text)
    if m:
        return m.group(1)
    start = raw_text.find('[')
    end = raw_text.rfind(']')
    if start != -1 and end != -1 and end > start:
        return raw_text[start:end+1]
    raise ValueError("Could not extract JSON array from Gemini output")


# 2. Generate structured conversation via Gemini
def generate_conversation(client, topic: str, model_name: str="gemini-2.5-flash") -> List[Dict[str, str]]:
    system_instruction = (
        "You are crafting a Gen Z-oriented, quirky dialogue between two personalities which goes into a brainrot generator AI pipeline: "
        "Sydney Sweeney and Cristiano Ronaldo. The conversation should revolve around the given a phrase about something the user does not understand and clear their confusion about it, "
        "be engaging, educational, and short (40-60 seconds total). "
        "Remember this text is going into a lip sync model and an audio generation model. Cool Abbreviations and Exclamations or custom expressions that are not part of the english language those models are trained on will not come out well in the final product. Avoid those."
        "Avoid excliamations, the audio model goes really sharp and jerky when there are moments of exclaimation so dont add any '!', '*' marks in thr script."
        "Also when using words that are pronounced differently, for example the scientific letter 'mu' is pronouned as mew. In the script write the wrong spelling if neccesary so that the audio model pronounces it right. This does not mean split the word, for example collinear does not need to be co-lli-near, just write it without hyphens and stuff cz otherwise the audio model cannot recognise it."
        "Dont use abbreviated names and words."
        "An example of conversation is when ronaldo asks what is a matrix in mathematics"
        "Sydney replies saying Matrices are pretty straightforward Ronaldo, dont worry too much goat. Just look into my eyes and let me help you understand it. It is just a way of representing data that has both spatial and inherent meaning. The number 1, \"alone\"  has only an inherent value of 1 but when you write it inside the square bracket is now having a position information of 0 and inherent information of the value of 1. Just keep this analogy in mind and you will be able to apply this anywhere."
        "You see how this very technical + educational and still is trying to be a little cool and quirky, keep that tone, maybe take it up by a few notches."
        "Return a JSON array where each element has: \n"
        "  'speaker': 'sydney' or 'ronaldo',\n"
        "  'text': string of their utterance,\n"
        "  'plot_code': matplotlib code to generate a diagram for that utterance. Just genrate random example diagram \"RELATED TO THE Explaination\" \n"
        "Only return the JSON array."
        "The first screen/conversation should definitely have a matplotlib plot. If there is nothing relevant to the explaination then try to have some relevant plot for each conversation. Please dont make random plots."
        "The pipeline which you are part of goes like this. You generate a script for the 2 characters which is used to first generate the audio"
        "this audio is then used with  areference video to enable lipsync and then the matplotlib comes on top half of the screen and the conversation in the bottom half. Like a typical brain rot video generator for tik tok but educational."
    )
    user_prompt = f"Need exmplaination with: {topic}"
    full_prompt = system_instruction + "\n\n" + user_prompt
    response = client.models.generate_content(model=model_name, contents=full_prompt)
    text = response.text
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


def render_plot_from_code(code: str, output_path: str) -> str:
    """
    Attempts to render matplotlib plot from code and save to output_path.
    If it fails, returns path to fallback video (e.g., Subway Surfers).
    """
    import contextlib
    import io

    fallback_video = "/workspace/brain_rot_db/subway_surfers.mp4"  # You'll replace this with your actual path

    with contextlib.redirect_stdout(io.StringIO()):
        exec_globals = {"plt": plt}
        try:
            exec(code, exec_globals)
            plt.savefig(output_path, bbox_inches='tight')
            plt.clf()
            return output_path
        except Exception as e:
            print(f"[Plot Generation Failed] Falling back to Subway Surfers: {e}")
            plt.clf()
            return fallback_video


# 4. Combine plot + character clip

def combine_diagram_and_skit(diagram_path: str, skit_path: str, subway_loop_path: str, output_path: str, subtitle_input: str):
    # Load the bottom clip (dialogue skit) and get dimensions
    bottom_clip = VideoFileClip(skit_path)
    skit_width, skit_height = bottom_clip.size
    top_height = skit_height
    final_width = skit_width
    final_height = skit_height * 2

    # Load and resize subway surfers background
    subway_clip = VideoFileClip(subway_loop_path)
    subway_clip = subway_clip.resize(width=final_width)
    if subway_clip.h < top_height:
        subway_clip = subway_clip.resize(height=top_height)
    subway_clip = subway_clip.crop(y1=subway_clip.h - top_height, y2=subway_clip.h)
    subway_clip = loop(subway_clip, duration=bottom_clip.duration)

    # Always use subway as the background layer
    top_layers = [subway_clip]

    # Overlay the diagram (if exists)
    if diagram_path and diagram_path.endswith(".png") and os.path.exists(diagram_path):
        diagram_clip = ImageClip(diagram_path).set_duration(bottom_clip.duration)

        # Resize the diagram so it's smaller than the top half
        max_width = int(final_width * 0.7)
        max_height = int(top_height * 0.7)

        if diagram_clip.w > max_width:
            diagram_clip = diagram_clip.resize(width=max_width)
        if diagram_clip.h > max_height:
            diagram_clip = diagram_clip.resize(height=max_height)

        # Centered placement
        diagram_clip = diagram_clip.set_position(("center", "center"))
        top_layers.append(diagram_clip)

    # Compose the top half
    top_clip = CompositeVideoClip(top_layers, size=(final_width, top_height))

    # subtitle_txt = os.path.splitext(os.path.basename(skit_path))[0]  # default to filename
    # if "_tmp_" in skit_path:
    #     subtitle_txt = os.path.basename(skit_path).split("_tmp_")[-1].replace("_", " ").replace(".mp4", "")

    # You can pass the actual subtitle text explicitly instead of guessing
    subtitle_txt = subtitle_input

    # Create subtitle clip
    # subtitle_clip = TextClip(
    #     subtitle_txt,
    #     fontsize=40,
    #     color='white',
    #     stroke_color='black',
    #     stroke_width=2,
    #     font="Arial-Bold"
    # ).set_duration(bottom_clip.duration).set_position(("center", int(skit_height * 0.9)))

    # Add subtitle to bottom_clip
    bottom_clip = CompositeVideoClip([bottom_clip, subtitle_clip])

    # Final composite
    final = CompositeVideoClip([
        top_clip.set_position(("center", "top")),
        bottom_clip.set_position(("center", "bottom"))
    ], size=(final_width, final_height))

    # Export the video
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")

    # Cleanup
    final.close()
    top_clip.close()
    bottom_clip.close()
    subway_clip.close()


# 5. Synthesize clips with diagram on top
def synthesize_clips(conversation: List[Dict[str, str]], output_dir: str) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    clip_paths = []
    for idx, turn in enumerate(conversation):
        speaker = turn["speaker"].lower()
        text = turn["text"]
        print(f"Synthesizing for {speaker}, turn {idx}: \"{text}\"")

        video_path = gen_video_with_audio(speaker, text)
        base_name = os.path.basename(video_path)
        vid_tmp_path = os.path.join(output_dir, f"tmp_{idx:03d}_{speaker}_{base_name}")
        os.replace(video_path, vid_tmp_path)

        final_path = os.path.join(output_dir, f"{idx:03d}_{speaker}.mp4")

        # Optional: generate plot and combine
        plot_path = os.path.join(output_dir, f"plot_{idx:03d}.png")
        if "plot_code" in turn and turn["plot_code"]:
            # print(turn["plot_code"])
            render_plot_from_code(turn["plot_code"], plot_path)
            combine_diagram_and_skit(plot_path, vid_tmp_path, "/workspace/brain_rot_db/subway_surfers.mp4" , final_path, text)
        else:
            combine_diagram_and_skit("", vid_tmp_path, "/workspace/brain_rot_db/subway_surfers.mp4", final_path, text)

        clip_paths.append(final_path)
        time.sleep(0.5)
    return clip_paths


# 6. Stitch video clips in sequence
def stitch_clips(clip_paths: List[str], final_output_path: str):
    clips = [VideoFileClip(p) for p in clip_paths]
    if not clips:
        raise RuntimeError("No clips to stitch.")
    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(final_output_path, codec="libx264", audio_codec="aac")
    for c in clips:
        c.close()
    final.close()


# 7. Main pipeline
def create_brain_rot_video(
    topic: str,
    temp_dir: str = "tmp_conversation_clips",
    final_output: str = "sydney_ronaldo_convo.mp4",
    model_name: str = "gemini-2.5-flash"
):
    client = configure_gemini()
    print(f"Generating conversation on topic: {topic}")
    conversation = generate_conversation(client, topic, model_name=model_name)
    print(f"Received {len(conversation)} turns.")
    clip_paths = synthesize_clips(conversation, temp_dir)
    print("Stitching clips...")
    stitch_clips(clip_paths, final_output)
    print(f"Final video saved to {final_output}")


# 8. CLI interface
if __name__ == "__main__":
    import time

    # start_time = time.time()
    # create_brain_rot_video(
    #     topic="Why cant we make a traingle if 3 points are collinear",
    #     temp_dir="temp_staging_folder_1",
    #     final_output=f"sydney_ronaldo_with_plot_collinear.mp4",
    #     model_name="gemini-2.5-flash"
    # )
    # print(f"Time taken: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    create_brain_rot_video(
        topic="GLoVE embedding model",
        temp_dir="temp_staging_folder_1",
        final_output=f"embedding_space_cr7syd.mp4",
        model_name="gemini-2.5-flash"
    )
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

    # start_time = time.time()
    # create_brain_rot_video(
    #     topic="Not able to understand elastic collisions",
    #     temp_dir="temp_staging_folder_2",
    #     final_output="sydney_ronaldo_with_plot_elasticCollision.mp4",
    #     model_name="gemini-2.5-flash"
    # )
    # print(f"Time taken: {time.time() - start:.2f} seconds")