"""
Usage:
    $ ./venv/bin/streamlit run director.py
"""

import json
import os
import subprocess
import typing

# Import streamlit first so we can read secrets before loading google-genai
import streamlit as st

# Load API key from Streamlit Secrets if available, otherwise fall back to
# the hardcoded key (useful for local development).
_api_key = st.secrets.get(
    "GOOGLE_API_KEY", "AIzaSyB1L_a3_vZVp-BcR2bJgx-DPo3rWNRXwhI"
)
_use_vertexai = st.secrets.get("GOOGLE_GENAI_USE_VERTEXAI", "0")

# Force Gemini API mode (not Vertex AI) before importing google-genai
os.environ["GOOGLE_API_KEY"] = _api_key
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = _use_vertexai

from google import genai
from google.genai import types
from google.genai import types as genai_types

from avatars import AVATAR_TO_CHARACTERISTICS, AVATAR_TO_ID
from gemini_prompt_cues import LOUDNESS_CUE, PITCH_CUE, RESPELLING, TEMPO_CUE
from wellsaid_tts import generate_tts_sync

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "avatar": {
            "type": "STRING",
            "description": (
                f"Voice avatar selection. Please choose a voice with "
                f"appropriate characteristics given the user directive.\n"
                f"Voice descriptions:\n"
                f"{AVATAR_TO_CHARACTERISTICS}"
            ),
            "type": "STRING",
            "enum": list(AVATAR_TO_ID.keys()),
        },
        "prosodic_segments": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "word_indices": {
                        "type": "ARRAY",
                        "items": {"type": "INTEGER"},
                        "description": (
                            "List of indices for a group of words. "
                            "IMPORTANT: You MUST group consecutive words "
                            "together into a single segment if they share "
                            "the same prosody settings."
                        ),
                    },
                    "pitch": {"type": "INTEGER", "description": PITCH_CUE},
                    "tempo": {"type": "NUMBER", "description": TEMPO_CUE},
                    "loudness": {
                        "type": "INTEGER",
                        "description": LOUDNESS_CUE,
                    },
                },
                "required": ["word_indices"],
            },
        },
        "respellings": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "word_idx": {
                        "type": "INTEGER",
                        "description": ("Index of word to be respelled."),
                    },
                    "respelling": {
                        "type": "STRING",
                        "description": RESPELLING,
                    },
                },
                "required": ["word_idx", "respelling"],
            },
        },
        "post_processing": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "effect": {
                        "type": "STRING",
                        "enum": [
                            "eq",
                            "compression",
                            "reverb",
                            "normalize",
                            "de_ess",
                        ],
                    },
                },
                "required": ["effect"],
            },
        },
    },
    "required": ["prosodic_segments", "avatar"],
}

SYSTEM_PROMPT = """You are a prosody director for text-to-speech.
Given a piece of text and a desired style/emotion, output a structured
specification describing how each segment should be spoken and what
post-processing to apply. Break text into logical segments and only
include adjustments that differ from defaults.

GROUPING RULE:
If three words in a row are all 'fast' and 'high pitch', do NOT create
three segments. Instead, create ONE segment with all three word indices
(e.g., [0, 1, 2]). ONLY create a new segment when the prosody settings
change.

WORD ORDER:
Do NOT omit any words. Do NOT change word order.
"""


def get_prosody_plan(
    client: genai.Client,
    text: str,
    directive: str,
    model: str,
    message_history: list,
) -> typing.Tuple[typing.Dict, list]:
    """Get a structured prosody plan using Gemini's structured output."""
    words = text.split()
    indexed_text = "\n".join([f"{i}: {word}" for i, word in enumerate(words)])

    if len(message_history) == 0:
        prompt_content = f"""
        ORIGINAL TEXT (INDEXED):
        {indexed_text}

        USER DIRECTIVE: {directive}

        INSTRUCTIONS:
        Assign prosody adjustments using the word indices above. Do not skip
        any words; every word must be included in exactly one segment.
        """
    else:
        prompt_content = f"""
        ORIGINAL TEXT (INDEXED):
        {indexed_text}

        NEW USER DIRECTIVE: {directive}

        INSTRUCTIONS:
        Update the prosody plan based on the new directive.
        Ensure every word index is covered.
        """

    response = client.models.generate_content(
        model=model,
        contents=message_history
        + [
            types.Content(
                role="user", parts=[types.Part.from_text(text=prompt_content)]
            )
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=SCHEMA,
        ),
    )

    assert response.text is not None
    assert response.candidates is not None
    plan = json.loads(response.text)

    # Update history
    new_history = message_history + [
        types.Content(
            role="model", parts=[types.Part.from_text(text=response.text)]
        ),
    ]

    # 2. Validation Check
    # all_expected_indices = set(range(len(words)))
    # received_indices = []
    # for prosodic_segment in plan.get("prosodic_segments", []):
    #     received_indices.extend(prosodic_segment.get("word_indices", []))

    # received_set = set(received_indices)

    # # Check for missing or extra/duplicate indices
    # missing = all_expected_indices - received_set
    # extra = received_set - all_expected_indices
    # duplicates = len(received_indices) != len(received_set)

    # if missing:
    #     st.warning(f"Note: Gemini missed word indices: {missing}")
    # if extra:
    #     st.error(f"Error: Gemini hallucinated indices that don't exist: {extra}")
    # if duplicates:
    #     st.warning("Note: Some word indices were included in multiple segments.")

    return plan, new_history


def get_postprocessing_plan(
    client: genai.Client,
    schema: typing.Dict,
    system_prompt: str,
    directive: str,
    audio_part: types.Part,
    message_history: list,
    model: str = "gemini-2.0-flash",
):
    """Get a list of ffmpeg/sox commands using Gemini's structured output."""
    prompt = f"""
                Listen to this audio file, and recommend post \
                processing effects to make the audio \
                    fit this directive better: \
                    {directive}

                Post-processing effects can be done in three passes \
                and should be returned as three sox commands \
                to be run in a specific order.
                """
    response = client.models.generate_content(
        model=model,
        contents={
            "role": "user",
            "parts": [
                {"text": str(message_history) + prompt},
                {
                    "inline_data": {
                        "mime_type": "audio/mp3",
                        "data": audio_part,
                    }
                },
            ],
        },
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )

    new_history = message_history + [
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)]),
        types.Content(
            role="model", parts=[types.Part.from_text(text=response.text)]
        ),
    ]
    return response, new_history


# === Tag Wrappers ===


def wrap_with_pitch(
    text: str, value: int, min_val: int = -45, max_val: int = 100
) -> str:
    """
    Wraps text with a <pitch> tag. Value range: -45 to +100.
    """
    value = max(value, min_val)
    value = min(value, max_val)
    return f'<pitch value="{value}">{text}</pitch>'


def wrap_with_tempo(
    text: str, value: float, min_val: float = 0.7, max_val: float = 2.3
) -> str:
    """
    Wraps text with a <tempo> tag. Value range: 0.5 to 2.5.
    """
    value = max(value, min_val)
    value = min(value, max_val)
    return f'<tempo value="{value}">{text}</tempo>'


def wrap_with_loudness(
    text: str, value: int, min_val: int = -15, max_val: int = 9
) -> str:
    """
    Wraps text with a <loudness> tag. Value range: -20 to +10.
    """
    value = max(value, min_val)
    value = min(value, max_val)
    return f'<loudness value="{value}">{text}</loudness>'


def apply_respelling(word: str, phonetic: str) -> str:
    """
    Wraps a word with a <respell> tag to provide a custom pronunciation.
    """
    return f'<respell value="{phonetic}">{word}</respell>'


# === Multi-tag Composition ===


def apply_all_prosodic_tags(
    text: str,
    pitch: int | None = None,
    tempo: float | None = None,
    loudness: int | None = None,
) -> str:
    """
    Applies multiple AI Director tags in nested order.
    """
    if pitch is not None:
        text = wrap_with_pitch(text, pitch)
    if tempo is not None:
        text = wrap_with_tempo(text, tempo)
    if loudness is not None:
        text = wrap_with_loudness(text, loudness)
    return text


def convert_plan_to_xml(text: str, plan: typing.Dict) -> str:
    """Use the prosody plan generated by the LLM to wrap the input text in XML
    tags.
    """
    words = text.split()
    idx_to_word = {i: w for i, w in enumerate(words)}

    # Apply respellings
    if "respellings" in plan:
        for respelling in plan["respellings"]:
            word_idx = respelling["word_idx"]
            phonetic = respelling["respelling"]
            respelled_word = apply_respelling(
                word=idx_to_word[word_idx], phonetic=phonetic
            )
            words[word_idx] = respelled_word
            idx_to_word[word_idx] = respelled_word

    # Apply prosodic tags
    tagged_segments = []
    for segment in plan["prosodic_segments"]:
        word_idxs = segment["word_indices"]
        words = [idx_to_word[idx] for idx in word_idxs]
        segment_text = " ".join(words)
        pitch, tempo, loudness = None, None, None
        if "respelling" in segment:
            respelling = segment["respelling"]
        if "pitch" in segment:
            pitch = segment["pitch"]
        if "tempo" in segment:
            tempo = segment["tempo"]
        if "loudness" in segment:
            loudness = segment["loudness"]
        tagged = apply_all_prosodic_tags(segment_text, pitch, tempo, loudness)
        tagged_segments.append(tagged)
    xml = " ".join(tagged_segments)

    return xml


def main():
    st.set_page_config(page_title="TTS Director Chat", layout="wide")
    st.title("🎙️ Iterative TTS Director")

    # --- Initialization ---
    if "message_history" not in st.session_state:
        st.session_state.message_history = []
    if "interactions" not in st.session_state:
        # This list will store dicts: {"directive": str, "xml": str, "audio": bytes, "avatar": str}
        st.session_state.interactions = []
    # if "current_plan" not in st.session_state:
    #     st.session_state.current_plan = None
    # if "last_xml" not in st.session_state:
    #     st.session_state.last_xml = ""
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    client = genai.Client(
        api_key=_api_key,
        vertexai=False,
    )

    # --- Sidebar Settings ---
    with st.sidebar:
        st.header("Settings")
        model = st.selectbox(
            label="Model",
            options=[
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-2.5-flash-lite",
            ],
        )
        st.session_state.input_text = st.text_area(
            label="Input Text", value=st.session_state.input_text
        )
        avatar = st.selectbox(
            label="Avatar", options=[None] + list(AVATAR_TO_ID.keys())
        )
        if st.button("Clear Conversation"):
            st.session_state.message_history = []
            st.session_state.current_plan = None
            st.session_state.interactions = []
            st.rerun()

    # --- Chat Interface ---
    for interaction in st.session_state.interactions:
        with st.chat_message("user"):
            st.write(interaction["directive"])
            # display_text = (
            #     msg.parts[0].text.split("NEW DIRECTIVE:")[-1].strip()
            # )
        with st.chat_message("assistant"):
            st.caption(f"Voice: {interaction['avatar']}")
            with st.expander("View XML"):
                # st.code(interaction["xml"], language="xml", wrap_lines=True)
                st.code(interaction["xml"], language="xml")
            st.markdown("Audio with no cues:")
            st.audio(interaction["audio_orig"], format="audio/mp3")
            st.markdown("Audio with AI cues:")
            st.audio(interaction["audio"], format="audio/mp3")
            if "post_commands" in interaction:
                with st.expander("View Post-Processing Commands"):
                    for command in interaction["post_commands"]:
                        st.code(command)
                if "audio_post" in interaction:
                    st.markdown("Post-processed audio:")
                    st.audio(interaction["audio_post"], format="audio/mp3")

    # User Input
    if directive := st.chat_input(
        "e.g., 'Make it sound more excited' or 'Change the voice to a deep male voice'"
    ):
        if not st.session_state.input_text:
            st.error("Please enter Input Text in the sidebar first.")
            return

        with st.chat_message("user"):
            st.write(directive)

        with st.chat_message("assistant"):
            with st.spinner("Generating prosody plan..."):
                plan, updated_history = get_prosody_plan(
                    client,
                    st.session_state.input_text,
                    directive,
                    model,
                    st.session_state.message_history,
                )
                # st.session_state.current_plan = plan
                st.session_state.message_history = updated_history

            # Process the new plan
            xml = convert_plan_to_xml(st.session_state.input_text, plan)
            # st.session_state.last_xml = xml

            if avatar is None:
                avatar = plan.get("avatar")
            assert avatar is not None
            avatar_id = AVATAR_TO_ID.get(avatar)
            assert avatar_id is not None

            # st.code(xml, language="xml", wrap_lines=True)
            st.code(xml, language="xml")

            st.caption(
                f"Voice: {avatar} ({AVATAR_TO_CHARACTERISTICS.get(avatar)})"
            )

            # Generate Audio
            with st.spinner("Generating audio..."):
                audio_orig = generate_tts_sync(
                    speaker_id=avatar_id, text=st.session_state.input_text
                )
                audio = generate_tts_sync(speaker_id=avatar_id, text=xml)
                st.markdown("Audio with no cues:")
                st.audio(audio_orig, format="audio/mp3")
                st.markdown("Audio with AI cues:")
                st.audio(audio, format="audio/mp3")

                st.session_state.interactions.append(
                    {
                        "directive": directive,
                        "xml": xml,
                        "audio_orig": audio_orig,
                        "audio": audio,
                        "avatar": avatar,
                    }
                )
                st.session_state.message_history = updated_history

            with st.spinner(
                "Sending audio to gemini for analysis and post-processing"
            ):
                pp_schema = {
                    "type": "ARRAY",
                    "items": {
                        "type": "STRING",
                        "description": (
                            "A sox command to be run as a subprocess, will be run "
                            "individually as a pass over the audio in the order it "
                            "appears in the array. The first pass should take as input"
                            "a file called input.mp3 and the final pass should output"
                            "a file called output.mp3."
                        ),
                    },
                }

                success = False
                while not success:
                    pp_result, updated_history = get_postprocessing_plan(
                        client=client,
                        schema=pp_schema,
                        directive=directive,
                        system_prompt=SYSTEM_PROMPT,
                        audio_part=audio,
                        model=model,
                        message_history=st.session_state.message_history,
                    )

                    # Remove old audio files that might be lying around
                    paths = ["input.mp3", "output.mp3"]
                    for path in paths:
                        if os.path.exists(path):
                            os.remove(path)

                    with open("input.mp3", "wb") as f:
                        f.write(audio)
                    assert pp_result.text is not None
                    commands = json.loads(pp_result.text)
                    for command in commands:
                        try:
                            subprocess.run(
                                command,
                                shell=True,
                                check=True,
                                text=True,
                                capture_output=True,
                            )
                            success = True

                        except subprocess.CalledProcessError as e:
                            success = False
                            error_history = [
                                types.Content(
                                    role="model",
                                    parts=[
                                        types.Part.from_text(
                                            text=(
                                                f"Command executed: {command}"
                                                f"\nError message: {e.stderr}"
                                            )
                                        )
                                    ],
                                ),
                                types.Content(
                                    role="user",
                                    parts=[
                                        types.Part.from_text(
                                            text=(
                                                "That command didn't work. "
                                                "Can you correct it and try "
                                                "again?"
                                            )
                                        )
                                    ],
                                ),
                            ]
                            st.session_state.message_history += error_history
                            # st.error("post-processing failed; trying again")
                            break

                    if success:
                        for command in commands:
                            st.code(command)
                        with open("output.mp3", "rb") as f:
                            audio = f.read()
                        st.audio(audio, format="audio/mp3")
                        st.session_state.interactions[-1]["audio_post"] = audio
                        st.session_state.interactions[-1][
                            "post_commands"
                        ] = commands
                        st.session_state.message_history += updated_history


if __name__ == "__main__":
    main()
