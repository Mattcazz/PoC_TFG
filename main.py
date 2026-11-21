import sys
import os

from dotenv import load_dotenv

load_dotenv()  # before importing langfuse so the keys are picked up

from langfuse import get_client
from langfuse.openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

MODEL = os.getenv('MODEL')

SYSTEM_PROMPT = (
    "You are a dialog writer. You are going to receive two characters and a story. "
    "Your job is to generate a conversation between the two based on their personalities. /no_think"
)


def read_file(path):
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def main():
    history_file = sys.argv[1] if len(sys.argv) > 1 else "test_1/story.txt"
    hero_file = sys.argv[2] if len(sys.argv) > 2 else "test_1/hero.txt"

    story = read_file(history_file)
    heroes = read_file(hero_file)

    user_prompt = f"Story:\n{story}\n\nCharacters:\n{heroes}"

    response = client.chat.completions.create(
        name="dialog-writer",
        metadata={"story_file": history_file, "hero_file": hero_file, "langfuse_user_id": os.getenv('USERNAME'),},
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
        stream_options={"include_usage": True},  # so token counts get recorded
        reasoning_effort="none",  # replaces think=False, see note below
    )

    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()

    get_client().flush()  # short-lived script: send the trace before exit


if __name__ == "__main__":
    main()
