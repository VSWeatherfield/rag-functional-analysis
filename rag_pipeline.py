import os
import sys

sys.path.insert(0, "/kaggle/input/aoa-utils")

from subprocess_runner import run_command
from file_operations import (
    parse_file,
    sanitize_text_for_api,
    save_embeddings,
    load_embeddings,
)

from embedding_operations import (
    get_chunks_embeddings,
    get_prompt_embedding,
    find_most_similar,
    query_model_with_context,
    query_model_without_context,
)

os.chdir("/kaggle/working/")
print(os.getcwd())

run_command("curl -fsSL https://ollama.com/install.sh | sh")
os.system("/usr/local/bin/ollama serve &")

run_command("ollama pull mistral")
# run_command("ollama pull all-minilm:l12-v2")
run_command(
    'curl http://127.0.0.1:11434/api/chat -d \'{"model": "mistral", "stream": false, "messages": [{ "role": "user", "content": "Write a sentence as J. D. Salinger would write it" }]}\''
)


def main():
    filename = "data/lecture_notes_functional_analysis.txt"

    paragraphs = parse_file(filename)
    processed_paragraphs = sanitize_text_for_api(paragraphs)

    SYSTEM_PROMPT = """You are a helpful teaching assistant that answers only in russian,
        on functional analisys questions based on snippets of text provided in context.
        Answer only using the context provided, being as concise as possible. Give
        straight formal answers on discrete analysis questions as the answer was presented.
        Here are excerpts from the notes, some, i.e. not all of which you may find useful for yourself.
        Don't be afraid to use them:
    """

    embeddings = get_chunks_embeddings(filename, "mistral", processed_paragraphs)
    print("Embeddings generated successfully!")

    prompt = input("what do you want to know? -> ")

    prompt_embedding = get_prompt_embedding(prompt, modelname="mistral")
    most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:8]

    response = query_model_with_context(
        modelname="mistral",
        system_prompt=SYSTEM_PROMPT,
        prompt=prompt,
        most_similar_chunks=most_similar_chunks,
        paragraphs=processed_paragraphs,
    )

    if response is not None:
        print("Model response:")
        print(response["content"])
    else:
        print("Failed to get a response from the model.")


if __name__ == "__main__":
    main()
