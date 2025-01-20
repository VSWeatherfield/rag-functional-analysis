import os
import sys

sys.path.insert(0, "/kaggle/input/rag-utils")

from subprocess_runner import run_command
from file_operations import parse_file, save_embeddings, load_embeddings
from embedding_operations import (
    get_chunks_embeddings,
    get_prompt_embedding,
    find_most_similar,
    query_model_with_context,
)

os.chdir("/kaggle/working/")
print(os.getcwd())

run_command("curl -fsSL https://ollama.com/install.sh | sh")
os.system("/usr/local/bin/ollama serve &")


def main():
    SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions 
        based on snippets of text provided in context. Answer only using the context provided, 
        being as concise as possible. If you're unsure, just say that you don't know.
        Context:
    """

    filename = "data/peter_pan.txt"
    paragraphs = parse_file(filename)

    embeddings = get_embeddings(filename, "nomic-embed-text", paragraphs)

    prompt = input("what do you want to know? -> ")

    prompt_embedding = get_prompt_embedding(prompt, modelname="mistral")
    most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:5]

    response = query_model_with_context(
        modelname="mistral",
        system_prompt=SYSTEM_PROMPT,
        prompt=prompt,
        most_similar_chunks=most_similar_chunks,
        paragraphs=paragraphs,
    )

    if response is not None:
        print("Model response:")
        print(response["content"])
    else:
        print("Failed to get a response from the model.")


if __name__ == "__main__":
    main()
