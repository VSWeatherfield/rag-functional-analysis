
import json
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

from file_operations import save_embeddings, load_embeddings
from subprocess_runner import run_command, make_api_request


def parse_json_lines(raw_response):
    """
    Parses a raw response containing JSON objects per line.

    Args:
        raw_response (str): Raw API response.

    Returns:
        str: Aggregated content from the parsed JSON objects.
    """
    aggregated_content = ""
    for line in raw_response.splitlines():
        try:
            chunk = json.loads(line)
            if "message" in chunk and "content" in chunk["message"]:
                aggregated_content += chunk["message"]["content"]
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line: {line}")
    return aggregated_content


def get_chunks_embeddings(filename, modelname, chunks):
    """
    Fetches embeddings for the given chunks of text.

    Args:
        filename (str): Filename for saving/loading embeddings.
        modelname (str): Name of the embedding model.
        chunks (list of str): Text chunks to process.

    Returns:
        list: Generated embeddings.
    """
    if (embeddings := load_embeddings(filename)) is not False:
        return embeddings

    embeddings = []
    for chunk in tqdm(chunks, desc="Processing chunks", unit="chunk"):
        try:
            payload = {"model": modelname, "prompt": chunk}
            response = make_api_request("http://127.0.0.1:11434/api/embeddings", payload)

            if "error" in response:
                print(f"API error: {response['error']}")
                print(f"chunk:{chunk}")
                continue

            embeddings.append(response["embedding"])
        except (RuntimeError, json.JSONDecodeError) as e:
            print(f"Error processing chunk: {chunk}\nDetails: {e}")
            continue

    save_embeddings(filename, embeddings)
    return embeddings


def get_prompt_embedding(prompt, modelname="mistral"):
    """
    Fetches the embedding for a single prompt.

    Args:
        prompt (str): Input text prompt.
        modelname (str): Name of the embedding model.

    Returns:
        list: Embedding vector for the given prompt.
    """
    try:
        payload = {"model": modelname, "prompt": prompt}
        response = make_api_request("http://127.0.0.1:11434/api/embeddings", payload)

        if "error" in response:
            raise RuntimeError(f"API error: {response['error']}")

        return response["embedding"]
    except (RuntimeError, json.JSONDecodeError) as e:
        print(f"Error fetching embedding for prompt: {prompt}\nDetails: {e}")
        return None


def query_model_with_context(
    modelname, system_prompt, prompt, most_similar_chunks, paragraphs
):
    """
    Queries the model with context from similar chunks.

    Args:
        modelname (str): Name of the model to query.
        system_prompt (str): System-level instructions for the model.
        prompt (str): User's prompt.
        most_similar_chunks (list of tuples): Indices of similar chunks.
        paragraphs (list of str): Text chunks from the book.

    Returns:
        dict: Aggregated response from the model.
    """
    try:
        system_content = system_prompt + "\n".join(
            paragraphs[item[1]] for item in most_similar_chunks
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]

        payload = {"model": modelname, "messages": messages}
        escaped_payload = json.dumps(payload).replace('"', '\\"')
        raw_response = run_command(
            f'curl -s http://127.0.0.1:11434/api/chat -d "{escaped_payload}"'
        )

        aggregated_content = parse_json_lines(raw_response)
        return {"content": aggregated_content}
    except (RuntimeError, json.JSONDecodeError, TypeError, ValueError) as e:
        print(f"Error querying model with context: {prompt}\nDetails: {e}")
        return None


def query_model_without_context(modelname, system_prompt, prompt):
    try:
        system_content = system_prompt

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]

        payload = {"model": modelname, "messages": messages}
        escaped_payload = json.dumps(payload).replace('"', '\\"')
        raw_response = run_command(
            f'curl -s http://127.0.0.1:11434/api/chat -d "{escaped_payload}"'
        )

        aggregated_content = parse_json_lines(raw_response)
        return {"content": aggregated_content}
    except (RuntimeError, json.JSONDecodeError, TypeError, ValueError) as e:
        print(f"Error querying model with context: {prompt}\nDetails: {e}")
        return None


def find_most_similar(needle, haystack):
    """
    Finds the most similar embeddings.

    Args:
        needle (np.ndarray): Embedding to search for.
        haystack (list of np.ndarray): List of embeddings to search in.

    Returns:
        list of tuple: Sorted list of (similarity_score, index).
    """
    needle_norm = norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)
