import os
import json
import re

def sanitize_text_for_api(paragraphs):
    """
    Sanitizes text paragraphs for API compatibility by replacing problematic characters,
    including double quotes, with placeholders or safe equivalents.
    
    Parameters:
        paragraphs (list of str): List of text paragraphs to sanitize.

    Returns:
        list of str: Sanitized text paragraphs.
    """
    processed = []
    for paragraph in paragraphs:
        paragraph = re.sub(r'[“”]', '§q§', paragraph)  # Normalize and replace double quotes
        paragraph = re.sub(r"[‘’]", "§s§", paragraph)

        paragraph = paragraph.replace("$", "§")  # Replace $ with a placeholder
        paragraph = paragraph.replace("{", "<").replace("}", ">")  # Replace {} with <>
        paragraph = paragraph.replace("[", "(").replace("]", ")")  # Replace [] with ()

        paragraph = re.sub(r"(?<!\\)\|", r"\\|", paragraph)  # Escape |
        paragraph = re.sub(r"(?<!\\)\*", r"\\*", paragraph)  # Escape *
        paragraph = re.sub(r"(?<!\\)%", r"\\%", paragraph)  # Escape %

        paragraph = re.sub(r"\\", "", paragraph)

        paragraph = re.sub(r"[^a-zA-Z0-9а-яА-Я\s<>\(\)\§\|\\*%§q§§s§]", "", paragraph)

        paragraph = paragraph.strip()
        processed.append(paragraph)

    return processed





def parse_file(filename):
    """
    Parses a text file into paragraphs.

    Args:
        filename (str): Path to the text file.

    Returns:
        list of str: List of paragraphs.
    """
    with open(filename, encoding="utf-8-sig") as f:
        paragraphs = []
        buffer = []
        for line in f.readlines():
            line = line.strip()
            if line:
                buffer.append(line)
            elif len(buffer):
                paragraphs.append(" ".join(buffer))
                buffer = []
        if len(buffer):
            paragraphs.append(" ".join(buffer))
        return paragraphs


def save_embeddings(filename, embeddings):
    """
    Saves embeddings to a JSON file.

    Args:
        filename (str): Filename for saving embeddings.
        embeddings (list): Embeddings data.

    Raises:
        OSError: If there is an issue creating the directory or file.
    """
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")

    base_filename = os.path.basename(filename)
    output_path = f"embeddings/{base_filename}.json"
    with open(output_path, "w") as f:
        json.dump(embeddings, f)


def load_embeddings(filename):
    """
    Loads embeddings from a JSON file.

    Args:
        filename (str): Original input filename (including its path).

    Returns:
        list or bool: Embeddings data, or False if the file doesn't exist.
    """
    base_filename = os.path.basename(filename)

    filepath = f"embeddings/{base_filename}.json"

    if not os.path.exists(filepath):
        return False

    with open(filepath, "r") as f:
        return json.load(f)
