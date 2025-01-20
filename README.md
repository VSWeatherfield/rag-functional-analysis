# LLM Course - Second Task: RAG Pipeline Using Free LLM Models (vLLM, Ollama, Mistral)

This project implements a Retrieval-Augmented Generation (RAG) pipeline leveraging free large language models (LLMs) and APIs. It is designed to retrieve knowledge and generate context-aware responses efficiently.

---

## Knowledge Database

The pipeline uses the following resources as the knowledge base:

- [Lecture Notes on Functional Analysis](https://vk.com/wall-213738964_17) by [Candy Club](https://vk.com/mipt_candy_club?from=search&search_track_code=60f1250bKk10A7-rofXLcAYI_5tZbmowNHgMCb0bmh9_NCd4JX18t9_xKWJKMj4oZ2qe5RUwNGRcOFFR-Uz0LWopPBAkHQ)
- [Lecture Notes on Functional Analysis](https://vk.com/wall-213738964_8) by the same group ❤️  
- [Peter Pan](https://www.gutenberg.org/cache/epub/16/pg16-images.html) from Project Gutenberg  

---

## Models Used

The following models were employed in this project:

- **Mistral**: Used for generating embeddings and answering queries.  
- **BGE-Base**: Used for embeddings creation.  

---

## RAG Pipeline Overview

Since this project was developed on a system without GPU support, most calculations were performed in a Kaggle notebook. The Ollama API was used as a subprocess, and queries were addressed via the `curl` command.

This is the initial commit of the project, focusing on the correctness of the pipeline. Apart from `main.py`, check out the accompanying notebook, **`data/rag_kaggle.ipynb`**, for a step-by-step explanation and interactive outputs.

---

## Process

1. **Install and Set Up**  
   - Download and run Ollama using `curl` as a subprocess.  
   - Similarly, download Mistral and BGE-Base for embedding generation.

2. **Knowledge Base Embeddings**  
   - Compute embeddings for the knowledge database using one of the supported models.

3. **Retrieve Relevant Chunks**  
   - Identify the top-k samples with the highest cosine similarity to the query.

4. **Create Query Prompt**  
   - Combine the retrieved chunks into a query using the following format:

   ```python
   prompt = f"""
   Context information is below.
   ---------------------
   {retrieved_chunk}
   ---------------------
   Given the context information and not prior knowledge, answer the query.
   Query: {question}
   Answer:
   """
   ```

5. **Generate Response**
   - Send the query to the model via the curl command and obtain the answer.

## Example Output

The interactive notebook contains detailed outputs for multiple sample queries, demonstrating the pipeline's functionality.

## Further Questions

For any questions, feedback, or permissions to use parts of this code or data, please contact:
**Vladimir Smirnov**  
`voff.smirnoff@gmail.com`