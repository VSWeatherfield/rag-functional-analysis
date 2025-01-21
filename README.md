# LLM Course - Second Task: RAG Pipeline Using Free LLM Models (vLLM, Ollama, Mistral)

This project implements a Retrieval-Augmented Generation (RAG) pipeline leveraging free large language models (LLMs) and APIs. It is designed to retrieve knowledge and generate context-aware responses efficiently.


## Knowledge Database

The pipeline uses the following resources as the knowledge base:

- [Lecture Notes on Functional Analysis](https://vk.com/wall-213738964_17) by [Candy Club](https://vk.com/mipt_candy_club?from=search&search_track_code=60f1250bKk10A7-rofXLcAYI_5tZbmowNHgMCb0bmh9_NCd4JX18t9_xKWJKMj4oZ2qe5RUwNGRcOFFR-Uz0LWopPBAkHQ)
- [Lecture Notes on Discrete Analysis](https://vk.com/wall-213738964_8) by the same group ❤️  
- [Peter Pan](https://www.gutenberg.org/cache/epub/16/pg16-images.html) from Project Gutenberg  


## Models Used

The following models were employed in this project:

- **Mistral**: Used for generating embeddings and answering queries.  
- **BGE-Base**: Used for embeddings creation.  
- **all-minilm:l12-v2**: Tested for russian embeddings creation (but didn't succeed much :( )


## RAG Pipeline Overview

Since this project was developed on a system without GPU support, most calculations were performed in a Kaggle notebook. The Ollama API was used as a subprocess, and queries were addressed via the `curl` command.

Apart from `rag_pipeline.py`, containing the pipeline, check out the accompanying notebook, **`notebooks/rag_kaggle.ipynb`**, for a step-by-step code with interactive outputs.


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


## Results

RAG Results with Lecture Notes on Discrete Analysis

| **PROMPT**   | **Mistral (with basic prompt)**   | **Mistral (with basic prompt and discrete analysis lecture notes)**   |
|--------------|-----------------------------------|-----------------------------------------------------------------------|
|`В каком году была доказана Теорема Алсведе-Хачатряна`| Теорема Алсведе-Хачатрян была доказана в 1983 году (неверно)| 1996 год.| 
|`В каком году была доказана теорема Франкла Вилсона`| 1936 год (неверно)| 1981 год.|
|`Напиши мне теоремы о кликовом числе и числе независимости`| 1. Теорема об эквивалентности числа клик и числа независимости (Теорема Эрдёша-Корадя) утверждает, что для любого фиксированного числа $k$ существуют графы с $n$ вершинами такими, что их число клик больше или равно их числу независимости, и существуют графы с $n$ вершинами такие, что их число клик меньше или равно их числу независимости. Эта теорема показывает, что число клик не всегда определяет структуру графа, а может быть связано с более общими свойствами графа.<br><br>2. Теорема Кёрчера-Стоуна утверждает, что для любого $n\geq 3$, существует граф, в котором число клик равно 1 и число независимости равно n-1. Эта теорема показывает, что можно построить граф с малым числом клик и большим числом независимости, то есть граф со специальной структурой.<br><br>3. Теорема об ограниченном росте кликового числа утверждает, что для любого фиксированного $k$ существуют $n_0$, такое, что любые графы с $n\geq n_0$ вершинами имеют не более чем k клик. Эта теорема показывает, что количество клик в больших графах ограничено фиксированным числом $k$. | 1. Теорема о кликовом числе:<br>§§omega(G) leqslant sum<k0><p> Cmk < s(p)§§<br>Эта теорема утверждает, что кликовое число графа G (ω(G)) не превышает сумму по k от 0 до p степеней многочлена m(k), где m(k) - многочлен, заданный в форме суммы Cmk.<br><br>2. Теорема о числе независимости:<br>§frac<Cn<(frac<2  sqrt<2>><2>)>><sum<k0><p1> Cnk>§<br>Эта теорема утверждает, что число независимых множеств в графе G (α(G)) меньше, чем сумма по k от 0 до p1 степеней многочлена n(k), где n(k) - многочлен, заданный в форме sum<k0><p1> Cnk. При этом коэффициент перед n^k в многочлене n(k) больше, чем корень из 2 по степени k. |


Results on Functional Analysis Lecture Notes 

| **PROMPT**   | **Mistral (with basic prompt)**   | **Mistral (with basic prompt and discrete analysis lecture notes)**   |
|--------------|-----------------------------------|-----------------------------------------------------------------------|
|`Что такое §mathcal<L>(E1 E2)§ ?`|Функция секция $\mathcal{L}(E_1, E_2)$ - это функция, определяющая множество всех возможных функций $f$, таких что $f: E_1 \to E_2$. В других словах, это функция, которая принимает на вход два множества и возвращает множество всех функций от первого множества во второе. Эта концепция является важной для функционального анализа.|Пространство линейных ограниченных операторов действующих из $E1$ в $E2$. Формально, это линейное пространство над полем $mathbb{K}$, состоящее из всех таких линейных операторов, которые можно представить как комбинацию скалярных множителей и сумм отдельных операторов.| 
|`Дай определение компактного метрического пространства`|Компактное метрическое пространство — это топологическое пространство, в котором каждая открытая покрывающая система имеет конечное подпокрытие. В metrical space означает пространство с метрикой (разстоянием между элементами), а топологическое пространство — пространство с заданной структурой окрестностей каждого элемента, определяемой открытыми множествами. Такая структура позволяет говорить об особых свойствах набора точек (или элементов), например о компактности.|Метрическое пространство $X$ называется **компактным**, если для любой системы открытых множеств $\left\{ G_\alpha \right\}_{\alpha \in A} \subset 2X$, где $A$ - некоторое индексное множество, такой что $\bigcup_{\alpha \in A} G_\alpha = X$ существует конечный набор $\left\{ \alpha_1, ..., \alpha_n \right\} \subset A$, такой что $\bigcup_{k=1}^n G_{\alpha_k} = X$.|
|`Что такое оператор Вольтерра`|Оператор Вольтерра - это линейный дифференциальный оператор второго порядка, используемый в теории уравнений дифференциальных уравнений. Он определяется как следующий: L[f] = d²f/dt² - df/dt - k^2f где f(t) - функция, k - константа (коэффициент Лапласа). Данный оператор определяет вид решения дифференциального уравнения 2-го порядка с конечными производными по старшим точкам. Он назван в честь французского математика Жозефа Лювилля (Joseph Liouville), который изучал его приложения к физике, а также итальянского математика Оресте Вильгельма Вольтерра, который ранее изучал схожие операторы.|Оператор Вольтерра - это линейный непрерывный опертор из пространства $E1 = C(0,1)$ в себя же, заданный формулой $(Af)(x) = \int\limits_0^x f(t)dt$. Таким образом, если $f \in E1$, то $Af$ будет функцией из $E2 = C(0,1)$, а именно интегралом $\int\limits_0^x f(t)dt$.|

As you see, turns our Mistral knows answers to some of the questions too, but the one with RAG are right (or at least better)

## Further Questions

For any questions, feedback, or permissions to use parts of this code or data, please contact: \
**Vladimir Smirnov**  \
`voff.smirnoff@gmail.com`