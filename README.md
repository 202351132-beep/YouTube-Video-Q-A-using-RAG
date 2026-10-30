# YouTube Video Q&A with RAG

Ask questions about a YouTube video and get answers taken from what was actually said in it.

The notebook pulls a video's transcript, splits it into chunks, embeds them into a FAISS vector store, and uses GPT-4o-mini to answer questions using only the chunks most relevant to each question. If the transcript doesn't contain the answer, the model is told to say it doesn't know instead of guessing.

Tested on the Lex Fridman podcast episode with Demis Hassabis (video ID `Gfr50f6ZBvo`), a conversation of over two hours.

## Example

**Question:** Is the topic of nuclear fusion discussed in this video? If yes, what was discussed?

**Answer:**
> Yes, the topic of nuclear fusion is discussed in the video. The discussion includes collaboration with EPFL in Switzerland, which has a test reactor used for various experiments. The speaker mentions identifying bottleneck problems in fusion and applying AI methods to address them. They highlight a specific achievement of holding plasma in specific shapes for a record amount of time...

The answer comes from the part of the podcast where Hassabis talks about DeepMind's plasma-control work with EPFL. That section was retrieved from 168 chunks.

## How it works

```
YouTube video ID
      │
      ▼
1. Load        youtube-transcript-api  →  full transcript as plain text
      │
      ▼
2. Split       RecursiveCharacterTextSplitter (1,000 chars, 200 overlap)  →  168 chunks
      │
      ▼
3. Embed       OpenAI text-embedding-3-small  →  one vector per chunk
      │
      ▼
4. Store       FAISS vector index
      │
      ▼
5. Retrieve    top 4 chunks most similar to the question
      │
      ▼
6. Generate    GPT-4o-mini (temperature 0.2) answers from those 4 chunks only
```

Steps 5 and 6 are combined into one LangChain chain:

```python
parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})
main_chain = parallel_chain | prompt | llm | StrOutputParser()

main_chain.invoke("Who is Demis?")
```

The prompt keeps the model grounded in the transcript:

```
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.
```

## Settings

| Setting | Value | Why |
|---|---|---|
| Chunk size | 1,000 characters | About a short paragraph of speech, usually one idea per chunk |
| Chunk overlap | 200 characters | A sentence cut at a boundary still appears whole in the next chunk |
| Embedding model | `text-embedding-3-small` | Cheap and good enough for semantic search over one transcript |
| Retriever | FAISS similarity search, k = 4 | Enough context for a specific question without adding noise |
| LLM | `gpt-4o-mini`, temperature 0.2 | Low temperature keeps answers close to the source text |

## Setup

Requires Python 3.9+ and an OpenAI API key. Easiest to run in Google Colab.

```bash
pip install youtube-transcript-api langchain-community langchain-openai faiss-cpu tiktoken python-dotenv
```

Open `rag_using_langchain.ipynb` and run the cells in order. The first cell asks for your OpenAI API key, or reads it from the `OPENAI_API_KEY` environment variable if it's already set. Don't hard-code the key in the notebook.

To use a different video, change the ID in the indexing cell. The ID is the part after `v=` in the URL:

```python
video_id = "Gfr50f6ZBvo"   # youtube.com/watch?v=Gfr50f6ZBvo
```

Then ask your own question:

```python
main_chain.invoke("What does he say about AlphaFold?")
```

## Limitations

- **Needs captions.** It only works on videos that have a transcript, either uploaded or auto-generated. Videos with captions disabled will raise `TranscriptsDisabled`.
- **Auto-caption errors carry through.** In the test video, "Demis Hassabis" is transcribed as "demus hasabis" and "AlphaGo Zero" as "alfred zero". Semantic search handles this better than keyword search, but quotes in answers can still contain these errors.
- **Weak at whole-video questions.** "Summarise the video" only sees 4 of the 168 chunks, about 2% of the transcript, so the summary covers a few topics and misses most of the conversation. RAG suits specific questions.
- **No formal evaluation yet.** Answers were checked by hand against the video on a handful of questions.
- **English only tested.** The transcript API can fetch other languages, but this hasn't been tried.

## Possible improvements

- Fall back to speech-to-text (e.g. Whisper) for videos without captions
- Map-reduce summarisation for whole-video questions
- Keep transcript timestamps as metadata so each answer can link to the exact moment in the video
- A small test set of questions with known answers, to measure retrieval and answer accuracy
- A simple Streamlit front end where you paste a URL and chat

## Tech stack

Python, LangChain, OpenAI API (GPT-4o-mini, text-embedding-3-small), FAISS, youtube-transcript-api, Google Colab
