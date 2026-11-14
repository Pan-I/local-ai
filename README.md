# local-ai

A small Python chat app that talks to Ollama. Needs Python 3.11 or newer and `pip install -r requirements.txt`.

## Run it

From this folder:

```
python -m app.main --check     # test the connection and list installed models
python -m app.main             # start chatting
```

Inside the chat: `/models`, `/model qwen3:0.6b`, `/think on|off|default`, `/history`, `/load <id>`, `/reset`, `/kb on|off`, `/sources`, `/index`, `/embed <name>`, `/help`, `/exit`.
After each reply it prints tokens, tokens/sec, model load time, and total time, which is handy for the performance baseline.

`inference.think` is left empty by default, so qwen3 reasons first and Ollama hides the reasoning.
That reasoning is most of the wait on CPU. Setting it to `false` (or `--think off`, `/think off`) asks the
model to skip it, but the current qwen3 on the ai-pc reasons anyway and prints it.

## Saved conversations

Every chat is saved to a SQLite file at `data/databases/local_ai.db` (set by `storage.database`).
The folder is created the first time you chat, and the file is ignored by git.

- `/history` lists your recent conversations with their ids.
- `/load 3` continues conversation 3, with the model you're using now.
- `/reset` starts a new conversation. The old one stays saved.
- `python -m app.main --no-save` chats without saving, or set `storage.save_conversations: false`.

Only your messages and the replies are saved, along with each reply's stats (tokens, tok/s, time).
The model's hidden thinking text isn't saved.

To look inside the file, any SQLite viewer works, such as "DB Browser for SQLite".

## Answering from your documents

The app can look things up in your own files and pass the relevant passages to the model,
then list which files (and lines or pages) it used. Nothing leaves your network: the
embeddings are made by Ollama on the ai-pc, and the index is a SQLite file on this PC.

How it works, briefly: each document is cut into chunks of about 1,000 characters. A small
*embedding model* turns each chunk into a list of numbers that describes its meaning, so chunks
about similar things get similar numbers. When you ask a question, it gets numbers too, and
the closest chunks are added to your message as numbered sources.

One-time setup:

1. On the ai-pc, install the embedding model: `ollama pull nomic-embed-text`
2. On this PC: `pip install -r requirements.txt` (adds `pypdf` and `python-docx` for PDF and Word files).
3. Put documents in the `knowledge/` folder, or list other folders under `knowledge.folders`
   in `config/local.yaml`. Only those folders are read. The contents of `knowledge/` are ignored by git.

Then:

```
python -m app.main --index           # read new and changed files into the index
python -m app.main --rebuild-index   # start the index over
python -m app.main --no-kb           # chat without your documents
```

Run `--index` again whenever files change. Unchanged files are skipped, and deleted ones are dropped.
It reads text, Markdown, code, PDF and Word (.docx) files, and skips hidden folders such as `.git`.
Scanned PDFs have no text to read, so they're reported as skipped.

In the chat: `/kb on|off` turns document lookup on or off, `/sources` shows the passages the last
answer was given with a preview, and `/index` updates the index without leaving the chat.
After each answer the app prints the sources and a match score from 0 to 1. Passages below
`knowledge.min_score` (0.5) aren't used. That number is a first guess, so check the scores with
`/sources` on a few real questions and raise or lower it.

Sources are sent with that one question only, so they don't fill up the model's memory in later
turns, and the saved conversation records which sources each reply used (`message_sources` table).
### Trying other embedding models

`/embed` shows the current embedding model and the ones installed on the ai-pc, and
`/embed qwen3-embedding:0.6b` switches to another for this chat (`--embed-model <name>` does
the same from the start). Each model keeps its own index file, such as
`data/indexes/knowledge-nomic-embed-text.db`, because one model's numbers can't be compared with
another's. The first time you switch to a model, run `/index` to build its index. After that,
switching back and forth is instant. Match scores differ between models, so a good
`min_score` for one may not suit another. To keep a model as the default, set
`knowledge.embedding_model` in `config/local.yaml`.

The index files are safe to delete; `--index` rebuilds them.

## Pointing it at the ai-pc

Settings come from `config/config.example.yaml`, with `config/local.yaml` merged on top for this machine.
The server address is `inference.base_url`. To use a different address for one run:

```
python -m app.main --base-url http://localhost:11434
```

### One-time setup on the ai-pc

By default Ollama only listens on localhost, so the primary PC can't reach it until you:

1. Set a system environment variable `OLLAMA_HOST` = `0.0.0.0` (Settings > System > About > Advanced system settings > Environment Variables), then quit Ollama from the tray and start it again.
2. Allow port 11434 in Windows Firewall from the primary PC only. In an admin PowerShell on the ai-pc:

   ```
   New-NetFirewallRule -DisplayName "Ollama LAN" -Direction Inbound -Protocol TCP -LocalPort 11434 -RemoteAddress <primary-pc-ip> -Action Allow
   ```

3. From the primary PC, run `python -m app.main --check`.

## Tests

```
python -m unittest discover tests
```

The tests use a fake Ollama server, so they pass without the ai-pc.

## Files

- `app/config.py` loads the YAML config from `config/`
- `app/ollama_client.py` is the only code that knows Ollama's API
- `app/conversation.py` keeps the message history for a chat
- `app/main.py` is the command-line chat
- `storage/database.py` saves conversations to SQLite
- `retrieval/readers.py` reads text from text, code, PDF and Word files
- `retrieval/chunker.py` cuts text into chunks with line numbers
- `retrieval/indexer.py` finds files in the approved folders and embeds them
- `retrieval/store.py` keeps chunks and embeddings in SQLite and searches them
- `retrieval/retriever.py` finds passages for a question and adds them to the prompt
- `prompts/system/system.txt` is the system prompt
