# local-ai

A small Python chat app that talks to Ollama. Needs Python 3.11 or newer and `pip install -r requirements.txt`.

## Run it

From this folder:

```
python -m app.main --check     # test the connection and list installed models
python -m app.main             # start chatting
```

Inside the chat: `/models`, `/model qwen3:0.6b`, `/think on|off|default`, `/reset`, `/help`, `/exit`.
After each reply it prints tokens, tokens/sec, model load time, and total time, which is handy for the performance baseline.

`inference.think` is left empty by default, so qwen3 reasons first and Ollama hides the reasoning.
That reasoning is most of the wait on CPU. Setting it to `false` (or `--think off`, `/think off`) asks the
model to skip it, but the current qwen3 on the ai-pc reasons anyway and prints it.

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
- `prompts/system/system.txt` is the system prompt
