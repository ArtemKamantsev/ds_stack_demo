---
name: chatgpt-share-import
description: Import a conversation from a ChatGPT shared-chat URL into the current task context, or save it to the file when requested.
---

# Import a ChatGPT shared conversation

These instructions contain all commands you need to execute, including chat import.
If the user provides a ChatGPT share link (named further as <share-link>) and asks to import or use its history, 
then follow next steps:

1. Accept links on `chatgpt.com/share/` and the legacy `chat.openai.com/share/` host.
2. Execute the command:
```bash
npx chatgpt2codex --codex-home ./import-chat-gpt-temp --force  <share-link>
```
3. The full path to the created .jsonl file is located after the "File:" prefix in console output.
4. If the user requested to save the conversation to the file, 
move the created file to the requested location with the requested name (if any provided).
5. Otherwise, add the created file to the current task context. 
Do not replace it with a summary unless the user asks for one.
6. Remove the temporary directory after the file is moved or added to the context:
```bash
rm -rf ./import-chat-gpt-temp
```
7. Execute requested actions on the imported information (if any was requested).