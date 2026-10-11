---
description: Summarize the OptChat history with a compactor subagent
---

Start one `optchat-compactor` subagent in the background and continue with the current task. When it reports rotate, start one new compactor. When it reports done, waiting or blocked, tell me the result in one line. Do not suggest another run for messages that arrived during the run.
