---
description: Summarize the OptChat history, on API credit when a key works, or else with a compactor subagent
---

Run `{RUN} compact` with Bash. When it exits with status 0, tell me its output in one line and stop. Otherwise start one `optchat-compactor` subagent in the background and continue with the current task. When it reports rotate, start one new compactor. When it reports done, waiting or blocked, tell me the result in one line, together with the reason from the output of `{RUN} compact`. Do not suggest another run for messages that arrived during the run.
