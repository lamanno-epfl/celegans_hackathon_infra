# Chat backups

Raw Claude Code session transcripts (JSONL, one message per line). Useful for
recovering decision context. Inspect with `jq`:

```bash
jq -r '.message.content[]?.text // .message.content' 2026-04-15_session.jsonl | less
```
