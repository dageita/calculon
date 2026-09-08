#!/usr/bin/env python3
"""Normalize local WikiText/ShareGPT assets to Megatron's {"text": ...} JSONL."""
import argparse, json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--source", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--limit", type=int, default=0)
a=p.parse_args(); a.output.parent.mkdir(parents=True, exist_ok=True)

def text_of(row):
  if isinstance(row, dict) and isinstance(row.get("text"), str): return row["text"]
  messages = row.get("messages", row.get("conversations", [])) if isinstance(row, dict) else []
  parts=[]
  for message in messages:
    if not isinstance(message, dict): continue
    role=message.get("role", message.get("from", "unknown"))
    content=message.get("content", message.get("value", ""))
    if isinstance(content, str): parts.append(f"{role}: {content}")
  return "\n".join(parts)

written=0
with a.output.open("w") as out:
  if a.source.suffix == ".jsonl":
    source=(json.loads(line) for line in a.source.open(errors="ignore") if line.strip())
  else:
    payload=json.load(a.source.open())
    source=payload if isinstance(payload, list) else [payload]
  for row in source:
    text=text_of(row).strip()
    if not text: continue
    out.write(json.dumps({"text": text}, ensure_ascii=False)+"\n")
    written += 1
    if a.limit and written >= a.limit: break
print(json.dumps({"source": str(a.source), "output": str(a.output), "documents": written}))
