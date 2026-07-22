# Tokenizer-only validation — scope clarification (H4.7)

Companion to `docs/B1_TOKENIZER_ONLY_VALIDATION.json`. Three distinct
tokenizer-related operations exist on this branch. They are not the same
operation, and no one of them proves another:

## 1. The earlier tokenizer-only validation operation (the JSON record)

What `B1_TOKENIZER_ONLY_VALIDATION.json` records: a one-time, isolated,
copy-mode download of ONLY the tokenizer/config assets of
`deepseek-ai/DeepSeek-R1-Distill-Llama-8B@6a6f4aa4…` into a throwaway
temp directory, with a complete before/after file inventory proving no
weight file was fetched. It validated: the pinned tokenizer loads, its
chat template hash, the rendered prompt hash, and the prompt token
count/hash for the frozen example.

**What it does NOT prove:** it did not exercise
`kvcot.discovery.snapshot_boundary.resolve_local_snapshot` (it used an
isolated `local_dir` copy, not the shared HF cache), so it is *not*
evidence that the production local-snapshot resolution path works on the
GPU host, and it is *not* the local-only prompt-verification path the
coordinator runs at execute time.

## 2. The production `resolve_local_snapshot` path

`kvcot.discovery.snapshot_boundary.resolve_local_snapshot` resolves an
exact 40-SHA revision from the ordinary Hugging Face cache with
`local_files_only=True`, verifies identity via public cache metadata,
validates tokenizer/model file inventories (including safetensors
index/shard accounting), and exports the full F8 integrity evidence
(inventory hash, per-file sizes, index content hashes, referenced shards).
This path is exercised on this machine only by CPU unit tests against
synthetic directories; it has **never** been executed against a real
cached Llama-8B model snapshot, because no model weights have ever been
downloaded here.

## 3. The current local-only prompt-verification path

`kvcot.discovery.b2a_execute._verify_resolved_prompt_identity` (Gate H4.5
repair) resolves the exact local tokenizer snapshot via
`resolve_local_snapshot` first and then re-renders/re-tokenizes/re-hashes
the frozen prompt from that verified local path with
`local_files_only=True`. It runs inside the coordinator at execute time.
On this machine it has been exercised only by CPU tests with injected
fakes; it has never run against the real pinned tokenizer through the
production cache, because the shared-cache snapshot required by path (2)
has never been populated here.

## Summary table

| Operation | Executed for real on this machine? | Proves |
|---|---|---|
| (1) isolated tokenizer-only validation | Yes (2026-07, record in the JSON) | pinned tokenizer loads; prompt identity hashes; no weight download |
| (2) `resolve_local_snapshot` | CPU tests only (synthetic dirs) | code path correctness, not real-cache behavior |
| (3) execute-time local prompt verification | CPU tests only (injected fakes) | code path correctness, not real-cache behavior |

Claiming (1) as evidence for (2) or (3) — or vice versa — is an
overclaim; each requires its own execution evidence on the eventual GPU
host before any B2A gate can rely on it.
