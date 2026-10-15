# Questions to Revisit

## Chunking
- [ ] How exactly does the overlap logic work? (the `buffer_tokens[:] = ...` line)
- [ ] Why does chunk size range from 123 to 921 tokens instead of being uniform?

## Embeddings (before we start)
- [ ] What does "1536-dimensional vector" actually mean visually?
- [ ] How does "similar meaning = similar numbers" work in practice?

## General
- [ ] Why do we use virtual environments?
- [ ] How does tiktoken tokenize differently from splitting on spaces?

## Concepts I need to understand at a deeper level someday
- [ ] Cosine similarity vs Euclidean distance
- [ ] What is BPE (Byte Pair Encoding)?

## thresholds are heuristic. 0.45 vs 0.4 is less meaningful than 0.7 vs 0.5."

## Known issues / future improvements

### Q5 — Judge false negative
- **Question:** "What is the process for re-examination?"
- **System output:** Correct answer citing pages 29-30
- **Judge output:** faithfulness=0.00 (flagged as unsupported)
- **Root cause:** Judge context truncated at 3000 chars; source text likely cut
- **Fix:** Increase judge context to full chunks, or use a stronger judge model
- **Status:** Documented as limitation, not fixed

### Retrieval accuracy baseline: 100% (10/10)
### Avg latency: 1.36s
### Avg faithfulness: 0.83
### Avg relevancy: 0.93