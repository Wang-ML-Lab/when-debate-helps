# Data format

The code intentionally does not redistribute benchmark data. Convert each split to UTF-8 JSONL with one record per question:

```json
{"id":"question-1","question":"...","choices":["...","...","...","..."],"answer":"B"}
```

Required fields are `id`, `question`, and `answer`. `choices` is required for `multiple_choice` tasks and omitted for free-form exact-answer tasks. An optional `metadata` mapping is preserved by the loader. IDs must be unique within a file.

Run `wdh validate-data path/to/split.jsonl` before launching an experiment. Construction and test files must be disjoint. Dataset licenses and access conditions remain the user's responsibility.
