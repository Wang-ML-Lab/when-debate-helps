# Implementation notes

This repository is an independent implementation based on the method description in *When Debate Helps: Proposal Supply and Verification-Aware Readout in Multi-Agent Reasoning*. It does not contain source files from the earlier experimental repository.

## Perturbations

For seed `s` and absolute noise scale `sigma`, the implementation evaluates `theta + sigma * epsilon(s)`, where `epsilon` is standard Gaussian noise. A persistent CPU snapshot of every selected floating-point tensor is created after model loading. Each candidate is applied in place and restored from that snapshot in a `finally` block. With `verify_restore: true`, every restored tensor is checked using exact `torch.equal` comparison.

This restoration strategy prioritizes correctness and requires host RAM approximately equal to the selected model weights. `include` and `exclude` regular expressions can restrict the perturbed tensors. The paper configuration perturbs all floating-point parameters.

## Coverage selection

AC-Greedy assigns score 1 to a correct candidate-question prediction and 0 otherwise. SC-Greedy clusters by normalized extracted answer and assigns each valid prediction the fraction of valid pool predictions in its cluster. An invalid extraction always receives score 0; it never forms an answer cluster. Both methods greedily maximize the mean question-level maximum score. AC-Greedy breaks equal objectives by construction accuracy and then candidate ID. SC-Greedy breaks equal objectives with a pseudorandom choice seeded by `--tie-break-seed` (default 42); it does not consult labels, correctness, or construction accuracy.

## Profile sharding

The multi-GPU launcher records every worker process and exits before merging if any worker fails. A successful merge validates that its candidate-ID set exactly equals the input seed bank, preventing partial or stale shards from silently producing an incomplete society pool.

## Debate and readout

Every initial response and round response is retained in `traces.jsonl`. The finalizer returns a unique majority answer when one exists. A tie or a fully invalid vote is sent to the unperturbed base model with a verification-style judge prompt, and the judge response is also retained.

Thicket agents use `generation.eval`, which is deterministic in the paper configuration. Sampled self-consistency and Vanilla Debate use the separate `generation.base` block with temperature 0.7 and top-p 0.95. The tie-break judge uses deterministic `generation.judge` settings.

The analyzer reports proposal hit, initial vote, final accuracy, recoverable headroom, conditional recovery, new correctness, and damage. It verifies the identity `gain = recovery_mass + new_correctness_mass - damage_mass` for every run.
