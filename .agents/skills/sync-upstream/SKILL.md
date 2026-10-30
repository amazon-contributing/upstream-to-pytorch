---
name: sync-upstream
description: Sync this fork (amazon-contributing/upstream-to-pytorch) with pytorch/pytorch main by merging upstream/main, resolving conflicts together with the user, and opening a draft PR. Use when the user asks to sync, update, or pull in upstream, or to resolve fork-vs-upstream merge conflicts.
---

# Sync the fork with upstream PyTorch

The fork carries device-agnostic test changes for PrivateUse1 (Neuron). A sync
conflicts mostly where upstream has since landed an equivalent change. The user
must stay in the loop on every conflict: show it, recommend, wait for approval.

## Rules

- **Merge, never rebase.** Rebase replays every fork commit and re-conflicts on
  the same files; commits that cancel out (add then revert) conflict for no
  reason. The fork has always synced with merge commits.
- **Land it with a merge commit, never squash or rebase-merge.** Both drop the
  `upstream/main` parent, so the next sync re-conflicts on everything. The repo
  setting `allow_merge_commit` may be off; check with
  `gh api repos/amazon-contributing/upstream-to-pytorch -q .allow_merge_commit`
  and ask the user to enable it (or fast-forward `main`, which is protected and
  needs a code-owner review).
- **Sides are swapped vs a rebase.** In a merge, `HEAD` ("current") is the
  fork and "incoming" is upstream. Always label hunks "fork (HEAD)" and
  "upstream (incoming)"; never say "current/incoming" alone.
- **Do not resolve silently.** For each file show the hunks, say which side you
  recommend and why, and wait. Do not `git add`, commit, push, or open a PR
  without the user asking (see `CLAUDE.md` AI policy: PR text must be approved
  and the PR opened as a draft).
- No `git push` to `main`; do not bypass branch protection.

## Procedure

1. **Preconditions.** `git status` must have no tracked changes (untracked
   fork files like `UPSTREAMING_TRACKER.md` and `patches/` are fine). Remote
   `upstream` must be `https://github.com/pytorch/pytorch.git`; if missing,
   ask. Use zsh arrays for file lists (`files=(a b)`; `$files` is not split).
2. **Branch and merge.** `git checkout -b sync/<date>`, `git fetch upstream
   main`, `git merge upstream/main`. Record the upstream SHA. Do not abort a
   half-resolved merge without asking; `git merge --abort` discards work.
3. **Triage.** List `git diff --name-only --diff-filter=U` with hunk counts
   (`grep -cE '^<<<<<<<' f`). Group by area and resolve shared infra first
   (`torch/testing/_internal/*`), since tests depend on it. Look at the shape
   of big files before going hunk by hunk: 30+ hunks that differ by one token
   usually have one mechanical answer.
4. **Per-file analysis** (read upstream before recommending):
   - `git show upstream/main:<file>` to see how upstream does it now.
   - If upstream already has an equivalent (`instantiate_device_type_tests`,
     `torch.accelerator`, `get_default_backend_for_device`, DeviceInterface
     registry), recommend taking upstream.
   - Keep the fork side only where it adds PrivateUse1 behavior upstream lacks
     (e.g. whitelists of NCCL/XCCL that exclude other backends); say so.
   - Modify/delete conflicts: check why upstream deleted
     (`git log upstream/main --diff-filter=D -- <file>`); if the code is gone,
     `git rm` the fork's copy.
   - Prefer upstream names. A fork-only rename (`devices_for_rank`) is churn and
     breaks files that auto-merged; if reverting, `git grep` every caller.
   - A file marker-free after the user's edit may still carry stale fork hunks
     and silently-merged fork changes: check
     `git diff upstream/main -- <file>` before staging.
5. **NCCL-only test files.** Upstream makes `test_c10d_nccl.py` NCCL-specific.
   Do not genericize it in place. Before taking upstream wholesale, check what
   the Neuron nightly runs from it (below); if tests pass there only because of
   fork edits, move those into a separate fork-owned file
   (`test/distributed/test_c10d_privateuse1.py`) instead of fighting 30+ hunks
   every sync. Save the fork delta first:
   `git diff $(git merge-base HEAD upstream/main) HEAD -- <file> > agent_space/<name>.patch`.
6. **Stage when marker-free.** Strict check:
   `grep -qE '^(<<<<<<<|>>>>>>>)' f` (a bare `=======` can be an rst heading).
   Re-run the unmerged list after each batch and report what is left.
7. **Verify** (say plainly what could not be run): `git grep` for any renamed
   or removed symbol; `git diff --cached upstream/main --stat` to see the
   remaining fork delta; `lintrunner -a` on files you created (new test
   classes need `hw_classification`); `python3 scripts/codeowners/check_taxonomy.py`
   after the commit (it fails mid-merge by design). Tests need a Neuron
   machine; do not claim they pass otherwise.
8. **Commit** only when asked. Message: why the sync, notable resolutions, Test
   Plan with the literal commands actually run, AI-assistance disclosure. Run
   `lintrunner -a` first; if you edit staged files afterwards, re-add and
   `git commit --amend` (unpushed only).
9. **PR.** Push the branch (`git push -u origin sync/<date>`), then show the
   user the exact title/body (AI-marked) and open a draft PR only after
   approval. Mention that review is required (code owner) and that it must be
   merged with "Create a merge commit".

## Neuron nightly coverage (aws-neuron/private-torch-neuronx)

Read-only, via `gh`: `gh run list --repo aws-neuron/private-torch-neuronx
--workflow "Nightly Wheel Build"`, then the job log through
`gh api repos/aws-neuron/private-torch-neuronx/actions/jobs/<id>/logs`. The
test list is `tests/pytorch_tests_native/upstream_pytorch_test_distribution_config.json`
(whole dirs: `distributed`, `distributed/fsdp`, `dynamo`, `profiler`, ...).
Count PASSED/SKIPPED/XFAIL per file from the log (`] PASSED ... test/<file>::Class::name`).
Skip and xfail lists there are keyed by test id: after moving tests to a new
file, tell the user those lists need updating (do not edit that repo).

## After the PR merges

Update the nightly skip/xfail ids, add rows to `UPSTREAMING_TRACKER.md`, and
note any fork change that upstream made redundant so it can be dropped.
