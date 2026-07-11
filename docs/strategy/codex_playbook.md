# Codex Agentic Coding Playbook

## What’s new in Codex
- Continuity: Sessions persist across the Codex Terminal, IDE, web, and mobile interfaces so an agent can swap devices without losing context or running jobs.[^codex-blog]
- Performance & reliability: Upgrades reduce latency spikes, make tool execution more predictable, and keep longer terminal sessions alive so background work can finish.[^codex-blog]
- Workspace awareness: Codex keeps track of open files, terminal history, and browser tabs, which lets agents “time travel” to earlier checkpoints instead of re-running work.[^codex-blog]

## Operational tactics for agentic coding
- **Task sizing**
  - Break goals into <90-minute “missions.” Anything larger should become a parent plan with explicitly sequenced sub-tasks and acceptance criteria.
  - Start every mission with a quick asset inventory (repo layout, outstanding TODOs) and a success checklist.
- **Checkpointing & state management**
  - After each significant change, summarize: files touched, open questions, follow-up tests. Store the summary in the task log so future agents can resume.
  - Use Git intentionally: feature-sized commits with message structure `context -> action -> validation`. Push partial work to a draft branch only when you have failing tests you need to discuss.
- **CI gates & validation**
  - Declare the tests or linters that must pass before hand-off; rerun “fast” suites (formatters, unit tests) locally and reserve “slow” integration suites for remote CI when >10 minutes.
  - Capture command output inline with the log and cite it to speed PR review.
- **Local vs. cloud execution**
  - Prefer local execution for linting, formatting, and unit tests under 5 minutes to iterate faster.
  - Offload container builds, large dataset downloads, and GPU-bound training to cloud jobs; add a tracking comment with job ID and estimated completion.
- **Handling long-running jobs**
  - Before launching, document the command, expected runtime, and success signal.
  - Schedule check-ins (e.g., `tail -f` windows or periodic status commands) and plan a “kill & resume” strategy (checkpoint artifacts, persisted logs).
- **Incident & failure recovery patterns**
  - If the agent is interrupted, replay the latest checkpoint summary, re-validate assumptions (tool versions, environment variables), and re-run only the minimal reproducer before continuing.
  - When a command fails unexpectedly, capture stdout/stderr, note hypotheses, and branch into a new scratch file to test fixes before touching production code.

## Prompting patterns that work
- Seed Codex with leading tokens (`#!/usr/bin/env python`, `// Tests:`) so it emits syntactically correct scaffolding.[^prompt-code]
- Request stepwise outputs: ask for a plan, confirm it, then let Codex implement each step (“Plan -> Diff -> Tests”).[^prompt-code]
- Provide structured scratchpads (`Thought:`, `Command:`, `Result:`) to channel tool usage and enforce verification loops.[^prompt-code]
- Highlight invariants (style guides, dependency versions) and known failure modes so the model can reason about guardrails before writing code.[^prompt-code]

### Example prompts
1. **Planning**
   ```
   You are pairing on a TypeScript monorepo. Goal: add pagination to `apps/portal/src/tasks/list.tsx`.
   1. Inspect existing data loaders.
   2. Outline the React state changes and API updates.
   3. List files to edit and tests to run.
   Think step-by-step before proposing code.
   ```
2. **Guarded implementation loop**
   ```
   Role: senior backend engineer.
   Constraint: follow the repo’s `CONTRIBUTING.md` lint/test workflow.
   Process:
   Thought -> Plan -> Command (if needed) -> Result -> Code diff -> Tests -> Next step.
   Stop after each diff and wait for confirmation.
   ```

### Failure-recovery playbook
- Maintain a rolling “assumptions” list; when results deviate, update the list and describe the new plan before coding again.
- For flaky tests, capture the failure seed/log, bisect to isolate the change, and annotate the PR with mitigation steps or follow-up issues.
- If Codex output drifts, restate the minimal failing example and re-run with a narrower prompt or direct code editing.

## 10-point Do/Don’t checklist
- ✅ **Do** confirm the latest requirements and constraints before editing.
- ✅ **Do** capture every executed command and result in the task log.
- ✅ **Do** gate hand-off on documented tests and code review notes.
- ✅ **Do** keep prompts short, scoped, and grounded in repo context.
- ✅ **Do** leave restart instructions (next command, open files, blockers).
- ❌ **Don’t** start coding without a validated plan and success metric.
- ❌ **Don’t** run long jobs without documenting checkpoints and timers.
- ❌ **Don’t** mix unrelated fixes in one commit or prompt iteration.
- ❌ **Don’t** overwrite Codex output without noting why it was rejected.
- ❌ **Don’t** close a task while questions, failing tests, or TODOs remain.

[^codex-blog]: [Introducing upgrades to Codex](https://openai.com/index/introducing-upgrades-to-codex/).
[^prompt-code]: [OpenAI prompt engineering best practices for code](https://platform.openai.com/docs/guides/prompt-engineering/code).
