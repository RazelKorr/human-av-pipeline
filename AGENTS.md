# Working in this repo

- **Never run `git reset --hard` here.** On 2026-10-01 and twice on 2026-10-03, stray `git reset --hard` runs wiped tracked-file changes mid-task (cause unknown; leading theory is one of our own subagents). The audit recovered the wiped files from a dangling stash, and the canon is committed — but uncommitted work is one reset away from the void. Commit early and often; use `git stash` if you need a clean tree instead of reset.
- Sealed predictions before any sim/measurement run: write the expected outcome down first, then run, then compare.
- The standing review protocol lives at `REVIEW-PROTOCOL.md` (beat map before events; claims cite timestamp + visual + beat anchor; red-team the draft).
