# Embodied recursive toy — later build idea (2026-09-30)

Mykal's idea, captured for later — not started.

## The idea
A simple version of the H|A/V seeing/listening/answering loop, but with **recursive capability on a very small scale**: the loop's outputs feed back as inputs, with a little persistent state between turns. Put it on a Raspberry Pi, embody it in a toy droid or a Furby shell, and let it just do its thing — autonomous, emergent behavior.

## The lineage
- **Grey Walter's tortoises** (Elmer & Elsie, 1948–49) — the original "early light-sensing turtlebots": phototaxis, obstacle avoidance, recharging, all from simple analog wiring.
- **Braitenberg vehicles** (1984) — the thought-experiment version: complex lifelike behavior from direct sensor-motor coupling.
- **The upgrade here**: Walter's tortoises had fixed wiring. A Pi loop carries *state* — it can habituate, get "bored," prefer, remember. That's the jump from reactive to lifelike, and it's exactly what the H|A/V turn loop already does with its carried context.

## Why it's buildable
- H|A/V already has the sense → think → act → sense architecture; the recursive bit maps to the turn loop with state carried between turns.
- Pi Zero 2 W + camera module + mic + small speaker fits in a Furby shell; the 1998 Furbies are a well reverse-engineered hacking platform with an existing community.
- Start tiny: light/sound-seeking + a small state vector (arousal, familiarity, "boredom"), then grow the recursion from there.

## Open questions (for when we pick it up)
- How small can the "brain" be while still feeling recursive rather than reactive? (Tiny local model vs. rules + state?)
- Furby shell (existing servos/sensors to hijack) vs. droid toy (cleaner slate)?
- Tethered to the H|A/V agent handoff model, or fully standalone on the Pi?

## Self-modifying weights — research notes (2026-10-01)

Mykal asked how a small LLM might be rigged to its own weights for real-time changes. Candidate approaches, roughly front-ranked by seriousness:

- **Test-time training (Titans, Google 2024).** A neural long-term memory module that takes gradient steps on the live input stream, gated by surprise — writes only on unexpected input, old memories decay. The principled version of real-time weight change.
- **Online LoRA (the buildable one).** Freeze the base model, attach a LoRA adapter (rank 8–16), take one gradient step on the adapter every N tokens using plain next-token loss — the stream supervises itself, no labels needed. Only a few million params move. Persist the adapter to disk periodically and the model literally wakes up changed.
- **Fast weights (Schmidhuber, now mainstream in disguise).** Linear attention and SSMs (Mamba, RWKV) are mathematically equivalent to a network rewriting its own fast weights each step — the hidden state *is* the writable part. A small recurrent model is already a self-modifying system.
- **The honest cheap version.** In-context learning already is real-time adaptation: context window as writable memory, weights frozen. The notebook-as-buffer pattern is this at the symbolic level, with the advantage of being inspectable.

**Build recipe (candidate):** SmolLM or Qwen 0.5–1.5B, frozen base, LoRA r=8–16, surprise-gated updates, weight decay back toward zero, versioned adapter checkpoints for rollback. **Caveats:** needs a GPU to be actually real-time (CPU gradient steps won't keep up); self-supervised updates adapt *style*, not understanding — vocabulary and cadence, not new reasoning; ungated self-training drifts into echoing recent inputs.

**Resonance with the tortoise brief:** the fast/slow split maps onto the persistent-self question — frozen base as the long self, adapter as the day's self. This partially answers the "how small can the brain be" open question above, but the ruling stays open until Mykal picks it up.

## Architecture notes (2026-10-01)

Mykal asked whether the LM can shrink if the tortoise never outputs words — at most motor control. Answer: yes, and it should. An LLM is the wrong-shaped tool when no text comes out; billions of params for a text prior you never query.

- **Two-layer design.** Fast loop: tiny RNN — or rules plus a state vector (arousal, familiarity, boredom) — at tens of Hz, sensors straight to motors, subsumption-style. Thousands of params, not billions. Slow loop: a small LM at low frequency reading a summary of the fast loop's state and writing back drives (*seek light, rest, investigate the new thing*). The slow layer is where inner-life texture comes from.
- **Where the self-modifying weights live.** The online-LoRA adapter belongs on the slow layer: the tortoise's personality adapting over days, while the fast layer stays reactive.
- **The tradeoff, honestly.** Smaller = more designed, less emergent-for-free. LLMs hand you richness; tiny models give you exactly what you build, nothing more.
- **The key insight.** For an embodied toy the world does half the compute — a thousand-param controller in a real room (shifting light, obstacles, a curious cat) feels more alive than a billion-param model in a chat window. Lifelikeness lives in the closed sensorimotor loop, not the parameter count.

**Framing (Mykal, 2026-10-01):** this isn't companion-droid work — it's *interesting digital entities*. Entity-ness over utility or companionship.
