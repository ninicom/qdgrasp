# S10 — runs v9–v11: is the sanctioned fallback backend even possible?

## Why these runs exist

Section 3.7 allows two resolutions when no MJWarp version is clean: a fallback
backend that passes capability and parity, or a blocked gate. Three documents in
this repo said the fallback was "new work needing its own plan" without anyone
checking whether it was possible. That is an assumption wearing the clothes of a
finding, and the wall-press scene had already shown what that costs.

So MJX — the other batched GPU backend for MuJoCo — was probed against the one
capability the contract cannot do without: per-contact force, readable.

## Answer: inconclusive, with a measured lower bound

`mujoco-mjx` installs. `jax.jit(mjx.step)` on the LEAP contact-rich scene did
**not finish compiling within 1800 s** on a T4, so the probe never reached the
capability question at all.

That does not condemn MJX. It does say an MJX fallback is not a drop-in: on our
actual models, compile cost alone is a serious open risk, and a fallback backend
would need its own budget for it. The claim "new work needing its own plan" now
rests on a measurement instead of an assumption, which is what these runs were
for.

## Cost of getting here

v9 and v10 both died on my own generated-source bugs, not on physics:

- v9: `SyntaxError: unterminated string literal` — a patch script ate one level
  of escaping, so `\n` reached the notebook as a real newline. Identical to v4.
- v10: `TypeError: unhashable type: 'dict'` — `{{"attempted": True}}`, a doubled
  brace in a string that was not an f-string. This *parses*; it is a set
  containing a dict, so the parse guard added after v9 waved it through.

Both are now impossible to ship: the builder refuses to write a notebook whose
code cells fail to parse, and refuses any cell containing a doubled brace. Two
runs at roughly forty minutes each bought that guard.

## Incidental: the solver finding reproduced twice more

v9 and v11 both re-ran the solver matrix independently of v8:

| variant | v8 | v9 |
|---|---|---|
| `baseline` | 66 813 | 64 017 |
| `ls_iterations=1` | 12 788 | 13 928 |
| `solver_cg` | 850 170 | 824 757 |
| `ls_parallel` | did not run | did not run |

Same conclusion from three samples: no solver configuration avoids the defect.
