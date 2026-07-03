---
name: Bug report
about: Something behaves wrongly (or two tiers disagree)
labels: bug
---

**Nova program that reproduces it** (as small as possible):

```nova
fn main() {
  // ...
}
```

**What happened / what you expected:**

**Which tiers disagree?** (the gold standard: `nova run` vs `nova vm` vs
`nova vm --no-jit` vs `nova build --aot`) — paste both outputs if they differ;
a tier divergence is always a high-priority bug.

**Version:** output of `nova version`, OS, and how you installed.
