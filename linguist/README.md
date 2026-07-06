# Getting Nova recognized by GitHub (Linguist)

GitHub decides the language of a file with [github/linguist]. Because Nova is a
new language it isn't in Linguist's database yet, so `.nova` files currently show
up as **Other**. This directory is the ready-to-submit contribution kit that
makes GitHub label the repo **Nova**.

Two halves:

## 1. In-repo (already active): `.gitattributes`
The repo root `.gitattributes` marks `*.nova` as `linguist-language=Nova
linguist-detectable=true` and excludes generated/vendored trees from the stats.
`linguist-language` overrides detection **once the language exists upstream**;
until then GitHub ignores an unknown name. So the second half is required for the
label to actually appear.

## 2. Upstream (the real fix): a github/linguist PR
Linguist only recognizes languages defined in its own database. Submit:

1. **`nova.tmLanguage.json`** → copy to `vendor/grammars/` (or reference an
   existing grammar submodule) and register it in `vendor/README.md` +
   `grammars.yml`. This file is a complete TextMate grammar for Nova (keywords,
   storage/modifiers, primitive + user types, numbers with all bases/suffixes,
   strings incl. raw `r#"…"#`, f-string interpolation, `json|sql|re` tagged
   strings, char + unicode escapes, lifetimes, attributes `#[…]`, and the full
   operator set incl. `->>` `>>>` `<-` `|>`).
2. **`languages.yml.entry`** → add to `lib/linguist/languages.yml` (alphabetical
   order). Run `bundle exec rake` / `script/update-ids` so maintainers allocate
   the canonical `language_id`.
3. **`samples/Nova/`** → copy to `samples/Nova/` in linguist. These real Nova
   programs (from the corpus/std/examples) train the Bayesian classifier so
   `.nova` is detected even without the `.gitattributes` override.

Then open the PR against <https://github.com/github/linguist>. Per their
CONTRIBUTING guide, a new language should be **in use in hundreds of
repositories** before it is accepted; the `.gitattributes` half already gives the
correct label the moment the upstream entry merges.

## Honest status
- ✅ `.gitattributes` mapping + this complete, valid contribution kit are in the
  repo now.
- ⏳ GitHub will render the language bar as **Nova** only after the upstream
  linguist PR is merged (adoption-gated) — that step is outside this repository.

[github/linguist]: https://github.com/github/linguist
