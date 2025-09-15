# Slugification Strategies: Shell Pipelines vs. `python-slugify`

## Context within Super Alita
- [`src/sdd/constitutional_pipeline.py` (lines 207-213)](src/sdd/constitutional_pipeline.py#L207-L213) and [`spec_kit.py` (lines 347-353)](spec_kit.py#L347-L353) apply a regex-driven helper that lowercases input, strips non-word characters, normalizes contiguous whitespace or hyphens to a single hyphen, removes leading/trailing separators, and truncates to 50 characters.
- [`src/tools/spec_generator.py` (lines 47-60)](src/tools/spec_generator.py#L47-L60) exposes `sanitize_slug`, which retains underscores, replaces disallowed characters with hyphens, trims the ends, and limits slugs to 60 characters.
- These helpers mirror common shell-based slugification recipes that combine `tr`, `sed`, `awk`, or `perl` to lowercase, sanitize, and coalesce separators. Understanding shell trade-offs helps decide when to adopt a dedicated library such as [`python-slugify`](https://github.com/un33k/python-slugify) for richer Unicode handling.

## Shell-Based Slugification Patterns
| Pattern | Typical Command | Strengths | Operational Risks |
| --- | --- | --- | --- |
| POSIX transliteration + sanitization | `echo "$text" | iconv -t ascii//translit | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | tr -s '-'` | Widely available, easy to inline in scripts, replicates current regex helpers. | Depends on system locales and `iconv` availability; transliteration varies by OS; failure to handle combining characters can drop content entirely. |
| Pure `tr`/`sed` ASCII-only | `echo "$text" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-|-$//g'` | Minimal dependencies, deterministic on ASCII input, fast in pipelines. | Drops or mangles any non-ASCII bytes; repeated runs can reintroduce multiple separators if input already includes hyphens or underscores. |
| `perl`/`python` one-liners | `perl -pe "s/\W+/-/g; s/^-|-$//g"` | Better Unicode support via modules (e.g., `Text::Unidecode`), easier to package as a reusable script. | Requires interpreter availability and module installation; error handling varies per environment. |
| Locale-aware `awk` pipelines | `LC_ALL=en_US.UTF-8 awk '{print tolower($0)}' | sed ...` | Slightly improved Unicode awareness under UTF-8 locales. | Locale must be set explicitly; fallback locales (e.g., `C`) break lowercase conversion and may produce uppercase ASCII output. |

### Shell Edge-Case Behavior
- **Spaces**: POSIX pipelines usually collapse runs of whitespace into a single hyphen after `sed` replacements, but leading/trailing whitespace must be stripped explicitly or shell expansions may trim them unexpectedly.
- **Punctuation**: Simple `sed` expressions treat all punctuation uniformly; retaining select characters (e.g., underscores) requires custom classes per shell. This increases maintenance overhead compared to Python regex character classes.
- **Unicode**: Transliteration strongly depends on `iconv` tables. For example, `é` → `e` on GNU `iconv`, but macOS may emit `e'` or drop the accent entirely. Emojis and CJK characters are usually stripped, producing empty slugs.
- **Repeated separators**: `tr -s '-'` or chained `sed -E 's/-{2,}/-/g'` collapse duplicates, yet separators introduced across pipes (e.g., `iconv` turning `ß` into `ss`) can produce double hyphens unless all stages reapply the collapsing rule.

## `python-slugify` Characteristics
| Capability | Details |
| --- | --- |
| Unicode normalization | Uses `Unidecode` to transliterate characters into ASCII, handling accents, CJK, Cyrillic, Greek, emoji fallbacks, and more. |
| Configurability | Accepts custom separator characters, lowercase toggles, stopword removal, and regex patterns for character replacement. |
| Deterministic output | Normalizes whitespace, strips leading/trailing separators, and collapses duplicates internally; easier to unit test than multi-stage shell pipelines. |
| Safety | Pure Python implementation avoids shell quoting pitfalls and command injection vulnerabilities when dealing with user input. |
| Dependency footprint | Requires installing the `python-slugify` package (and `Unidecode`); increases application size slightly versus inlining regex helpers.

### Comparative Edge-Case Handling
| Scenario | Shell Pipelines | `python-slugify` |
| --- | --- | --- |
| **Spaces** | Must explicitly trim and collapse using `sed` or `tr`; behavior differs if tabs or non-breaking spaces appear. | Normalizes any whitespace character class to a single separator; non-breaking spaces are handled correctly. |
| **Punctuation** | Default regex replacements either drop or replace every non-alphanumeric character; fine-grained keep/drop rules require custom scripts. | Supports `regex_pattern` overrides and default punctuation stripping; underscores or other symbols can be preserved via configuration. |
| **Unicode** | Depends on `iconv` tables and locale; unsupported characters are often removed, possibly yielding an empty slug. | Predictable transliteration to ASCII; characters without transliteration become separator placeholders, preventing blank outputs unless string is exclusively unsupported. |
| **Repeated separators** | Needs manual collapsing (`tr -s '-'`); if multiple scripts run sequentially the guarantee can be lost. | Automatically deduplicates separators and enforces leading/trailing trimming. |
| **Maximum length** | Achieved via `cut`/`head` or shell parameter expansion; easy to forget when copying recipes. | Built-in `max_length` parameter can enforce limits consistently. |

## Recommendations
1. **Retain lightweight regex helpers** for constrained pipelines that already sanitize ASCII-only inputs (e.g., existing regex functions in Super Alita) but document locale assumptions and maximum length truncation.
2. **Adopt `python-slugify`** when:
   - User input includes diverse Unicode scripts or emoji.
   - Reproducibility across Linux/macOS environments is mandatory.
   - Centralized configuration (separator, stopwords, casing rules) is valuable.
3. **Wrap shell pipelines** in integration tests if they remain, ensuring locale is pinned via `LC_ALL` and commands fail fast when prerequisites (like `iconv`) are missing.

## Property-Based Test Plan
These cases target both current regex helpers and prospective `python-slugify` integration. They should live alongside deterministic unit tests in `tests/runtime/` to guard our streaming and planning flows.

1. **Character Set Contract**
   - *Generators*: `st.text(alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"), min_size=1, max_size=256)`
   - *Property*: Output only includes lowercase ASCII letters, digits, underscores (when allowed), and the configured separator (`-` by default). Both regex helpers and `python-slugify` should satisfy this under comparable configuration.

2. **Idempotency**
   - *Property*: Applying the slugifier twice yields the same result (`slug(slug(x)) == slug(x)`). Validates separator normalization and trimming invariants for repeated separators.

3. **Whitespace Normalization**
   - *Property*: All Unicode whitespace code points (including NBSP, thin spaces, tabs) collapse to a single separator; leading/trailing whitespace never produces boundary separators. Hypothesis strategy: inject whitespace clusters via `st.builds("".join, st.lists(st.sampled_from(WHITESPACE_CHARS), min_size=1, max_size=5))` interleaved with alphanumerics.

4. **Punctuation Stability**
   - *Property*: Removing ASCII punctuation from the input does not change the slug when the slugifier is configured to drop punctuation (verifies that punctuation is always translated to separators rather than literal characters).

5. **Unicode Transliteration**
   - *Property*: For inputs restricted to Unicode letters with `unicodedata.category(ch).startswith("L")`, the slug is non-empty. Compare regex helper output against `python-slugify`; mismatch highlights where shell-style regex drops transliterated forms. This guards against silent data loss.

6. **Length Boundaries**
   - *Property*: Output length never exceeds the configured cap (50 or 60 characters for existing helpers). Hypothesis can target boundary lengths by generating long random strings and asserting truncation occurs exactly at the limit.

7. **Separator Coalescing**
   - *Property*: No substring contains the separator repeated twice. Hypothesis: generate inputs with repeated punctuation/whitespace (e.g., using `st.lists(st.text(min_size=1, max_size=5), min_size=2)` joined by `st.sampled_from(["-", "_", " ", "--"])`).

8. **Round-Trip Compatibility**
   - *Property*: For ASCII-only inputs, manual regex helpers and `python-slugify` should match exactly. Failures highlight differences where adopting the library could break existing downstream expectations.

Implement each property with `@given` tests and include `example()` seeds capturing problematic cases (e.g., emoji-only strings, repeated hyphens). Pair property tests with explicit regression fixtures sourced from production feature names to detect accidental behavior shifts.
