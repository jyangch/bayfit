# Python Package Docstring Standard v1.2

> This document is a pure style specification covering docstring and inline-comment conventions. Workflow instructions (target scope, skipped directories, override policy) live in the project root `CLAUDE.md`.

> **v1.2 changelog** (2026-04-29):
> - §1 adds the *ASCII-only punctuation* rule; rewrites the conflict-resolution clause as a decision tree to remove the ambiguity in v1.1's priority phrasing.
> - §2 fixes the en dash in the example; section title unified to `Example:` (Google-style compliant).
> - §3 clarifies the priority order between the two `__doc__` injection styles.
> - §4 the factory classmethod subsection adds Example-decision guidance.
> - §6 adds an explicit stance on the `XXX` marker.
> - §8 sharpens the description of display protocols (distinguishing dunder methods from IPython `_repr_*_` protocols).
> - §9 adds an operational definition of "description is accurate".
> - §11 adds an `allowed-confusables` recipe for physics-unit characters in prose.

---

## 1. General Principles

- **Style baseline**: Google-style docstrings (compatible with Sphinx, mkdocstrings, pdoc).
- **Language**: docstring bodies and inline comments are written in **English**; identifiers, type names, library names, and proper nouns are kept verbatim. Prefer concise, active voice, present tense.
- **Type information**: PEP 484 type hints are mandatory; **never** repeat types inside the docstring.
- **Coverage**: every public API (module, class, public function/method) must have a docstring; members prefixed with `_` are documented on demand.
- **Docstring tier requirements**:
  - **Required**: modules; classes; classmethods / staticmethods; public functions; computed `@property` (multi-line body, or name with ambiguous semantics or implicit units).
  - **Recommended**: single-line public `@property`. Even when the implementation is a one-liner like `np.mean(...)`, write a one-line summary so IDE hovers and generated docs have something to show.
  - **Optional**: pure pass-through `@property` (`return self._x`) and its setter; display-protocol methods (see §8); inner wrappers decorated with `@functools.wraps` that inherit the original `__doc__`.
- **Summary line**: ≤ 80 characters, imperative mood (`Return`, `Compute`, `Handle`...), terminated with a period, no line breaks.
- **Conflict resolution** (decide top-down; do not rely on priority slogans):
  1. Member is in the **Required** tier → write something even if it feels thin; brevity beats absence.
  2. Member is in the **Recommended** tier but you cannot describe it accurately → leave it empty; never produce stub docstrings like `"This is a function."`.
  3. Member is in the **Optional** tier → leave empty by default, unless the behavior is counter-intuitive.
- **English style rules**:
  - Summary uses imperative mood (`Compute...`, `Return...`, `Raise...`); **do not** start with `This function...` or `Returns the...`.
  - Args / Returns / Raises descriptions are full sentences, capitalized, period-terminated.
  - Refer to code symbols using double backticks: ``` ``src`` ```, ``` ``None`` ```.
  - Avoid colloquialisms (`gonna`, `kinda`) and vague filler words (`various`, `stuff`).
- **ASCII-only punctuation**: docstrings, inline comments, and string literals **must not** contain the following ambiguous Unicode characters (they trigger ruff `RUF001/002/003`); use ASCII equivalents:
  - en dash `–` (U+2013) → `-`; em dash `—` (U+2014) → `--`
  - curly quotes `‘ ’ “ ”` → `' "`; ellipsis `…` → `...`
  - minus sign `−` (U+2212) → `-`; bullets `·` `•` → `*` or `-`
  - numeric ranges are written `0-13`, never `0–13`.
  - **Exception**: domain-essential characters such as physics units and Greek letters follow the §11 `allowed-confusables` whitelist; do not rewrite `cm²` as `cm^2` purely to satisfy the linter.

---

## 2. Module-level Docstring

Placed at the very top of the file, before any imports.

```python
"""<One-line summary of the module's responsibility>.

<Optional 2-4 lines of detail: design intent, core concepts, relations to
other modules.>

Example:
    from mypkg.user import UserService
    svc = UserService(repo)
    svc.get(1)

Attributes:
    DEFAULT_TIMEOUT (int): Only list public module-level constants here.
"""
```

> Section titles must use the canonical Google-style words (`Example:` / `Examples:` / `Attributes:` / `Args:` / `Returns:` / `Raises:`). Sphinx's napoleon extension only recognizes that vocabulary; custom titles like `Typical usage:` will not be parsed as structured sections.

---

## 3. Class Docstring

Placed on the line directly below the `class` statement; describes the **responsibility**, not the implementation. Parameter docs for `__init__` live **inside** `__init__` and must not be duplicated in the class docstring.

```python
class UserService:
    """Handle CRUD and permission checks for the user domain.

    Uses an injected repository; thread-safe. Caching layer is optional.

    Attributes:
        repo: User persistence repository.
        cache_ttl: Cache expiry in seconds.

    Example:
        >>> svc = UserService(repo=repo)
        >>> svc.get(user_id=1)
        User(id=1, ...)
    """
```

### Dynamically generated classes

Classes synthesized via `type(name, bases, dct)` or class factories **do not** automatically receive a docstring. Use one of the two injection styles below, in the stated priority order:

**Preferred**: pass `'__doc__'` directly inside `dct`. Single-step, picked up by IDE static analysis and documentation tools.

```python
new_class = type(
    f'XS_{name}', (Model,),
    {'__init__': make_init(name, cls), 'func': func,
     '__doc__': f'XSPEC {name} model bridged into bayspec.'})
```

**Fallback**: monkey-patch the attribute after assignment, only when modifying `dct` is awkward inside the factory (e.g., when reusing a third-party metaclass).

```python
globals()[f'XS_{name}'] = new_class
new_class.__doc__ = f'XSPEC {name} model bridged into bayspec.'
```

---

## 4. Function / Method Docstring

**Section order** (skip missing sections; never leave empty stubs):

1. Summary line
2. Extended description (optional)
3. `Args:`
4. `Returns:` or `Yields:`
5. `Raises:`
6. `Example:` (**recommended**; required in the two cases below)
7. `Note:` / `Warning:` (optional)

**Example is required when**:
- (a) The function is standalone or a pure utility that can be demonstrated without other domain objects.
- (b) The call is easy to misuse, or the method has multiple equivalent signatures.

When you cannot write a real, executable Example, omit the section entirely - a fabricated Example is worse than a missing one.

```python
def transfer(src: Account, dst: Account, amount: Decimal) -> TransferResult:
    """Perform an atomic transfer between two accounts.

    Uses a database transaction for consistency; any failure rolls back
    the entire operation.

    Args:
        src: Source account; must be authenticated.
        dst: Destination account; cross-currency is supported.
        amount: Transfer amount; must be positive.

    Returns:
        The transfer result, including transaction id and fee.

    Raises:
        InsufficientFundsError: If ``src`` has insufficient balance.
        AccountFrozenError: If either account is frozen.

    Example:
        >>> transfer(a, b, Decimal("100"))
        TransferResult(tx_id='...', fee=Decimal('0.5'))
    """
```

### Factory classmethods

Constructor-style classmethods such as `from_xxx` get their own docstring - do not rely on the class docstring or `__init__`. Typically they include Summary + Args + Raises; when the body is just `return cls(...)`, the Returns section may be omitted.

```python
class Spectrum:
    @classmethod
    def from_src(cls, src_file):
        """Load a source spectrum from a single-row OGIP PHA file.

        Falls back to ``RATE`` if ``COUNTS`` is absent, and synthesizes
        Poisson errors if ``STAT_ERR`` is missing.

        Args:
            src_file: Path to the PHA file, or a ``BytesIO``.

        Raises:
            ValueError: If ``src_file`` is neither a string nor ``BytesIO``.
        """
```

Decide on Examples for factory methods using the §4 main rule. When the factory accepts multiple input shapes (path / `BytesIO` / dict / pre-loaded object), provide one doctest line per shape so that the expected behavior is unambiguous - this is a textbook case of rule (b).

### Wrappers returned by decorators

Inner wrappers decorated with `@functools.wraps(func)` inherit the original `__doc__`, so their own docstring may be omitted. The outer decorator factory itself still follows the public-function rules.

---

## 5. Inline Comment Rules

- Comments explain **why**, not **what**. Self-documenting code needs no comment.
- Same-line comments: at least two spaces after the code, then `# `.
- Standalone multi-line comments: separated from surrounding code by one blank line on each side.
- Magic numbers and special algorithms: must cite a source (paper, issue, doc URL).

**Bad**: `i += 1  # increment i` → forbidden.
**Good**: `sleep(0.3)  # avoid downstream rate limit, see #1287`.

---

## 6. Special Markers (canonical vocabulary)

| Marker | Meaning | Citation required |
|------|------|------------------|
| `TODO(name)` | Pending work | Optional |
| `FIXME(name)` | Known bug | **Issue number required** |
| `HACK(name)` | Workaround | **Trigger condition required** |
| `NOTE` | Design decision or counter-intuitive behavior | Recommended |
| `DEPRECATED` | Scheduled for removal | **Removal version required** |

Example: `# TODO(alice): switch to a connection pool, see #342`

**Forbidden marker**: `XXX` is not used. Its meaning sits ambiguously between `HACK` and `FIXME`; any occurrence is treated as a style violation and must be rewritten using one of the markers above.

---

## 7. Type Annotation Conventions

- All public functions must annotate parameters and return types.
- Container types must be parameterized: `list[User]`, never bare `list`.
- Optional types prefer `X | None` (Python 3.10+); fall back to `Optional[X]` only on Python ≤ 3.9.
- Complex types are extracted as a `TypeAlias` at the top of the module.
- **Legacy migration exception**: files in active migration may describe types in shorthand inside the docstring (e.g., `src: Source spectrum or path string.`) until type hints are added; new files and any function whose signature has been edited follow the rules above unconditionally. Once PEP 484 hints are in place, the docstring must not duplicate the type information.

---

## 8. Forbidden Practices

- ❌ Docstring contradicts actual code behavior.
- ❌ Stub docstrings produced by copy-paste templates.
- ❌ Repeating types in the docstring that are already in type hints.
- ❌ Mixing English and another language inside the same docstring (proper nouns excepted).
- ❌ Writing filler docstrings on pure pass-through getters/setters and on **display-protocol methods**.
  - Display-protocol methods include: dunders `__str__` / `__repr__` / `__format__`; and IPython/Jupyter rich-display protocols `_repr_html_` / `_repr_mimebundle_` / `_repr_png_` and other single-underscore `_repr_*_` siblings.
  - ⚠ **Behavioral** dunders - `__add__` / `__sub__` / `__mul__` / `__truediv__` / `__call__` / `__getitem__` / `__contains__` and friends - **must** be documented when they form part of the public API (operator algebra, container protocols, etc.).
- ❌ Modifying business logic, identifier names, or import order.
- ❌ Using any of the ambiguous Unicode characters listed in §1 *ASCII-only punctuation* inside docstrings, comments, or string literals (triggers ruff `RUF001/002/003`).

---

## 9. Iterating on Existing Comments

Whether to rewrite an existing docstring is decided by the table below, eliminating per-case judgment. Workflow concerns (whether to show a diff and ask for confirmation) live in `CLAUDE.md`; this section governs only the **content/style** rewrite triggers.

| Trigger | Action |
|---|---|
| Style is not Google (NumPy `Parameters:` underline / RST `:param:` `:return:` / `@param`) | **Rewrite**, no diff needed |
| Mixed languages, colloquialisms (`gonna`, `kinda`, etc.), or `NOTE:` without source | **Rewrite** |
| Inconsistent with current code behavior, or description is stale | **Rewrite**, flag in commit message |
| > 10 lines and member is part of public API | Show diff and confirm before rewriting |
| Style-compliant and accurate | **Keep** |
| Compliant but could be more precise (missing Raises / units / etc.) | **Augment**, keep the original summary |

**Operational definition of "description is accurate"**: the original Summary and Args contain **no contradiction** with the current signature and return value across the four axes below -

1. **Type**: stated parameter and return types match the actual types.
2. **Unit**: physical quantities, durations, percentages, etc. use units consistent with the code.
3. **Side effects**: whether the function mutates inputs, performs disk I/O, or makes network calls matches actual behavior.
4. **Exceptions**: every exception explicitly raised in code is listed in `Raises:`.

A mismatch on any axis counts as inaccurate and routes the docstring to the rewrite branch.

Edge case: when a docstring is half-right, half-wrong, or carries valuable domain notes (paper citations, historical decisions), **preserve the cited content** and rewrite the rest. It is better to split the docstring into Summary + `Note:` than to discard the original information.

---

## 10. Group Docstrings for Repeated Member Families

When several public methods or properties form a recognizable family - differing only by suffix (`_re` / `_error` / `_f64`) or by statistic name (`mean_*` / `median_*` / `best_*`) - you may write a single group-level explanation on the **first member** and omit docstrings on the rest.

```python
@property
def conv_ctsrate(self):
    """Per-unit convolved count rate for the paired model.

    Every ``conv_*``, ``*_at_rsp``, ``cts_to_*``, ``deconv_*`` family
    on this class mirrors the same-named ``Model`` property but uses
    the paired ``data`` so the values are consistent with the current
    ``Pair``.
    """
    return self._convolve()


@property
def conv_re_ctsrate(self):
    # Inherits the family explanation above; docstring omitted on purpose.
    return self._re_convolve()
```

Constraints:
- The group explanation must enumerate the **naming patterns covered by this family**; "siblings follow the same pattern" alone is not acceptable.
- The pattern does not apply when family members are spread across different `class` definitions or files - each class needs at least one group statement.
- The family head (first member) must still describe its **own individual behavior** accurately; it cannot degenerate into a group statement only.

---

## 11. Mathematical Notation

Mathematical formulas inside docstrings use the reST `:math:` role or short ASCII; **do not** write bare `$...$` - rendering across sphinx / mkdocstrings / pdoc is inconsistent.

- **Inline**: `` :math:`N(E)` ``, `` :math:`\nu F_\nu` ``.
- **Short ASCII** (only when the formula is simple and contains no Greek letters): `` ``N(E)`` ``, `` ``2 + alpha`` ``.
- **Block**: use the reST `.. math::` directive on its own paragraph for multi-line formulas.

LaTeX backslashes in Python strings need either a raw string (`r"""..."""`) or doubled backslashes (`\\nu`). Inside Google-style docstrings, raw strings are preferred:

```python
def flxspec(self, E, T=None):
    r"""Energy flux density :math:`F_\nu = E \, N(E)` in erg/cm^2/s/keV."""
```

Other LaTeX styles - `$...$`, `\(...\)`, `\begin{equation}` - are discouraged; they do not interoperate with the current toolchain.

### Whitelisting physics-unit characters

§1 *ASCII-only punctuation* forbids characters such as `²` `³` `Å` `μ` in ordinary prose, but physics units and Greek letters are unavoidable in scientific code (writing `cm^2` is ugly and breaks rendering). Allow them explicitly in `pyproject.toml`:

```toml
[tool.ruff.lint]
allowed-confusables = [
    "²", "³",                  # Superscript digits (unit exponents)
    "Å",                       # Angstrom
    "α", "β", "γ", "δ", "ε",  # Lowercase Greek (common)
    "θ", "λ", "μ", "ν", "π",
    "ρ", "σ", "τ", "φ", "χ", "ω",
    "Δ", "Σ", "Ω",            # Uppercase Greek (common)
]
```

Trim the list to what your project actually needs. If `:math:\`\\nu\`` works, prefer it over a bare `ν`; the narrower the whitelist, the better it catches accidental ambiguity.
