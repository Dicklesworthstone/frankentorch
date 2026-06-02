# UBS Security Scanner Policy

FrankenTorch uses [UBS (Ultimate Bug Scanner)](https://github.com/nightowlqa/ubs) for
security scanning of Rust code.

## Pre-commit Hook

The pre-commit hook runs `ubs --staged --only=rust` on staged files.

### Installation

```bash
cp hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

### Large File Handling

`crates/ft-api/src/lib.rs` is ~86K lines and takes 4-5 minutes for a full UBS scan.
The pre-commit hook handles this by:

1. Detecting staged files >50K lines
2. Extending timeout to 300 seconds for large files
3. Providing a global `UBS_SKIP=1` escape hatch when needed
4. Treating the known `ubs --staged` zero-file/zero-finding failure as a hook pass

**For ft-api/lib.rs changes**: Use `UBS_SKIP=1` for local commits. The file is
scanned in CI before merge. This is the recommended workflow for monolithic files
that exceed reasonable pre-commit timeouts.

### Manual Scanning

For full project scans (CI or manual):

```bash
# Full project (may take several minutes)
ubs --only=rust

# Staged files only (recommended for pre-commit)
ubs --staged --only=rust

# Specific file with extended timeout
timeout 300 ubs --only=rust crates/ft-api/src/lib.rs
```

### Bypassing

If the hook times out or UBS staged mode is misconfigured and you already ran an
equivalent manual scan:

```bash
UBS_SKIP=1 git commit -m "message"
```

**Important**: Always run a manual UBS scan on the changed Rust files before
bypassing the hook.

## CI Integration

CI pipelines should run UBS without timeout constraints:

```bash
ubs --only=rust --format=sarif --ci
```
