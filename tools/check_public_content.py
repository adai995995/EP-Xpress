#!/usr/bin/env python3
"""Reject private material before committing or publishing this repository.

Usage: python3 tools/check_public_content.py --staged --history

With no mode, scan tracked and unignored worktree files. History scanning visits
every reachable commit, including blobs later deleted or replaced. Private
organization identifiers can be supplied as a JSON list of literal fragments
via EP_XPRESS_PRIVATE_PATTERNS_FILE. That file must remain outside the repository.
Matches are case insensitive. Findings show only a repository path and rule,
never matched text. This is a conservative local gate, not a proof that content
contains no private information; review the complete diff before publishing.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import unicodedata


MAX_FILE_BYTES = 2 * 1024 * 1024
PUBLIC_NAME = "EP-Xpress Contributors"
PUBLIC_EMAIL = "contributors@users.noreply.github.com"
_IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
_PATHS = re.compile(r"/(?:Users|home)/[^\s/\"'<>]+|/(?:mnt/ufs[^\s/]*|nfs)(?:/|\b)")
_PRIVATE_KEY = re.compile(r"-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----")
_TOKENS = re.compile(
    r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|"
    r"(?:AKIA|ASIA)[A-Z0-9]{16})\b"
)
_CREDENTIAL_NAME = (
    r"(?:(?:[a-z][a-z0-9]*_)*"
    r"(?:password|passwd|token|secret|api[_-]?key|access[_-]?token|auth[_-]?token|"
    r"private[_-]?key|secret[_-]?key|secret[_-]?access[_-]?key))"
)
_QUOTED_CREDENTIAL = re.compile(
    r"(?im)(?<![\w])(?:[\"']?" + _CREDENTIAL_NAME + r"[\"']?)\s*[:=]\s*"
    r"(?:[rubf]{0,2})?([\"'])([^\r\n]*?)\1"
)
_BARE_CREDENTIAL = re.compile(
    r"(?im)^\s*(?:export\s+)?" + _CREDENTIAL_NAME + r"\s*[:=]\s*([^\s#\"']+)"
)
_SAFE_LITERAL = re.compile(r"^(?:\$\{[A-Z_][A-Z0-9_]*\}|\$[A-Z_][A-Z0-9_]*|<[A-Z_][A-Z0-9_ -]*>)$")
_ARTIFACT_SUFFIXES = (
    ".log", ".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".pkl",
    ".pickle", ".npy", ".npz", ".onnx", ".whl", ".zip", ".tar",
    ".tar.gz", ".tgz", ".so", ".dylib", ".dll", ".o", ".a", ".pyc",
    ".pem", ".key", ".sqlite", ".sqlite3", ".db", ".nsys-rep", ".ncu-rep",
)
_ARTIFACT_NAMES = {".env", ".ds_store", "id_rsa", "id_ed25519", "known_hosts"}
_ARTIFACT_DIRS = {"__pycache__", ".venv", "venv", "node_modules", "logs", "checkpoints"}
_CONFIG_SUFFIXES = {".md", ".txt", ".yaml", ".yml", ".toml", ".ini", ".conf", ".env", ".sh"}


class ScanError(Exception):
    """An inspection failed; callers must fail closed."""


def _git(root: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(root), *args], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise ScanError("git-read-failed")
    return result.stdout


def _safe_literal(value: str) -> bool:
    return not value or value in {"REDACTED", "PLACEHOLDER"} or bool(_SAFE_LITERAL.fullmatch(value))


def _text_rules(value: str, private_patterns) -> set:
    rules = set()
    if _PATHS.search(value):
        rules.add("private-absolute-path")
    if _PRIVATE_KEY.search(value):
        rules.add("private-key")
    if _TOKENS.search(value):
        rules.add("credential-token")
    for match in _IPV4.finditer(value):
        address = match.group(0)
        try:
            parsed = ipaddress.IPv4Address(address)
        except ipaddress.AddressValueError:
            continue
        if str(parsed) not in {"127.0.0.1", "0.0.0.0"}:
            rules.add("nonlocal-ipv4")
    if any(not _safe_literal(match.group(2)) for match in _QUOTED_CREDENTIAL.finditer(value)):
        rules.add("hardcoded-credential")
    if any(ord(char) > 127 and unicodedata.category(char).startswith("L") for char in value):
        rules.add("non-english-text")
    folded = value.casefold()
    if any(pattern in folded for pattern in private_patterns):
        rules.add("private-pattern")
    return rules


def _artifact_path(path: str) -> bool:
    parts = Path(path).parts
    lower = path.lower()
    return (lower.endswith(_ARTIFACT_SUFFIXES)
            or Path(lower).name in _ARTIFACT_NAMES
            or any(part.lower() in _ARTIFACT_DIRS for part in parts))


def load_private_patterns(root: Path):
    configured = os.environ.get("EP_XPRESS_PRIVATE_PATTERNS_FILE")
    if not configured:
        result = subprocess.run(["git", "-C", str(root), "config", "--get",
                                 "epxpress.privatePatternsFile"], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, check=False)
        if result.returncode not in (0, 1):
            raise ScanError("private-pattern-configuration-unreadable")
        if result.returncode == 0:
            configured = result.stdout.decode("utf-8").strip()
    if not configured:
        return []
    try:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve(strict=True)
        if os.path.commonpath((str(root.resolve()), str(path))) == str(root.resolve()):
            raise ScanError("private-pattern-file-must-be-outside-repository")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ScanError("private-pattern-file-too-large")
        patterns = json.loads(path.read_text(encoding="utf-8"))
    except ScanError:
        raise
    except (OSError, ValueError, UnicodeError):
        raise ScanError("private-pattern-file-invalid") from None
    if not isinstance(patterns, list) or any(not isinstance(p, str) or not p for p in patterns):
        raise ScanError("private-pattern-file-invalid")
    return [pattern.casefold() for pattern in patterns]


class Scanner:
    def __init__(self, root: Path, private_patterns=()):
        self.root = root
        self.private_patterns = [p.casefold() for p in private_patterns]
        self.findings = set()
        self._blob_rules = {}

    def add(self, path: str, rule: str):
        # A filename can itself contain private information; suppress it then.
        if _text_rules(path, self.private_patterns):
            path = "<redacted-path>"
        self.findings.add((path, rule))

    def inspect_path(self, path: str, mode: str = "100644"):
        for rule in _text_rules(path, self.private_patterns):
            self.add(path, rule)
        if _artifact_path(path):
            self.add(path, "generated-or-private-artifact")
        if mode not in {"100644", "100755"}:
            self.add(path, "unsupported-file-type")

    def content_rules(self, path: str, data: bytes):
        if len(data) > MAX_FILE_BYTES:
            return {"oversized-file"}
        try:
            content = data.decode("utf-8")
        except UnicodeError:
            return {"binary-file"}
        if "\x00" in content:
            return {"binary-file"}
        rules = _text_rules(content, self.private_patterns)
        if Path(path).suffix.lower() in _CONFIG_SUFFIXES:
            for match in _BARE_CREDENTIAL.finditer(content):
                value = match.group(1)
                if value not in {"None", "null", "true", "false"} and not _safe_literal(value):
                    rules.add("hardcoded-credential")
        return rules

    def inspect_content(self, path: str, data: bytes):
        for rule in self.content_rules(path, data):
            self.add(path, rule)

    def inspect_blob(self, path: str, oid: str, mode: str):
        self.inspect_path(path, mode)
        if mode not in {"100644", "100755"}:
            return
        # Preserve only rule names in memory. Config files use an extra check.
        cache_key = (oid, Path(path).suffix.lower() in _CONFIG_SUFFIXES)
        if cache_key not in self._blob_rules:
            size = int(_git(self.root, "cat-file", "-s", oid).strip())
            if size > MAX_FILE_BYTES:
                self._blob_rules[cache_key] = {"oversized-file"}
            else:
                data = _git(self.root, "cat-file", "blob", oid)
                self._blob_rules[cache_key] = self.content_rules(path, data)
        for rule in self._blob_rules[cache_key]:
            self.add(path, rule)

    def inspect_identity(self, name: str, email: str, label: str):
        if (name, email) != (PUBLIC_NAME, PUBLIC_EMAIL):
            self.add(label, "nonpublic-git-identity")
        for rule in _text_rules(name + "\n" + email, self.private_patterns):
            self.add(label, rule)

    def staged(self):
        for record in _git(self.root, "ls-files", "--stage", "-z").split(b"\x00"):
            if not record:
                continue
            metadata, raw_path = record.split(b"\t", 1)
            mode, oid, stage = metadata.decode("ascii").split()
            path = raw_path.decode("utf-8", errors="replace")
            if stage != "0":
                self.add(path, "unmerged-index")
                continue
            self.inspect_blob(path, oid, mode)
        for kind in ("AUTHOR", "COMMITTER"):
            ident = _git(self.root, "var", "GIT_" + kind + "_IDENT").decode("utf-8", errors="replace")
            match = re.fullmatch(r"(.*) <([^<>]*)> \d+ [+-]\d{4}\n?", ident)
            if not match:
                raise ScanError("git-identity-unreadable")
            self.inspect_identity(match.group(1), match.group(2), "<pending-" + kind.lower() + ">")

    def history(self):
        for oid_bytes in _git(self.root, "rev-list", "--all").splitlines():
            oid = oid_bytes.decode("ascii")
            label = "<commit-" + oid[:12] + ">"
            raw = _git(self.root, "show", "-s", "--format=%an%x00%ae%x00%cn%x00%ce%x00%B", oid)
            fields = raw.decode("utf-8", errors="replace").split("\x00", 4)
            if len(fields) != 5:
                raise ScanError("commit-metadata-unreadable")
            self.inspect_identity(fields[0], fields[1], label + ":author")
            self.inspect_identity(fields[2], fields[3], label + ":committer")
            for rule in _text_rules(fields[4], self.private_patterns):
                self.add(label + ":message", rule)
            for record in _git(self.root, "ls-tree", "-r", "-z", oid).split(b"\x00"):
                if not record:
                    continue
                metadata, raw_path = record.split(b"\t", 1)
                mode, kind, blob = metadata.decode("ascii").split()
                path = raw_path.decode("utf-8", errors="replace")
                if kind != "blob":
                    self.add(path, "unsupported-file-type")
                    continue
                self.inspect_blob(path, blob, mode)

    def worktree(self):
        paths = _git(self.root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
        for raw_path in sorted(set(paths.split(b"\x00")) - {b""}):
            path = raw_path.decode("utf-8", errors="replace")
            full_path = self.root / path
            try:
                info = full_path.lstat()
            except FileNotFoundError:
                continue  # A tracked deletion contains no worktree payload.
            except OSError:
                self.add(path, "file-unreadable")
                continue
            self.inspect_path(path)
            if not stat.S_ISREG(info.st_mode):
                self.add(path, "unsupported-file-type")
                continue
            if info.st_size > MAX_FILE_BYTES:
                self.add(path, "oversized-file")
                continue
            try:
                self.inspect_content(path, full_path.read_bytes())
            except OSError:
                self.add(path, "file-unreadable")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="scan the index and pending Git identity")
    parser.add_argument("--history", action="store_true", help="scan all reachable commit metadata and historical blobs")
    parser.add_argument("--worktree", action="store_true", help="scan tracked and unignored files")
    args = parser.parse_args(argv)
    try:
        root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel").decode("utf-8").strip())
        scanner = Scanner(root, load_private_patterns(root))
        if args.staged:
            scanner.staged()
        if args.history:
            scanner.history()
        if args.worktree or not (args.staged or args.history):
            scanner.worktree()
        for path, rule in sorted(scanner.findings):
            print(json.dumps(path, ensure_ascii=True) + ": " + rule)
        return 1 if scanner.findings else 0
    except (ScanError, OSError, ValueError):
        print('"<scanner>": inspection-failed', file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
