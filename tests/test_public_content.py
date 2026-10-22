"""Privacy-gate fixtures are synthetic and assembled to avoid self-matches."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCANNER = Path(__file__).resolve().parents[1] / "tools" / "check_public_content.py"
PUBLIC_NAME = "EP-Xpress Contributors"
PUBLIC_EMAIL = "contributors@users.noreply.github.com"


class PublicContentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name) / "repository"
        self.root.mkdir()
        self.environment = dict(os.environ)
        for key in list(self.environment):
            if key.startswith(("GIT_", "EP_XPRESS_PRIVATE_")):
                self.environment.pop(key)
        self.environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
        self.git("init", "-q")
        self.git("config", "user.name", PUBLIC_NAME)
        self.git("config", "user.email", PUBLIC_EMAIL)

    def tearDown(self):
        self.directory.cleanup()

    def git(self, *args):
        result = subprocess.run(["git", "-c", "commit.gpgsign=false", "-c",
                                 "core.hooksPath=" + os.devnull, *args], cwd=self.root,
                                env=self.environment, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, check=False)
        self.assertEqual(result.returncode, 0, "fixture Git operation failed")
        return result.stdout

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")

    def scan(self, *modes):
        return subprocess.run([sys.executable, str(SCANNER), *modes], cwd=self.root,
                              env=self.environment, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, check=False)

    def commit(self, message="Add fixture"):
        self.git("add", "-A")
        self.git("commit", "-qm", message)

    def test_empty_history_and_clean_documentation_pass(self):
        self.assertEqual(self.scan("--history").returncode, 0)
        self.write("README.md", "# Public project\nRun torchrun with 127.0.0.1.\n")
        self.write("sample.py", "import os\nvalue = os.environ['ACCESS_TOKEN']\n")
        self.write("example.env", "API_KEY=${API_KEY}\n")
        self.write("settings.md", "secret: <PLACEHOLDER>\n")
        self.assertEqual(self.scan("--worktree").returncode, 0)
        self.commit()
        result = self.scan("--staged", "--history", "--worktree")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_staged_reads_index_even_when_worktree_is_cleaned(self):
        address = ".".join(["198", "51", "100", "25"])
        self.write("settings.txt", "server=" + address)
        self.git("add", "settings.txt")
        self.write("settings.txt", "server=127.0.0.1")
        self.assertEqual(self.scan("--worktree").returncode, 0)
        result = self.scan("--staged")
        self.assertEqual(result.returncode, 1)
        self.assertIn("nonlocal-ipv4", result.stdout)
        self.assertNotIn(address, result.stdout + result.stderr)

    def test_deleted_historical_secret_still_blocks_publication(self):
        credential = "ghp_" + "a" * 36
        self.write("old.txt", credential)
        self.commit()
        (self.root / "old.txt").unlink()
        self.write("README.md", "Public documentation.\n")
        self.commit("Remove obsolete file")
        self.assertEqual(self.scan("--staged", "--worktree").returncode, 0)
        result = self.scan("--history")
        self.assertEqual(result.returncode, 1)
        self.assertIn("credential-token", result.stdout)
        self.assertIn("old.txt", result.stdout)
        self.assertNotIn(credential, result.stdout + result.stderr)

    def test_private_absolute_paths_and_artifacts_fail(self):
        private_paths = ["/" + "Users" + "/" + "fixture-user" + "/project",
                         "/" + "home" + "/" + "fixture-user" + "/project",
                         "/mnt/" + "ufs" + "00/project", "/" + "nfs" + "/project"]
        self.write("paths.txt", "\n".join(private_paths))
        self.write("output.log", "Timing: 10 ms")
        self.write("weights.safetensors", b"fixture")
        self.write("binary.dat", b"\x00\x01\xff")
        result = self.scan("--worktree")
        self.assertEqual(result.returncode, 1)
        self.assertIn("private-absolute-path", result.stdout)
        self.assertIn("generated-or-private-artifact", result.stdout)
        self.assertIn("binary-file", result.stdout)
        for value in private_paths:
            self.assertNotIn(value, result.stdout + result.stderr)

    def test_hardcoded_credentials_key_headers_and_cloud_keys_fail(self):
        label = "pass" + "word"
        credential = "synthetic" + "-fixture-value"
        key_header = "-----BEGIN " + "PRIVATE KEY-----"
        cloud_id = "AKIA" + "A" * 16
        self.write("config.py", label + " = " + repr(credential) + "\n")
        self.write("config.yaml", label + ": " + credential + "\n")
        self.write("settings.json", json.dumps({"service_" + label: credential}))
        self.write("material.txt", key_header + "\n" + cloud_id)
        result = self.scan("--worktree")
        self.assertEqual(result.returncode, 1)
        for rule in ("hardcoded-credential", "private-key", "credential-token"):
            self.assertIn(rule, result.stdout)
        for value in (credential, key_header, cloud_id):
            self.assertNotIn(value, result.stdout + result.stderr)

    def test_sensitive_author_and_committer_metadata_fail(self):
        identity = "Fixture" + " Person"
        email = "fixture" + "@" + "example.invalid"
        self.git("config", "user.name", identity)
        self.git("config", "user.email", email)
        self.write("README.md", "Public documentation.\n")
        pending = self.scan("--staged")
        self.assertEqual(pending.returncode, 1)
        self.assertIn("nonpublic-git-identity", pending.stdout)
        self.commit()
        result = self.scan("--history")
        self.assertEqual(result.returncode, 1)
        self.assertIn("nonpublic-git-identity", result.stdout)
        self.assertNotIn(identity, result.stdout + result.stderr)
        self.assertNotIn(email, result.stdout + result.stderr)

    def test_commit_message_is_checked(self):
        self.write("README.md", "Public documentation.\n")
        private_path = "/" + "Users" + "/" + "fixture-user" + "/project"
        self.commit("Copy from " + private_path)
        result = self.scan("--history")
        self.assertEqual(result.returncode, 1)
        self.assertIn("private-absolute-path", result.stdout)
        self.assertIn(":message", result.stdout)
        self.assertNotIn(private_path, result.stdout + result.stderr)

    def test_external_private_fragments_and_fail_closed_configuration(self):
        marker = "fictional" + "-private-project"
        external = Path(self.directory.name) / "private-patterns.json"
        external.write_text(json.dumps([marker]), encoding="utf-8")
        self.environment["EP_XPRESS_PRIVATE_PATTERNS_FILE"] = str(external)
        self.write("README.md", "Project: " + marker.upper())
        result = self.scan("--worktree")
        self.assertEqual(result.returncode, 1)
        self.assertIn("private-pattern", result.stdout)
        self.assertNotIn(marker.upper(), result.stdout + result.stderr)
        self.environment.pop("EP_XPRESS_PRIVATE_PATTERNS_FILE")
        self.git("config", "epxpress.privatePatternsFile", str(external))
        self.assertEqual(self.scan("--worktree").returncode, 1)
        internal = self.root / "private-patterns.json"
        internal.write_text(json.dumps([marker]), encoding="utf-8")
        self.environment["EP_XPRESS_PRIVATE_PATTERNS_FILE"] = str(internal)
        self.assertEqual(self.scan("--worktree").returncode, 2)
        self.environment["EP_XPRESS_PRIVATE_PATTERNS_FILE"] = str(external)
        external.write_text("not JSON", encoding="utf-8")
        result = self.scan("--worktree")
        self.assertEqual(result.returncode, 2)
        self.assertNotIn(str(external), result.stdout + result.stderr)

    def test_large_files_and_non_english_letters_fail(self):
        self.write("large.txt", b"x" * (2 * 1024 * 1024 + 1))
        self.write("notes.md", chr(0x4E2D))
        result = self.scan("--worktree")
        self.assertEqual(result.returncode, 1)
        self.assertIn("oversized-file", result.stdout)
        self.assertIn("non-english-text", result.stdout)


if __name__ == "__main__":
    unittest.main()
