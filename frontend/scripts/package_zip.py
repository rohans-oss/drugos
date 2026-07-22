#!/usr/bin/env python3
"""
Package the upgraded DrugOS codebase into a zip file for delivery.

Includes:
- src/ (frontend + API routes)
- prisma/ (schema)
- public/ (static assets)
- scripts/ (utility scripts)
- package.json, tsconfig.json, next.config.ts, tailwind.config.ts,
  postcss.config.mjs, eslint.config.mjs, components.json, Caddyfile
- .env.example, README.md, UPGRADE_NOTES.md, worklog.md

Excludes:
- node_modules/, .next/, .git/, .zscripts/, db/, dev.log, server.log
- src_backup/, prisma_backup/, public_backup/, scripts_backup/, uploaded_code/
- tests/, examples/ (kept separate, not required to run the app)
"""

import os
import zipfile
import sys
from pathlib import Path

PROJECT_DIR = Path("/home/z/my-project")
OUT_ZIP = Path("/home/z/my-project/download/drugos_v0.6.0_upgraded.zip")

INCLUDE_DIRS = [
    "src",
    "prisma",
    "public",
    "scripts",
]

INCLUDE_FILES = [
    "package.json",
    "tsconfig.json",
    "next.config.ts",
    "tailwind.config.ts",
    "postcss.config.mjs",
    "eslint.config.mjs",
    "components.json",
    "Caddyfile",
    ".env.example",
    ".gitignore",
    "README.md",
    "UPGRADE_NOTES.md",
    "worklog.md",
]

EXCLUDE_PATTERNS = [
    "node_modules",
    ".next",
    ".git",
    "__pycache__",
    ".DS_Store",
    "*.log",
    "*.db",
    "*.db-journal",
]


def should_exclude(path: Path) -> bool:
    parts = path.parts
    for pattern in EXCLUDE_PATTERNS:
        if pattern.startswith("*"):
            if path.name.endswith(pattern[1:]):
                return True
        elif pattern in parts:
            return True
        elif path.name == pattern:
            return True
    return False


def main():
    # Make sure output dir exists
    OUT_ZIP.parent.mkdir(parents=True, exist_ok=True)

    if OUT_ZIP.exists():
        OUT_ZIP.unlink()

    file_count = 0
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Write a top-level folder `drugos_v0.6.0_upgraded/` so unzip creates a clean dir.
        TOP = "drugos_v0.6.0_upgraded"

        # Directories
        for d in INCLUDE_DIRS:
            src_dir = PROJECT_DIR / d
            if not src_dir.exists():
                print(f"  SKIP (missing): {d}/")
                continue
            for root, dirs, files in os.walk(src_dir):
                # Prune excluded dirs in-place so os.walk doesn't descend into them.
                dirs[:] = [x for x in dirs if x not in ("node_modules", ".next", ".git", "__pycache__")]
                for fname in files:
                    abs_fp = Path(root) / fname
                    rel = abs_fp.relative_to(PROJECT_DIR)
                    if should_exclude(rel):
                        continue
                    arcname = f"{TOP}/{rel.as_posix()}"
                    zf.write(abs_fp, arcname)
                    file_count += 1

        # Files
        for f in INCLUDE_FILES:
            src_fp = PROJECT_DIR / f
            if not src_fp.exists() or not src_fp.is_file():
                print(f"  SKIP (missing): {f}")
                continue
            arcname = f"{TOP}/{f}"
            zf.write(src_fp, arcname)
            file_count += 1

        # Add a fresh .env.example if .env.example doesn't exist but .env does
        if not (PROJECT_DIR / ".env.example").exists():
            env_example = """# DrugOS Environment Configuration
# Copy this file to .env and fill in the required values.
#   cp .env.example .env

# REQUIRED: SQLite database file location
DATABASE_URL=file:./db/custom.db

# REQUIRED: JWT signing secret. Generate with: openssl rand -hex 32
JWT_SECRET=replace_me_with_a_long_random_hex_string_at_least_32_bytes_long

# OPTIONAL: biomedical data source API keys
NCBI_API_KEY=
PATENTSVIEW_API_KEY=

# OPTIONAL: standalone ML service URLs (Phase 1, 2, 4 of build plan)
KG_SERVICE_URL=
DATASET_SERVICE_URL=
RL_SERVICE_URL=

# Application
NODE_ENV=development
PORT=3000
"""
            zf.writestr(f"{TOP}/.env.example", env_example)
            file_count += 1

    size_mb = OUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"\n✓ Created {OUT_ZIP}")
    print(f"  Files: {file_count}")
    print(f"  Size:  {size_mb:.2f} MB")


if __name__ == "__main__":
    sys.exit(main())
