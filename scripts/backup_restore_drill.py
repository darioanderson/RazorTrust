from __future__ import annotations

import argparse
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def _run(*args: str, capture: bool = False) -> str:
    result = subprocess.run(
        list(args),
        check=True,
        text=True,
        capture_output=capture,
    )
    return result.stdout.strip() if capture else ""


def run_drill(
    *,
    compose_file: Path,
    backup_path: Path,
    restore_database: str = "razortrust_restore_drill",
    keep_restore_database: bool = False,
) -> dict[str, object]:
    if restore_database == "razortrust":
        raise ValueError("restore drill database must not be the live razortrust database")
    if not compose_file.is_file():
        raise FileNotFoundError(compose_file)

    container_id = _run(
        "docker", "compose", "-f", str(compose_file), "ps", "-q", "postgres", capture=True
    )
    if not container_id:
        raise RuntimeError("postgres service is not running")

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    container_backup = f"/tmp/razortrust-backup-{stamp}.dump"

    try:
        _run(
            "docker",
            "exec",
            container_id,
            "pg_dump",
            "--format=custom",
            "--dbname=razortrust",
            "--username=razortrust",
            f"--file={container_backup}",
        )
        _run("docker", "cp", f"{container_id}:{container_backup}", str(backup_path))
        if not backup_path.is_file() or backup_path.stat().st_size == 0:
            raise RuntimeError("backup is empty")

        _run(
            "docker",
            "exec",
            container_id,
            "dropdb",
            "--if-exists",
            f"--dbname={restore_database}",
            "--username=razortrust",
        )
        _run(
            "docker",
            "exec",
            container_id,
            "createdb",
            f"--dbname={restore_database}",
            "--username=razortrust",
        )
        _run(
            "docker",
            "exec",
            container_id,
            "pg_restore",
            "--exit-on-error",
            "--no-owner",
            f"--dbname={restore_database}",
            "--username=razortrust",
            container_backup,
        )
        table_count_text = _run(
            "docker",
            "exec",
            container_id,
            "psql",
            f"--dbname={restore_database}",
            "--username=razortrust",
            "--tuples-only",
            "--no-align",
            "--command=SELECT count(*) FROM information_schema.tables WHERE table_schema='public';",
            capture=True,
        )
        table_count = int(table_count_text)
        if table_count <= 0:
            raise RuntimeError("restored database contains no public tables")
        alembic_version = _run(
            "docker",
            "exec",
            container_id,
            "psql",
            f"--dbname={restore_database}",
            "--username=razortrust",
            "--tuples-only",
            "--no-align",
            "--command=SELECT version_num FROM alembic_version LIMIT 1;",
            capture=True,
        )
        if not alembic_version:
            raise RuntimeError("restored database has no Alembic version")
        return {
            "backup_path": str(backup_path.resolve()),
            "backup_bytes": backup_path.stat().st_size,
            "restore_database": restore_database,
            "restored_public_tables": table_count,
            "alembic_version": alembic_version,
            "restore_verified": True,
        }
    finally:
        if container_id:
            if not keep_restore_database:
                subprocess.run(
                    [
                        "docker",
                        "exec",
                        container_id,
                        "dropdb",
                        "--if-exists",
                        f"--dbname={restore_database}",
                        "--username=razortrust",
                    ],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            subprocess.run(
                ["docker", "exec", container_id, "rm", "-f", container_backup],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Back up RazorTrust and prove a real fresh restore."
    )
    parser.add_argument("--compose-file", required=True, type=Path)
    parser.add_argument("--backup-path", required=True, type=Path)
    parser.add_argument("--restore-database", default="razortrust_restore_drill")
    parser.add_argument("--keep-restore-database", action="store_true")
    args = parser.parse_args()
    result = run_drill(
        compose_file=args.compose_file,
        backup_path=args.backup_path,
        restore_database=args.restore_database,
        keep_restore_database=args.keep_restore_database,
    )
    print("Backup + fresh restore drill PASSED")
    for key, value in result.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
