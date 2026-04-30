import argparse
import sqlite3


def main() -> None:
    """Generate an organization count database from an mbox mailbox file."""
    parser = argparse.ArgumentParser(
        description="Count emails by organization from an mbox file."
    )
    parser.add_argument(
        "mbox_path",
        nargs="?",
        default="mbox.txt",
        help="Path to the mbox file (default: mbox.txt).",
    )
    parser.add_argument(
        "database_path",
        nargs="?",
        default="orgcount.sqlite",
        help="Path to the SQLite database (default: orgcount.sqlite).",
    )
    args = parser.parse_args()

    org_counts = {}
    try:
        with open(args.mbox_path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.startswith("From "):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                email = parts[1]
                if "@" not in email:
                    continue
                org = email.split("@", 1)[1].lower()
                org_counts[org] = org_counts.get(org, 0) + 1
    except FileNotFoundError as exc:
        raise SystemExit(
            f"mbox file not found: {args.mbox_path}"
        ) from exc

    try:
        with sqlite3.connect(args.database_path) as conn:
            cur = conn.cursor()
            cur.execute("DROP TABLE IF EXISTS Counts")
            cur.execute("CREATE TABLE Counts (org TEXT PRIMARY KEY, count INTEGER)")
            cur.executemany(
                "INSERT INTO Counts (org, count) VALUES (?, ?)",
                org_counts.items(),
            )
    except sqlite3.Error as exc:
        raise SystemExit(
            f"Failed to write database at {args.database_path}: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
