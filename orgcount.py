import argparse
import sqlite3


def main() -> None:
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

    conn = sqlite3.connect(args.database_path)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS Counts")
    cur.execute("CREATE TABLE Counts (org TEXT PRIMARY KEY, count INTEGER)")

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
            org = email.split("@", 1)[1]
            cur.execute(
                "INSERT INTO Counts (org, count) VALUES (?, 1) "
                "ON CONFLICT(org) DO UPDATE SET count = count + 1",
                (org,),
            )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()
