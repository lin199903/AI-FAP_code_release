"""Secret-free MIMIC-IV Postgres DSN resolver (shared by all 04B scripts).

Resolution order (no credential is ever stored in source):
  1. PG_DSN environment variable
  2. a local .env file containing a PG_DSN=... line; either
       - the path in the MDAP_ENV_FILE environment variable, or
       - ./.env.local next to this file
  3. otherwise raise, with instructions.

For public release this file contains no host, user, or password.
"""
import os


def get_dsn():
    dsn = os.getenv("PG_DSN")
    if dsn:
        return dsn
    candidates = []
    if os.getenv("MDAP_ENV_FILE"):
        candidates.append(os.getenv("MDAP_ENV_FILE"))
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.local"))
    for env_path in candidates:
        if env_path and os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("PG_DSN="):
                        return line.split("=", 1)[1].strip()
    raise RuntimeError(
        "No PostgreSQL DSN found. Set the PG_DSN environment variable, e.g.\n"
        "  set PG_DSN=postgresql://USER:PASSWORD@HOST:5432/mimiciv   (Windows)\n"
        "  export PG_DSN=postgresql://USER:PASSWORD@HOST:5432/mimiciv (Unix)\n"
        "or point MDAP_ENV_FILE at a file containing a 'PG_DSN=...' line."
    )
