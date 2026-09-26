from config import REV
from lib_digest import build_digest, digest_text

if __name__ == "__main__":
    d = build_digest(REV)
    (REV / "revision_digest.json").write_text(digest_text(d), encoding="utf-8")
    print({k: len(v) for k, v in d.items()})
