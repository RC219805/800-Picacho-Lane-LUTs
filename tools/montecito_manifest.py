import hashlib
from pathlib import Path

def _md5(path: Path) -> str:
    """Calculate MD5 hash with usedforsecurity=False for non-cryptographic use."""
    digest = hashlib.md5(usedforsecurity=False)  # Fixed: Added usedforsecurity=False
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()
