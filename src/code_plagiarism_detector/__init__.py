"""Code plagiarism detection using perceptual hashing and AST fingerprinting."""

from .code_hasher import CodeHasher
from .database import CodeHashDatabase

__version__ = "0.1.0"
__all__ = ["CodeHasher", "CodeHashDatabase"]
