# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Type definitions for Magika content type detection."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class MagikaStatus(str, Enum):
    """Status codes for Magika detection results."""

    OK = "ok"
    EMPTY_FILE = "empty_file"
    TOO_SMALL_FILE = "too_small_file"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class ContentTypeInfo:
    """Information about a detected content type."""

    label: str
    """Short label identifying the content type (e.g., 'python', 'pdf')."""

    mime_type: str
    """MIME type string (e.g., 'text/x-python', 'application/pdf')."""

    group: str
    """High-level group/category (e.g., 'code', 'document', 'image')."""

    description: str
    """Human-readable description of the content type."""

    extensions: list[str] = field(default_factory=list)
    """Common file extensions associated with this content type."""

    is_text: bool = False
    """Whether this content type is text-based."""

    def __str__(self) -> str:
        return f"{self.label} ({self.mime_type})"


@dataclass
class MagikaResult:
    """Result of a Magika content type detection operation."""

    path: Optional[Path]
    """Path to the file that was analyzed, or None for raw bytes input."""

    dl: ContentTypeInfo
    """Content type determined by the deep learning model."""

    output: ContentTypeInfo
    """Final output content type (may differ from dl if overrides apply)."""

    score: float
    """Confidence score in the range [0.0, 1.0]."""

    status: MagikaStatus = MagikaStatus.OK
    """Status of the detection operation."""

    def ok(self) -> bool:
        """Return True if the detection completed without errors."""
        return self.status == MagikaStatus.OK

    def __str__(self) -> str:
        path_str = str(self.path) if self.path else "<bytes>"
        return (
            f"MagikaResult(path={path_str!r}, label={self.output.label!r}, "
            f"mime_type={self.output.mime_type!r}, score={self.score:.4f}, "
            f"status={self.status.value!r})"
        )


@dataclass
class ModelConfig:
    """Configuration parameters for the Magika ONNX model."""

    beg_size: int
    """Number of bytes to sample from the beginning of the file."""

    mid_size: int
    """Number of bytes to sample from the middle of the file."""

    end_size: int
    """Number of bytes to sample from the end of the file."""

    use_inputs_at_offsets: bool
    """Whether the model uses inputs sampled at specific offsets."""

    medium_confidence_threshold: float
    """Score threshold below which a result is considered medium confidence."""

    min_file_size_for_dl: int
    """Minimum file size in bytes required to use the deep learning model."""

    padding_token: int = 256
    """Token value used to pad input sequences shorter than the required size."""
