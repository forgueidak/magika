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

"""Magika: AI-powered file type detection.

Magika is a novel AI-powered file type detection tool that relies on the
latest generation of AI to provide accurate detection. Under the hood,
Magika uses a custom, highly optimized deep-learning model that allows
it to perform file identification with high accuracy and confidence.

Basic usage:
    >>> from magika import Magika
    >>> m = Magika()
    >>> result = m.identify_bytes(b"# Hello\nprint('world')")
    >>> print(result.output.ct_label)
    'python'

Batch usage:
    >>> from magika import Magika
    >>> m = Magika()
    >>> results = m.identify_paths([Path("file1.py"), Path("file2.js")])
    >>> for result in results:
    ...     print(result.path, result.output.ct_label)
"""

from magika.magika import Magika
from magika.types import (
    MagikaResult,
    MagikaOutputFields,
    ModelFeatures,
    ModelOutput,
    ModelOutputFields,
    PredictionMode,
    Status,
    StatusOr,
)

__version__ = "0.6.0"
__author__ = "Google LLC"
__license__ = "Apache-2.0"

__all__ = [
    "Magika",
    "MagikaResult",
    "MagikaOutputFields",
    "ModelFeatures",
    "ModelOutput",
    "ModelOutputFields",
    "PredictionMode",
    "Status",
    "StatusOr",
    "__version__",
]
