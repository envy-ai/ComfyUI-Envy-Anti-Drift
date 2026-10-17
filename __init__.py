# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Bart

from .anti_drift import EnvyKSamplerAdvancedAntiDrift
from .qwen_text_encode import TextEncodeQwenImage21Experimental

NODE_CLASS_MAPPINGS = {"EnvyKSamplerAdvancedAntiDrift": EnvyKSamplerAdvancedAntiDrift}
NODE_DISPLAY_NAME_MAPPINGS = {"EnvyKSamplerAdvancedAntiDrift": "Envy KSampler Advanced Anti-Drift"}

_schema = TextEncodeQwenImage21Experimental.GET_SCHEMA()
NODE_CLASS_MAPPINGS[_schema.node_id] = TextEncodeQwenImage21Experimental
NODE_DISPLAY_NAME_MAPPINGS[_schema.node_id] = _schema.display_name
