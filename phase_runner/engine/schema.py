"""JSON schema definitions for Phase Runner rule packs."""

from __future__ import annotations

from typing import Any, Dict


RULE_PACK_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Phase Runner Rule Pack",
    "type": "object",
    "required": ["metadata", "rules"],
    "additionalProperties": False,
    "properties": {
        "metadata": {
            "type": "object",
            "required": ["payer", "pack_version", "engine_version"],
            "additionalProperties": False,
            "properties": {
                "payer": {"type": "string", "minLength": 1},
                "pack_version": {"type": "string", "minLength": 1},
                "engine_version": {"type": "string", "minLength": 1},
                "description": {"type": "string"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "uniqueItems": True,
                },
                "last_reviewed": {"type": "string", "format": "date"},
            },
        },
        "rules": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["id", "when", "then"],
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "description": {"type": "string"},
                    "when": {"$ref": "#/$defs/condition"},
                    "then": {
                        "type": "object",
                        "required": ["decision"],
                        "additionalProperties": False,
                        "properties": {
                            "decision": {
                                "type": "string",
                                "enum": ["approve", "deny", "pend", "manual_review"],
                            },
                            "reason": {"type": "string"},
                            "references": {
                                "type": "array",
                                "items": {"type": "string"},
                                "uniqueItems": True,
                            },
                            "severity": {
                                "type": "string",
                                "enum": ["info", "warning", "error"],
                                "default": "warning",
                            },
                        },
                    },
                    "flags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "uniqueItems": True,
                    },
                },
            },
            "uniqueItems": True,
        },
    },
    "$defs": {
        "condition": {
            "type": "object",
            "oneOf": [
                {
                    "required": ["all"],
                    "properties": {
                        "all": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"$ref": "#/$defs/condition"},
                        }
                    },
                    "additionalProperties": False,
                },
                {
                    "required": ["any"],
                    "properties": {
                        "any": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"$ref": "#/$defs/condition"},
                        }
                    },
                    "additionalProperties": False,
                },
                {
                    "required": ["not"],
                    "properties": {
                        "not": {"$ref": "#/$defs/condition"}
                    },
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "required": ["field", "op"],
                    "properties": {
                        "field": {"type": "string", "minLength": 1},
                        "op": {
                            "type": "string",
                            "enum": [
                                "eq",
                                "ne",
                                "lt",
                                "lte",
                                "gt",
                                "gte",
                                "in",
                                "not_in",
                                "exists",
                                "missing",
                                "matches",
                                "starts_with",
                                "ends_with",
                            ],
                        },
                        "value": {},
                    },
                    "allOf": [
                        {
                            "if": {
                                "properties": {
                                    "op": {"enum": ["exists", "missing"]}
                                }
                            },
                            "then": {
                                "not": {"required": ["value"]}
                            },
                            "else": {
                                "required": ["value"]
                            },
                        }
                    ],
                    "additionalProperties": False,
                },
            ],
        }
    },
}


__all__ = ["RULE_PACK_SCHEMA"]
