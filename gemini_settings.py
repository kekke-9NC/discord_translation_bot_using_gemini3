"""Gemini model configuration shared by translation and reply generation."""

import os


GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemma-4-26b-a4b-it")
