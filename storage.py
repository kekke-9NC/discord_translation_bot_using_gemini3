"""Persistent per-server configuration and message/thread correspondence."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TranslationPair:
    guild_id: int
    source_channel_id: int
    target_channel_id: int
    source_language: str
    target_language: str


class Store:
    """Small SQLite store; all keys are scoped to a Discord guild."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self):
        with self._connect() as db:
            db.executescript("""
                PRAGMA foreign_keys = ON;
                CREATE TABLE IF NOT EXISTS glossary (
                    guild_id INTEGER NOT NULL,
                    source_term TEXT NOT NULL COLLATE NOCASE,
                    target_term TEXT NOT NULL,
                    source_language TEXT NOT NULL,
                    target_language TEXT NOT NULL,
                    PRIMARY KEY (guild_id, source_term, source_language, target_language)
                );
                CREATE TABLE IF NOT EXISTS translation_pairs (
                    guild_id INTEGER NOT NULL,
                    source_channel_id INTEGER NOT NULL,
                    target_channel_id INTEGER NOT NULL,
                    source_language TEXT NOT NULL,
                    target_language TEXT NOT NULL,
                    PRIMARY KEY (guild_id, source_channel_id),
                    UNIQUE (guild_id, target_channel_id)
                );
                CREATE TABLE IF NOT EXISTS message_links (
                    guild_id INTEGER NOT NULL,
                    source_channel_id INTEGER NOT NULL,
                    source_message_id INTEGER NOT NULL,
                    target_channel_id INTEGER NOT NULL,
                    target_message_id INTEGER NOT NULL,
                    PRIMARY KEY (guild_id, source_channel_id, source_message_id),
                    UNIQUE (guild_id, target_channel_id, target_message_id)
                );
                CREATE TABLE IF NOT EXISTS thread_links (
                    guild_id INTEGER NOT NULL,
                    source_thread_id INTEGER NOT NULL,
                    target_thread_id INTEGER NOT NULL,
                    PRIMARY KEY (guild_id, source_thread_id),
                    UNIQUE (guild_id, target_thread_id)
                );
            """)

    def set_glossary_term(self, guild_id: int, source: str, target: str,
                          source_language: str, target_language: str):
        source, target = source.strip(), target.strip()
        source_language, target_language = source_language.strip(), target_language.strip()
        if not source or not target:
            raise ValueError("Glossary terms cannot be empty")
        if not source_language or not target_language:
            raise ValueError("Glossary languages cannot be empty")
        with self._connect() as db:
            db.execute("""INSERT INTO glossary VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(guild_id, source_term, source_language, target_language)
                DO UPDATE SET target_term=excluded.target_term""",
                (guild_id, source, target, source_language, target_language))

    def remove_glossary_term(self, guild_id: int, source: str,
                             source_language: str, target_language: str) -> bool:
        source_language, target_language = source_language.strip(), target_language.strip()
        with self._connect() as db:
            cursor = db.execute("DELETE FROM glossary WHERE guild_id=? AND source_term=? COLLATE NOCASE AND source_language=? AND target_language=?",
                (guild_id, source.strip(), source_language, target_language))
            return cursor.rowcount > 0

    def get_glossary(self, guild_id: int, source_language: str,
                     target_language: str) -> list[tuple[str, str]]:
        source_language, target_language = source_language.strip(), target_language.strip()
        with self._connect() as db:
            rows = db.execute("SELECT source_term, target_term FROM glossary WHERE guild_id=? AND source_language=? AND target_language=? ORDER BY length(source_term) DESC, source_term COLLATE NOCASE",
                (guild_id, source_language, target_language)).fetchall()
            return [(row[0], row[1]) for row in rows]

    def add_pair(self, pair: TranslationPair):
        source_language = pair.source_language.strip()
        target_language = pair.target_language.strip()
        if not source_language or not target_language:
            raise ValueError("Source and target languages cannot be empty")
        if pair.source_channel_id == pair.target_channel_id:
            raise ValueError("Source and target channels must differ")
        if source_language.casefold() == target_language.casefold():
            raise ValueError("Source and target languages must differ")
        with self._connect() as db:
            conflict = db.execute("""SELECT 1 FROM translation_pairs
                WHERE guild_id=? AND source_channel_id<>?
                  AND (source_channel_id IN (?, ?) OR target_channel_id IN (?, ?))
                LIMIT 1""", (pair.guild_id, pair.source_channel_id,
                    pair.source_channel_id, pair.target_channel_id,
                    pair.source_channel_id, pair.target_channel_id)).fetchone()
            if conflict:
                raise ValueError("A channel can belong to only one translation pair per server")
            db.execute("INSERT INTO translation_pairs VALUES (?, ?, ?, ?, ?) ON CONFLICT(guild_id, source_channel_id) DO UPDATE SET target_channel_id=excluded.target_channel_id, source_language=excluded.source_language, target_language=excluded.target_language",
                (pair.guild_id, pair.source_channel_id, pair.target_channel_id, source_language, target_language))

    def get_pair(self, guild_id: int, channel_id: int) -> TranslationPair | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM translation_pairs WHERE guild_id=? AND source_channel_id=? UNION ALL SELECT guild_id, target_channel_id AS source_channel_id, source_channel_id AS target_channel_id, target_language AS source_language, source_language AS target_language FROM translation_pairs WHERE guild_id=? AND target_channel_id=? LIMIT 1",
                (guild_id, channel_id, guild_id, channel_id)).fetchone()
        return TranslationPair(*row) if row else None

    def save_message_link(self, guild_id: int, source_channel_id: int,
                          source_message_id: int, target_channel_id: int,
                          target_message_id: int):
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO message_links VALUES (?, ?, ?, ?, ?)",
                (guild_id, source_channel_id, source_message_id, target_channel_id, target_message_id))

    def get_linked_message(self, guild_id: int, channel_id: int, message_id: int):
        with self._connect() as db:
            row = db.execute("SELECT target_channel_id, target_message_id FROM message_links WHERE guild_id=? AND source_channel_id=? AND source_message_id=? UNION ALL SELECT source_channel_id, source_message_id FROM message_links WHERE guild_id=? AND target_channel_id=? AND target_message_id=? LIMIT 1",
                (guild_id, channel_id, message_id, guild_id, channel_id, message_id)).fetchone()
        return tuple(row) if row else None

    def get_forwarded_message(self, guild_id: int, channel_id: int, message_id: int):
        """Return a target only when this message was the original source."""
        with self._connect() as db:
            row = db.execute("SELECT target_channel_id, target_message_id FROM message_links WHERE guild_id=? AND source_channel_id=? AND source_message_id=?",
                (guild_id, channel_id, message_id)).fetchone()
        return tuple(row) if row else None

    def save_thread_link(self, guild_id: int, source_thread_id: int, target_thread_id: int):
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO thread_links VALUES (?, ?, ?)", (guild_id, source_thread_id, target_thread_id))

    def get_linked_thread(self, guild_id: int, thread_id: int) -> int | None:
        with self._connect() as db:
            row = db.execute("SELECT target_thread_id FROM thread_links WHERE guild_id=? AND source_thread_id=? UNION ALL SELECT source_thread_id FROM thread_links WHERE guild_id=? AND target_thread_id=? LIMIT 1",
                (guild_id, thread_id, guild_id, thread_id)).fetchone()
        return int(row[0]) if row else None
