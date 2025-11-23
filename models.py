#!/usr/bin/env python3
"""
SQLAlchemy Models for Protein Interaction Database

Tables:
- proteins: Core protein entities with query tracking
- interactions: Protein-protein relationships with full JSONB payload
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sqlalchemy.dialects.postgresql import JSONB
from typing import Optional

db = SQLAlchemy()


class Protein(db.Model):
    """
    Protein entity with query tracking and metadata.

    Invariants:
    - symbol is unique (enforced by DB constraint)
    - query_count increments on each query
    - total_interactions updated after sync
    """
    __tablename__ = 'proteins'

    # Primary key
    id = db.Column(db.Integer, primary_key=True)

    # Protein identifier (unique, indexed for fast lookups)
    symbol = db.Column(db.String(50), unique=True, nullable=False, index=True)

    # Query tracking
    first_queried = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    last_queried = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    query_count = db.Column(db.Integer, default=0, nullable=False)
    total_interactions = db.Column(db.Integer, default=0, nullable=False)

    # Flexible metadata storage (JSONB for schema evolution)
    # Note: Using 'extra_data' instead of 'metadata' (reserved by SQLAlchemy)
    extra_data = db.Column(JSONB, server_default='{}', nullable=False)

    # Audit timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships (one-to-many with interactions)
    interactions_as_a = db.relationship(
        'Interaction',
        foreign_keys='Interaction.protein_a_id',
        backref='protein_a_obj',
        cascade='all, delete-orphan',
        lazy='dynamic'
    )
    interactions_as_b = db.relationship(
        'Interaction',
        foreign_keys='Interaction.protein_b_id',
        backref='protein_b_obj',
        cascade='all, delete-orphan',
        lazy='dynamic'
    )

    def __repr__(self) -> str:
        return f'<Protein {self.symbol}>'


class Interaction(db.Model):
    """
    Protein-protein interaction with full JSONB payload.

    Invariants:
    - (protein_a_id, protein_b_id) is unique
    - protein_a_id != protein_b_id (no self-interactions)
    - data JSONB contains full pipeline output (evidence, functions, PMIDs)
    - interaction_type: 'direct' (physical) or 'indirect' (cascade/pathway)
    - upstream_interactor: required for indirect interactions, null for direct
    - mediator_chain: array of mediator proteins for multi-hop paths
    - depth: 1=direct, 2+=indirect (number of hops from query protein)
    - chain_context: stores interaction from all protein perspectives in chain

    Dual-Track System (for indirect chains):
    - function_context: 'direct' (pair-specific validation), 'net' (NET effect via chain), null (legacy)
    - Example: ATXN3→RHEB→MTOR chain creates TWO records:
      1. ATXN3→MTOR: interaction_type='indirect', function_context='net' (chain NET effect)
      2. RHEB→MTOR: interaction_type='direct', function_context='direct', _inferred_from_chain=True (extracted mediator link)
    """
    __tablename__ = 'interactions'

    # Primary key
    id = db.Column(db.Integer, primary_key=True)

    # Foreign keys (protein pair)
    protein_a_id = db.Column(
        db.Integer,
        db.ForeignKey('proteins.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    protein_b_id = db.Column(
        db.Integer,
        db.ForeignKey('proteins.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )

    # Denormalized fields for fast filtering (extracted from data JSONB)
    confidence = db.Column(db.Numeric(3, 2), index=True)  # 0.00 to 1.00
    direction = db.Column(db.String(20))  # 'bidirectional', 'main_to_primary', 'primary_to_main'
    arrow = db.Column(db.String(50))  # 'binds', 'activates', 'inhibits', 'regulates' (BACKWARD COMPAT: primary arrow)
    arrows = db.Column(JSONB, nullable=True)  # NEW (Issue #4): Multiple arrow types per direction {'main_to_primary': ['activates', 'inhibits'], ...}
    interaction_type = db.Column(db.String(20))  # 'direct' (physical) or 'indirect' (cascade/pathway)
    upstream_interactor = db.Column(db.String(50), nullable=True)  # Upstream protein symbol for indirect interactions
    function_context = db.Column(db.String(20), nullable=True)  # 'direct' (pair-specific), 'net' (NET effect via chain), null (legacy/unvalidated)

    # Chain metadata for multi-level indirect interactions
    mediator_chain = db.Column(JSONB, nullable=True)  # Full chain path e.g., ["VCP", "LAMP2"] for ATXN3→VCP→LAMP2→target
    depth = db.Column(db.Integer, default=1, nullable=False)  # 1=direct, 2=first indirect, 3=second indirect, etc.
    chain_context = db.Column(JSONB, nullable=True)  # Stores full chain context from all protein perspectives
    chain_with_arrows = db.Column(JSONB, nullable=True)  # NEW (Issue #2): Chain with typed arrows [{"from": "VCP", "to": "IκBά", "arrow": "inhibits"}, ...]

    # FULL PAYLOAD - Stores complete interactor JSON from pipeline
    # Contains: evidence[], functions[], pmids[], support_summary, etc.
    # Dual-track flags: _inferred_from_chain, _net_effect, _direct_mediator_link, _display_badge
    data = db.Column(JSONB, nullable=False)

    # Discovery metadata
    discovered_in_query = db.Column(db.String(50))  # Which protein query found this
    discovery_method = db.Column(db.String(50), default='pipeline')  # 'pipeline', 'requery', 'manual'

    # Audit timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Constraints and indexes
    __table_args__ = (
        # Prevent duplicate interactions
        db.UniqueConstraint('protein_a_id', 'protein_b_id', name='interaction_unique'),
        # Prevent self-interactions
        db.CheckConstraint('protein_a_id != protein_b_id', name='interaction_proteins_different'),
        # Indexes for chain queries
        db.Index('idx_interactions_depth', 'depth'),
        db.Index('idx_interactions_interaction_type', 'interaction_type'),
    )

    # Relationships (many-to-one with proteins)
    protein_a = db.relationship('Protein', foreign_keys=[protein_a_id], overlaps="interactions_as_a,protein_a_obj")
    protein_b = db.relationship('Protein', foreign_keys=[protein_b_id], overlaps="interactions_as_b,protein_b_obj")

    def __repr__(self) -> str:
        a_symbol = self.protein_a.symbol if self.protein_a else '?'
        b_symbol = self.protein_b.symbol if self.protein_b else '?'
        return f'<Interaction {a_symbol} ↔ {b_symbol}>'
