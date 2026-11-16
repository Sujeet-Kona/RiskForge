from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

from sqlalchemy import text
from sqlalchemy.engine import make_url

from db.config import DatabaseConfig
from db.repository import AuditRecord, AuditRepository
from db.session import AuditDatabase


def database() -> AuditDatabase:
    configured = os.environ.get("DATABASE_URL", "")
    url = make_url(configured)
    if (
        url.get_backend_name() != "postgresql"
        or url.database != "riskforge_e2e"
        or url.host not in {"localhost", "127.0.0.1", "::1"}
    ):
        raise SystemExit("E2E fixtures require riskforge_e2e on a loopback host.")
    return AuditDatabase.create(DatabaseConfig.from_environment())


def reset(db: AuditDatabase) -> None:
    with db.engine.begin() as connection:
        unrelated = connection.scalar(
            text(
                "SELECT count(*) FROM risk_decisions WHERE transaction_id NOT LIKE 'E2E-%'"
            )
        )
        if unrelated:
            raise SystemExit(
                "Refusing to reset a database containing non-E2E decisions."
            )
        connection.execute(
            text(
                "TRUNCATE TABLE risk_idempotency_records, risk_review_events, "
                "risk_review_cases, risk_decisions RESTART IDENTITY"
            )
        )


def seed(db: AuditDatabase) -> None:
    repository = AuditRepository(db)
    for name in ("APPROVE", "REJECT"):
        repository.create_audit_record(
            AuditRecord(
                transaction_id=f"E2E-{name}",
                risk_probability=0.09,
                decision="REVIEW",
                reason_codes=[
                    {
                        "code": "RC-CARD-VELOCITY-1H",
                        "title": "Synthetic card velocity signal",
                        "type": "behavioral",
                        "description": "Deterministic synthetic review fixture.",
                        "feature": "card_txn_count_1h",
                        "contribution": 0.41,
                        "direction": "risk_increasing",
                        "n_transformed_columns": 1,
                    }
                ],
                risk_decreasing_signals=[],
                decision_reason_code={
                    "code": "DEC-REVIEW",
                    "title": "Synthetic review-band decision",
                    "type": "decision_rule",
                    "description": "Synthetic fixture in the review band.",
                },
                decision_explanation="Synthetic fixture: 0.09 is in the REVIEW band.",
                model_detail={
                    "base_value": -3.1,
                    "raw_margin": -2.3,
                    "raw_model_probability": 0.0911,
                    "attribution_space": "raw_margin",
                    "attribution_space_note": "Synthetic raw-margin fixture.",
                    "total_risk_increasing": 0.41,
                    "total_risk_decreasing": 0.0,
                },
                policy_detail={"review_threshold": 0.075, "block_threshold": 0.11},
                scored_inputs={"TransactionAmt": 117.5, "ProductCD": "W"},
                top_k=3,
                model_version="e2e-synthetic",
                feature_version="behavioral-v1",
                calibration_method="isotonic",
                policy_version="policy-001",
                feature_count=28,
                history_transactions_used=0,
            )
        )


def snapshot(db: AuditDatabase) -> dict:
    cases, total = AuditRepository(db).list_review_cases(
        status=None, limit=100, offset=0
    )
    with db.engine.connect() as connection:
        events = (
            connection.execute(
                text(
                    "SELECT review_case_id, action, reviewer_id, comment "
                    "FROM risk_review_events ORDER BY id"
                )
            )
            .mappings()
            .all()
        )
    return {
        "total": total,
        "cases": [asdict(case) for case in cases],
        "events": [dict(e) for e in events],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("reset", "seed", "snapshot"))
    args = parser.parse_args()
    db = database()
    try:
        if args.action == "reset":
            reset(db)
        elif args.action == "seed":
            seed(db)
        else:
            print(json.dumps(snapshot(db), default=str))
    finally:
        db.dispose()
