import datetime

class ClaimsLedgerDatabase:
    """
    Simulates an atomic PostgreSQL connection ledger pool.
    Tracks state transitions, commits, and rollback compensation actions.
    """
    def __init__(self):
        # In-memory session tracking simulating active row-state mutations
        self.mock_ledger_table = {}

    def insert_initial_transaction_state(self, tx_id: str, policy_id: str, amount: float) -> str:
        """Creates an uncommitted row placeholder under strict isolation parameters."""
        self.mock_ledger_table[tx_id] = {
            "policy_holder_id": policy_id,
            "claim_amount": amount,
            "ledger_status": "STATE_INITIALIZED",
            "approved_room_rent": 0.0,
            "timestamp": datetime.datetime.now().isoformat()
        }
        print(f"[PostgreSQL Sync]: Row created for TX {tx_id} -> Status: STATE_INITIALIZED")
        return "STATE_INITIALIZED"

    def commit_approved_transaction(self, tx_id: str, calculated_rent: float) -> str:
        """Finalizes the distributed ledger row with approval status attributes."""
        if tx_id in self.mock_ledger_table:
            self.mock_ledger_table[tx_id]["ledger_status"] = "CLAIM_APPROVED_SUCCESSFULLY"
            self.mock_ledger_table[tx_id]["approved_room_rent"] = calculated_rent
            print(f"[PostgreSQL Sync]: TX {tx_id} updated -> Status: CLAIM_APPROVED_SUCCESSFULLY")
            return "CLAIM_APPROVED_SUCCESSFULLY"
        return "ERROR_TX_NOT_FOUND"

    def execute_compensating_rollback(self, tx_id: str, violation_reason: str) -> str:
        """
        The Core Saga Compensating Action Node. Reverses mutations, 
        flags anomalies for audit trail telemetry, and rolls back balance states.
        """
        if tx_id in self.mock_ledger_table:
            self.mock_ledger_table[tx_id]["ledger_status"] = f"REJECTED_ROLLBACK_{violation_reason}"
            self.mock_ledger_table[tx_id]["approved_room_rent"] = 0.0
            print(f"[PostgreSQL Saga Rollback]: TX {tx_id} REVERSED -> Status: REJECTED_ROLLBACK_{violation_reason}")
            return f"REJECTED_ROLLBACK_{violation_reason}"
        return "ERROR_TX_NOT_FOUND"