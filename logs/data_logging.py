import os
import json
import hashlib
import logging
from datetime import datetime

class data_logger:
    """
    Centralized JSON-based logger for RFP pipeline.
    Each RFP is identified by a stable document ID derived from metadata.
    """

    def __init__(self, log_filename: str = "rfp_logs.json", app_log: str = "rfp_app.log"):
        self.LOG_FILE = os.path.join(os.path.dirname(__file__), log_filename)
        log_path = os.path.join(os.path.dirname(__file__), app_log)

        # Get a dedicated logger for RFP pipeline
        self.logger = logging.getLogger("RFPLogger")
        self.logger.setLevel(logging.INFO)

        # Avoid duplicate handlers if already added
        if not self.logger.handlers:
            file_handler = logging.FileHandler(log_path)
            stream_handler = logging.StreamHandler()

            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            stream_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(stream_handler)

    # ------------------------
    # Helpers
    # ------------------------
    def _load_logs(self) -> dict:
        if not os.path.exists(self.LOG_FILE):
            return {}
        try:
            with open(self.LOG_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            self.logger.error(f"Corrupted log file {self.LOG_FILE}. Resetting.")
            return {}

    def _save_logs(self, logs: dict):
        with open(self.LOG_FILE, "w") as f:
            json.dump(logs, f, indent=4)

    def _generate_doc_id(self, rfp_number: str, issue_date: str, client_name: str) -> str:
        stable_data = f"{rfp_number}_{issue_date}_{client_name}"
        return hashlib.sha256(stable_data.encode("utf-8")).hexdigest()

    def _update_tool(self, logs: dict, rfp_id: str, tool: str, result: dict):
        """Add or update results for a specific tool in logs."""
        if rfp_id not in logs:
            raise ValueError(f"RFP ID {rfp_id} not found in logs.")

        logs[rfp_id]["tools"][tool] = {
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
        logs[rfp_id]["last_updated"] = datetime.now().isoformat()
        return logs

    # ------------------------
    # Public APIs
    # ------------------------
    def log_rfp(self, document_name: str, extracted_data: dict,
                rfp_number: str, issue_date: str, client_name: str) -> str:
        logs = self._load_logs()
        rfp_id = self._generate_doc_id(rfp_number, issue_date, client_name)

        if rfp_id not in logs:
            logs[rfp_id] = {
                "document_name": document_name,
                "rfp_number": rfp_number,
                "issue_date": issue_date,
                "client_name": client_name,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "tools": {"summary": {"timestamp": datetime.now().isoformat(),
                                      "result": extracted_data}}
            }
            self.logger.info(f"New RFP logged: {document_name} ({rfp_id})")
        else:
            logs[rfp_id]["tools"]["summary"] = {
                "timestamp": datetime.now().isoformat(),
                "result": extracted_data
            }
            logs[rfp_id]["last_updated"] = datetime.now().isoformat()
            self.logger.info(f"Updated summary for RFP {rfp_id}")

        self._save_logs(logs)
        return rfp_id

    def log_match(self, rfp_id: str, result: dict):
        logs = self._load_logs()
        logs = self._update_tool(logs, rfp_id, "matching", result)
        self._save_logs(logs)
        self.logger.info(f"Matching results logged for RFP {rfp_id}")
        return rfp_id

    def log_quotation(self, rfp_id: str, result: dict):
        logs = self._load_logs()
        logs = self._update_tool(logs, rfp_id, "quotation", result)
        self._save_logs(logs)
        self.logger.info(f"Quotation logged for RFP {rfp_id}")
        return rfp_id

    def log_proposal(self, rfp_id: str, result: dict):
        logs = self._load_logs()
        logs = self._update_tool(logs, rfp_id, "proposal", result)
        self._save_logs(logs)
        self.logger.info(f"proposal logged for RFP {rfp_id}")
        return rfp_id

    def log_cutsheet(self, rfp_id: str, result: dict):
        logs = self._load_logs()
        logs = self._update_tool(logs, rfp_id, "cutsheet", result)
        self._save_logs(logs)
        self.logger.info(f"Cutsheet logged for RFP {rfp_id}")
        return rfp_id
    
    def log_email(self, rfp_id: str, result: dict):
        logs = self._load_logs()

        # Ensure structure exists
        logs.setdefault(rfp_id, {}).setdefault("tools", {}).setdefault("email", {}).setdefault("result", {})

        # Expecting result in the form: {"rfq_email": {to_email: {quotation_id: rfp_id}}}
        if "rfq_email" in result:
            rfq_email_data = result.get("rfq_email", {})
            for to_email, quotations in rfq_email_data.items():
                logs[rfp_id]["tools"]["email"]["result"].setdefault("rfq_email", {}).setdefault(to_email, {})
                exist = list(logs[rfp_id]["tools"]["email"]["result"]["rfq_email"][to_email])
                exist.append(quotations)
                logs[rfp_id]["tools"]["email"]["result"]["rfq_email"][to_email] = exist
        else:
            logs[rfp_id]["tools"]["email"]["result"]["submission_email"] = result
        self._save_logs(logs)
        self.logger.info(f"📧 Email logged for RFP {rfp_id}")
        return rfp_id

    def get_rfp_data(self, rfp_id: str) -> dict:
        logs = self._load_logs()
        return logs.get(rfp_id, {})

    def list_all_logs(self) -> dict:
        return self._load_logs()