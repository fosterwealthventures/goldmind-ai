# notification_system.py
import logging
from typing import Dict

class NotificationManager:
    """
    Manages and sends notifications, alerts, and reports for the GoldMIND system.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger("NotificationManager")
        self.logger.info("NotificationManager initialized with configuration.")

    def send_alert(self, alert_type: str, message: str, severity: str = "INFO"):
        """
        Sends an alert based on system events.
        
        Args:
            alert_type (str): The type of alert (e.g., 'system_error', 'performance_degradation').
            message (str): The detailed message of the alert.
            severity (str): The severity level ('INFO', 'WARNING', 'ERROR', 'CRITICAL').
        """
        self.logger.log(getattr(logging, severity.upper(), logging.INFO), 
                        f"ALERT: [{alert_type.upper()}] - {message}")

    def send_report(self, report_type: str, data: Dict):
        """
        Sends a detailed report.
        
        Args:
            report_type (str): The type of report (e.g., 'daily_summary', 'model_performance').
            data (Dict): The data to be included in the report.
        """
        self.logger.info(f"REPORT: [{report_type.upper()}] - Data: {data}")

# Example of how to use it
if __name__ == "__main__":
    mock_config = {}
    nm = NotificationManager(mock_config)
    nm.send_alert("system_startup", "GoldMIND AI has started.", severity="INFO")
    nm.send_alert("model_failure", "LSTM model failed to generate a prediction.", severity="ERROR")
    nm.send_report("daily_summary", {"total_requests": 1500, "errors": 5})
