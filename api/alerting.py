import os
import requests
import json
from datetime import datetime
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")


def send_slack_alert(anomalies: list[dict], summary: dict) -> bool:
    """
    Send a Slack alert when anomalies are detected.
    
    Args:
        anomalies: list of anomalous prediction records
        summary: dict with total_records, anomaly_count, anomaly_rate_pct
    
    Returns:
        True if alert sent successfully, False otherwise
    """
    if not anomalies:
        return False

    # Build anomaly details text
    details = ""
    for a in anomalies[:5]:  # show max 5 to keep message clean
        details += (
            f"• `{a['timestamp']}` — "
            f"score: *{a['anomaly_score']}* | "
            f"confidence: *{a['confidence']}* | "
            f"cpu1: {a['cpu_util_1']} | "
            f"cpu2: {a['cpu_util_2']} | "
            f"rds: {a['rds_cpu']}\n"
        )

    if len(anomalies) > 5:
        details += f"_...and {len(anomalies) - 5} more anomalies_\n"

    # Severity color — red if high confidence, orange if medium, yellow if low
    max_score = max(a["anomaly_score"] for a in anomalies)
    threshold = summary["threshold_used"]

    if max_score > threshold * 1.2:
        color   = "#FF0000"   # red
        severity = "🔴 CRITICAL"
    elif max_score > threshold * 1.05:
        color   = "#FF8C00"   # orange
        severity = "🟠 WARNING"
    else:
        color   = "#FFD700"   # yellow
        severity = "🟡 CAUTION"

    # Slack message payload
    payload = {
        "attachments": [
            {
                "color": color,
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"{severity} — Anomalies Detected"
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*Total records scored:*\n{summary['total_records']}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Anomalies detected:*\n{summary['anomaly_count']}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Anomaly rate:*\n{summary['anomaly_rate_pct']}%"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Max anomaly score:*\n{round(max_score, 4)}"
                            },
                        ]
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Anomalous timestamps:*\n{details}"
                        }
                    },
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": (
                                    f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC | "
                                    f"Threshold: {threshold} | "
                                    f"Model: Isolation Forest"
                                )
                            }
                        ]
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(
            SLACK_WEBHOOK_URL,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        if response.status_code == 200:
            print(f"[alert] Slack alert sent — {summary['anomaly_count']} anomalies")
            return True
        else:
            print(f"[alert] Slack error: {response.status_code} {response.text}")
            return False
    except Exception as e:
        print(f"[alert] Failed to send Slack alert: {e}")
        return False