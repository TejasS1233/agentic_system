"""
Cron/Reminder Tool - Generate cron schedules and iCalendar (.ics) reminder files
from natural language descriptions.

Uses LLM (Groq API direct) to parse natural language into structured schedules.
Outputs both cron expressions and downloadable .ics calendar files.
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional
from pydantic import BaseModel, Field


class CronReminderToolArgs(BaseModel):
    description: str = Field(
        ...,
        description=(
            "Natural language description of the reminder. "
            "E.g. 'Check arXiv for new diffusion papers every Monday at 9am', "
            "'Remind me to review code every weekday at 3pm', "
            "'Monthly team meeting first Friday of each month at 2pm'"
        ),
    )
    output_format: Optional[str] = Field(
        "both",
        description="Output format: 'cron', 'ics', or 'both' (default 'both').",
    )
    output_path: Optional[str] = Field(
        None,
        description="Output file path for the .ics file.",
    )


class CronReminderTool:
    """
    Generate cron schedules and iCalendar (.ics) reminder files from
    natural language descriptions.

    Features:
    - Natural language to cron expression conversion
    - iCalendar (.ics) file generation for importing into any calendar app
    - Supports recurring events (daily, weekly, monthly)
    - Human-readable schedule explanation

    Uses LLM for natural language parsing, outputs standard formats.
    """

    name = "cron_reminder"
    description = (
        "Generate cron schedules and calendar reminder files (.ics) from natural language. "
        "Say something like 'Remind me to check arXiv every Monday at 9am' and get "
        "a cron expression plus a downloadable .ics calendar file."
    )
    args_schema = CronReminderToolArgs

    # Common schedule patterns for fallback parsing
    DAY_MAP = {
        "monday": 1,
        "tuesday": 2,
        "wednesday": 3,
        "thursday": 4,
        "friday": 5,
        "saturday": 6,
        "sunday": 0,
        "mon": 1,
        "tue": 2,
        "wed": 3,
        "thu": 4,
        "fri": 5,
        "sat": 6,
        "sun": 0,
    }

    CRON_DAY_MAP = {
        0: "SUN",
        1: "MON",
        2: "TUE",
        3: "WED",
        4: "THU",
        5: "FRI",
        6: "SAT",
    }

    ICAL_DAY_MAP = {
        0: "SU",
        1: "MO",
        2: "TU",
        3: "WE",
        4: "TH",
        5: "FR",
        6: "SA",
    }

    def __init__(self, output_dir: str = "/output"):
        self.output_dir = output_dir
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError:
            self.output_dir = "/tmp"

    def _call_llm(self, prompt: str) -> str:
        """Call Groq API directly."""
        import requests

        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            return ""

        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 512,
                    "temperature": 0.1,
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return ""

    def _parse_schedule(self, description: str) -> dict:
        """Use LLM to extract schedule details from natural language."""
        prompt = f"""Parse this reminder/schedule into a JSON object.

OUTPUT ONLY valid JSON, no markdown fences, no explanation.

Required fields:
- "task": string (what to do)
- "cron": string (standard 5-field cron expression: minute hour day_of_month month day_of_week)
- "frequency": string (human readable: "Every Monday at 9:00 AM")
- "hour": int (0-23)
- "minute": int (0-59)
- "days_of_week": array of ints (0=Sunday, 1=Monday, ..., 6=Saturday) — empty for daily/monthly
- "recurrence": string ("daily", "weekly", "monthly", "yearly", "weekdays")

INPUT: {description}

JSON:"""

        content = self._call_llm(prompt)
        if content:
            if content.startswith("```"):
                lines = content.split("\n")[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines)
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "cron" in data:
                    return data
            except json.JSONDecodeError:
                pass

        # Fallback: basic parsing
        return self._fallback_parse(description)

    def _fallback_parse(self, description: str) -> dict:
        """Basic regex-based schedule parsing."""
        import re

        desc_lower = description.lower()

        hour, minute = 9, 0  # default 9:00 AM
        # Try to find time
        time_match = re.search(r"(\d{1,2})\s*(?::(\d{2}))?\s*(am|pm)?", desc_lower)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2)) if time_match.group(2) else 0
            if time_match.group(3) == "pm" and hour < 12:
                hour += 12
            elif time_match.group(3) == "am" and hour == 12:
                hour = 0

        days = []
        for day_name, day_num in self.DAY_MAP.items():
            if day_name in desc_lower:
                if day_num not in days:
                    days.append(day_num)

        if "weekday" in desc_lower or "every day" in desc_lower:
            recurrence = "weekdays" if "weekday" in desc_lower else "daily"
            cron_dow = "1-5" if recurrence == "weekdays" else "*"
            cron = f"{minute} {hour} * * {cron_dow}"
            freq = f"Every {'weekday' if recurrence == 'weekdays' else 'day'} at {hour}:{minute:02d}"
        elif "monthly" in desc_lower or "month" in desc_lower:
            recurrence = "monthly"
            cron = f"{minute} {hour} 1 * *"
            freq = f"Monthly on the 1st at {hour}:{minute:02d}"
        elif days:
            recurrence = "weekly"
            cron_days = ",".join(self.CRON_DAY_MAP[d] for d in sorted(days))
            day_names = ", ".join(self.CRON_DAY_MAP[d] for d in sorted(days))
            cron = f"{minute} {hour} * * {cron_days}"
            freq = f"Every {day_names} at {hour}:{minute:02d}"
        else:
            recurrence = "weekly"
            cron = f"{minute} {hour} * * MON"
            freq = f"Every Monday at {hour}:{minute:02d}"
            days = [1]

        # Extract task
        task = description
        for word in [
            "remind me to",
            "reminder to",
            "remind me",
            "alert me to",
            "every ",
            "at ",
        ]:
            task = re.sub(rf"(?i){word}", "", task).strip()
        task = task[:100] if task else description[:100]

        return {
            "task": task,
            "cron": cron,
            "frequency": freq,
            "hour": hour,
            "minute": minute,
            "days_of_week": days,
            "recurrence": recurrence,
        }

    def _generate_ics(self, schedule: dict, description: str) -> str:
        """Generate an iCalendar (.ics) file content."""
        now = datetime.utcnow()
        task = schedule.get("task", description)
        frequency = schedule.get("frequency", "")
        recurrence = schedule.get("recurrence", "weekly")
        hour = schedule.get("hour", 9)
        minute = schedule.get("minute", 0)
        days = schedule.get("days_of_week", [])

        # Generate UID
        uid = hashlib.md5(f"{task}{now.isoformat()}".encode()).hexdigest()

        # Start date: next occurrence
        dtstart = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if dtstart <= now:
            dtstart += timedelta(days=1)

        dtstart_str = dtstart.strftime("%Y%m%dT%H%M%S")
        dtend = dtstart + timedelta(minutes=30)
        dtend_str = dtend.strftime("%Y%m%dT%H%M%S")
        dtstamp = now.strftime("%Y%m%dT%H%M%SZ")

        # Build RRULE
        if recurrence == "daily":
            rrule = "RRULE:FREQ=DAILY"
        elif recurrence == "weekdays":
            rrule = "RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR"
        elif recurrence == "monthly":
            rrule = "RRULE:FREQ=MONTHLY;BYMONTHDAY=1"
        elif recurrence == "yearly":
            rrule = "RRULE:FREQ=YEARLY"
        elif days:
            ical_days = ",".join(self.ICAL_DAY_MAP[d] for d in sorted(days))
            rrule = f"RRULE:FREQ=WEEKLY;BYDAY={ical_days}"
        else:
            rrule = "RRULE:FREQ=WEEKLY;BYDAY=MO"

        ics = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//IASCIS//CronReminderTool//EN
CALSCALE:GREGORIAN
METHOD:PUBLISH
BEGIN:VEVENT
UID:{uid}@iascis
DTSTAMP:{dtstamp}
DTSTART:{dtstart_str}
DTEND:{dtend_str}
{rrule}
SUMMARY:{task}
DESCRIPTION:{description}
BEGIN:VALARM
TRIGGER:-PT10M
ACTION:DISPLAY
DESCRIPTION:Reminder: {task}
END:VALARM
END:VEVENT
END:VCALENDAR"""

        return ics

    def run(
        self,
        description: str,
        output_format: str = "both",
        output_path: str = None,
    ) -> str:
        """Generate cron schedule and/or .ics file from natural language.

        Args:
            description: Natural language reminder description
            output_format: 'cron', 'ics', or 'both'
            output_path: Output .ics file path

        Returns:
            JSON with cron expression, schedule details, and .ics file path.
        """
        output_format = (output_format or "both").lower()
        if output_format not in ("cron", "ics", "both"):
            output_format = "both"

        # Parse the schedule
        schedule = self._parse_schedule(description)

        result = {
            "success": True,
            "task": schedule.get("task", description),
            "cron_expression": schedule.get("cron", ""),
            "frequency": schedule.get("frequency", ""),
            "recurrence": schedule.get("recurrence", ""),
        }

        # Generate .ics if requested
        if output_format in ("ics", "both"):
            ics_content = self._generate_ics(schedule, description)

            # Always under /output/ so it persists on host volume
            if output_path and not output_path.startswith(self.output_dir):
                output_path = os.path.join(
                    self.output_dir, os.path.basename(output_path)
                )
            if not output_path:
                existing = [
                    f
                    for f in os.listdir(self.output_dir)
                    if f.startswith("reminder_") and f.endswith(".ics")
                ]
                idx = len(existing) + 1
                output_path = os.path.join(self.output_dir, f"reminder_{idx}.ics")

            os.makedirs(
                os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
                exist_ok=True,
            )

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(ics_content)

            result["ics_path"] = output_path
            result["ics_file_size_bytes"] = len(ics_content.encode("utf-8"))

        result["message"] = (
            f"Schedule: {schedule.get('frequency', '')}. "
            f"Cron: {schedule.get('cron', '')}. "
            + (
                f"Calendar file saved to {output_path}."
                if output_format in ("ics", "both")
                else ""
            )
        )

        return json.dumps(result, indent=2)
