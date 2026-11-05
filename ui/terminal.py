import re
import sys


class TerminalUI:
    # Use a warm orange accent with a plain-text fallback when Rich is unavailable.
    def __init__(self, stream=None, color=True):
        self.stream = stream or sys.stderr
        self.console = None
        try:
            from rich.console import Console

            self.console = Console(
                file=self.stream, no_color=not color, highlight=False
            )
        except ImportError:
            pass

    # Remove control characters and treat model and database text as plain content.
    @staticmethod
    def _safe(value):
        text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(value))
        return "".join(character for character in text if character.isprintable())[:400]

    # Print one styled line without interpreting embedded terminal markup.
    def _line(self, value="", style=None):
        text = self._safe(value)
        if self.console:
            from rich.text import Text

            self.console.print(Text(text, style=style))
        else:
            print(text, file=self.stream)

    # Show the ticket and model before the investigation starts.
    def show_header(self, ticket_id, model, budget):
        self._line()
        self._line("  > ERP Ticket Agent", "bold #D97757")
        self._line(f"    {ticket_id} | {model} | {budget} tool call(s)", "dim")
        self._line("    Investigating...", "dim")

    # Show tool execution status without exposing hidden model reasoning.
    def show_tool_result(self, record, used, budget):
        status = record["observation"].get("status", "tool_error")
        success = status == "ok"
        marker = "+" if success else "!"
        cached = " (cached)" if record.get("cache_hit") else ""
        self._line(
            f"    {marker} [{used}/{budget}] {record['tool_name']} -> {status}{cached}",
            "green" if success else "yellow",
        )

    # Display a compact provisional diagnosis with truthful evidence and token labels.
    def show_summary(self, result):
        diagnosis, metrics = result.diagnosis, result.metrics
        headers = ("Ticket", "Cause (provisional)", "Calls", "Time")
        values = (
            diagnosis["ticket_id"],
            diagnosis["root_cause_code"],
            f"{metrics['tool_calls']}/{metrics['tool_budget']}",
            f"{metrics['elapsed_seconds']:.2f}s",
        )
        self._line()
        if self.console:
            from rich.table import Table
            from rich.text import Text

            table = Table(box=None, padding=(0, 2), header_style="bold #D97757")
            for header in headers:
                table.add_column(header)
            table.add_row(*(Text(self._safe(value)) for value in values))
            self.console.print(table)
        else:
            self._line(" | ".join(headers))
            self._line(" | ".join(str(value) for value in values))
        self._line("  Evidence     Not verified (verification step pending)", "yellow")
        self._line(
            f"  Action       {diagnosis['proposed_fix']['action']} (recommendation only)"
        )
        self._line(f"  Escalation   {'Yes' if diagnosis['escalate'] else 'No'}")
        self._line(f"  Stop reason  {metrics['termination_reason']}", "dim")
        tokens = sum(metrics["reported_tokens"].values())
        qualifier = "reported" if metrics["usage_complete"] else "partially reported"
        self._line(
            f"  {metrics['model_calls']} model call(s) | {tokens} {qualifier} tokens | {metrics['retries']} retries",
            "dim",
        )
        self._line(
            "  Results require human review until evidence verification is implemented.",
            "dim",
        )
