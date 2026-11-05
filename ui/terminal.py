import os
import re
import shutil
import sys
import textwrap
import threading
import time


class TerminalActivity:
    # Keep the waiting indicator separate from agent logic and output rendering.
    def __init__(self, terminal):
        self.terminal = terminal
        self.stop = threading.Event()
        self.thread = None

    # Draw a small spinner only while waiting in an interactive terminal.
    def _spin(self):
        frames = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
        index = 0
        while not self.stop.is_set():
            with self.terminal.lock:
                text = f"  {frames[index % len(frames)]} Investigating..."
                if self.terminal.color:
                    text = self.terminal.COLORS["muted"] + text + "\033[0m"
                self.terminal.stream.write("\r\033[2K" + text)
                self.terminal.stream.flush()
                self.terminal.activity_visible = True
            index += 1
            self.stop.wait(0.1)

    # Start the animation without blocking model requests or database lookups.
    def __enter__(self):
        if self.terminal.animate:
            self.thread = threading.Thread(target=self._spin, daemon=True)
            self.thread.start()
        return self

    # Clear the waiting line even when an error or keyboard interrupt occurs.
    def __exit__(self, error_type, error, traceback):
        self.stop.set()
        if self.thread:
            self.thread.join()
        with self.terminal.lock:
            if self.terminal.activity_visible:
                self.terminal.stream.write("\r\033[2K")
                self.terminal.stream.flush()
                self.terminal.activity_visible = False
        return False


class TerminalUI:
    COLORS = {
        "accent": "\033[1;38;5;214m",
        "green": "\033[1;38;5;114m",
        "teal": "\033[38;5;80m",
        "muted": "\033[90m",
        "bold": "\033[1m",
        "yellow": "\033[38;5;221m",
    }

    # Keep the same layout without extra libraries and color only real terminals.
    def __init__(self, stream=None, color=True, width=None, animate=True):
        self.stream = stream or sys.stderr
        self.color = (
            color
            and bool(getattr(self.stream, "isatty", lambda: False)())
            and "NO_COLOR" not in os.environ
            and os.environ.get("TERM") != "dumb"
        )
        available = width or shutil.get_terminal_size((100, 24)).columns
        self.width = max(32, min(100, available - 2))
        self.animate = (
            animate
            and bool(getattr(self.stream, "isatty", lambda: False)())
            and os.environ.get("TERM") != "dumb"
        )
        self.lock = threading.Lock()
        self.activity_visible = False

    # Remove terminal controls and leave model and database text as literal content.
    @staticmethod
    def _safe(value):
        text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(value))
        return "".join(character for character in text if character.isprintable())[:400]

    # Write trusted colors around sanitized text without interpreting embedded markup.
    def _parts(self, parts):
        output = []
        for value, style in parts:
            text = self._safe(value)
            if self.color and style in self.COLORS:
                text = self.COLORS[style] + text + "\033[0m"
            output.append(text)
        with self.lock:
            if self.activity_visible:
                self.stream.write("\r\033[2K")
                self.activity_visible = False
            print("".join(output), file=self.stream, flush=True)

    # Provide a waiting indicator that always stops when the investigation ends.
    def waiting(self):
        return TerminalActivity(self)

    # Reveal completed explanation sentences smoothly without delaying research metrics.
    def _reveal(self, value, indent=2):
        text = self._safe(value)
        if not self.animate:
            self._line(text, indent=indent)
            return
        sentences = re.split(r"(?<=[.!?])\s+", text)
        delay = min(0.035, 1.5 / max(1, len(text.split())))
        for index, sentence in enumerate(sentences):
            for line in textwrap.wrap(sentence, width=max(10, self.width - indent)) or [
                ""
            ]:
                with self.lock:
                    self.stream.write(" " * indent)
                    for word in re.findall(r"\S+\s*", line):
                        self.stream.write(word)
                        self.stream.flush()
                        time.sleep(delay)
                    self.stream.write("\n")
                    self.stream.flush()
            if index < len(sentences) - 1:
                time.sleep(0.18)

    # Wrap ordinary paragraphs to fit the terminal instead of producing long lines.
    def _line(self, value="", style=None, indent=0):
        lines = textwrap.wrap(
            self._safe(value), width=max(10, self.width - indent)
        ) or [""]
        for line in lines:
            self._parts([(" " * indent + line, style)])

    # Draw the reference-style header with a highlighted model and muted metadata.
    def show_header(self, ticket_id, model, budget):
        friendly = {
            "openai/gpt-oss-20b": "GPT-OSS 20B via Groq",
            "openai/gpt-oss-120b": "GPT-OSS 120B via Groq",
        }.get(model, model)
        self._line()
        self._parts([("  Model: ", "bold"), (friendly, "accent")])
        self._parts([("  Tool-call budget: ", "muted"), (str(budget), "teal")])
        self._line()
        panel_width = min(self.width, 82)
        inside = panel_width - 4
        rows = [
            (">_ ERP Ticket Agent", "accent"),
            ("", None),
            (f"ticket: {self._safe(ticket_id)}", "muted"),
            (f"model:  {self._safe(friendly)}", "muted"),
        ]
        self._line("╭" + "─" * (panel_width - 2) + "╮", "muted")
        for value, style in rows:
            for line in textwrap.wrap(value, width=inside) or [""]:
                self._parts(
                    [("│ ", "muted"), (line.ljust(inside), style), (" │", "muted")]
                )
        self._line("╰" + "─" * (panel_width - 2) + "╯", "muted")
        self._line()
        self._line(
            "I'll inspect the ERP records linked to this ticket and gather evidence."
        )
        self._line(
            "Read-only investigation · one tool at a time · recommendations only",
            "muted",
        )

    # Render short evidence trees with teal values and consistent continuation lines.
    def _tree(self, items):
        for index, (label, value) in enumerate(items):
            last = index == len(items) - 1
            branch = "└─ " if last else "├─ "
            continuation = "   " if last else "│  "
            label = self._safe(label)
            if len(label) > self.width - 15:
                label = label[: self.width - 19] + "... "
            lines = textwrap.wrap(
                self._safe(value), width=max(1, self.width - len(label) - 7)
            ) or [""]
            self._parts([("  " + branch, "muted"), (label, None), (lines[0], "teal")])
            for line in lines[1:]:
                self._parts(
                    [("  " + continuation + " " * len(label), "muted"), (line, "teal")]
                )

    # Turn a successful tool observation into a few useful facts instead of raw JSON.
    def _record_facts(self, observation):
        for key in ("invoice", "order", "sales_order", "customer", "price_list"):
            row = observation.get(key)
            if isinstance(row, dict):
                fields = [
                    name
                    for name in (
                        "id",
                        "status",
                        "blocked_reason",
                        "block_reason",
                        "on_hold",
                        "tax_id",
                        "credit_limit",
                        "credit_used",
                        "valid_from",
                        "valid_to",
                    )
                    if name in row and (row[name] is not None or name == "tax_id")
                ]
                return [
                    (name.replace("_", " ") + ": ", row[name]) for name in fields[:5]
                ]
        for key in ("matches", "deliveries", "payments", "job_runs"):
            rows = observation.get(key)
            if isinstance(rows, list):
                facts = []
                for row in rows[:3]:
                    if not isinstance(row, dict):
                        continue
                    details = [
                        f"{name.replace('_', ' ')}={row[name]}"
                        for name in (
                            "status",
                            "amount",
                            "applied_to_limit",
                            "shipped_qty",
                            "ordered_qty",
                        )
                        if name in row
                    ]
                    facts.append(
                        (
                            f"{row.get('id', 'record')}: ",
                            " · ".join(details) or row.get("entity", "retrieved"),
                        )
                    )
                if len(rows) > 3:
                    facts.append(("More records: ", str(len(rows) - 3)))
                if observation.get("truncated"):
                    facts.append(
                        ("Search: ", "More matches exist; refine the reference")
                    )
                return facts
        return []

    # Show real tool results with green markers only for successful database reads.
    def show_tool_result(self, record, used, budget):
        observation = record["observation"]
        success = observation.get("status") == "ok"
        cached = " · cached" if record.get("cache_hit") else ""
        self._line()
        if self.width < 60:
            self._line(
                f"● {record['tool_name']}  {used}/{budget} calls{cached}",
                "green" if success else "yellow",
            )
        else:
            self._parts(
                [
                    ("● ", "green" if success else "yellow"),
                    (record["tool_name"], "green" if success else "yellow"),
                    (f"  {used}/{budget} calls{cached}", "muted"),
                ]
            )
        facts = (
            self._record_facts(observation)
            if success
            else [
                ("Result: ", observation.get("status")),
                ("Details: ", observation.get("message", "No matching records")),
            ]
        )
        self._tree(facts)

    # Keep the four-column summary and stack its fields on narrow terminals.
    def _summary_table(self, diagnosis, metrics):
        values = [
            diagnosis["ticket_id"],
            diagnosis["root_cause_code"],
            f"{metrics['tool_calls']}/{metrics['tool_budget']}",
            f"{metrics['elapsed_seconds']:.2f}s",
        ]
        headers = ["Ticket", "Cause (provisional)", "Calls", "Time"]
        if self.width < 78:
            self._tree(
                [(header + ": ", value) for header, value in zip(headers, values)]
            )
            return
        widths = [14, self.width - 38, 8, 8]
        self._parts(
            [
                (self._safe(header).ljust(width) + "  ", "bold")
                for header, width in zip(headers, widths)
            ]
        )
        cells = [
            textwrap.wrap(self._safe(value), width=width) or [""]
            for value, width in zip(values, widths)
        ]
        for index in range(max(map(len, cells))):
            self._parts(
                [
                    (
                        (cell[index] if index < len(cell) else "").ljust(width) + "  ",
                        "accent" if column == 1 else None,
                    )
                    for column, (cell, width) in enumerate(zip(cells, widths))
                ]
            )

    # Present provisional findings, evidence claims, recommendations, and measured usage.
    def show_summary(self, result):
        diagnosis, metrics = result.diagnosis, result.metrics
        self._line()
        self._line("● Provisional diagnosis · human review required", "yellow")
        self._line()
        self._summary_table(diagnosis, metrics)
        self._line()
        self._line("Evidence", "bold", indent=2)
        evidence = diagnosis.get("evidence", [])
        if evidence:
            self._tree(
                [
                    (
                        f"{item['entity']} {item['id']}: ",
                        f"{item['field']} = {item['observed']}",
                    )
                    for item in evidence[:5]
                ]
            )
            if len(evidence) > 5:
                self._line(
                    f"{len(evidence) - 5} additional citations in the JSON result.",
                    "muted",
                    indent=2,
                )
        else:
            self._line(
                "No supporting records cited in the final answer.", "muted", indent=2
            )
        self._line(
            "Not verified · record and value checks are not implemented yet.",
            "yellow",
            indent=2,
        )
        self._line()
        self._reveal(diagnosis.get("root_cause_explanation", ""), indent=2)
        self._line()
        self._line("Recommended action", "bold", indent=2)
        self._tree(
            [
                ("Action: ", diagnosis["proposed_fix"]["action"]),
                ("Target: ", diagnosis["proposed_fix"]["target"]),
                ("Escalation: ", "Yes" if diagnosis["escalate"] else "No"),
            ]
        )
        self._line("No database changes were made.", "muted", indent=2)
        self._line()
        usage = metrics["reported_tokens"]
        qualifier = "reported" if metrics["usage_complete"] else "partially reported"
        if self.width < 78:
            self._line(
                f"{usage['input_tokens']} input · {usage['output_tokens']} output · {sum(usage.values())} {qualifier} tokens",
                "teal",
                indent=2,
            )
            self._line(
                f"Completed in {metrics['elapsed_seconds']:.2f}s · {metrics['model_calls']} model calls · {metrics['retries']} retries",
                "muted",
                indent=2,
            )
            self._line(
                f"Stop reason: {metrics['termination_reason']}", "muted", indent=2
            )
            self._line()
            return
        self._parts(
            [
                ("  ", None),
                (str(usage["input_tokens"]), "teal"),
                (" input · ", "muted"),
                (str(usage["output_tokens"]), "teal"),
                (" output · ", "muted"),
                (str(sum(usage.values())), "teal"),
                (f" {qualifier} tokens", "muted"),
            ]
        )
        self._parts(
            [
                ("  Completed in ", "muted"),
                (f"{metrics['elapsed_seconds']:.2f}s", "teal"),
                (
                    f" · {metrics['model_calls']} model calls · {metrics['retries']} retries",
                    "muted",
                ),
            ]
        )
        self._line(f"Stop reason: {metrics['termination_reason']}", "muted", indent=2)
        self._line()
