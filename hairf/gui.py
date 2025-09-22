"""Tkinter-based graphical interface for the HAIRF framework."""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from typing import Any

from .api import answer_question


_MODEL_OPTIONS: dict[str, Any] = {
    "GPT-5": "gpt-5",
    "GPT-5 Pro (Open Think Pro)": {
        "model": "gpt-5-pro",
        "provider": "open-think-pro",
    },
    "Gemini 2.5 Pro": "gemini-2.5-pro",
    "Gemini Deep Think (Open Think Pro)": {
        "model": "gemini-deep-think",
        "provider": "open-think-pro",
    },
}


class HAIRFGui(ttk.Frame):
    """Main application frame used by :func:`run_gui`."""

    def __init__(self, master: tk.Tk):
        super().__init__(master, padding=16)
        self.master = master
        self.master.title("Open Think Pro – HAIRF Interface")

        self.model_var = tk.StringVar(value=next(iter(_MODEL_OPTIONS)))
        self.query_var = tk.StringVar()

        self._create_widgets()

    def _create_widgets(self) -> None:
        self.grid(sticky=tk.NSEW)
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # Model selector
        selector_frame = ttk.Frame(self)
        selector_frame.grid(row=0, column=0, sticky=tk.EW, pady=(0, 12))
        selector_frame.columnconfigure(1, weight=1)

        ttk.Label(selector_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        self.model_dropdown = ttk.Combobox(
            selector_frame,
            textvariable=self.model_var,
            values=list(_MODEL_OPTIONS.keys()),
            state="readonly",
        )
        self.model_dropdown.grid(row=0, column=1, sticky=tk.EW)

        # Output display
        ttk.Label(self, text="Response:").grid(row=1, column=0, sticky=tk.W)
        self.output_text = tk.Text(self, wrap=tk.WORD, height=12, state=tk.DISABLED)
        self.output_text.grid(row=2, column=0, sticky=tk.NSEW)

        # Query input
        ttk.Label(self, text="Enter your question:").grid(row=3, column=0, sticky=tk.W, pady=(12, 0))
        self.query_entry = ttk.Entry(self, textvariable=self.query_var)
        self.query_entry.grid(row=4, column=0, sticky=tk.EW)
        self.query_entry.bind("<Return>", self._on_submit_event)

        # Submit button
        self.submit_button = ttk.Button(self, text="Ask", command=self._on_submit)
        self.submit_button.grid(row=5, column=0, sticky=tk.E, pady=(12, 0))

        # Configure resizing
        self.rowconfigure(2, weight=1)

    def _on_submit_event(self, event: tk.Event | None = None) -> None:
        self._on_submit()

    def _on_submit(self) -> None:
        question = self.query_var.get().strip()
        if not question:
            messagebox.showerror("Invalid input", "Please enter a question before submitting.")
            return

        model_label = self.model_var.get()
        llm_config = _MODEL_OPTIONS.get(model_label, None)

        self._set_output("Processing your request...")
        self.submit_button.config(state=tk.DISABLED)

        thread = threading.Thread(
            target=self._process_question,
            args=(question, llm_config),
            daemon=True,
        )
        thread.start()

    def _process_question(self, question: str, llm_config: Any | None) -> None:
        try:
            result = answer_question(question, llm=llm_config)
        except Exception as exc:  # noqa: BLE001
            self.after(0, self._handle_error, exc)
            return

        self.after(0, self._display_result, result.answer)

    def _display_result(self, answer: str) -> None:
        self._set_output(answer)
        self.submit_button.config(state=tk.NORMAL)

    def _handle_error(self, exc: Exception) -> None:
        self.submit_button.config(state=tk.NORMAL)
        messagebox.showerror("Processing error", str(exc))

    def _set_output(self, message: str) -> None:
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, message)
        self.output_text.config(state=tk.DISABLED)


def run_gui() -> None:
    """Launch the Tkinter interface for the HAIRF framework."""

    root = tk.Tk()
    HAIRFGui(root)
    root.mainloop()


__all__ = ["run_gui", "HAIRFGui"]
