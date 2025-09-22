"""Tkinter-based graphical interface for the HAIRF framework."""

from __future__ import annotations

import os
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


_API_ENV_VARS: list[tuple[str, str]] = [
    ("OpenAI", "OPENAI_API_KEY"),
    ("Gemini", "GOOGLE_API_KEY"),
    ("DeepSeek", "DEEPSEEK_API_KEY"),
    ("Qwen", "QWEN_API_KEY"),
]


class HAIRFGui(ttk.Frame):
    """Main application frame used by :func:`run_gui`."""

    def __init__(self, master: tk.Tk):
        super().__init__(master, padding=16)
        self.master = master
        self.master.title("Open Think Pro – HAIRF Interface")

        self.model_var = tk.StringVar(value=next(iter(_MODEL_OPTIONS)))
        self.query_var = tk.StringVar()
        self._settings_window: tk.Toplevel | None = None
        self._settings_vars: dict[str, tk.StringVar] = {}

        self._create_widgets()

    def _create_widgets(self) -> None:
        self.grid(sticky=tk.NSEW)
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # Menu bar
        self._create_menu()

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

    def _create_menu(self) -> None:
        menubar = tk.Menu(self.master)
        settings_menu = tk.Menu(menubar, tearoff=False)
        settings_menu.add_command(label="API Keys…", command=self._open_settings_dialog)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        self.master.config(menu=menubar)

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

    def _open_settings_dialog(self) -> None:
        if self._settings_window and self._settings_window.winfo_exists():
            self._settings_window.focus_set()
            return

        self._settings_window = tk.Toplevel(self)
        self._settings_window.title("Configure API Keys")
        self._settings_window.transient(self.master)
        self._settings_window.resizable(False, False)
        self._settings_window.grab_set()
        self._settings_window.protocol("WM_DELETE_WINDOW", self._close_settings_dialog)

        container = ttk.Frame(self._settings_window, padding=16)
        container.grid(sticky=tk.NSEW)
        container.columnconfigure(1, weight=1)

        self._settings_vars = {}
        first_entry: ttk.Entry | None = None
        for row, (provider_label, env_var) in enumerate(_API_ENV_VARS):
            ttk.Label(container, text=f"{provider_label} API key:").grid(
                row=row, column=0, sticky=tk.W, padx=(0, 8), pady=(0, 8)
            )
            var = tk.StringVar(value=os.getenv(env_var, ""))
            entry = ttk.Entry(container, textvariable=var, width=40)
            entry.grid(row=row, column=1, sticky=tk.EW, pady=(0, 8))
            self._settings_vars[env_var] = var
            if first_entry is None:
                first_entry = entry

        if first_entry is not None:
            first_entry.focus_set()

        ttk.Label(
            container,
            text="Leave a field blank to clear the stored key for that provider.",
        ).grid(row=len(_API_ENV_VARS), column=0, columnspan=2, sticky=tk.W, pady=(0, 12))

        button_row = ttk.Frame(container)
        button_row.grid(row=len(_API_ENV_VARS) + 1, column=0, columnspan=2, sticky=tk.E)

        ttk.Button(button_row, text="Cancel", command=self._close_settings_dialog).grid(
            row=0, column=0, padx=(0, 8)
        )
        ttk.Button(button_row, text="Save", command=self._save_settings).grid(row=0, column=1)

    def _close_settings_dialog(self) -> None:
        if self._settings_window is None:
            return
        try:
            self._settings_window.grab_release()
        except tk.TclError:
            pass
        self._settings_window.destroy()
        self._settings_window = None
        self._settings_vars = {}

    def _save_settings(self) -> None:
        updated_labels: list[str] = []
        cleared_labels: list[str] = []
        for provider_label, env_var in _API_ENV_VARS:
            var = self._settings_vars.get(env_var)
            value = var.get().strip() if var is not None else ""
            if value:
                os.environ[env_var] = value
                updated_labels.append(provider_label)
            else:
                os.environ.pop(env_var, None)
                cleared_labels.append(provider_label)

        message_parts: list[str] = []
        if updated_labels:
            message_parts.append(
                "Updated keys for: " + ", ".join(updated_labels)
            )
        if cleared_labels:
            message_parts.append(
                "Cleared keys for: " + ", ".join(cleared_labels)
            )
        if not message_parts:
            message_parts.append("No changes were made to API keys.")

        messagebox.showinfo("API keys updated", "\n".join(message_parts))
        self._close_settings_dialog()

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
