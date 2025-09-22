import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Callable, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

START_MODEL_PATH = "./finbert-finetuned"
START_LABEL_MAP = {
    0: "News",
    1: "Earnings Call",
    2: "Central Bank Speech",
}

def lazy_import_transformers():
    from transformers import pipeline
    import torch
    return pipeline, torch

def extract_text_from_pdf(path: Path) -> str:
    from pypdf import PdfReader
    text_parts = []
    reader = PdfReader(str(path))
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            text_parts.append(t)
    return "\n\n".join(text_parts).strip()

class StartClassifierWorker(QtCore.QThread):
    done = QtCore.Signal(str, float, str)
    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.text = text
    def _humanize(self, raw_label: str) -> str:
        if raw_label.startswith("LABEL_"):
            try:
                lid = int(raw_label.split("_")[-1])
                return START_LABEL_MAP.get(lid, f"Class {lid}")
            except Exception:
                pass
        low = raw_label.strip().lower()
        if "earn" in low:
            return "Earnings Call"
        if "central" in low or "bank" in low or "speech" in low:
            return "Central Bank Speech"
        if "news" in low:
            return "News"
        return raw_label
    def run(self):
        try:
            if not self.text.strip():
                self.done.emit("", 0.0, "No text to classify.")
                return
            mp = START_MODEL_PATH
            if not Path(mp).exists():
                self.done.emit("", 0.0, f"Model path not found: {mp}")
                return
            pipeline, torch = lazy_import_transformers()
            device = 0 if torch.cuda.is_available() else -1
            clf = pipeline("text-classification", model=mp, tokenizer=mp, device=device)
            out = clf(self.text, truncation=True, max_length=512)[0]
            raw_label = str(out.get("label", ""))
            score = float(out.get("score", 0.0))
            pretty = self._humanize(raw_label)
            self.done.emit(pretty, score, "")
        except Exception as e:
            self.done.emit("", 0.0, f"{type(e).__name__}: {e}")

class SummarizerWorker(QtCore.QThread):
    done = QtCore.Signal(str)
    failed = QtCore.Signal(str)

    def __init__(self, class_name: str, text: str, parent=None):
        super().__init__(parent)
        self.class_name = (class_name or "").strip()
        self.text = text

    def _resolve_func(self, module, names):
        for n in names:
            fn = getattr(module, n, None)
            if callable(fn):
                return fn
        return None

    def run(self):
        try:
            if not self.text.strip():
                self.failed.emit("No text to summarize.")
                return

            cls = self.class_name.lower()
            if "earn" in cls or "company" in cls:
                mod_path = "summarizers.runnerPEGASUS"
                fn_names = ["summarize_earnings", "summarize_pegasus", "summarize_text", "summarize"]
            elif "central" in cls or "bank" in cls or "speech" in cls:
                mod_path = "summarizers.runnerT5"
                fn_names = ["summarize_cb", "summarize_t5", "summarize_text", "summarize"]
            else:
                mod_path = "summarizers.runnerBART"
                fn_names = ["summarize_news", "summarize_bart", "summarize_text", "summarize"]

            module = __import__(mod_path, fromlist=["*"])
            fn = self._resolve_func(module, fn_names)
            if fn is None:
                self.failed.emit(f"No summarizer function found in {mod_path}. Tried: {', '.join(fn_names)}")
                return

            summary = fn(self.text)
            self.done.emit(summary if isinstance(summary, str) else str(summary))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")

class SentimentWorker(QtCore.QThread):
    done = QtCore.Signal(str, float)
    failed = QtCore.Signal(str)

    def __init__(self, class_name: str, text: str, module_override: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.class_name = (class_name or "").strip()
        self.text = text
        self.module_override = module_override

    def _resolve_func(self, module) -> Optional[Callable[[str], Tuple[str, float]]]:
        fn = getattr(module, "predict", None)
        return fn if callable(fn) else None

    def run(self):
        try:
            if not self.text.strip():
                self.failed.emit("No text to analyze.")
                return

            if self.module_override:
                mod_path = self.module_override
            else:
                cls = self.class_name.lower()
                if "earn" in cls or "company" in cls:
                    mod_path = "sentimentClassifiers.runnerCompany"
                elif "central" in cls or "bank" in cls or "speech" in cls:
                    mod_path = "sentimentClassifiers.runnerBank"
                else:
                    mod_path = "sentimentClassifiers.runnerNews"

            module = __import__(mod_path, fromlist=["*"])
            fn = self._resolve_func(module)
            if fn is None:
                self.failed.emit(f"No predict() in {mod_path}")
                return

            label, conf = fn(self.text)
            pretty = label.strip().capitalize() if label else label
            self.done.emit(pretty, float(conf))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Financial Text Analyzer")
        self.resize(1500, 950)

        v_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, self)
        v_split.setHandleWidth(6)

        top_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, v_split)
        top_split.setHandleWidth(6)

        bottom_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, v_split)
        bottom_split.setHandleWidth(6)

        v_split.addWidget(top_split)
        v_split.addWidget(bottom_split)
        v_split.setStretchFactor(0, 2)
        v_split.setStretchFactor(1, 1)

        self.upload_panel = QtWidgets.QWidget()
        tl = QtWidgets.QVBoxLayout(self.upload_panel)
        tl.setContentsMargins(14, 14, 14, 14)
        tl.setSpacing(10)

        title = QtWidgets.QLabel("Upload TXT or PDF")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        tl.addWidget(title)

        self.file_path_edit = QtWidgets.QLineEdit()
        self.file_path_edit.setPlaceholderText("No file selected")
        self.file_path_edit.setReadOnly(True)
        tl.addWidget(self.file_path_edit)

        row = QtWidgets.QHBoxLayout()
        self.btn_browse = QtWidgets.QPushButton("Browse…")
        self.btn_browse.clicked.connect(self.on_browse)
        row.addWidget(self.btn_browse)

        self.btn_clear = QtWidgets.QPushButton("Clear")
        self.btn_clear.clicked.connect(self.clear_all)
        row.addWidget(self.btn_clear)
        row.addStretch(1)
        tl.addLayout(row)

        hint = QtWidgets.QLabel("Tip: You can also drag & drop a .txt or .pdf file here.")
        hint.setStyleSheet("color: #666;")
        tl.addWidget(hint)
        tl.addStretch(1)

        self.upload_panel.setAcceptDrops(True)
        self.upload_panel.dragEnterEvent = self._drag_enter
        self.upload_panel.dropEvent = self._drop

        self.preview_summary = QtWidgets.QWidget()
        pr_layout = QtWidgets.QVBoxLayout(self.preview_summary)
        pr_layout.setContentsMargins(14, 14, 14, 14)
        pr_layout.setSpacing(10)

        self.preview_box = QtWidgets.QGroupBox("Extracted Text Preview")
        pv_box_layout = QtWidgets.QVBoxLayout(self.preview_box)
        self.text_preview = QtWidgets.QPlainTextEdit()
        self.text_preview.setReadOnly(True)
        pv_box_layout.addWidget(self.text_preview)
        pr_layout.addWidget(self.preview_box, stretch=3)

        self.summary_box = QtWidgets.QGroupBox("Summary")
        sm_box_layout = QtWidgets.QVBoxLayout(self.summary_box)
        self.summary_view = QtWidgets.QPlainTextEdit()
        self.summary_view.setReadOnly(True)
        sm_box_layout.addWidget(self.summary_view)
        pr_layout.addWidget(self.summary_box, stretch=2)

        top_split.addWidget(self.upload_panel)
        top_split.addWidget(self.preview_summary)
        top_split.setStretchFactor(0, 1)
        top_split.setStretchFactor(1, 2)

        bottom_left_stack = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        bottom_left_stack.setHandleWidth(6)

        self.start_panel = QtWidgets.QGroupBox("Start Classifier (Routing)")
        bl = QtWidgets.QGridLayout(self.start_panel)
        bl.setContentsMargins(12, 12, 12, 12)
        bl.setHorizontalSpacing(8)
        bl.setVerticalSpacing(10)

        info = QtWidgets.QLabel(f"Model: {START_MODEL_PATH}")
        info.setStyleSheet("color: #666;")
        bl.addWidget(info, 0, 0, 1, 2)

        self.btn_run_start = QtWidgets.QPushButton("Run on extracted text")
        self.btn_run_start.clicked.connect(self.on_browse if False else self.on_run_start_classifier)
        bl.addWidget(self.btn_run_start, 1, 0, 1, 2)

        self.lbl_class = QtWidgets.QLabel("Class: —")
        self.lbl_conf  = QtWidgets.QLabel("Confidence: —")
        self.lbl_status = QtWidgets.QLabel("")
        self.lbl_status.setStyleSheet("color: #666;")
        bl.addWidget(self.lbl_class, 2, 0, 1, 2)
        bl.addWidget(self.lbl_conf, 3, 0, 1, 2)
        bl.addWidget(self.lbl_status, 4, 0, 1, 2)

        self.sent_panel = QtWidgets.QGroupBox("Sentiment")
        sl = QtWidgets.QGridLayout(self.sent_panel)
        sl.setContentsMargins(12, 12, 12, 12)
        sl.setHorizontalSpacing(8)
        sl.setVerticalSpacing(10)
        self.lbl_sent_label = QtWidgets.QLabel("Label: —")
        self.lbl_sent_conf  = QtWidgets.QLabel("Confidence: —")
        self.lbl_sent_status = QtWidgets.QLabel("")
        self.lbl_sent_status.setStyleSheet("color: #666;")
        sl.addWidget(self.lbl_sent_label, 0, 0, 1, 2)
        sl.addWidget(self.lbl_sent_conf,  1, 0, 1, 2)
        sl.addWidget(self.lbl_sent_status,2, 0, 1, 2)

        self.cb_custom_panel = QtWidgets.QGroupBox("CB Custom Sentiment (HAN)")
        cl = QtWidgets.QGridLayout(self.cb_custom_panel)
        cl.setContentsMargins(12, 12, 12, 12)
        cl.setHorizontalSpacing(8)
        cl.setVerticalSpacing(10)
        self.lbl_cb_label = QtWidgets.QLabel("Label: —")
        self.lbl_cb_conf  = QtWidgets.QLabel("Confidence: —")
        self.lbl_cb_status = QtWidgets.QLabel("")
        self.lbl_cb_status.setStyleSheet("color: #666;")
        cl.addWidget(self.lbl_cb_label,  0, 0, 1, 2)
        cl.addWidget(self.lbl_cb_conf,   1, 0, 1, 2)
        cl.addWidget(self.lbl_cb_status, 2, 0, 1, 2)
        self.cb_custom_panel.setVisible(False)

        bottom_left_stack.addWidget(self.start_panel)
        bottom_left_stack.addWidget(self.sent_panel)
        bottom_left_stack.addWidget(self.cb_custom_panel)
        bottom_left_stack.setStretchFactor(0, 1)
        bottom_left_stack.setStretchFactor(1, 1)
        bottom_left_stack.setStretchFactor(2, 1)

        self.log_panel = QtWidgets.QGroupBox("Logs")
        br = QtWidgets.QVBoxLayout(self.log_panel)
        br.setContentsMargins(12, 12, 12, 12)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        br.addWidget(self.log)

        bottom_split.addWidget(bottom_left_stack)
        bottom_split.addWidget(self.log_panel)
        bottom_split.setStretchFactor(0, 1)
        bottom_split.setStretchFactor(1, 1)

        self.setCentralWidget(v_split)

        self.current_file: Optional[Path] = None
        self.current_text: str = ""
        self.last_class_name: str = ""

        self.worker = None
        self.sum_worker = None
        self.sent_worker = None
        self.cb_worker = None

    def on_browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select TXT or PDF", str(Path.cwd()),
            "Documents (*.txt *.pdf);;All files (*.*)"
        )
        if path:
            self.load_file(Path(path))

    def _drag_enter(self, e: QtGui.QDragEnterEvent):
        if e.mimeData().hasUrls():
            for u in e.mimeData().urls():
                p = Path(u.toLocalFile())
                if p.suffix.lower() in {".txt", ".pdf"}:
                    e.acceptProposedAction()
                    return
        e.ignore()

    def _drop(self, e: QtGui.QDropEvent):
        for u in e.mimeData().urls():
            p = Path(u.toLocalFile())
            if p.suffix.lower() in {".txt", ".pdf"}:
                self.load_file(p)
                break

    def load_file(self, path: Path):
        try:
            if not path.exists():
                self.append_log(f"[ERROR] File not found: {path}")
                return
            self.current_file = path
            self.file_path_edit.setText(str(path))
            if path.suffix.lower() == ".txt":
                text = path.read_text(encoding="utf-8", errors="ignore")
            elif path.suffix.lower() == ".pdf":
                text = extract_text_from_pdf(path)
            else:
                self.append_log(f"[ERROR] Unsupported file type: {path.suffix}")
                return

            text = (text or "").strip()
            self.current_text = text
            self.text_preview.setPlainText(text)
            self.summary_view.clear()

            self.lbl_class.setText("Class: —")
            self.lbl_conf.setText("Confidence: —")
            self.lbl_status.setText("")
            self.sent_panel.setTitle("Sentiment")
            self.lbl_sent_label.setText("Label: —")
            self.lbl_sent_conf.setText("Confidence: —")
            self.lbl_sent_status.setText("")
            self.cb_custom_panel.setVisible(False)
            self.lbl_cb_label.setText("Label: —")
            self.lbl_cb_conf.setText("Confidence: —")
            self.lbl_cb_status.setText("")

            self.append_log(f"[OK] Loaded {path.name} ({len(text)} chars).")
        except Exception as e:
            self.append_log(f"[ERROR] {type(e).__name__}: {e}")

    def clear_all(self):
        self.current_file = None
        self.current_text = ""
        self.file_path_edit.clear()
        self.text_preview.clear()
        self.summary_view.clear()
        self.lbl_class.setText("Class: —")
        self.lbl_conf.setText("Confidence: —")
        self.lbl_status.setText("")
        self.sent_panel.setTitle("Sentiment")
        self.lbl_sent_label.setText("Label: —")
        self.lbl_sent_conf.setText("Confidence: —")
        self.lbl_sent_status.setText("")
        self.cb_custom_panel.setVisible(False)
        self.lbl_cb_label.setText("Label: —")
        self.lbl_cb_conf.setText("Confidence: —")
        self.lbl_cb_status.setText("")
        self.append_log("[INFO] Cleared.")

    def on_run_start_classifier(self):
        if not self.current_text.strip():
            self.append_log("[WARN] No extracted text. Load a file first.")
            self.lbl_status.setText("Load a file with text first.")
            return

        self.btn_run_start.setEnabled(False)
        self.lbl_status.setText("Running classifier…")
        self.summary_view.setPlainText("Summarizer will run automatically after classification.")
        self.append_log(f"[RUN] Start Classifier using model: {START_MODEL_PATH}")

        self.worker = StartClassifierWorker(self.current_text, self)
        self.worker.done.connect(self._on_start_done, QtCore.Qt.ConnectionType.UniqueConnection)
        self.worker.start()

    def _on_start_done(self, class_name: str, conf: float, error: str):
        if self.sender() is not getattr(self, "worker", None):
            return
        try:
            self.worker.done.disconnect(self._on_start_done)
        except Exception:
            pass
        self.worker = None

        self.btn_run_start.setEnabled(True)
        if error:
            self.lbl_status.setText(error)
            self.append_log(f"[ERROR] {error}")
            return

        self.last_class_name = class_name or "News"
        self.lbl_class.setText(f"Class: {self.last_class_name}")
        self.lbl_conf.setText(f"Confidence: {conf:.4f}")
        self.lbl_status.setText("Classification complete. Starting summarizer & sentiment…")
        self.append_log(f"[OK] Predicted: {self.last_class_name}  |  confidence={conf:.4f}")

        self.summary_view.setPlainText(f"[RUN] Summarizing ({self.last_class_name}) …")
        self.sum_worker = SummarizerWorker(self.last_class_name, self.current_text, self)
        self.sum_worker.done.connect(self._on_summary_done, QtCore.Qt.ConnectionType.UniqueConnection)
        self.sum_worker.failed.connect(self._on_summary_failed, QtCore.Qt.ConnectionType.UniqueConnection)
        self.sum_worker.start()

        cls = (self.last_class_name or "").lower()
        if "central" in cls or "bank" in cls or "speech" in cls:
            self.sent_panel.setTitle("Sentiment (DeBERTa)")
            self.lbl_sent_status.setText("Running sentiment (DeBERTa)…")
            self.sent_worker = SentimentWorker(self.last_class_name, self.current_text, None, self)
            self.sent_worker.done.connect(self._on_sentiment_done, QtCore.Qt.ConnectionType.UniqueConnection)
            self.sent_worker.failed.connect(self._on_sentiment_failed, QtCore.Qt.ConnectionType.UniqueConnection)
            self.sent_worker.start()

            self.cb_custom_panel.setVisible(True)
            self.lbl_cb_status.setText("Running custom CB sentiment (HAN)…")
            self.cb_worker = SentimentWorker(
                self.last_class_name, self.current_text,
                module_override="sentimentClassifiers.runnerCustom",
                parent=self
            )
            self.cb_worker.done.connect(self._on_cb_custom_done, QtCore.Qt.ConnectionType.UniqueConnection)
            self.cb_worker.failed.connect(self._on_cb_custom_failed, QtCore.Qt.ConnectionType.UniqueConnection)
            self.cb_worker.start()
        else:
            self.sent_panel.setTitle("Sentiment")
            self.lbl_sent_status.setText("Running sentiment…")
            self.cb_custom_panel.setVisible(False)
            self.sent_worker = SentimentWorker(self.last_class_name, self.current_text, None, self)
            self.sent_worker.done.connect(self._on_sentiment_done, QtCore.Qt.ConnectionType.UniqueConnection)
            self.sent_worker.failed.connect(self._on_sentiment_failed, QtCore.Qt.ConnectionType.UniqueConnection)
            self.sent_worker.start()

    @QtCore.Slot(str)
    def _on_summary_done(self, summary: str):
        if self.sender() is not getattr(self, "sum_worker", None):
            return
        if not summary.strip():
            summary = "[No summary produced.]"
        self.summary_view.setPlainText(summary)
        self.lbl_status.setText("Done.")
        self.append_log("[OK] Summarization complete.")
        try:
            self.sum_worker.done.disconnect(self._on_summary_done)
            self.sum_worker.failed.disconnect(self._on_summary_failed)
        except Exception:
            pass
        self.sum_worker = None

    @QtCore.Slot(str)
    def _on_summary_failed(self, err: str):
        if self.sender() is not getattr(self, "sum_worker", None):
            return
        self.summary_view.setPlainText("[Failed to summarize]")
        self.lbl_status.setText("Summarizer error.")
        self.append_log(f"[ERROR] Summarizer failed: {err}")
        try:
            self.sum_worker.done.disconnect(self._on_summary_done)
            self.sum_worker.failed.disconnect(self._on_summary_failed)
        except Exception:
            pass
        self.sum_worker = None

    @QtCore.Slot(str, float)
    def _on_sentiment_done(self, label: str, conf: float):
        if self.sender() is not getattr(self, "sent_worker", None):
            return
        self.lbl_sent_label.setText(f"Label: {label or '—'}")
        self.lbl_sent_conf.setText(f"Confidence: {conf:.4f}" if label else "Confidence: —")
        self.lbl_sent_status.setText("Done.")
        self.append_log(f"[OK] Sentiment: {label}  |  confidence={conf:.4f}")
        try:
            self.sent_worker.done.disconnect(self._on_sentiment_done)
            self.sent_worker.failed.disconnect(self._on_sentiment_failed)
        except Exception:
            pass
        self.sent_worker = None

    @QtCore.Slot(str)
    def _on_sentiment_failed(self, err: str):
        if self.sender() is not getattr(self, "sent_worker", None):
            return
        self.lbl_sent_label.setText("Label: —")
        self.lbl_sent_conf.setText("Confidence: —")
        self.lbl_sent_status.setText("Error.")
        self.append_log(f"[ERROR] Sentiment failed: {err}")
        try:
            self.sent_worker.done.disconnect(self._on_sentiment_done)
            self.sent_worker.failed.disconnect(self._on_sentiment_failed)
        except Exception:
            pass
        self.sent_worker = None

    @QtCore.Slot(str, float)
    def _on_cb_custom_done(self, label: str, conf: float):
        if self.sender() is not getattr(self, "cb_worker", None):
            return
        self.lbl_cb_label.setText(f"Label: {label or '—'}")
        self.lbl_cb_conf.setText(f"Confidence: {conf:.4f}" if label else "Confidence: —")
        self.lbl_cb_status.setText("Done.")
        self.append_log(f"[OK] CB Custom Sentiment (HAN): {label}  |  confidence={conf:.4f}")
        try:
            self.cb_worker.done.disconnect(self._on_cb_custom_done)
            self.cb_worker.failed.disconnect(self._on_cb_custom_failed)
        except Exception:
            pass
        self.cb_worker = None

    @QtCore.Slot(str)
    def _on_cb_custom_failed(self, err: str):
        if self.sender() is not getattr(self, "cb_worker", None):
            return
        self.lbl_cb_label.setText("Label: —")
        self.lbl_cb_conf.setText("Confidence: —")
        self.lbl_cb_status.setText("Error.")
        self.append_log(f"[ERROR] CB Custom Sentiment failed: {err}")
        try:
            self.cb_worker.done.disconnect(self._on_cb_custom_done)
            self.cb_worker.failed.disconnect(self._on_cb_custom_failed)
        except Exception:
            pass
        self.cb_worker = None

    def append_log(self, msg: str):
        self.log.appendPlainText(msg)
        cursor = self.log.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        self.log.setTextCursor(cursor)

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Financial Text Analyzer")
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
