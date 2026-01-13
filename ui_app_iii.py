import os
import copy

from PySide6.QtCore import QObject, QThread, Signal

os.environ.setdefault("OLLAMA_HOST", "http://localhost:11434")
os.environ.setdefault("LLM_MODEL", "gpt-oss:20b")
os.environ.setdefault(
    "HF_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

import sys
import time
import csv
from typing import List
import math
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

np.random.seed(42)

# PyQt6 imports

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStackedWidget, QSlider,
    QLabel, QPushButton, QFileDialog, QMessageBox, QTextEdit,
    QListWidget, QTableWidget, QTableWidgetItem, QLineEdit,
    QComboBox, QCheckBox, QScrollArea, QVBoxLayout, QHeaderView,
    QHBoxLayout, QGridLayout, QDialog, QFrame, QSizePolicy, QGroupBox,
    QStyleOptionButton, QStyle, QStyleOptionHeader
)
from PySide6.QtWidgets import QProgressDialog
from PySide6.QtCore import Qt, QRect, Signal
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtWidgets import QProgressBar

# Try to import PIL for logo support
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

#Helpers
from helpers import (infer_jd_experience_range, infer_experience, two_phase_rank, extract_skills_from_taxonomy )

#Matching
class MatchingWorker(QObject): 
    finished = Signal()        
    error = Signal(str)

    def __init__(self, controller):
        super().__init__()
        self.controller = controller

    def run(self):
        try:
            self.controller.run_matching()
            self.finished.emit()  
        except Exception as e:
            self.error.emit(str(e))

#Loading the model when the start button is clicked
class ModelInitWorker(QObject):
    finished = Signal(object)  
    error = Signal(str)

    def run(self):
        try:
            import stratos_i as model_mod
            embedder, ok, err = model_mod.try_load_embedder()
            if not ok:
                raise RuntimeError(err)
            self.finished.emit(embedder)
        except Exception as e:
            self.error.emit(str(e))

class EmbeddingWorker(QObject):
    progress = Signal(int, float)
    finished = Signal(int, object)   
    error = Signal(str)

    def __init__(self, embedder, texts, job_id, batch_size=32):
        super().__init__()
        self.embedder = embedder
        self.texts = texts
        self.job_id = job_id
        self.batch_size = batch_size

    def run(self):
        try:
            import stratos_i as model_mod
            total = len(self.texts)
            all_embs = []
            start_time = time.time()

            for i in range(0, total, self.batch_size):

                batch = self.texts[i:i + self.batch_size]
                emb = model_mod.embed_texts_cached(self.embedder, batch)
                all_embs.append(emb)

                done = min(i + len(batch), total)
                elapsed = time.time() - start_time
                rate = done / max(elapsed, 1e-6)
                eta = (total - done) / max(rate, 1e-6)

                self.progress.emit(int(done / total * 100), eta)

            self.finished.emit(self.job_id, np.vstack(all_embs))


        except Exception as e:
            self.error.emit(str(e))

#Checkbox Toggle
class CheckBoxHeader(QHeaderView):
    toggled = Signal(bool)

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self._checked = False
        self.setSectionsClickable(True)

    def paintSection(self, painter, rect, logicalIndex):
        if logicalIndex != 0:
            super().paintSection(painter, rect, logicalIndex)
            return

        painter.save()

        opt = QStyleOptionHeader()
        self.initStyleOption(opt)
        opt.rect = rect
        opt.section = logicalIndex
        self.style().drawControl(QStyle.CE_Header, opt, painter)

        margin = 6

        # Checkbox
        cb_opt = QStyleOptionButton()
        cb_opt.state = QStyle.State_Enabled | (
            QStyle.State_On if self._checked else QStyle.State_Off
        )

        checkbox_size = self.style().pixelMetric(
            QStyle.PM_IndicatorWidth, cb_opt, self
        )

        cb_rect = QRect(
            rect.right() - checkbox_size - margin,
            rect.center().y() - checkbox_size // 2,
            checkbox_size,
            checkbox_size,
        )
        cb_opt.rect = cb_rect

        self.style().drawControl(QStyle.CE_CheckBox, cb_opt, painter)

        # Header text 
        text_rect = QRect(
            rect.left() + margin,
            rect.top(),
            rect.width() - checkbox_size - margin * 2,
            rect.height(),
        )

        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            self.model().headerData(
                logicalIndex, Qt.Orientation.Horizontal
            ),
        )

        painter.restore()

    def mousePressEvent(self, event):
        idx = self.logicalIndexAt(event.pos())
        if idx == 0:
            self._checked = not self._checked
            self.toggled.emit(self._checked)
            self.viewport().update()
        super().mousePressEvent(event)

    def setChecked(self, state: bool):
        self._checked = state
        self.viewport().update()

#Decision
def decide_by_skill_and_experience(
    skill_coverage,
    years_experience,
    jd_min_exp
):
    # Base decision from skill match
    if skill_coverage >= 0.70:
        decision = "Consider"
    elif skill_coverage >= 0.40:
        decision = "Maybe"
    else:
        return "Reject"

    # Experience gate 
    if jd_min_exp is not None:
        if years_experience is None or years_experience < jd_min_exp:
            if decision == "Consider":
                return "Maybe"
            elif decision == "Maybe":
                return "Reject"

    return decision

#Reason
def build_reason_with_decision(
    matched,
    missing,
    skill_coverage,
    decision,
    years_experience,
    jd_min_exp,
    semantic_score
):
    parts = []

    # Skill explanation
    pct = int(skill_coverage * 100)
    if matched:
        parts.append(
            f"Matched {len(matched)} key JD skills ({pct}% coverage): "
            f"{', '.join(matched[:6])}"
        )
    else:
        parts.append("No meaningful JD skill matches identified")

    if missing:
        parts.append(
            f"Missing important skills: {', '.join(missing[:4])}"
        )

    # Experience explanation
    if jd_min_exp is not None:
        if years_experience is None:
            parts.append(
                f"Experience not specified (JD requires minimum {jd_min_exp} years)"
            )
        elif years_experience < jd_min_exp:
            parts.append(
                f"Experience ({years_experience} years) below JD requirement ({jd_min_exp}+ years)"
            )
        else:
            parts.append(
                f"Experience ({years_experience} years) meets JD requirement"
            )

    # Semantic support (secondary)
    if semantic_score >= 0.75:
        parts.append("Strong alignment with JD responsibilities")
    elif semantic_score >= 0.55:
        parts.append("Moderate alignment with JD responsibilities")
    else:
        parts.append("Limited alignment with JD responsibilities")

    # Explicit decision justification
    if decision == "Consider":
        parts.append(
            "Recommended for consideration due to strong skill match and acceptable experience"
        )
    elif decision == "Maybe":
        parts.append(
            "Potential candidate, but gaps exist in skills or experience"
        )
    else:
        parts.append(
            "Not recommended due to insufficient skill match and/or experience mismatch"
        )

    parts.append(
    "Final ranking determined using two-phase validation, "
    "which prioritizes skill sufficiency and experience fit "
    "over raw similarity score."
    )


    return ". ".join(parts) + "."

# Main App
class App(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Competency Matching System")

        screen = QApplication.primaryScreen()
        screen_geo = screen.availableGeometry()

        app_w = min(1280, screen_geo.width() - 100)
        app_h = min(800, screen_geo.height() - 120)

        self.resize(app_w, app_h)
        self.setMinimumSize(1100, 700)

        # Controller state 
        self.cv_files = []
        self.jd_text = ""
        self.results = []
        self.metrics = {}
        self.ground_truth_relevance = {}
        self.validation_warning = False
        self.embedder = None
        self.model_loading = False
        self.resume_embeddings = None
        self.embedding_job_id = 0
        self.embedding_ready_job_id = None
        self.embed_thread = None
        self.embed_worker = None

        # Central stacked pages
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.frames = {}

        for PageClass in (PageAbout, PageCVUpload, PageJDInput, PageResults, PageComparison, PageRankingComparison, PageEvaluation ):

            page = PageClass(self)
            self.frames[PageClass.__name__] = page
            self.stack.addWidget(page)

        self.show_frame("PageAbout")

    def show_frame(self, name: str):
        frame = self.frames.get(name)
        if not frame:
            return
        if hasattr(frame, "on_show"):
            frame.on_show()
        self.stack.setCurrentWidget(frame)

    def start_and_init_model(self):
        if self.embedder is not None or self.model_loading:
            self.show_frame("PageCVUpload")
            return

        self.model_loading = True

        self.progress = QProgressDialog(
            "Initializing model… Please wait.",
            None,
            0,
            0,
            self
        )
        self.progress.setWindowTitle("Loading")
        self.progress.setWindowModality(Qt.ApplicationModal)
        self.progress.setCancelButton(None)
        self.progress.setMinimumDuration(0)
        self.progress.show()

        self.init_thread = QThread()
        self.init_worker = ModelInitWorker()
        self.init_worker.moveToThread(self.init_thread)

        self.init_thread.started.connect(self.init_worker.run)
        self.init_worker.finished.connect(self.on_model_ready)
        self.init_worker.error.connect(self.on_model_init_error)

        self.init_worker.finished.connect(self.init_thread.quit)
        self.init_worker.finished.connect(self.init_worker.deleteLater)
        self.init_thread.finished.connect(self.init_thread.deleteLater)

        self.init_thread.start()

    def on_model_ready(self, embedder):
        self.embedder = embedder
        self.model_loading = False
        if hasattr(self, "progress"):
            self.progress.close()
        self.show_frame("PageCVUpload")

    def on_model_init_error(self, msg):
        self.model_loading = False

        if hasattr(self, "progress"):
            self.progress.close()

        QMessageBox.critical(
            self,
            "Model Initialization Failed",
            msg
        )

    # Model execution  
    def run_matching(self):
        if not self.jd_text.strip() or not self.cv_files:
            return

        try:
            import stratos_i as model_mod
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return
        
        if self.embedder is None:
            embedder, ok, err = model_mod.try_load_embedder()
            if not ok:
                QMessageBox.critical(self, "Embedding Error", str(err))
                return
            self.embedder = embedder
        else:
            embedder = self.embedder
        
        jd_min_exp, jd_max_exp = infer_jd_experience_range(self.jd_text)
        self.jd_exp_range = (jd_min_exp, jd_max_exp)
         
        if hasattr(self, "results") and self.results:
            self.prev_results = copy.deepcopy(self.results)
        else:
            self.prev_results = []


        rows = []
        for i, cv in enumerate(self.cv_files, start=1):
            years_exp, exp_level = infer_experience(cv["text"])

            rows.append({
                "id": i,
                "name": cv["name"],
                "resume_text": cv["text"],
                "exp_level": exp_level,
                "years_experience": years_exp,
                "skills": extract_skills_from_taxonomy(cv["text"])
            })

        df = pd.DataFrame(rows)

        if self.embedding_ready_job_id != self.embedding_job_id:
            QMessageBox.warning(
            None,
            "Embedding mismatch",
            "CV embeddings are out of sync. Please re-embed."
            )
            return

        results_df, hyre = model_mod.rank_candidates_for_jd(
            df_resumes=df,
            resume_embeddings=self.resume_embeddings,  
            embedder=embedder,           
            jd_text=self.jd_text
        )

        if results_df is None:
            QMessageBox.critical(self, "Matching failed", hyre)
            return

        jd_skills = extract_skills_from_taxonomy(self.jd_text)
        ui_rows = []

        for _, r in results_df.iterrows():
            matched = [s for s in jd_skills if s in r["skills"]]
            missing = [s for s in jd_skills if s not in r["skills"]]

            jd_skill_count = len(jd_skills)
            skill_coverage = len(matched) / (jd_skill_count + 1e-6)


            experience = (
                f"{r['years_experience']} years"
                if r.get("years_experience") is not None
                else r.get("exp_level") or "Not specified"
            )

            sem = float(r.get("semantic_score", 0.0))

            decision = decide_by_skill_and_experience(
                skill_coverage=skill_coverage,
                years_experience=r.get("years_experience"),
                jd_min_exp=jd_min_exp
            )

            reason = build_reason_with_decision(
                matched=matched,
                missing=missing,
                skill_coverage=skill_coverage,
                decision=decision,
                years_experience=r.get("years_experience"),
                jd_min_exp=jd_min_exp,
                semantic_score=sem
            )


            ui_rows.append({
                "name": r["name"],
                "semantic_score": sem,
                "score": round(sem * 100, 2),

                "skill_score": round(float(r.get("skill_score", 0.0)) * 100, 2),
                "hybrid_score": round(float(r.get("hybrid_score", 0.0)) * 100, 2),

                "matched": matched,
                "missing": missing,
                "skill_coverage": skill_coverage,
                "experience": experience,
                "years_experience": r.get("years_experience"),
                "reason": reason,
                "decision": decision
            })

        self.model_results = copy.deepcopy(ui_rows)
        self.final_results = two_phase_rank(ui_rows, top_k=len(ui_rows))
        self.validation_warning = False
        self.results = self.final_results

        for i, r in enumerate(self.model_results, start=1):
            r["model_rank"] = i

        for i, r in enumerate(self.final_results, start=1):
            r["final_rank"] = i

        print(f"[INFO] CV files loaded: {len(self.cv_files)}")

    def closeEvent(self, event):
        if hasattr(self, "embed_thread") and self.embed_thread:
            if self.embed_thread.isRunning():
                self.embed_thread.quit()
                self.embed_thread.wait()

        if hasattr(self, "init_thread") and self.init_thread:
            if self.init_thread.isRunning():
                self.init_thread.quit()
                self.init_thread.wait()

        event.accept()

# PageAbout
class PageAbout(QWidget):
    LOGO_PATH = "logo.png"

    def __init__(self, controller: App):
        super().__init__()
        self.controller = controller

        base_font = QFont("Segoe UI", 11)
        self.setFont(base_font)

        title_font = QFont("Segoe UI", 15, QFont.Weight.Bold)
        bold_font = QFont("Segoe UI", 11, QFont.Weight.Bold)
        meta_font = QFont("Segoe UI", 10)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 24, 28, 24)
        outer.setSpacing(12)

        header_row = QHBoxLayout()
        header_row.setSpacing(4)  

        logo_lbl = QLabel()
        logo_lbl.setAlignment(Qt.AlignmentFlag.AlignTop)

        logo_path = (
            os.path.join(os.path.dirname(__file__), self.LOGO_PATH)
            if "__file__" in globals()
            else self.LOGO_PATH
        )

        if os.path.exists(logo_path):
            pix = QPixmap(logo_path).scaled(
                200, 200,  
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            logo_lbl.setPixmap(pix)
        else:
            logo_lbl.setText("◇")
            logo_lbl.setFont(QFont("Segoe UI", 40))

        header_row.addWidget(logo_lbl)

        header_col = QVBoxLayout()
        header_col.setSpacing(0)  

        title = QLabel("Stratos")
        title.setFont(title_font)
        header_col.addWidget(title)

        subtitle = QLabel("Intelligent matching, confident hiring.")
        header_col.addWidget(subtitle)

        version = QLabel("Version: 0.9.0  •  Last updated: 2025-12-12")
        version.setFont(meta_font)
        version.setStyleSheet("color: white; font-style: italic;")
        header_col.addWidget(version)

        header_row.addLayout(header_col)
        outer.addLayout(header_row)

        desc = QLabel(
            "Stratos is an intelligent resume–job matching system that integrates semantic embeddings, "
            "skill taxonomy extraction, and two-phase validation logic. The platform prioritizes "
            "skill sufficiency and experience fit over raw similarity scores, producing explainable "
            "and reproducible candidate rankings."
        )
        desc.setWordWrap(True)
        desc.setMaximumWidth(1000)
        outer.addWidget(desc)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #ddd;")
        outer.addWidget(line)

        # Development Team 
        dev_title = QLabel("Development Team")
        dev_title.setFont(bold_font)
        dev_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(dev_title)

        dev_row = QHBoxLayout()
        dev_row.setSpacing(80)

        # Developer A
        left = QVBoxLayout()
        left.setSpacing(4)

        name_a = QLabel("Developer: Kalidindi Ritika")
        name_a.setFont(bold_font)
        name_a.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left.addWidget(name_a)

        para_a = QLabel(
            "Ritika implemented the UI/UX, evaluation pipeline, and export/reporting features. " 
            "She focused on building a robust evaluation flow for Recall@K and NDCG@K."

        )
        para_a.setWordWrap(True)
        para_a.setMaximumWidth(460)
        left.addWidget(para_a)

        meta_a = QLabel(
            "Contact: kalidindi.ritika@sasken.com  •  Role: Frontend & Evaluation"
        )
        meta_a.setFont(meta_font)
        meta_a.setStyleSheet("color: white; font-style: italic;")
        left.addWidget(meta_a)

        dev_row.addLayout(left)

        # Developer B
        right = QVBoxLayout()
        right.setSpacing(4)

        name_b = QLabel("Developer: Rishita Battula")
        name_b.setFont(bold_font)
        name_b.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right.addWidget(name_b)

        para_b = QLabel(
            "Rishita worked on the matching model integration and the data ingestion pipeline. " 
            "She prepared prototypes for model outputs and ranking interfaces."
        )
        para_b.setWordWrap(True)
        para_b.setMaximumWidth(460)
        right.addWidget(para_b)

        meta_b = QLabel(
            "Contact: rishita.battula@sasken.com  •  Role: Backend & Model"
        )
        meta_b.setFont(meta_font)
        meta_b.setStyleSheet("color: white; font-style: italic;")
        right.addWidget(meta_b)

        dev_row.addLayout(right)
        outer.addLayout(dev_row)

        # Manager
        mgr_container = QVBoxLayout()
        mgr_container.setAlignment(Qt.AlignmentFlag.AlignCenter)

        mgr_title = QLabel("Manager / Project Guide")
        mgr_title.setFont(bold_font)
        mgr_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mgr_title.setTextFormat(Qt.TextFormat.PlainText)
        mgr_title.setStyleSheet("color: white !important;")
        mgr_container.addWidget(mgr_title)

        mgr = QLabel(
            "Selvaraj Vadivelu — Dean, Delivery Management\n"
            "Oversaw project milestones, provided domain guidance and validated evaluation methodology."
        )

        mgr.setTextFormat(Qt.TextFormat.PlainText)    
        mgr.setWordWrap(True)
        mgr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mgr.setMaximumWidth(800)
        mgr.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        mgr.setStyleSheet("color: white !important;")
        mgr.setFont(QFont("Segoe UI", 11))

        mgr_container.addWidget(mgr)
        outer.addLayout(mgr_container)

        meta_m = QLabel(
            "Contact: selvaraj.vadivelu@sasken.com"
        )
        meta_m.setFont(meta_font)
        meta_m.setStyleSheet("color: white; font-style: italic;")
        mgr_container.addWidget(meta_m)
        meta_m.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Footer meta
        footer = QHBoxLayout()

        tech = QLabel("Tech: Python 3 • Tkinter • numpy • sklearn")
        tech.setFont(meta_font)
        tech.setStyleSheet("color: white; font-style: italic;")
        footer.addWidget(tech)

        license_lbl = QLabel("License: Internal / Proprietary")
        license_lbl.setFont(meta_font)
        license_lbl.setStyleSheet("color: white; font-style: italic;")
        license_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        footer.addWidget(license_lbl)

        outer.addLayout(footer)

        # Start button
        start_btn = QPushButton("Start ▶")
        start_btn.setFixedSize(170, 42)
        start_btn.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        start_btn.setStyleSheet("""
            QPushButton {
                background-color: #1a73e8;
                color: white;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #1666cc; }
            QPushButton:pressed { background-color: #0f5bd6; }
        """)
        start_btn.clicked.connect(self.controller.start_and_init_model)

        outer.addWidget(start_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        help_lbl = QLabel(
            "For help: See 'Notes' in Evaluation page or contact the dev team."
        )
        help_lbl.setFont(meta_font)
        help_lbl.setStyleSheet("color: white; font-style: italic;")
        help_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(help_lbl)

# PageCVUpload
class PageCVUpload(QWidget):
    def __init__(self, controller: App):
        super().__init__()
        self.controller = controller

        # Layout
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        title = QLabel("Step 1: Select CV Folder")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        outer.addWidget(title)

        desc = QLabel(
            "Choose or type the path to a folder that contains candidate CV files (text-like files)."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 10px;")
        outer.addWidget(desc)

        # Folder entry row
        entry_row = QHBoxLayout()

        entry_label = QLabel("CV Folder Path:")
        entry_label.setStyleSheet("font-size: 10px; font-weight: bold;")
        entry_row.addWidget(entry_label)

        self.folder_entry = QLineEdit()
        entry_row.addWidget(self.folder_entry, stretch=1)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_cv_folder)
        entry_row.addWidget(browse_btn)

        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self.load_folder_from_entry)
        entry_row.addWidget(load_btn)

        outer.addLayout(entry_row)

        # CV list
        list_label = QLabel("CV files:")
        list_label.setStyleSheet("font-size: 10px; font-weight: bold;")
        outer.addWidget(list_label)

        self.cv_list = QListWidget()
        self.cv_list.itemSelectionChanged.connect(self.on_select_cv)
        outer.addWidget(self.cv_list, stretch=1)

        # Footer row
        footer = QHBoxLayout()

        clear_btn = QPushButton("Clear folder")
        clear_btn.clicked.connect(self.clear_folder_selection)
        footer.addWidget(clear_btn)

        footer.addStretch()

        self.count_label = QLabel("No folder selected.")
        footer.addWidget(self.count_label)

        self.embed_bar = QProgressBar()
        self.embed_bar.setRange(0, 100)
        self.embed_bar.setValue(0)
        self.embed_bar.setVisible(False)
        self.embed_bar.setTextVisible(True)
        self.embed_bar.setFormat("Embedding CVs… %p%")

        footer.addWidget(self.embed_bar)

        self.embed_status = QLabel("Idle")
        self.embed_status.setStyleSheet("""
            QLabel {
                padding: 4px 10px;
                border-radius: 10px;
                background-color: #e0e0e0;
                font-size: 10px;
            }
        """)

        self.embed_eta = QLabel("ETA: —")
        self.embed_eta.setFont(QFont("Segoe UI", 9))

        footer.addSpacing(12)
        footer.addWidget(self.embed_status)
        footer.addSpacing(8)
        footer.addWidget(self.embed_eta)

        outer.addLayout(footer)

        # Navigation
        nav = QHBoxLayout()
        nav.addStretch()

        self.next_btn = QPushButton("Next ▶ Page 2: Job Description")
        self.next_btn.clicked.connect(
            lambda: controller.show_frame("PageJDInput")
        )
        nav.addWidget(self.next_btn)

        outer.addLayout(nav)

    # Actions
    def browse_cv_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select CV folder"
        )
        if not folder:
            return
        self.folder_entry.setText(folder)

    def load_folder_from_entry(self):
        # Prevent starting a new embedding while one is running
        if (
            hasattr(self.controller, "embed_thread")
            and self.controller.embed_thread is not None
            and self.controller.embed_thread.isRunning()
        ):
            QMessageBox.warning(
                self,
                "Embedding in progress",
                "Please wait for the current embedding to finish."
            )
            return

        self.embed_bar.setValue(0)
        self.embed_bar.setVisible(False)
        self.embed_bar.setFormat("Embedding CVs… %p%")
        self.embed_status.setText("Idle")
        self.embed_eta.setText("ETA: —")

        self.controller.embedding_job_id += 1
        self.controller.embedding_ready_job_id = None
        current_job_id = self.controller.embedding_job_id
        self.controller.resume_embeddings = None
        folder = self.folder_entry.text().strip()
        if not folder:
            QMessageBox.warning(
                self, "No folder",
                "Please type or select a folder path to load."
            )
            return

        if not os.path.isdir(folder):
            QMessageBox.critical(
                self, "Invalid folder",
                f"The path is not a valid directory:\n{folder}"
            )
            return

        self.controller.cv_files = []
        self.cv_list.clear()

        try:
            files = os.listdir(folder)
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to list folder:\n{e}"
            )
            self.count_label.setText("Failed to read folder.")
            return

        text_like_exts = {".txt", ".md", ".log", ".pdf", ".docx"}
        cv_files = [
            f for f in files
            if os.path.isfile(os.path.join(folder, f))
            and os.path.splitext(f)[1].lower() in text_like_exts
        ]

        for fname in sorted(cv_files):
            QApplication.processEvents()
            full = os.path.join(folder, fname)
            ext = os.path.splitext(fname)[1].lower()
            
            try:
                if ext == ".pdf":
                    from pypdf import PdfReader
                    reader = PdfReader(full)
                    text = " ".join(
                        page.extract_text() or "" for page in reader.pages
                    )

                elif ext == ".docx":
                    from docx import Document
                    doc = Document(full)
                    text = " ".join(p.text for p in doc.paragraphs)

                else:  # txt / md / log
                    with open(full, "r", encoding="utf-8", errors="ignore") as fh:
                        text = fh.read()

            except Exception as e:
                print(f"[WARN] Failed to read {fname}: {e}")
                text = ""

            self.controller.cv_files.append({
                "name": fname,
                "path": full,
                "text": text
            })
            self.cv_list.addItem(fname)

        if not cv_files:
            self.cv_list.addItem(
                "No .txt/.md/.log files found in folder."
            )
            self.count_label.setText("0 CV files found.")
        else:
            self.count_label.setText(
                f"{len(cv_files)} CV(s) loaded from folder."
            )

        self.next_btn.setEnabled(False)
        self.folder_entry.setEnabled(False)

        try:
            import stratos_i as model_mod

            #progress bar
            if self.controller.embedder is not None and self.controller.cv_files:

                texts = [cv["text"] for cv in self.controller.cv_files]

                self.controller.embed_thread = QThread()

                self.controller.embed_worker = EmbeddingWorker(
                    embedder=self.controller.embedder,
                    texts=texts,
                    job_id=current_job_id,
                    batch_size=32
                )

                self.controller.embed_worker.moveToThread(self.controller.embed_thread)

                self.controller.embed_worker.progress.connect(self._on_embed_progress)
                self.controller.embed_worker.finished.connect(self._on_embeddings_ready)
                self.controller.embed_worker.error.connect(self._on_embed_error)

                self.controller.embed_thread.started.connect(self.controller.embed_worker.run)
                self.controller.embed_worker.finished.connect(self.controller.embed_thread.quit)
                self.controller.embed_worker.finished.connect(self.controller.embed_worker.deleteLater)
                self.controller.embed_thread.finished.connect(self.controller.embed_thread.deleteLater)

                self.embed_bar.setVisible(True)
                self.embed_bar.setValue(0)
                self.embed_bar.setEnabled(True)
                self.embed_status.setText("Starting…")
                self.embed_status.setStyleSheet("background:#ffe082;")
                self.embed_eta.setText("ETA: calculating…")

                QApplication.processEvents()

                self.controller.embed_thread.start()

        except Exception as e:
            print("[WARN] Pre-embedding skipped:", e)

    def _on_embed_error(self, msg):
        self.embed_status.setText("Failed")
        self.embed_status.setStyleSheet("background:#ffcdd2;")
        self.next_btn.setEnabled(True)
        self.controller.embed_thread = None
        self.controller.embed_worker = None
        QMessageBox.critical(self, "Embedding failed", msg)
        self.folder_entry.setEnabled(True)

    def _on_embed_progress(self, percent, eta):
        self.embed_bar.setVisible(True)
        self.embed_bar.setValue(percent)

        if eta < 60:
            self.embed_eta.setText(f"ETA: {int(eta)}s")
        else:
            self.embed_eta.setText(
                f"ETA: {int(eta // 60)}m {int(eta % 60)}s"
            )

        self.embed_status.setText("Embedding")
        self.embed_status.setStyleSheet("background:#ffcc80;")

    def _on_embeddings_ready(self, job_id, embeddings):
        if job_id != self.controller.embedding_job_id:
            print("[INFO] Ignoring stale embedding result")
            return

        self.controller.resume_embeddings = embeddings
        self.controller.embedding_ready_job_id = job_id

        self.embed_bar.setValue(100)
        self.embed_bar.setFormat("Embeddings cached ✓")
        self.embed_bar.setEnabled(False)
        self.embed_status.setText("Cached")
        self.embed_status.setStyleSheet("background:#c8e6c9;")
        self.embed_eta.setText("ETA: done")

        self.next_btn.setEnabled(True)

        if self.controller.embed_thread is not None:
            self.controller.embed_thread.quit()
            self.controller.embed_thread.wait()

        self.controller.embed_thread = None
        self.controller.embed_worker = None

        print("[INFO] Resume embeddings cached (validated)")
        print(f"[INFO] Cached embeddings for {len(embeddings)} CVs (job {job_id})")

        self.folder_entry.setEnabled(True)

    def clear_folder_selection(self):
        self.folder_entry.setText("")
        self.controller.cv_files = []
        self.cv_list.clear()
        self.count_label.setText("No folder selected.")
        self.embed_bar.setVisible(False)
        self.embed_status.setText("Idle")
        self.embed_eta.setText("ETA: —")
        self.next_btn.setEnabled(False)

    def on_select_cv(self):
        pass

    def on_show(self):
        self.cv_list.clear()
        for cv in getattr(self.controller, "cv_files", []):
            self.cv_list.addItem(cv["name"])

        n = len(getattr(self.controller, "cv_files", []))
        self.count_label.setText(
            "No folder selected." if n == 0 else f"{n} CV(s) loaded."
        )

        if self.controller.resume_embeddings is not None:
            self.embed_status.setText("Cached")
            self.embed_bar.setValue(100)
            self.embed_bar.setVisible(True)
        else:
            self.embed_bar.setVisible(False)

        self.next_btn.setEnabled(False)

# PageJDInput
class PageJDInput(QWidget):
    def __init__(self, controller: App):
        super().__init__()
        self.controller = controller
        self.thread = None
        self._matching_in_progress = False

        # Layout
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        title = QLabel("Step 2: Job Description")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        outer.addWidget(title)

        desc = QLabel(
            "Upload a JD file or paste the Job Description text below."
        )
        desc.setStyleSheet("font-size: 10px;")
        outer.addWidget(desc)

        # Top controls
        top_row = QHBoxLayout()

        load_btn = QPushButton("Load JD from file...")
        load_btn.clicked.connect(self.load_jd_file)
        top_row.addWidget(load_btn)

        self.jd_status = QLabel("No JD loaded yet.")
        self.jd_status.setStyleSheet("font-size: 9px;")
        top_row.addWidget(self.jd_status)

        top_row.addStretch()
        outer.addLayout(top_row)

        # JD text area
        self.text = QTextEdit()
        self.text.setStyleSheet("font-family: Consolas; font-size: 10px;")
        outer.addWidget(self.text, stretch=1)

        # Navigation
        nav = QHBoxLayout()

        back_btn = QPushButton("◀ Back to Page 1")
        back_btn.clicked.connect(
            lambda: controller.show_frame("PageCVUpload")
        )
        nav.addWidget(back_btn)

        nav.addStretch()

        next_btn = QPushButton("Next ▶ Page 3: Results")
        next_btn.clicked.connect(self.go_next)
        nav.addWidget(next_btn)

        outer.addLayout(nav)

    # Lifecycle
    def on_show(self):
        self.text.clear()
        if self.controller.jd_text:
            self.text.setPlainText(self.controller.jd_text)

    # Actions
    def load_jd_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select JD file",
            "",
            "Text files (*.txt);;All files (*.*)"
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to read JD file:\n{e}"
            )
            return

        self.text.clear()
        self.text.setPlainText(text)
        self.jd_status.setText(
            f"Loaded from: {os.path.basename(path)}"
        )

    def go_next(self):
        jd_text = self.text.toPlainText().strip()

        if not jd_text:
            QMessageBox.warning(
                self, "JD missing",
                "Please paste or load a Job Description."
            )
            return
        
        if self.controller.resume_embeddings is None:
            QMessageBox.information(
            self,
            "Embedding in progress",
            "Please wait for CV embedding to complete."
            )
            return


        if not self.controller.cv_files:
            QMessageBox.warning(
                self, "No CVs",
                "Please upload CVs in Page 1 before continuing."
            )
            return
        
        if self._matching_in_progress:
            return
        
        self.controller.jd_text = jd_text

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        self.progress = QProgressDialog(
            "Running matching… Please wait.",
            None,
            0,
            0,
            self
        )
        self.progress.setWindowTitle("Processing")
        self.progress.setWindowModality(Qt.ApplicationModal)
        self.progress.setCancelButton(None)
        self.progress.setMinimumDuration(0)
        self.progress.show()


        self.thread = QThread()
        self.worker = MatchingWorker(self.controller)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_matching_done)
        self.worker.error.connect(self.on_matching_error)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.error.connect(self.thread.quit)
        self.worker.error.connect(self.worker.deleteLater)

        self._matching_in_progress = True 

        self.thread.start()

    def on_matching_done(self):
        self._matching_in_progress = False
        QApplication.restoreOverrideCursor()

        if hasattr(self, "progress"):
            self.progress.close()

        self.controller.show_frame("PageResults")

    def on_matching_error(self, msg):
        self._matching_in_progress = False
        QApplication.restoreOverrideCursor()

        if hasattr(self, "progress"):
            self.progress.close()

        QMessageBox.critical(self, "Matching failed", msg)


# PageResults
class PageResults(QWidget):
    """
    Page 3 – Results table aligned to model outputs:
    Columns:
    Select | Rank | CV Name | Experience | Semantic (%) | Reason | Recommended Action
    """

    def __init__(self, controller: App):
        super().__init__()
        self.controller = controller
        self.selection = {}
        self.current_filtered = []

        # Layout
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        title = QLabel("Step 3: Matching Results")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        outer.addWidget(title)

        subtitle = QLabel(
            "Candidates are ranked using a two-phase validation model. "
            "Hybrid score is shown for transparency, not direct ordering.")
        subtitle.setStyleSheet("font-size: 10px; color: #666;")
        outer.addWidget(subtitle)

        # Filters bar
        filters = QHBoxLayout()
        filters.setSpacing(8)

        filters.addWidget(QLabel("Skill"))
        self.skill_combo = QComboBox()
        self.skill_combo.addItem("Any")
        filters.addWidget(self.skill_combo)

        filters.addWidget(QLabel("Level"))
        self.level_combo = QComboBox()
        self.level_combo.addItem("Any")
        filters.addWidget(self.level_combo)

        filters.addWidget(QLabel("Min score (%)"))
        self.min_score_edit = QLineEdit("0")
        self.min_score_edit.setFixedWidth(55)
        filters.addWidget(self.min_score_edit)

        self.show_all_chk = QCheckBox("Show all candidates")
        filters.addWidget(self.show_all_chk)
        self.show_all_chk.stateChanged.connect(self.refresh_results)

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.refresh_results)
        filters.addWidget(apply_btn)

        compare_btn = QPushButton("Compare Selected")
        compare_btn.clicked.connect(lambda: self.controller.show_frame("PageComparison"))
        filters.addWidget(compare_btn)

        compare_rank_btn = QPushButton("Compare Rankings")
        compare_rank_btn.clicked.connect(lambda: self.controller.show_frame("PageRankingComparison"))
        filters.addWidget(compare_rank_btn)

        redo_btn = QPushButton("Redo Matching")
        redo_btn.clicked.connect(self.redo_matching)
        filters.addWidget(redo_btn)

        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self.export)
        filters.addWidget(self.export_btn)

        filters.addStretch()
        outer.addLayout(filters)

        # Results table
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(["Select", "Rank", "CV Name","Experience","Semantic (%)","Skill Overlap (%)","Hybrid Score (%)","Reason","Recommended Action"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.cellClicked.connect(self._on_cell_clicked)

        self.table.setColumnWidth(0, 60)
        self.table.setColumnWidth(1, 60)   # Rank
        self.table.setColumnWidth(2, 280)  # CV Name
        self.table.setColumnWidth(3, 110)  # Experience
        self.table.setColumnWidth(4, 120)  # Semnatic 
        self.table.setColumnWidth(5, 130)  # Skill overlap
        self.table.setColumnWidth(6, 130)  # Hybrid score
        self.table.setColumnWidth(7, 360)  # Reason
        self.table.setColumnWidth(8, 120)  # Decision


        outer.addWidget(self.table, stretch=1)

        self.table.setWordWrap(True)
        self.table.resizeRowsToContents()

        header = CheckBoxHeader(Qt.Orientation.Horizontal, self.table)
        self.table.setHorizontalHeader(header)
        header.toggled.connect(self._toggle_all_visible)

        # Navigation
        nav = QHBoxLayout()

        back_btn = QPushButton("◀ Back to Page 2")
        back_btn.clicked.connect(lambda: controller.show_frame("PageJDInput"))
        nav.addWidget(back_btn)

        nav.addStretch()

        next_btn = QPushButton("Next ▶ Page 4: Evaluation Metrics")
        next_btn.clicked.connect(lambda: controller.show_frame("PageEvaluation"))
        nav.addWidget(next_btn)

        outer.addLayout(nav)

    # Lifecycle
    def on_show(self):
        self.selection = {
            r.get("name", ""): False
            for r in self.controller.results
        }
        self._populate_filter_dropdowns()
        self.refresh_results()

        if getattr(self.controller, "validation_warning", False):
            QMessageBox.information(
                self,
                "Ranking Validation",
                "Two-phase validation found minor ranking disagreement.\n"
                "Top candidates were adjusted for consistency."
                )

    #Toggle for checkbox
    def _toggle_all_visible(self, checked: bool):
        for r in self.current_filtered:
            self.selection[r["name"]] = checked
        self.refresh_results()

    # Filters
    def _populate_filter_dropdowns(self):
        skills, levels = set(), set()

        for r in self.controller.results:
            for s in (r.get("matched") or []):
                skills.add(s)
            exp = r.get("experience")
            if exp and exp != "Not specified":
                levels.add(exp)

        self.skill_combo.clear()
        self.skill_combo.addItem("Any")
        self.skill_combo.addItems(sorted(skills))

        self.level_combo.clear()
        self.level_combo.addItem("Any")
        self.level_combo.addItems(sorted(levels))

    # Refresh table
    def refresh_results(self):
        self.table.setRowCount(0)
        self.current_filtered = []

        try:
            min_score = float(self.min_score_edit.text().strip())
        except Exception:
            min_score = 0.0

        sel_skill = self.skill_combo.currentText()
        sel_level = self.level_combo.currentText()

        results = self.controller.results

        total = len(results)
        if total == 0:
            return

        if not self.show_all_chk.isChecked():
            results = results[:20]

        for r in results:
            if r["score"] < min_score:
                continue
            if sel_skill != "Any" and sel_skill not in r["matched"]:
                continue
            if sel_level != "Any" and str(r["experience"]) != sel_level:
                continue

            self.current_filtered.append(r)

        for idx, r in enumerate(self.current_filtered, start=1):
            row = self.table.rowCount()
            self.table.insertRow(row)

            checked = "☑" if self.selection.get(r["name"]) else "☐"

            self.table.setItem(row, 0, QTableWidgetItem(checked))
            rank_val = r.get("final_rank", idx)
            self.table.setItem(row, 1, QTableWidgetItem(str(rank_val)))
            self.table.setItem(row, 2, QTableWidgetItem(r["name"]))
            self.table.setItem(row, 3, QTableWidgetItem(str(r.get("experience"))))
            self.table.setItem(row, 4, QTableWidgetItem(str(r.get("score"))))
            self.table.setItem(row, 5, QTableWidgetItem(str(r.get("skill_score"))))
            self.table.setItem(row, 6, QTableWidgetItem(str(r.get("hybrid_score"))))
            self.table.setItem(row, 7, QTableWidgetItem(r.get("reason", "")))
            decision_item = QTableWidgetItem(r.get("decision", ""))

            if r["decision"] == "Consider":
                decision_item.setForeground(Qt.GlobalColor.darkGreen)
            elif r["decision"] == "Maybe":
                decision_item.setForeground(Qt.GlobalColor.darkCyan)
            else:
                decision_item.setForeground(Qt.GlobalColor.red)
            self.table.setItem(row, 8, decision_item)

            any_selected = any(
                self.selection.get(r["name"])
                for r in self.current_filtered
            )
            self.export_btn.setText(
                "Export Selected" if any_selected else "Export All"
            )

    # Table click handling
    def _on_cell_clicked(self, row, col):
        name = self.table.item(row, 2).text()

        if col == 0:
            self.selection[name] = not self.selection.get(name, False)
            self.refresh_results()
        elif col == 7:
            self._show_reason_popup(
                name,
                self.table.item(row, 7).text()
            )
    
    def _sync_header_checkbox(self):
        if not self.current_filtered:
            return

        header = self.table.horizontalHeader()
        all_checked = all(
            self.selection.get(r["name"], False)
            for r in self.current_filtered
        )
        header.setChecked(all_checked)

    # Reason popup
    def _show_reason_popup(self, title, text):
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Explanation – {title}")
        dlg.resize(700, 400)

        layout = QVBoxLayout(dlg)
        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setText(text)
        layout.addWidget(txt)

        dlg.exec()

    # Selection helpers
    def select_all(self):
        for r in self.current_filtered:
            self.selection[r["name"]] = True
        self.refresh_results()

    def deselect_all(self):
        for r in self.current_filtered:
            self.selection[r["name"]] = False
        self.refresh_results()

    # Export
    def export(self):
        selected = [
            r for r in self.current_filtered
            if self.selection.get(r["name"])
        ]

        rows = selected if selected else self.current_filtered

        if not rows:
            QMessageBox.warning(
                self,
                "No rows", "No rows available to export."
            )
            return

        self._export_rows(rows)

    def _export_rows(self, rows):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "", "CSV files (*.csv)"
        )
        if not path:
            return

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "cv_name", "experience",
                "semantic_pct", "reason", "decision"
            ])
            for r in rows:
                writer.writerow([
                    r.get("name"),
                    r.get("experience"),
                    r.get("score"),
                    r.get("reason"),
                    r.get("decision")
                ])

        QMessageBox.information(
            self, "Export complete",
            f"Saved {len(rows)} rows to:\n{path}"
        )

    # Redo / Start over
    def redo_matching(self):
        self.controller.validation_warning = False
        self.controller.show_frame("PageJDInput")

    def start_over(self):
        if QMessageBox.question(
            self,
            "Start Over",
            "Clear CVs, JD, and results?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ) != QMessageBox.StandardButton.Yes:
            return
        self.controller.cv_files = []
        self.controller.jd_text = ""
        self.controller.results = []

        self.selection = {}
        self.current_filtered = []
        self.table.setRowCount(0)

        self.controller.show_frame("PageCVUpload")

def compute_model_metrics(results):
    scores = np.array([r["score"] for r in results])
    if len(scores) == 0:
        return {}

    top = scores[:max(1, len(scores)//4)]
    bottom = scores[-max(1, len(scores)//4):]

    return {
        "mean": np.mean(scores),
        "std": np.std(scores),
        "top_bottom_gap": np.mean(top) - np.mean(bottom),
        "confidence_ratio": np.mean(top) / (np.mean(bottom) + 1e-6),
        "entropy": -np.sum(
            (scores / scores.sum()) * np.log(scores / scores.sum() + 1e-9)
        )
    }

def compute_recall_at_k(relevances, k):
    if not relevances:
        return 0.0

    total_relevant = sum(1 for r in relevances if r > 0)
    if total_relevant == 0:
        return 0.0

    retrieved_relevant = sum(1 for r in relevances[:k] if r > 0)
    return retrieved_relevant / total_relevant


def compute_ndcg_at_k(relevances, k):
    def dcg(rel):
        return sum(
            rel[i] / math.log2(i + 2)
            for i in range(min(len(rel), k))
        )

    actual_dcg = dcg(relevances)

    ideal_relevances = sorted(relevances, reverse=True)
    ideal_dcg = dcg(ideal_relevances)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg

def compute_rbo(rank_a, rank_b, p=0.85):
    """
    Rank-Biased Overlap (RBO)
    rank_a, rank_b: ranked lists (best → worst)
    p: top-weighting factor (0.8–0.9 typical)
    """
    depth = max(len(rank_a), len(rank_b))
    score = 0.0
    seen_a, seen_b = set(), set()

    for d in range(1, depth + 1):
        if d <= len(rank_a):
            seen_a.add(rank_a[d - 1])
        if d <= len(rank_b):
            seen_b.add(rank_b[d - 1])

        overlap = len(seen_a.intersection(seen_b))
        score += (overlap / d) * (p ** (d - 1))

    return (1 - p) * score

# PageComparison
class PageComparison(QWidget):
    def __init__(self, controller: App):
        super().__init__()
        self.controller = controller

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.setSpacing(12)

        title = QLabel("Candidate Comparison")
        title.setStyleSheet("font-size:18px; font-weight:600;")
        outer.addWidget(title)

        subtitle = QLabel(
            "Side-by-side comparison of selected candidates "
            "to evaluate strengths, gaps, and suggest recommended actions."
        )
        subtitle.setStyleSheet("font-size:11px; color:#555;")
        outer.addWidget(subtitle)

        # Comparison table
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([
            "Attribute", "Candidate 1", "Candidate 2",
            "Candidate 3", "Candidate 4"
        ])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)

        self.table.setColumnWidth(0, 180)
        for i in range(1, 5):
            self.table.setColumnWidth(i, 240)

        outer.addWidget(self.table, stretch=1)

        self.table.setWordWrap(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.resizeRowsToContents()

        # Navigation
        nav = QHBoxLayout()

        back_btn = QPushButton("◀ Back to Results")
        back_btn.clicked.connect(
            lambda: controller.show_frame("PageResults")
        )
        nav.addWidget(back_btn)

        nav.addStretch()
        outer.addLayout(nav)

    # Lifecycle
    def on_show(self):
        # Pull selected candidates from PageResults
        results_page = self.controller.frames.get("PageResults")
        if not results_page:
            return

        selected = [
            r for r in self.controller.results
            if results_page.selection.get(r["name"])
        ]

        if len(selected) < 2:
            QMessageBox.information(
                self,
                "Not enough candidates",
                "Please select at least two candidates in the Results page."
            )
            self.table.setRowCount(0)
            return

        self.populate(selected[:4])

    # Populate table
    def populate(self, candidates):
        self.table.setRowCount(0)

        rows = [
            ("Name", lambda r: r.get("name", "")),
            ("Experience", lambda r: str(r.get("experience", ""))),
            ("Score (%)", lambda r: str(r.get("score", ""))),
            ("Decision", lambda r: r.get("decision", "")),
            ("Matched Skills", lambda r: ", ".join(r.get("matched", [])[:10])),
            ("Missing Skills", lambda r: ", ".join(r.get("missing", [])[:10])),
            ("Explanation", lambda r: r.get("reason", "")),
        ]

        for label, extractor in rows:
            row = self.table.rowCount()
            self.table.insertRow(row)

            header = QTableWidgetItem(label)
            header.setFlags(header.flags() ^ Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, header)

            for i, cand in enumerate(candidates):
                text = extractor(cand)
                cell = QTableWidgetItem(text)
                cell.setTextAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
                cell.setFlags(cell.flags() ^ Qt.ItemFlag.ItemIsEditable)
                cell.setToolTip(text)
                self.table.setItem(row, i + 1, cell)

        
        self.table.resizeRowsToContents()

#PageRankingComparison
class PageRankingComparison(QWidget):
    def __init__(self, controller: App):
        super().__init__()
        self.controller = controller

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.setSpacing(12)

        title = QLabel("Model vs Two-Phase Ranking Comparison")
        title.setStyleSheet("font-size:18px; font-weight:600;")
        outer.addWidget(title)

        subtitle = QLabel(
            "This table shows how the two-phase ranking adjusted the original model output."
        )
        subtitle.setStyleSheet("font-size:11px; color:#555;")
        outer.addWidget(subtitle)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels([
            "CV Name",
            "Model Rank",
            "Final Rank",
            "Δ Rank",
            "Semantic (%)",
            "Skill (%)",
            "Hybrid (%)",
            "Overridden?"
        ])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QTableWidget.NoSelection)

        self.table.setColumnWidth(0, 260)
        for i in range(1, 8):
            self.table.setColumnWidth(i, 110)

        outer.addWidget(self.table, stretch=1)

        nav = QHBoxLayout()
        back_btn = QPushButton("◀ Back to Results")
        back_btn.clicked.connect(
            lambda: controller.show_frame("PageResults")
        )
        nav.addWidget(back_btn)

        nav.addStretch()
        outer.addLayout(nav)
    
    def on_show(self):
        self.table.setRowCount(0)

        model = {
            r["name"]: r
            for r in getattr(self.controller, "model_results", [])
        }
        final = {
            r["name"]: r
            for r in getattr(self.controller, "final_results", [])
        }

        for name, m in model.items():
            f = final.get(name)
            if not f:
                continue

            model_rank = m.get("model_rank")
            final_rank = f.get("final_rank")
            delta = model_rank - final_rank

            row = self.table.rowCount()
            self.table.insertRow(row)

            values = [
                name,
                model_rank,
                final_rank,
                delta,
                m["score"],
                m["skill_score"],
                m["hybrid_score"],
                "Yes" if delta != 0 else "No"
            ]

            for col, v in enumerate(values):
                item = QTableWidgetItem(str(v))
                if col == 3 and delta != 0:
                    item.setForeground(Qt.GlobalColor.darkRed if delta < 0 else Qt.GlobalColor.darkGreen)
                self.table.setItem(row, col, item)

# PageEvaluation
class PageEvaluation(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller

        self.human_ranking = ["resume_0005.txt", "resume_0001.txt", "resume_0009.txt", "resume_0003.txt", "resume_0006.txt", 
                              "resume_0008.txt", "resume_0002.txt", "resume_0007.txt", "resume_0010.txt", "resume_0004.txt"]

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(16)

        title = QLabel("Step 4: Evaluation Metrics")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        main_layout.addWidget(title)

        subtitle = QLabel("Evaluate the ranking quality using information retrieval metrics.")
        subtitle.setStyleSheet("color: white;")
        main_layout.addWidget(subtitle)

        note = QLabel(
            "Note: Relevance is estimated using rule-based proxy logic "
            "based on skill match and experience alignment. "
            "These metrics do not represent human hiring decisions."
        )
        note.setStyleSheet("color: white; font-style: italic;")
        note.setWordWrap(True)
        main_layout.addWidget(note)

        controls = QHBoxLayout()

        self.export_csv_btn = QPushButton("Export CSV")
        self.export_csv_btn.clicked.connect(self.export_csv)
        controls.addWidget(self.export_csv_btn)

        controls.addWidget(QLabel("Select K:"))
        self.k_selector = QComboBox()
        self.k_selector.addItems(["5", "10", "20"])
        self.k_selector.setFixedWidth(80)
        controls.addWidget(self.k_selector)

        controls.addSpacing(20)

        controls.addWidget(QLabel("Relevance Threshold:"))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(55, 85)
        self.threshold_slider.setValue(70)
        self.threshold_slider.setFixedWidth(200)
        controls.addWidget(self.threshold_slider)

        self.threshold_label = QLabel("≥ 60%")
        self.threshold_label.setFixedWidth(50)
        controls.addWidget(self.threshold_label)

        controls.addSpacing(20)

        self.compute_btn = QPushButton("Compute Metrics")
        self.compute_btn.clicked.connect(self.compute_metrics)
        controls.addWidget(self.compute_btn)

        controls.addStretch()
        main_layout.addLayout(controls)

        summary_box = QGroupBox("Metric Summary")
        summary_layout = QHBoxLayout(summary_box)

        self.recall_label = QLabel("Recall@K: —")
        self.ndcg_label = QLabel("NDCG@K: —")

        self.rbo_label = QLabel("RBO (Human vs Model): —")
        self.rbo_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        summary_layout.addWidget(self.rbo_label)

        for lbl in (self.recall_label, self.ndcg_label):
            lbl.setStyleSheet("font-size: 14px;")
            summary_layout.addWidget(lbl)

        summary_layout.addStretch()
        main_layout.addWidget(summary_box)

        table_box = QGroupBox("Per-CV Contribution (optional)")
        table_layout = QVBoxLayout(table_box)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Rank", "CV Name", "Proxy Relevant (0/1)", "Proxy Contribution"])
        self.table.horizontalHeader().setStretchLastSection(True)
        table_layout.addWidget(self.table)

        plot_box = QGroupBox("Recall vs Threshold")
        plot_layout = QVBoxLayout(plot_box)

        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumWidth(420)
        plot_layout.addWidget(self.canvas)

        table_plot_row = QHBoxLayout()
        table_plot_row.setSpacing(12)
        table_plot_row.addWidget(table_box, stretch=3)
        table_plot_row.addWidget(plot_box, stretch=2)
        main_layout.addLayout(table_plot_row)

        nav = QHBoxLayout()
        back_btn = QPushButton("◀ Back to Results")
        back_btn.clicked.connect(lambda: controller.show_frame("PageResults"))
        nav.addWidget(back_btn)
        nav.addStretch()
        main_layout.addLayout(nav)

        self.threshold_slider.valueChanged.connect(self._on_threshold_change)
        self.threshold_slider.valueChanged.connect(self.compute_metrics)

    def on_show(self):
        self.controller.metrics = {}
        self.recall_label.setText("Recall@K: —")
        self.ndcg_label.setText("NDCG@K: —")
        self.table.setRowCount(0)

    def _on_threshold_change(self, value):
        self.threshold_label.setText(f"≥ {value}%")
    
    def compute_metrics(self):
        if not self.controller.results:
            return
        
        # --- RBO: Human vs Model ranking ---

        # Case 1: Human ranking not yet provided
        if not getattr(self, "human_ranking", None):
            self.rbo_label.setText("RBO (Human vs Model): Pending human ranking")

        else:
            model_ranking = [r["name"] for r in self.controller.results]

            # Align rankings (only common CVs)
            common = [cv for cv in self.human_ranking if cv in model_ranking]

            if len(common) < 2:
                self.rbo_label.setText("RBO (Human vs Model): N/A (insufficient overlap)")
            else:
                human_rank = common
                model_rank = [cv for cv in model_ranking if cv in common]

                rbo_score = compute_rbo(human_rank, model_rank)
                self.rbo_label.setText(
                    f"RBO (Human vs Model): {rbo_score:.3f}"
                )

        k = int(self.k_selector.currentText())
        threshold = self.threshold_slider.value()
        results = self.controller.results

        def proxy_relevance(r):
            # Rule-based proxy for recruiter judgment
            skill_ok = len(r.get("matched", [])) >= max(1, len(r.get("missing", [])))
            exp_ok = (
                r.get("years_experience") is not None and
                self.controller.jd_exp_range[0] <= r["years_experience"] <= self.controller.jd_exp_range[1]
            )
            return int(skill_ok and exp_ok)

        relevance = [proxy_relevance(r) for r in results]

        recall_k = compute_recall_at_k(relevance, k)
        ndcg_k = compute_ndcg_at_k(relevance, k)

        self.recall_label.setText(f"Recall@{k}: {recall_k:.3f}")
        self.ndcg_label.setText(f"NDCG@{k}: {ndcg_k:.3f}")
        
        top_k = relevance[:k]

        self.table.setRowCount(0)
        for rank, (r, rel) in enumerate(zip(results[:k], top_k), start=1):
            row = self.table.rowCount()
            self.table.insertRow(row)
            contribution = 1 / math.log2(rank + 1) if rel else 0.0
            self.table.setItem(row, 0, QTableWidgetItem(str(rank)))
            self.table.setItem(row, 1, QTableWidgetItem(r["name"]))
            self.table.setItem(row, 2, QTableWidgetItem(str(rel)))
            self.table.setItem(row, 3, QTableWidgetItem(f"{contribution:.4f}"))

        self.plot_recall_curve()

    def plot_recall_curve(self):
        if not self.controller.results:
            return

        scores = [r["score"] for r in self.controller.results]
        thresholds = list(range(0, 101, 5))
        recalls = []

        for t in thresholds:
            relevance = [1 if s >= t else 0 for s in scores]
            num_rel = sum(relevance)
            recalls.append(sum(relevance[:10]) / num_rel if num_rel > 0 else 0)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(thresholds, recalls, marker="o")
        ax.set_xlabel("Relevance Threshold (%)")
        ax.set_ylabel("Recall@10")
        ax.set_title("Recall vs Threshold")
        ax.grid(True)
        self.canvas.draw()

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
        if not path:
            return

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Rank", "CV Name", "Score", "Relevant"])
            for i, r in enumerate(self.controller.results, start=1):
                writer.writerow([i, r["name"], r["score"], int(r["score"] >= self.threshold_slider.value())])

# Run the app

if __name__ == "__main__":
    app = QApplication(sys.argv)

    win = App()
    win.show()

    sys.exit(app.exec())
