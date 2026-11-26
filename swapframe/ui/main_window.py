"""Main window."""
from __future__ import annotations

import os
import time
import traceback
from typing import Callable, List, Optional

import numpy as np
from PySide6.QtCore import QSettings, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QIcon
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
                               QListWidgetItem, QMainWindow, QMenu, QProgressBar, QPushButton, QScrollArea,
                               QSlider, QVBoxLayout, QWidget)

from .. import media, models
from ..engine import Face, Pipeline, Settings, available_devices
from .widgets import DropFrame, PreviewView, Segmented, Switch, ThumbList, section, short_name, thumb

SWAPPER_CHOICES = [('AlphaFace · best for angles', 'alphaface_256'),
                   ('HyperSwap 1A', 'hyperswap_1a_256'), ('HyperSwap 1B', 'hyperswap_1b_256'),
                   ('HyperSwap 1C', 'hyperswap_1c_256'), ('InSwapper 128', 'inswapper_128_fp16')]
ENHANCER_CHOICES = [('Off', None), ('GFPGAN 1.4', 'gfpgan_1.4'), ('GPEN 512', 'gpen_bfr_512')]
FACE_MODES = [('Everyone', 'all'), ('Chosen people', 'selected'), ('Largest face', 'largest')]
DETAIL = [('Standard', 1), ('High', 2), ('Max', 4)]
IMAGE_FILTER = 'Images (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff)'
MEDIA_FILTER = ('Photos and videos (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff '
                '*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.gif)')


def _bool(v) -> bool:
    return v in (True, 'true', 'True', 1, '1')


class Task(QThread):
    progress = Signal(str, int, int)
    frame = Signal(object)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable[['Task'], object], parent=None):
        super().__init__(parent)
        self.fn = fn
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def cancelled(self) -> bool:
        return self._cancel

    def run(self):
        try:
            self.done.emit(self.fn(self))
        except InterruptedError:
            self.failed.emit('Cancelled')
        except Exception as e:  # pragma: no cover
            traceback.print_exc()
            self.failed.emit(str(e) or e.__class__.__name__)


class MainWindow(QMainWindow):
    def __init__(self, colors: dict, family: str):
        super().__init__()
        self.colors = colors
        self.setWindowTitle('Swapframe')
        self.resize(1320, 880)
        self.setMinimumSize(1000, 680)
        self.store = QSettings('Swapframe', 'Swapframe')

        devices = available_devices()
        saved_device = self.store.value('device', devices[0])
        self.pipeline = Pipeline(saved_device if saved_device in devices else devices[0])

        # state
        self.source_paths: List[str] = []
        self.source_faces: List[Face] = []
        self.target_path: Optional[str] = None
        self.video_info: Optional[media.VideoInfo] = None
        self.frame: Optional[np.ndarray] = None
        self.result: Optional[np.ndarray] = None
        self.result_gen = -1
        self.gen = 0
        self.references: List[np.ndarray] = []
        self.strip_faces: List[Face] = []
        self.tasks: set = set()
        self.preview_task: Optional[Task] = None
        self.preview_pending = False
        self.render_task: Optional[Task] = None
        self.comparing = False
        self.last_output: Optional[str] = None

        self._build(devices)
        self._load_settings()

        self.preview_timer = QTimer(self, singleShot=True, interval=280, timeout=self._run_preview)
        self.seek_timer = QTimer(self, singleShot=True, interval=180, timeout=self._seek)

    # ------------------------------------------------------------------ UI --
    def _switch(self, text: str, tip: str) -> Switch:
        sw = Switch(text, accent=self.colors['accent'], track=self.colors['track'])
        sw.setToolTip(tip)
        sw.toggled.connect(self._settings_changed)
        return sw

    def _build(self, devices: List[str]):
        root = QWidget(objectName='root')
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(16)

        # ---- sidebar
        side_panel = QFrame(objectName='panel')
        side_panel.setFixedWidth(340)
        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side_inner = QWidget()
        side_inner.setStyleSheet('background: transparent;')
        side = QVBoxLayout(side_inner)
        side.setContentsMargins(18, 18, 18, 18)
        side.setSpacing(10)
        side_scroll.setWidget(side_inner)
        sp = QVBoxLayout(side_panel)
        sp.setContentsMargins(0, 0, 0, 0)
        sp.addWidget(side_scroll)

        side.addWidget(QLabel('Swapframe', objectName='title'))
        side.addSpacing(4)

        # source faces
        row = QHBoxLayout()
        row.addWidget(section('Face'))
        row.addStretch()
        row.addWidget(QPushButton('Clear', objectName='ghost', clicked=self._clear_sources))
        side.addLayout(row)
        self.src_drop = DropFrame()
        self.src_drop.files.connect(self._add_sources)
        sd = QVBoxLayout(self.src_drop)
        sd.setContentsMargins(10, 10, 10, 10)
        sd.setSpacing(6)
        self.src_list = ThumbList(64, flow_wrap=False)
        self.src_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.src_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.src_list.customContextMenuRequested.connect(self._source_menu)
        self.src_hint = QLabel('Drop 2–5 clear photos of the face from a few angles', objectName='hint')
        self.src_hint.setWordWrap(True)
        sd.addWidget(self.src_list)
        sd.addWidget(self.src_hint)
        sd.addWidget(QPushButton('Add photos', clicked=self._browse_sources))
        side.addWidget(self.src_drop)

        # target
        side.addSpacing(6)
        side.addWidget(section('Target'))
        self.tgt_drop = DropFrame()
        self.tgt_drop.files.connect(lambda p: self._set_target(p[0]))
        td = QHBoxLayout(self.tgt_drop)
        td.setContentsMargins(10, 10, 10, 10)
        self.tgt_thumb = QLabel()
        self.tgt_thumb.setFixedSize(52, 52)
        self.tgt_thumb.hide()
        td.addWidget(self.tgt_thumb)
        tcol = QVBoxLayout()
        tcol.setSpacing(2)
        self.tgt_name = QLabel('Photo or video')
        self.tgt_meta = QLabel('', objectName='hint')
        tcol.addWidget(self.tgt_name)
        tcol.addWidget(self.tgt_meta)
        td.addLayout(tcol, 1)
        td.addWidget(QPushButton('Browse', clicked=self._browse_target))
        side.addWidget(self.tgt_drop)

        # model
        side.addSpacing(6)
        side.addWidget(section('Model'))
        self.model_box = QComboBox()
        for label, key in SWAPPER_CHOICES:
            self.model_box.addItem(label, key)
        self.model_box.currentIndexChanged.connect(self._settings_changed)
        side.addWidget(self.model_box)

        side.addWidget(section('Detail'))
        self.detail = Segmented([d[0] for d in DETAIL])
        self.detail.changed.connect(self._settings_changed)
        side.addWidget(self.detail)

        # expression
        side.addWidget(section('Expression'))
        xrow = QHBoxLayout()
        self.expr_switch = self._switch('Keep target expression',
                                        "Copies the original smile, eyes, brows and mouth back onto the new face")
        self.expr_slider = QSlider(Qt.Orientation.Horizontal, minimum=10, maximum=100, value=80)
        self.expr_slider.setToolTip('How strongly to copy the original expression')
        self.expr_slider.setFixedWidth(90)
        self.expr_slider.sliderReleased.connect(self._settings_changed)
        xrow.addWidget(self.expr_switch, 1)
        xrow.addWidget(self.expr_slider)
        side.addLayout(xrow)

        side.addWidget(section('Enhance'))
        erow = QHBoxLayout()
        self.enh_box = QComboBox()
        for label, key in ENHANCER_CHOICES:
            self.enh_box.addItem(label, key)
        self.enh_box.currentIndexChanged.connect(self._settings_changed)
        self.enh_slider = QSlider(Qt.Orientation.Horizontal, minimum=10, maximum=100, value=80)
        self.enh_slider.setToolTip('Enhance strength')
        self.enh_slider.sliderReleased.connect(self._settings_changed)
        erow.addWidget(self.enh_box, 1)
        erow.addWidget(self.enh_slider, 1)
        side.addLayout(erow)

        side.addWidget(section('Faces'))
        self.face_box = QComboBox()
        for label, key in FACE_MODES:
            self.face_box.addItem(label, key)
        self.face_box.currentIndexChanged.connect(self._face_mode_changed)
        side.addWidget(self.face_box)

        # natural look
        side.addSpacing(6)
        side.addWidget(section('Natural look'))
        self.shape_switch = self._switch('Follow face shape', 'Blend along the real face outline instead of a box')
        self.tone_switch = self._switch('Match skin tone and lighting', 'Adapt the new face to the scene lighting')
        self.lm_switch = self._switch('Precise alignment', '68-point landmarks for turned heads and expressions')
        self.occl_switch = self._switch('Keep hands and hair in front', 'Objects covering the face stay on top')
        self.steady_switch = self._switch('Steady video tracking', 'Smooth jitter and keep faces through head turns')
        for sw in (self.shape_switch, self.tone_switch, self.lm_switch, self.occl_switch, self.steady_switch):
            side.addWidget(sw)

        side.addSpacing(4)
        side.addWidget(section('Device'))
        self.device_box = QComboBox()
        for d in devices:
            self.device_box.addItem(d, d)
        self.device_box.setCurrentIndex(max(0, devices.index(self.pipeline.device)))
        self.device_box.currentIndexChanged.connect(self._device_changed)
        side.addWidget(self.device_box)
        side.addStretch()
        side.addWidget(QPushButton('Open models folder', objectName='ghost',
                                   clicked=lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(models.models_dir()))),
                       alignment=Qt.AlignmentFlag.AlignLeft)
        outer.addWidget(side_panel)

        # ---- main area
        main_panel = QFrame(objectName='panel')
        main = QVBoxLayout(main_panel)
        main.setContentsMargins(16, 16, 16, 16)
        main.setSpacing(12)

        self.preview = PreviewView()
        self.preview.set_colors(self.colors)
        self.preview.files.connect(self._preview_drop)
        main.addWidget(self.preview, 1)

        self.scrub = QWidget()
        sl = QHBoxLayout(self.scrub)
        sl.setContentsMargins(0, 0, 0, 0)
        self.scrub_slider = QSlider(Qt.Orientation.Horizontal)
        self.scrub_slider.valueChanged.connect(self._scrub_moved)
        self.time_label = QLabel('0:00 / 0:00', objectName='muted')
        sl.addWidget(self.scrub_slider, 1)
        sl.addWidget(self.time_label)
        self.scrub.hide()
        main.addWidget(self.scrub)

        self.strip = QWidget()
        stl = QVBoxLayout(self.strip)
        stl.setContentsMargins(0, 0, 0, 0)
        stl.setSpacing(4)
        stl.addWidget(QLabel('Tap the people to swap', objectName='hint'))
        self.face_list = ThumbList(56, flow_wrap=False)
        self.face_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.face_list.itemSelectionChanged.connect(self._faces_picked)
        stl.addWidget(self.face_list)
        self.strip.hide()
        main.addWidget(self.strip)

        bar = QHBoxLayout()
        bar.setSpacing(10)
        self.compare_btn = QPushButton('Hold to compare')
        self.compare_btn.pressed.connect(lambda: self._compare(True))
        self.compare_btn.released.connect(lambda: self._compare(False))
        bar.addWidget(self.compare_btn)
        status_col = QVBoxLayout()
        status_col.setSpacing(4)
        self.status = QLabel('Add a face and a target to begin', objectName='muted')
        self.progress = QProgressBar(textVisible=False)
        self.progress.hide()
        status_col.addWidget(self.status)
        status_col.addWidget(self.progress)
        bar.addLayout(status_col, 1)
        self.open_btn = QPushButton('Show file', clicked=self._reveal_last)
        self.open_btn.hide()
        bar.addWidget(self.open_btn)
        self.go_btn = QPushButton('Save image', objectName='primary', clicked=self._go)
        self.go_btn.setEnabled(False)
        self.go_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        bar.addWidget(self.go_btn)
        main.addLayout(bar)
        outer.addWidget(main_panel, 1)

    # ------------------------------------------------------------ settings --
    def _load_settings(self):
        s = self.store
        widgets = (self.model_box, self.enh_box, self.enh_slider, self.expr_switch, self.expr_slider,
                   self.shape_switch, self.tone_switch, self.lm_switch, self.occl_switch, self.steady_switch)
        for w in widgets:
            w.blockSignals(True)
        self.model_box.setCurrentIndex(max(0, self.model_box.findData(s.value('swapper', 'alphaface_256'))))
        self.detail.set_index(int(s.value('detail', 1)))
        self.expr_switch.setChecked(_bool(s.value('expression_on', True)))
        self.expr_slider.setValue(int(s.value('expression', 80)))
        self.enh_box.setCurrentIndex(max(0, self.enh_box.findData(s.value('enhancer', None) or None)))
        self.enh_slider.setValue(int(s.value('enhance_blend', 80)))
        self.shape_switch.setChecked(_bool(s.value('face_shape', True)))
        self.tone_switch.setChecked(_bool(s.value('colour_match', True)))
        self.lm_switch.setChecked(_bool(s.value('landmarks', True)))
        self.occl_switch.setChecked(_bool(s.value('occlusion', False)))
        self.steady_switch.setChecked(_bool(s.value('stabilise', True)))
        for w in widgets:
            w.blockSignals(False)
        self._sync_sliders()

    def _save_settings(self):
        s = self.store
        s.setValue('swapper', self.model_box.currentData())
        s.setValue('detail', self.detail.index())
        s.setValue('expression_on', self.expr_switch.isChecked())
        s.setValue('expression', self.expr_slider.value())
        s.setValue('enhancer', self.enh_box.currentData() or '')
        s.setValue('enhance_blend', self.enh_slider.value())
        s.setValue('face_shape', self.shape_switch.isChecked())
        s.setValue('colour_match', self.tone_switch.isChecked())
        s.setValue('landmarks', self.lm_switch.isChecked())
        s.setValue('occlusion', self.occl_switch.isChecked())
        s.setValue('stabilise', self.steady_switch.isChecked())
        s.setValue('device', self.device_box.currentData())

    def _sync_sliders(self):
        self.enh_slider.setEnabled(self.enh_box.currentData() is not None)
        self.expr_slider.setEnabled(self.expr_switch.isChecked())

    def settings(self) -> Settings:
        return Settings(
            swapper=self.model_box.currentData(),
            boost=DETAIL[self.detail.index()][1],
            expression=self.expr_slider.value() / 100.0 if self.expr_switch.isChecked() else 0.0,
            enhancer=self.enh_box.currentData(),
            enhance_blend=self.enh_slider.value() / 100.0,
            occlusion=self.occl_switch.isChecked(),
            face_shape=self.shape_switch.isChecked(),
            landmarks=self.lm_switch.isChecked(),
            colour_match=0.8 if self.tone_switch.isChecked() else 0.0,
            face_mode=self.face_box.currentData(),
            reference=list(self.references),
            stabilise=self.steady_switch.isChecked(),
        )

    def _settings_changed(self, *_):
        self._sync_sliders()
        self._save_settings()
        self._schedule_preview()

    def _face_mode_changed(self, *_):
        self.strip.setVisible(self.face_box.currentData() == 'selected' and self.frame is not None)
        self._schedule_preview()

    def _device_changed(self, *_):
        self.pipeline.set_device(self.device_box.currentData())
        self._save_settings()
        self._schedule_preview()

    # ------------------------------------------------------------- tasks --
    def _start(self, fn, on_done=None, on_fail=None, on_progress=None, on_frame=None) -> Task:
        task = Task(fn, self)
        if on_done:
            task.done.connect(on_done)
        task.failed.connect(on_fail or self._task_failed)
        task.progress.connect(on_progress or self._download_progress)
        if on_frame:
            task.frame.connect(on_frame)
        self.tasks.add(task)
        task.finished.connect(lambda t=task: self.tasks.discard(t))
        task.start()
        return task

    def _task_failed(self, msg: str):
        self.preview.set_busy(False)
        self.progress.hide()
        self.status.setText(msg)

    def _download_progress(self, label: str, done: int, total: int):
        self.progress.show()
        if total:
            self.progress.setRange(0, 100)
            self.progress.setValue(int(done * 100 / total))
            self.status.setText(f'Downloading {label}  ·  {done / 1e6:.0f} / {total / 1e6:.0f} MB')
        else:
            self.progress.setRange(0, 0)
            self.status.setText(f'Downloading {label}  ·  {done / 1e6:.0f} MB')

    # ------------------------------------------------------------ sources --
    def _browse_sources(self):
        paths, _ = QFileDialog.getOpenFileNames(self, 'Choose face photos', self._last_dir(), IMAGE_FILTER)
        if paths:
            self._add_sources(paths)

    def _add_sources(self, paths: List[str]):
        new = [p for p in paths if media.is_image(p) and p not in self.source_paths]
        if not new:
            return
        self._remember_dir(new[0])
        self.source_paths.extend(new)
        self._analyse_sources()

    def _clear_sources(self):
        self.source_paths.clear()
        self.source_faces.clear()
        self.src_list.clear()
        self.src_hint.setText('Drop 2–5 clear photos of the face from a few angles')
        self.result = None
        self._show_frame()
        self._update_actions()

    def _source_menu(self, pos):
        item = self.src_list.itemAt(pos)
        if not item:
            return
        menu = QMenu(self)
        remove = menu.addAction('Remove')
        if menu.exec(self.src_list.mapToGlobal(pos)) == remove:
            self.source_paths.remove(item.data(Qt.ItemDataRole.UserRole))
            self._analyse_sources()

    def _analyse_sources(self):
        paths = list(self.source_paths)
        self.status.setText('Reading faces…')

        def job(task: Task):
            self.pipeline.prepare_core(task.progress.emit, task.cancelled)
            rows = []
            for p in paths:
                img = media.read_image(p)
                faces = self.pipeline.source_faces([img]) if img is not None else []
                rows.append((p, img, faces[0] if faces else None))
            return rows

        self._start(job, self._sources_ready)

    def _sources_ready(self, rows):
        self.progress.hide()
        self.src_list.clear()
        self.source_faces = []
        missing = 0
        for path, img, face in rows:
            if img is None:
                missing += 1
                continue
            crop = img
            if face is not None:
                self.source_faces.append(face)
                crop = _face_crop(img, face)
            else:
                missing += 1
            item = QListWidgetItem(QIcon(thumb(crop, 64)), '')
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(os.path.basename(path) + ('' if face is not None else '  ·  no face found'))
            self.src_list.addItem(item)
        n = len(self.source_faces)
        if n == 0:
            self.src_hint.setText('No face found. Try a clearer, front-facing photo')
        else:
            extra = f'  ·  {missing} skipped' if missing else ''
            self.src_hint.setText(f'{n} photo{"s" if n > 1 else ""} blended into one identity{extra}')
        self.status.setText('Ready' if self.frame is not None else 'Now add a target photo or video')
        self._schedule_preview()

    # ------------------------------------------------------------- target --
    def _browse_target(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Choose a photo or video', self._last_dir(), MEDIA_FILTER)
        if path:
            self._set_target(path)

    def _preview_drop(self, paths: List[str]):
        imgs = [p for p in paths if media.is_image(p)]
        if not self.source_paths and len(paths) >= 2 and len(imgs) == len(paths):
            self._add_sources([paths[0]])
            self._set_target(paths[1])
        else:
            self._set_target(paths[0])

    def _set_target(self, path: str):
        if not (media.is_image(path) or media.is_video(path)):
            self.status.setText('That file type is not supported')
            return
        self._remember_dir(path)
        self.target_path = path
        self.result = None
        self.references.clear()
        self.open_btn.hide()
        self.tgt_name.setText(short_name(path))
        self.tgt_meta.setText('Loading…')

        if media.is_image(path):
            img = media.read_image(path)
            if img is None:
                self.status.setText('Could not open that image')
                return
            self.video_info = None
            self._target_loaded(img, f'{img.shape[1]} × {img.shape[0]}')
        else:
            def job(_t):
                info = media.probe(path)
                return path, info, (media.grab_frame(path, 0) if info else None)
            self._start(job, self._video_loaded)

    def _video_loaded(self, payload):
        path, info, frame = payload
        if path != self.target_path:
            return
        if info is None or frame is None:
            self.status.setText('Could not open that video')
            self.tgt_meta.setText('')
            return
        self.video_info = info
        mins, secs = divmod(int(info.duration), 60)
        self._target_loaded(frame, f'{info.width} × {info.height}  ·  {info.fps:.0f} fps  ·  {mins}:{secs:02d}')

    def _target_loaded(self, frame: np.ndarray, meta: str):
        self.frame = frame
        self.tgt_meta.setText(meta)
        self.tgt_thumb.setPixmap(thumb(frame, 52))
        self.tgt_thumb.show()
        is_video = self.video_info is not None
        self.scrub.setVisible(is_video)
        self.steady_switch.setVisible(is_video)
        if is_video:
            self.scrub_slider.blockSignals(True)
            self.scrub_slider.setRange(0, self.video_info.frames - 1)
            self.scrub_slider.setValue(0)
            self.scrub_slider.blockSignals(False)
            self._update_time(0)
        self.go_btn.setText('Swap video' if is_video else 'Save image')
        self.strip.setVisible(self.face_box.currentData() == 'selected')
        self.face_list.clear()
        self._show_frame()
        self._update_actions()
        self.status.setText('Ready' if self.source_faces else 'Now add a face photo')
        self._schedule_preview()

    def _scrub_moved(self, value: int):
        self._update_time(value)
        self.seek_timer.start()

    def _update_time(self, value: int):
        if not self.video_info:
            return
        t = value / self.video_info.fps
        total = self.video_info.duration
        self.time_label.setText(f'{int(t // 60)}:{int(t % 60):02d} / {int(total // 60)}:{int(total % 60):02d}')

    def _seek(self):
        if not self.video_info:
            return
        t = self.scrub_slider.value() / self.video_info.fps
        path = self.target_path

        def job(_t):
            return media.grab_frame(path, t)

        def done(frame):
            if frame is not None and path == self.target_path:
                self.frame = frame
                self.result = None
                self._show_frame()
                self._schedule_preview()

        self._start(job, done)

    # ------------------------------------------------------------ preview --
    def _schedule_preview(self):
        self.gen += 1
        self._update_actions()
        if self.frame is not None and (self.source_faces or self.face_box.currentData() == 'selected'):
            self.preview_timer.start()

    def _run_preview(self):
        if self.render_task is not None or self.frame is None:
            return
        if self.preview_task is not None:
            self.preview_pending = True
            return
        gen, frame, settings = self.gen, self.frame, self.settings()
        sources = list(self.source_faces)
        want_strip = settings.face_mode == 'selected'
        self.preview.set_busy(True)

        def job(task: Task):
            self.pipeline.prepare(settings, task.progress.emit, task.cancelled)
            strip = self.pipeline.analyse(frame) if want_strip else None
            self.pipeline.reset_tracking()
            out = self.pipeline.process(frame, sources, settings) if sources else None
            return gen, frame, out, strip

        self.preview_task = self._start(job, self._preview_ready, self._preview_failed)

    def _preview_failed(self, msg: str):
        self.preview_task = None
        self._task_failed(msg)
        self._drain_preview()

    def _preview_ready(self, payload):
        self.preview_task = None
        self.progress.hide()
        gen, frame, out, strip = payload
        if frame is self.frame:
            if strip is not None:
                self._fill_strip(frame, strip)
            if out is not None:
                self.result, self.result_gen = out, gen
                self._show_frame()
                if self.render_task is None:
                    self.status.setText('Preview ready' if gen == self.gen else 'Updating…')
        self.preview.set_busy(False)
        self._update_actions()
        if gen != self.gen:
            self.preview_pending = True
        self._drain_preview()

    def _drain_preview(self):
        if self.preview_pending:
            self.preview_pending = False
            QTimer.singleShot(10, self._run_preview)

    def _fill_strip(self, frame: np.ndarray, faces: List[Face]):
        self.strip_faces = faces
        self.face_list.blockSignals(True)
        self.face_list.clear()
        for face in faces:
            item = QListWidgetItem(QIcon(thumb(_face_crop(frame, face), 56)), '')
            self.face_list.addItem(item)
            if self.references and face.normed is not None:
                if max(float(r @ face.normed) for r in self.references) > 0.4:
                    item.setSelected(True)
        self.face_list.blockSignals(False)
        if not faces:
            self.status.setText('No faces found in this frame')

    def _faces_picked(self):
        rows = sorted(self.face_list.row(i) for i in self.face_list.selectedItems())
        keep = [r for r in self.references
                if not any(f.normed is not None and float(r @ f.normed) > 0.4 for f in self.strip_faces)]
        picked = [self.strip_faces[r].normed for r in rows if self.strip_faces[r].normed is not None]
        self.references = keep + picked
        self._schedule_preview()

    def _compare(self, on: bool):
        self.comparing = on
        self._show_frame()

    def _show_frame(self):
        if self.frame is None:
            self.preview.set_image(None)
        elif self.comparing or self.result is None:
            self.preview.set_image(self.frame)
        else:
            self.preview.set_image(self.result)

    def _update_actions(self):
        ready = self.frame is not None and bool(self.source_faces)
        if self.render_task is not None:
            self.go_btn.setEnabled(True)
            return
        if self.video_info is None:
            self.go_btn.setEnabled(ready and self.result is not None and self.result_gen == self.gen)
        else:
            self.go_btn.setEnabled(ready)
        self.compare_btn.setEnabled(self.result is not None)

    # ------------------------------------------------------------- output --
    def _go(self):
        if self.render_task is not None:
            self.render_task.cancel()
            self.go_btn.setEnabled(False)
            self.status.setText('Stopping…')
            return
        if self.video_info is None:
            self._save_image()
        else:
            self._render_video()

    def _default_out(self, ext: str) -> str:
        base, _ = os.path.splitext(self.target_path)
        return f'{base}_swapped{ext}'

    def _save_image(self):
        ext = os.path.splitext(self.target_path)[1].lower()
        ext = ext if ext in ('.jpg', '.jpeg', '.png', '.webp') else '.png'
        path, _ = QFileDialog.getSaveFileName(self, 'Save image', self._default_out(ext),
                                              'PNG (*.png);;JPEG (*.jpg);;WebP (*.webp)')
        if not path:
            return
        try:
            media.write_image(path, self.result)
        except Exception as e:
            self.status.setText(f'Could not save: {e}')
            return
        self.last_output = path
        self.open_btn.show()
        self.status.setText(f'Saved {os.path.basename(path)}')

    def _render_video(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save video', self._default_out('.mp4'), 'MP4 video (*.mp4)')
        if not path:
            return
        if not path.lower().endswith('.mp4'):
            path += '.mp4'
        if os.path.abspath(path) == os.path.abspath(self.target_path):
            self.status.setText('Choose a different file name than the original')
            return
        settings, sources = self.settings(), list(self.source_faces)
        src, info = self.target_path, self.video_info
        self._set_controls_enabled(False)
        self.go_btn.setText('Cancel')
        self.open_btn.hide()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.show()

        def job(task: Task):
            self.pipeline.prepare(settings, task.progress.emit, task.cancelled)
            self.pipeline.reset_tracking()
            writer = media.VideoWriter(path, src, info)
            total, start, last_preview = info.frames, time.time(), 0.0
            try:
                for i, frame in enumerate(media.iter_frames(src, info)):
                    if task.cancelled():
                        raise InterruptedError
                    out = self.pipeline.process(frame, sources, settings, video=True)
                    writer.write(out)
                    now = time.time()
                    task.progress.emit(f'__render__{now - start:.3f}', i + 1, total)
                    if now - last_preview > 0.6:
                        last_preview = now
                        task.frame.emit(out)
                writer.close()
            except BaseException:
                writer.abort()
                try:
                    os.remove(path)
                except OSError:
                    pass
                raise
            return path

        self.render_task = self._start(job, self._render_done, self._render_failed,
                                       self._render_progress, self._render_frame)

    def _render_progress(self, label: str, done: int, total: int):
        if not label.startswith('__render__'):
            self._download_progress(label, done, total)
            return
        elapsed = float(label[10:])
        total = max(total, done)
        self.progress.setRange(0, 100)
        self.progress.setValue(int(done * 100 / total))
        fps = done / elapsed if elapsed > 0 else 0
        left = (total - done) / fps if fps > 0 else 0
        self.status.setText(f'Swapping frame {done} of {total}  ·  {fps:.1f} fps  ·  '
                            f'{int(left // 60)}:{int(left % 60):02d} left')

    def _render_frame(self, frame):
        self.result = frame
        self._show_frame()

    def _render_done(self, path: str):
        self.render_task = None
        self.last_output = path
        self.progress.hide()
        self._set_controls_enabled(True)
        self.go_btn.setText('Swap video')
        self.open_btn.show()
        self.status.setText(f'Saved {os.path.basename(path)}')
        self._update_actions()

    def _render_failed(self, msg: str):
        self.render_task = None
        self.progress.hide()
        self._set_controls_enabled(True)
        self.go_btn.setText('Swap video')
        self.status.setText('Stopped' if msg == 'Cancelled' else f'Something went wrong: {msg}')
        self._update_actions()

    def _set_controls_enabled(self, on: bool):
        for w in (self.src_drop, self.tgt_drop, self.model_box, self.detail, self.expr_switch, self.enh_box,
                  self.face_box, self.shape_switch, self.tone_switch, self.lm_switch, self.occl_switch,
                  self.steady_switch, self.device_box, self.scrub, self.strip):
            w.setEnabled(on)
        if on:
            self._sync_sliders()
        else:
            self.expr_slider.setEnabled(False)
            self.enh_slider.setEnabled(False)
        self.preview.setAcceptDrops(on)

    def _reveal_last(self):
        if self.last_output and os.path.exists(self.last_output):
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(self.last_output)))

    # -------------------------------------------------------------- misc --
    def _last_dir(self) -> str:
        return self.store.value('last_dir', os.path.expanduser('~'))

    def _remember_dir(self, path: str):
        self.store.setValue('last_dir', os.path.dirname(path))

    def closeEvent(self, e):
        for t in list(self.tasks):
            t.cancel()
        for t in list(self.tasks):
            t.wait(3000)
        super().closeEvent(e)


def _face_crop(img: np.ndarray, face: Face, pad: float = 0.35) -> np.ndarray:
    x1, y1, x2, y2 = face.bbox
    side = max(x2 - x1, y2 - y1) * (1 + pad)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    h, w = img.shape[:2]
    a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    c, d = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    crop = img[b:d, a:c]
    return crop if crop.size else img
