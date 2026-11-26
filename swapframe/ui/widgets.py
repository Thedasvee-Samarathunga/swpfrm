"""Reusable widgets."""
from __future__ import annotations

import os
from typing import List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QRectF, QSize, Qt, Signal, Property
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import (QAbstractButton, QButtonGroup, QFrame, QHBoxLayout, QLabel, QListWidget,
                               QListView, QPushButton, QSizePolicy, QWidget)


def to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def thumb(bgr: np.ndarray, size: int) -> QPixmap:
    h, w = bgr.shape[:2]
    side = min(h, w)
    y, x = (h - side) // 2, (w - side) // 2
    square = cv2.resize(bgr[y:y + side, x:x + side], (size * 2, size * 2), interpolation=cv2.INTER_AREA)
    pm = QPixmap.fromImage(to_qimage(square)).scaled(size, size, Qt.AspectRatioMode.IgnoreAspectRatio,
                                                     Qt.TransformationMode.SmoothTransformation)
    return rounded(pm, 10)


def rounded(pm: QPixmap, radius: float) -> QPixmap:
    out = QPixmap(pm.size())
    out.fill(Qt.GlobalColor.transparent)
    p = QPainter(out)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    path = QPainterPath()
    path.addRoundedRect(QRectF(0, 0, pm.width(), pm.height()), radius, radius)
    p.setClipPath(path)
    p.drawPixmap(0, 0, pm)
    p.end()
    return out


class DropFrame(QFrame):
    """Dashed drop zone that emits local file paths."""
    files = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('drop')
        self.setAcceptDrops(True)
        self.setProperty('hover', False)

    def _set_hover(self, on: bool):
        self.setProperty('hover', on)
        self.style().unpolish(self)
        self.style().polish(self)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self._set_hover(True)

    def dragLeaveEvent(self, e):
        self._set_hover(False)

    def dropEvent(self, e):
        self._set_hover(False)
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.files.emit(paths)


class Switch(QAbstractButton):
    """Label on the left, animated toggle on the right. Sizes itself to its text so nothing gets clipped."""
    TRACK_W, TRACK_H, GAP = 36, 20, 12

    def __init__(self, text: str = '', parent=None, accent='#5B5BF0', track='#D9DCE5'):
        super().__init__(parent)
        self.setCheckable(True)
        self.setText(text)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pos = 0.0
        self.accent, self.track = QColor(accent), QColor(track)
        self._anim = QPropertyAnimation(self, b'pos_', self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.toggled.connect(self._animate)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_colors(self, accent: str, track: str):
        self.accent, self.track = QColor(accent), QColor(track)
        self.update()

    def _animate(self, on: bool):
        self._anim.stop()
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()

    def setChecked(self, on: bool):  # noqa: N802
        super().setChecked(on)
        self._anim.stop()
        self._pos = 1.0 if on else 0.0
        self.update()

    def get_pos(self):
        return self._pos

    def set_pos(self, v):
        self._pos = v
        self.update()

    pos_ = Property(float, get_pos, set_pos)

    def sizeHint(self):
        fm = self.fontMetrics()
        return QSize(fm.horizontalAdvance(self.text()) + self.GAP + self.TRACK_W + 2, max(28, fm.height() + 10))

    def minimumSizeHint(self):
        return QSize(self.TRACK_W + 80, self.sizeHint().height())

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.TRACK_W, self.TRACK_H
        x = self.width() - w - 1
        y = (self.height() - h) / 2
        track = QColor(self.track)
        if self._pos > 0:
            a, t = self.accent, self.track
            track = QColor(int(t.red() + (a.red() - t.red()) * self._pos),
                           int(t.green() + (a.green() - t.green()) * self._pos),
                           int(t.blue() + (a.blue() - t.blue()) * self._pos))
        if not self.isEnabled():
            track.setAlpha(120)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(QRectF(x, y, w, h), h / 2, h / 2)
        p.setBrush(QColor('#FFFFFF'))
        d = h - 6
        p.drawEllipse(QRectF(x + 3 + (w - d - 6) * self._pos, y + 3, d, d))
        p.setPen(self.palette().color(self.foregroundRole()))
        text_w = int(x - self.GAP)
        text = self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, text_w)
        p.drawText(QRectF(0, 0, text_w, self.height()), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   text)
        p.end()


class Segmented(QFrame):
    changed = Signal(int)

    def __init__(self, labels: List[str], parent=None):
        super().__init__(parent)
        self.setObjectName('segwrap')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(3)
        self.group = QButtonGroup(self)
        for i, label in enumerate(labels):
            b = QPushButton(label)
            b.setObjectName('seg')
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setMinimumWidth(10)
            self.group.addButton(b, i)
            lay.addWidget(b)
        self.group.idClicked.connect(self.changed.emit)

    def set_index(self, i: int):
        btn = self.group.button(i)
        if btn:
            btn.setChecked(True)

    def index(self) -> int:
        return max(0, self.group.checkedId())


class PreviewView(QWidget):
    """Aspect-correct image canvas with placeholder + drop support."""
    files = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200, 200)
        self._image: Optional[QImage] = None
        self._placeholder = 'Drop a photo, video or GIF here'
        self._colors = dict(sunken='#EEF0F5', muted='#6A6F7E', accent='#5B5BF0')
        self._busy = False
        self._drag = False

    def sizeHint(self):
        return QSize(640, 480)

    def set_colors(self, colors: dict):
        self._colors = colors
        self.update()

    def set_image(self, bgr: Optional[np.ndarray]):
        self._image = to_qimage(bgr) if bgr is not None else None
        self.update()

    def set_busy(self, busy: bool):
        self._busy = busy
        self.update()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self._drag = True
            self.update()

    def dragLeaveEvent(self, e):
        self._drag = False
        self.update()

    def dropEvent(self, e):
        self._drag = False
        self.update()
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.files.emit(paths)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        r = QRectF(self.rect())
        path = QPainterPath()
        path.addRoundedRect(r, 14, 14)
        p.fillPath(path, QColor(self._colors['sunken']))
        if self._image is not None and not self._image.isNull():
            iw, ih = self._image.width(), self._image.height()
            scale = min((r.width() - 24) / iw, (r.height() - 24) / ih)
            w, h = iw * scale, ih * scale
            target = QRectF(r.x() + (r.width() - w) / 2, r.y() + (r.height() - h) / 2, w, h)
            clip = QPainterPath()
            clip.addRoundedRect(target, 10, 10)
            p.setClipPath(clip)
            p.drawImage(target, self._image)
            p.setClipping(False)
        else:
            p.setPen(QColor(self._colors['muted']))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, self._placeholder)
        if self._drag:
            pen = p.pen()
            pen.setColor(QColor(self._colors['accent']))
            pen.setWidth(2)
            p.setPen(pen)
            p.drawRoundedRect(r.adjusted(1, 1, -1, -1), 14, 14)
        if self._busy:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 110))
            pill = QRectF(r.right() - 112, r.top() + 14, 98, 28)
            p.drawRoundedRect(pill, 14, 14)
            p.setPen(QColor('#FFFFFF'))
            p.drawText(pill, Qt.AlignmentFlag.AlignCenter, 'Working…')
        p.end()


class ThumbList(QListWidget):
    """Row/flow list of rounded thumbnails."""

    def __init__(self, size: int = 64, flow_wrap: bool = True, parent=None):
        super().__init__(parent)
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setIconSize(QSize(size, size))
        self.setMovement(QListView.Movement.Static)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setWrapping(flow_wrap)
        if not flow_wrap:
            self.setFlow(QListView.Flow.LeftToRight)
            self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setSpacing(2)
        self.setUniformItemSizes(True)
        self.setFixedHeight(size + 26)
        self.setMinimumWidth(10)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # let drops fall through to the parent drop zone
        self.setDragEnabled(False)
        self.setAcceptDrops(False)
        self.viewport().setAcceptDrops(False)


class ElideLabel(QLabel):
    """Single-line label that shortens with '…' instead of forcing its parent wider."""

    def __init__(self, text: str = '', parent=None, **kw):
        super().__init__(text, parent, **kw)
        self._full = text
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(10)

    def setText(self, text: str):  # noqa: N802
        self._full = text
        self.setToolTip(text if len(text) > 40 else '')
        self._refresh()

    def text_full(self) -> str:
        return self._full

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._refresh()

    def _refresh(self):
        super().setText(self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideRight, max(10, self.width())))


def section(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setObjectName('section')
    return lbl


def short_name(path: str, limit: int = 34) -> str:
    name = os.path.basename(path)
    return name if len(name) <= limit else name[:limit - 10] + '…' + name[-9:]
