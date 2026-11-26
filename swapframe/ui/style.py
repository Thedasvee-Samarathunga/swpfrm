"""Palette + Qt stylesheet."""
from __future__ import annotations

LIGHT = dict(bg='#F4F5F9', panel='#FFFFFF', sunken='#EEF0F5', text='#1B1D24', muted='#6A6F7E',
             border='#E3E5EC', accent='#5B5BF0', accent_hover='#4B4BE0', accent_text='#FFFFFF',
             danger='#E5484D', track='#D9DCE5')
DARK = dict(bg='#121318', panel='#1B1D24', sunken='#23252E', text='#ECEDF2', muted='#949AAB',
            border='#2B2E38', accent='#7C7CFF', accent_hover='#8E8EFF', accent_text='#FFFFFF',
            danger='#FF6369', track='#353946')


def stylesheet(c: dict, family: str) -> str:
    return f"""
* {{ font-family: '{family}'; font-size: 13px; color: {c['text']}; }}
QMainWindow, QWidget#root {{ background: {c['bg']}; }}
QFrame#panel {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: 14px; }}
QLabel {{ background: transparent; }}
QLabel#title {{ font-size: 15px; font-weight: 600; }}
QLabel#section {{ font-size: 12px; font-weight: 600; color: {c['muted']}; }}
QLabel#muted {{ color: {c['muted']}; }}
QLabel#hint {{ color: {c['muted']}; font-size: 12px; }}

QFrame#drop {{ background: {c['sunken']}; border: 1.5px dashed {c['border']}; border-radius: 12px; }}
QFrame#drop[hover="true"] {{ border-color: {c['accent']}; }}

QPushButton {{ background: {c['sunken']}; border: 1px solid {c['border']}; border-radius: 9px;
              padding: 7px 14px; font-weight: 500; }}
QPushButton:hover {{ border-color: {c['accent']}; }}
QPushButton:disabled {{ color: {c['muted']}; }}
QPushButton#primary {{ background: {c['accent']}; color: {c['accent_text']}; border: none; font-weight: 600;
                      padding: 10px 22px; }}
QPushButton#primary:hover {{ background: {c['accent_hover']}; }}
QPushButton#primary:disabled {{ background: {c['track']}; color: {c['muted']}; }}
QPushButton#ghost {{ background: transparent; border: none; color: {c['muted']}; padding: 4px 8px; }}
QPushButton#ghost:hover {{ color: {c['text']}; }}
QPushButton#seg {{ border-radius: 8px; padding: 6px 0px; background: transparent; border: none; }}
QPushButton#seg:checked {{ background: {c['panel']}; color: {c['accent']}; font-weight: 600; }}
QFrame#segwrap {{ background: {c['sunken']}; border-radius: 10px; }}

QComboBox {{ background: {c['sunken']}; border: 1px solid {c['border']}; border-radius: 9px; padding: 6px 10px; }}
QComboBox:hover {{ border-color: {c['accent']}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {c['panel']}; border: 1px solid {c['border']};
                               selection-background-color: {c['sunken']}; selection-color: {c['text']};
                               outline: none; padding: 4px; }}

QSlider::groove:horizontal {{ height: 4px; background: {c['track']}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {c['accent']}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {c['panel']}; border: 2px solid {c['accent']}; width: 12px; height: 12px;
                              margin: -6px 0; border-radius: 8px; }}

QProgressBar {{ background: {c['track']}; border: none; border-radius: 3px; max-height: 6px; }}
QProgressBar::chunk {{ background: {c['accent']}; border-radius: 3px; }}

QListWidget {{ background: transparent; border: none; outline: none; }}
QListWidget::item {{ border-radius: 10px; padding: 3px; margin: 2px; }}
QListWidget::item:selected {{ background: {c['accent']}; }}
QListWidget::item:hover:!selected {{ background: {c['sunken']}; }}

QScrollArea {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 8px; }}
QScrollBar::handle:vertical {{ background: {c['track']}; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QToolTip {{ background: {c['panel']}; color: {c['text']}; border: 1px solid {c['border']}; padding: 5px; }}
QMenu {{ background: {c['panel']}; border: 1px solid {c['border']}; padding: 4px; }}
QMenu::item {{ padding: 6px 14px; border-radius: 6px; }}
QMenu::item:selected {{ background: {c['sunken']}; }}
"""
