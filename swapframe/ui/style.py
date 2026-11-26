"""Themes + Qt stylesheet."""
from __future__ import annotations


def _theme(bg, panel, sunken, text, muted, border, accent, accent_hover, track, accent_text='#FFFFFF',
           danger='#E5484D', dark=False):
    return dict(bg=bg, panel=panel, sunken=sunken, text=text, muted=muted, border=border, accent=accent,
                accent_hover=accent_hover, accent_text=accent_text, danger=danger, track=track, dark=dark)


THEMES = {
    'Light': _theme('#F4F5F9', '#FFFFFF', '#EEF0F5', '#1B1D24', '#6A6F7E', '#E3E5EC', '#5B5BF0', '#4B4BE0',
                    '#D9DCE5'),
    'Dark': _theme('#121318', '#1B1D24', '#23252E', '#ECEDF2', '#949AAB', '#2B2E38', '#7C7CFF', '#8E8EFF',
                   '#353946', danger='#FF6369', dark=True),
    'Midnight': _theme('#0B1020', '#121A2E', '#1A2440', '#E6ECFF', '#8E9AC0', '#22304F', '#4F8BFF', '#6A9DFF',
                       '#2A3858', danger='#FF6B7A', dark=True),
    'Forest': _theme('#0F1612', '#16201A', '#1E2B23', '#E5F0E8', '#8FA597', '#26352C', '#3FBF7F', '#55CC90',
                     '#2E4036', accent_text='#06140C', danger='#FF6B6B', dark=True),
    'Graphite': _theme('#18181A', '#202023', '#2A2A2E', '#EDEDED', '#9A9AA0', '#333338', '#F2A33A', '#F5B357',
                       '#3B3B40', accent_text='#1A1204', danger='#FF6B6B', dark=True),
    'Rose': _theme('#FBF4F6', '#FFFFFF', '#F6E9EE', '#2A1C22', '#86707A', '#EEDDE4', '#E0457B', '#CC3A6D',
                   '#EBD3DC'),
    'Sand': _theme('#F7F3EC', '#FFFDF9', '#F0E9DD', '#2B241B', '#857A69', '#E8DECE', '#C07A2C', '#A9691F',
                   '#E3D7C3'),
    'Mint': _theme('#F1F8F6', '#FFFFFF', '#E4F2EE', '#16231F', '#6B827B', '#D6E9E3', '#17A589', '#12907A',
                   '#CFE5DE'),
}
THEME_NAMES = ['System'] + list(THEMES)


def resolve(name: str, system_dark: bool) -> dict:
    if name in THEMES:
        return THEMES[name]
    return THEMES['Dark' if system_dark else 'Light']


def stylesheet(c: dict, family: str) -> str:
    return f"""
* {{ font-family: '{family}'; font-size: 13px; color: {c['text']}; }}
QMainWindow, QWidget#root {{ background: {c['bg']}; }}
QFrame#panel {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: 14px; }}
QFrame#card {{ background: transparent; border: none; border-top: 1px solid {c['border']}; }}
QLabel {{ background: transparent; }}
QLabel#title {{ font-size: 16px; font-weight: 700; }}
QLabel#section {{ font-size: 11px; font-weight: 700; color: {c['muted']}; letter-spacing: 0.6px; }}
QLabel#muted {{ color: {c['muted']}; }}
QLabel#hint {{ color: {c['muted']}; font-size: 12px; }}
QLabel#value {{ color: {c['muted']}; font-size: 12px; min-width: 34px; }}

QSplitter::handle {{ background: transparent; }}
QSplitter::handle:hover {{ background: {c['border']}; border-radius: 3px; }}

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
QSlider::sub-page:horizontal:disabled {{ background: {c['track']}; }}
QSlider::handle:horizontal:disabled {{ border-color: {c['track']}; }}

QProgressBar {{ background: {c['track']}; border: none; border-radius: 3px; max-height: 6px; }}
QProgressBar::chunk {{ background: {c['accent']}; border-radius: 3px; }}

QListWidget {{ background: transparent; border: none; outline: none; }}
QListWidget::item {{ border-radius: 10px; padding: 3px; margin: 2px; }}
QListWidget::item:selected {{ background: {c['accent']}; }}
QListWidget::item:hover:!selected {{ background: {c['sunken']}; }}

QScrollArea {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 4px 2px; }}
QScrollBar::handle:vertical {{ background: {c['track']}; border-radius: 3px; min-height: 30px; }}
QScrollBar:horizontal {{ background: transparent; height: 8px; }}
QScrollBar::handle:horizontal {{ background: {c['track']}; border-radius: 3px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{ height: 0; width: 0;
    background: transparent; }}
QToolTip {{ background: {c['panel']}; color: {c['text']}; border: 1px solid {c['border']}; padding: 5px; }}
QMenu {{ background: {c['panel']}; border: 1px solid {c['border']}; padding: 4px; }}
QMenu::item {{ padding: 6px 14px; border-radius: 6px; }}
QMenu::item:selected {{ background: {c['sunken']}; }}
QMessageBox {{ background: {c['panel']}; }}
"""
