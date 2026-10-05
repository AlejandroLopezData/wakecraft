"""Lectura de una sola tecla desde el terminal, sin dependencias externas.

``read_key()`` devuelve:
    "enter"      para ENTER
    "esc"        para ESC y para teclas especiales (flechas, F1..F12, etc.)
    "<letra>"    en minúscula para cualquier otro carácter

Lanza ``KeyboardInterrupt`` con Ctrl+C y ``EOFError`` si stdin se cierra.
Si stdin no es un terminal interactivo (por ejemplo una tubería), lee una
línea completa: línea vacía = "enter", si no, su primer carácter.
"""

from __future__ import annotations

import sys


def read_key() -> str:
    if not sys.stdin.isatty():
        return _read_key_line()
    if sys.platform == "win32":
        return _read_key_windows()
    return _read_key_posix()


def _read_key_line() -> str:
    line = input()  # EOFError si no hay más entrada
    line = line.strip().lower()
    return "enter" if not line else line[0]


def _read_key_posix() -> str:
    import os
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)  # sin eco ni modo línea; Ctrl+C sigue funcionando
        termios.tcflush(fd, termios.TCIFLUSH)  # descarta teclas pulsadas de antemano
        data = os.read(fd, 32)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    if not data:
        raise EOFError
    first = data[:1]
    if first == b"\x1b":
        return "esc"  # ESC o secuencia de escape (flechas, F-keys...)
    if first in (b"\r", b"\n"):
        return "enter"
    if first == b"\x03":
        raise KeyboardInterrupt
    if first == b"\x04":
        raise EOFError
    return data.decode("utf-8", errors="ignore")[:1].lower()


def _read_key_windows() -> str:
    import msvcrt

    while msvcrt.kbhit():  # descarta teclas pulsadas de antemano
        msvcrt.getwch()

    ch = msvcrt.getwch()
    if ch in ("\x00", "\xe0"):  # tecla especial: el segundo código va aparte
        msvcrt.getwch()
        return "esc"
    if ch == "\r":
        return "enter"
    if ch == "\x03":
        raise KeyboardInterrupt
    if ch == "\x1b":
        return "esc"
    return ch.lower()