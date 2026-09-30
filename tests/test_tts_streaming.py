"""Tests for streaming TTS playback (hva/conversation.py speak_stream).

The stub "player" is just python reading stdin to EOF -- no audio
device needed. Real tts synthesis runs (short phrases, ~1-2 s each).
"""
import os
import sys
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import hva.conversation as conv
from hva.conversation import Speaker


def _stub_player():
    """A player that consumes stdin to EOF and exits 0."""
    return [sys.executable, "-c", "import sys; sys.stdin.buffer.read()"]


def test_streaming_activates_with_stub_player():
    sp = Speaker(player_cmd=_stub_player())
    assert sp.streaming is True
    assert sp.player_cmd == _stub_player()


def test_default_speaker_stays_simulated(monkeypatch):
    monkeypatch.delenv("HVA_AUDIO_PLAYER", raising=False)
    sp = Speaker()
    assert sp.streaming is False
    assert sp.play_stream("hello", "/tmp/x.mp3", 0.0) is None
    assert sp.is_playing(0.0) is False


def test_fallback_on_missing_player_binary(monkeypatch):
    monkeypatch.delenv("HVA_AUDIO_PLAYER", raising=False)
    sp = Speaker(player_cmd="definitely-not-a-real-player-xyz")
    assert sp.streaming is False
    assert sp.player_cmd is None


def test_env_var_configures_player(monkeypatch):
    monkeypatch.delenv("HVA_AUDIO_PLAYER", raising=False)
    assert Speaker().streaming is False
    monkeypatch.setenv(
        "HVA_AUDIO_PLAYER",
        f'{sys.executable} -c "import sys; sys.stdin.buffer.read()"')
    sp = Speaker()
    assert sp.streaming is True
    assert sp.player_cmd[0] == sys.executable


def _long_phrase():
    """Long enough that tts is still streaming when the handshake
    returns (first byte ~2 s, total ~7 s of audio)."""
    return "This is a deliberately longer test phrase. " * 8


def test_play_stream_writes_record_and_reports_playing(tmp_path):
    sp = Speaker(player_cmd=_stub_player())
    out = str(tmp_path / "reply_01.mp3")
    handle = sp.play_stream(_long_phrase(), out, 0.0)
    try:
        assert handle is not None
        assert sp.is_playing(0.0) is True  # genuinely mid-stream here
        assert sp.current_file == out
    finally:
        sp.stop()
    assert sp.is_playing(0.0) is False
    assert os.path.getsize(out) > 0
    with open(out, "rb") as f:
        assert f.read(2) == b"\xff\xf3"  # MP3 frame sync


def test_stop_pipe_kills_both_processes(tmp_path):
    sp = Speaker(player_cmd=_stub_player())
    out = str(tmp_path / "reply_02.mp3")
    handle = sp.play_stream(_long_phrase(), out, 0.0)
    assert handle.poll() is None  # player alive mid-stream
    tts_proc, player_proc = handle.tts_proc, handle.player_proc
    sp.stop()
    assert sp.is_playing(0.0) is False
    assert sp.interrupted is True
    # Both processes reaped: no zombies linger.
    assert tts_proc.poll() is not None
    assert player_proc.poll() is not None
    # Idempotent: second stop is a no-op.
    sp.stop()
    assert os.path.getsize(out) > 0


def test_stream_finishes_naturally(tmp_path):
    sp = Speaker(player_cmd=_stub_player())
    out = str(tmp_path / "reply_03.mp3")
    sp.play_stream("hi", out, 0.0)
    finished = False
    for _ in range(200):  # up to ~20 s for synthesis + drain
        if sp.check_finished(1.0):
            finished = True
            break
        time.sleep(0.1)
    assert finished is True
    assert sp.is_playing(1.0) is False
    assert sp.interrupted is False
    assert os.path.getsize(out) > 0


def test_tts_failure_raises_runtime_error(monkeypatch, tmp_path):
    # Point TTS_BIN at python itself: `python speak --stream ...` tries
    # to run a script named "speak" and exits nonzero with no bytes.
    monkeypatch.setattr(conv, "TTS_BIN", sys.executable)
    sp = Speaker(player_cmd=_stub_player())
    with pytest.raises(RuntimeError, match="tts --stream failed"):
        sp.play_stream("hello", str(tmp_path / "reply_04.mp3"), 0.0)
    assert sp.is_playing(0.0) is False


def test_stop_not_hung_by_lingering_tts_child(monkeypatch, tmp_path):
    """The tts CLI forks daemon children (fork+setsid) that inherit its
    stdout and can outlive it by ~a minute, so the pump must never wait
    on EOF from the tts pipe. stop() has to return promptly and the
    pump thread must exit silently (no unhandled thread exception)."""
    import threading

    child_sleep_s = 8  # longer than StreamPlayback's join backstop
    stub = tmp_path / "fake_tts.py"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys, time\n"
        "sys.stdout.buffer.write(b'\\xff\\xf3' + b'\\x00' * 100000)\n"
        "sys.stdout.buffer.flush()\n"
        "pid = os.fork()\n"  # fork at once: the child must exist before
        "if pid == 0:\n"  # stop() can kill the parent
        f"    time.sleep({child_sleep_s})\n"
        "    os._exit(0)\n"
        "time.sleep(2)\n"  # parent lingers briefly, then exits by itself
        "sys.exit(0)\n"  # child holds tts stdout open past the parent
    )
    stub.chmod(0o755)
    monkeypatch.setattr(conv, "TTS_BIN", str(stub))

    errors = []
    old_hook = threading.excepthook
    threading.excepthook = lambda args: errors.append(args.exc_value)
    try:
        sp = Speaker(player_cmd=_stub_player())
        out = str(tmp_path / "reply_child.mp3")
        h = sp.play_stream("x" * 50, out, 0.0)
        assert h is not None
        t0 = time.time()
        sp.stop()
        dt = time.time() - t0
        assert dt < 4, f"stop() hung on lingering child: {dt:.1f}s"
        assert not h.pump_thread.is_alive()
        assert errors == [], f"pump thread raised: {errors!r}"
        assert os.path.getsize(out) >= 100000
    finally:
        threading.excepthook = old_hook
