"""Fast tests for the conversation captions (no models, GPU, microphone or Ollama needed).

    pytest -q
"""
from __future__ import annotations

import _thread
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from spectre import conv_eval, room
from spectre import conversation as cv
from spectre import live

SR = live.SR
RNG = np.random.default_rng(0)


def tone(freq: float, seconds: float, sr: int = SR, amp: float = 0.1) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def talk(freq: float, seconds: float, sr: int = SR) -> np.ndarray:
    """A tone with the tiny gaps of real speech (60 ms every 260 ms)."""
    gate = (np.arange(int(seconds * sr)) / sr) % 0.26 < 0.2
    return tone(freq, seconds, sr) * gate


def quiet(seconds: float, sr: int = SR) -> np.ndarray:
    return (1e-4 * RNG.standard_normal(int(seconds * sr))).astype(np.float32)


# A three-person conversation. Each line is a tone: the band says whose voice it is, the exact frequency which line.
SCRIPT = [
    ("Joana", 200, "Olá João! Há quanto tempo não te via."),
    ("João", 300, "Olá Joana, tudo bem contigo?"),
    ("Joana", 205, "Tudo ótimo. Olha, apresento-te o Pedro."),
    ("Pedro", 400, "Muito prazer, João. Eu sou o Pedro."),
    ("João", 305, "O prazer é meu, Pedro. A Rita também vem jantar?"),
    ("Joana", 210, "A Rita chega mais tarde."),
    ("Pedro", 405, "Então vamos andando."),
]
WHO = [who for who, *_ in SCRIPT]
BANDS = [(190, 230), (290, 330), (390, 430)]

# What a small local LLM answers for each line, including a typical mistake: a name copied from the context.
ANSWERS = {
    SCRIPT[0][2]: [("João", "addressed")],
    SCRIPT[1][2]: [("Joana", "addressed"), ("João", "mentioned")],     # "João" is not in this line
    SCRIPT[2][2]: [("Pedro", "introduced")],
    SCRIPT[3][2]: [("João", "addressed"), ("Pedro", "self")],
    SCRIPT[4][2]: [("Pedro", "addressed"), ("Rita", "mentioned")],
    SCRIPT[5][2]: [("Rita", "mentioned")],
    SCRIPT[6][2]: [],
}


def as_json(names) -> str:
    return json.dumps({"names": [{"name": n, "role": r} for n, r in names]}, ensure_ascii=False)


def dominant(wav: np.ndarray) -> float:
    return float(np.fft.rfftfreq(len(wav), 1 / SR)[np.argmax(np.abs(np.fft.rfft(wav)))])


class FakeEmbedder:
    """Voice = the band where the energy is (a turn with two voices gives a mixed embedding)."""
    model_id, device = "run_x/best.pt", SimpleNamespace(type="cpu")

    def default_threshold(self):
        return 0.5

    def __call__(self, wavs):
        out = []
        for w in [wavs] if isinstance(wavs, np.ndarray) else wavs:
            p = np.abs(np.fft.rfft(w)) ** 2
            f = np.fft.rfftfreq(len(w), 1 / SR)
            e = [p[(f >= lo) & (f <= hi)].sum() for lo, hi in BANDS]
            v = np.sqrt(np.array(e + [max(p.sum() - sum(e), 0.0)]))
            out.append(v / (np.linalg.norm(v) + 1e-12))
        return np.stack(out)


class FakeTranscriber:
    """The scripted text of the line whose frequency is in the audio ("" for noise)."""

    def __init__(self, on_call=None):
        self.on_call = on_call

    def __call__(self, audio):
        if self.on_call:
            self.on_call()
        f = dominant(audio)
        return next((text for _, freq, text in SCRIPT if abs(f - freq) < 2), "")


class FakeReader:
    def __init__(self):
        self.calls = []

    def check(self):
        return None

    def warm(self):
        pass

    def ask(self, context, line):
        self.calls.append((list(context), line))
        return as_json(ANSWERS.get(line, []))


def dialogue(sr: int = SR, gap: float = 1.0, seconds: float = 1.6) -> np.ndarray:
    parts = [quiet(1.0, sr)]
    for _, f, _ in SCRIPT:
        parts += [tone(f, seconds, sr), quiet(gap, sr)]
    return np.concatenate(parts)


def turns_of(wav: np.ndarray, sr: int, **kw) -> list[cv.Turn]:
    seg, block = cv.TurnSegmenter(sr, **kw), int(0.1 * sr)
    return [t for s in range(0, len(wav), block) for t in seg.push(wav[s: s + block])] + seg.flush()


RULES = dict(threshold=0.6, cut=0.6, merge_margin=0.2, follow_below_s=1.0, new_margin=0.3, min_cut_side_s=0.6)


def make_conversation(reader=None, enrolled=None) -> cv.Conversation:
    return cv.Conversation(FakeEmbedder(), FakeTranscriber(), reader, cv.Voices(cv.VoiceRules(**RULES), enrolled))


def process(conv: cv.Conversation, turns: list[cv.Turn]) -> None:
    """Like the Runner, but one step at a time: each line is read before the next turn is heard."""
    for turn in turns:
        for part in conv.split(turn):
            if (line := conv.hear(part)) is not None:
                conv.read(line)


def labels(conv: cv.Conversation) -> list[str]:
    return [conv.label(ln.speaker) for ln in conv.lines]


def line(spk: int, names=None, text: str | None = "…") -> cv.Line:
    return cv.Line(0.0, 1.0, spk, 1.0, text, names)


# --------------------------------------------------------------------------- turns
def test_segmenter_cuts_turns_at_pauses_at_48k():
    sr = 48000
    wav = np.concatenate([quiet(1, sr), tone(200, 1.5, sr), quiet(1, sr), tone(300, 1, sr), quiet(0.3, sr),
                          tone(300, 1, sr), quiet(1, sr), tone(500, 0.1, sr), quiet(1, sr)])
    a, b = turns_of(wav, sr)                     # the 0.3 s pause does not end a turn; a 0.1 s click is not speech
    assert a.start == pytest.approx(0.75, abs=0.03) and a.end == pytest.approx(2.6, abs=0.03)
    assert a.speech_s == pytest.approx(1.5, abs=0.05) and abs(len(a.audio) - (a.end - a.start) * SR) <= 2
    assert b.speech_s == pytest.approx(2.0, abs=0.06)
    assert len(b.cuts) == 1 and b.start + b.cuts[0] / SR == pytest.approx(4.65, abs=0.03)   # where it may be cut


def test_long_turns_end_at_a_short_pause_or_at_the_limit():
    monologue = np.concatenate([quiet(1)] + [np.concatenate([tone(200, 1.7), quiet(0.3)]) for _ in range(5)] + [quiet(1)])
    first, second = turns_of(monologue, SR)
    assert 6.0 <= first.end - first.start <= 6.4 and first.end <= second.start   # cut at the first pause after 6 s
    nonstop = np.concatenate([quiet(1), talk(200, 20), quiet(1)])
    first, second = turns_of(nonstop, SR)
    assert first.end - first.start == pytest.approx(15.0) and second.end == pytest.approx(21.06, abs=0.05)


def test_flush_returns_what_was_being_said():
    seg = cv.TurnSegmenter(SR)
    assert seg.push(np.concatenate([quiet(1), tone(200, 1.2)])) == [] and seg.speaking > 1.0
    assert len(seg.recent_audio()) > SR
    (turn,) = seg.flush()
    assert turn.speech_s == pytest.approx(1.2, abs=0.05) and seg.speaking == 0 and seg.flush() == []


def test_quick_replies_are_cut_where_the_voice_changes():
    emb = FakeEmbedder()
    audio = np.concatenate([tone(200, 1.5), quiet(0.3), tone(300, 1.5), quiet(0.3), tone(210, 1.5)])   # A, B, A
    turn = cv.Turn(10.0, 10.0 + len(audio) / SR, audio, 4.5, [int(1.65 * SR), int(3.45 * SR)])
    parts = cv.split_by_voice(turn, emb, 0.6)
    assert [round(p.start, 2) for p in parts] == [10.0, 11.65, 13.45]
    assert [round(dominant(p.audio)) for p in parts] == [200, 300, 210]
    assert sum(p.speech_s for p in parts) == pytest.approx(4.5)
    same = cv.Turn(0, 3.3, np.concatenate([tone(200, 1.5), quiet(0.3), tone(205, 1.5)]), 3.0, [int(1.65 * SR)])
    assert cv.split_by_voice(same, emb, 0.6)[0] is same                      # one voice: no cut
    edge = cv.Turn(0, 2.0, np.concatenate([tone(200, 0.3), quiet(0.2), tone(300, 1.5)]), 1.8, [int(0.4 * SR)])
    assert cv.split_by_voice(edge, emb, 0.6)[0] is edge                      # too little speech to judge


# --------------------------------------------------------------------------- voices
def unit(v):
    return v / np.linalg.norm(v)


def test_voices_new_speakers_merge_and_forget():
    a, b, other = np.eye(4)[0], np.eye(4)[1], np.eye(4)[3]
    v = cv.Voices(cv.VoiceRules(threshold=0.9, merge_margin=-0.15))       # merge at 0.75
    assert v.assign(a, 2.0) == (0, False) and v.assign(unit(a + 0.2 * b), 1.0) == (0, False)   # same speaker
    assert v.assign(b, 1.0) == (1, False) and v.clusters[1]["label"] == "Speaker 2"
    assert v.assign(unit(a + 0.6 * b), 1.0) == (2, False)                   # not sure yet: a new speaker...
    assert v.merge() == {2: 0} and sorted(v.clusters) == [0, 1]             # ...that turns out to be the first
    noise, _ = v.assign(other, 1.2)
    assert v.clusters[noise]["label"] == "Speaker 4"
    v.unassign(noise, other, 1.2)                                           # it was only noise
    assert sorted(v.clusters) == [0, 1] and v.next_label == 4
    assert v.closest(b) == (1, pytest.approx(1.0))


def test_short_turns_follow_the_closest_voice():
    a, b, c = np.eye(4)[0], np.eye(4)[1], np.eye(4)[2]
    v = cv.Voices(cv.VoiceRules(threshold=0.6, new_margin=0.3))             # a new voice below 0.3
    lines = [cv.Line(0, 2, *v.assign(a, 2.0)[:1], 2.0, emb=a)]
    short = unit(0.8 * a + c)                                               # 0.62 to A: the closest voice so far
    spk, follows = v.assign(short, 0.5)
    assert (spk, follows) == (0, True) and np.allclose(v.clusters[0]["sum"], 2 * a)   # A is not changed by it
    lines.append(cv.Line(2, 2.5, spk, 0.5, emb=short, follows=True))
    assert v.assign(b, 0.4) == (1, False)                                   # clearly nobody known: a new voice
    spk, _ = v.assign(c, 2.0)                                               # a long turn by someone new...
    lines.append(cv.Line(3, 5, spk, 2.0, emb=c))
    cv.relabel(lines, v)
    assert lines[1].speaker == spk                                          # ...was who said the short turn


def test_room_compensation():
    a, b, r = np.eye(4)[0], np.eye(4)[1], np.eye(4)[3]
    e1, e2 = unit(a + 2 * r), unit(b + 2 * r)                               # two people, one shared room
    assert e1 @ e2 > 0.75
    rules = cv.VoiceRules(room=r[None, :])
    p1, p2 = rules.project(np.stack([e1, e2]))
    assert abs(p1 @ p2) < 1e-6 and np.allclose(np.linalg.norm([p1, p2], axis=1), 1)
    assert np.allclose(cv.VoiceRules().project(e1), e1)                     # without a calibration: unchanged


def test_calibration_file(tmp_path):
    run = tmp_path / "run_x"
    run.mkdir()
    assert cv.VoiceRules.for_model(run / "best.pt", "run_x/best.pt") == cv.VoiceRules()
    (run / cv.CALIBRATION).write_text(json.dumps(
        {"model": "run_x/best.pt", "threshold": 0.4, "cut": 0.3, "merge_margin": 0.2, "follow_below_s": 1.0,
         "new_margin": 0.3, "min_cut_side_s": 0.8, "room_directions": [[0, 0, 0, 1.0]], "accuracy": 0.913}))
    rules = cv.VoiceRules.for_model(run / "best.pt", "run_x/best.pt")
    assert (rules.threshold, rules.cut, rules.merge, rules.new_voice) == (0.4, 0.3, pytest.approx(0.6), pytest.approx(0.1))
    assert rules.room.shape == (1, 4) and "91 %" in rules.summary
    assert cv.VoiceRules.for_model(run / "best.pt", "other/best.pt").room is None   # made for another model


def test_enrolled_voices_are_fixed():
    a, b = np.eye(4)[0], np.eye(4)[1]
    v = cv.Voices(cv.VoiceRules(threshold=0.6, merge_margin=-0.1), {"Ana": a, "Rui": unit(a + 0.5 * b)})
    assert v.merge() == {}                                                  # two enrolled people never merge
    assert v.assign(unit(a + 0.1 * b), 3.0) == (0, False) and np.allclose(v.clusters[0]["sum"], a)   # unchanged
    w = cv.Voices(cv.VoiceRules(threshold=0.9, merge_margin=-0.2), {"Ana": a})
    assert w.assign(unit(a + 0.7 * b), 2.0) == (1, False)
    assert w.merge() == {1: 0} and w.fixed_names() == {0: "Ana"}            # merging into Ana keeps Ana


# --------------------------------------------------------------------------- reading names
def test_parse_names_keeps_only_names_written_in_the_line():
    answer = 'Sure! {"names": [{"name": "Joana", "role": "addressed"}, {"name": "João", "role": "mentioned"}]}'
    assert cv.parse_names(answer, "Olá, Joana. Tudo bem?") == [("Joana", "addressed")]
    assert cv.written_in("João", "ola joao!") and cv.written_in("Inês", "a Ines chegou")
    assert cv.written_in("Catarina", "Olá Katarina")                           # a small spelling difference
    assert not cv.written_in("Joana", "Olá João") and not cv.written_in("Pedro", "O prazer é meu, Tedo.")
    both = '{"names": [{"name": "Pedro", "role": "mentioned"}, {"name": "Pedro", "role": "self"}]}'
    assert cv.parse_names(both, "Eu sou o Pedro") == [("Pedro", "self")]
    assert cv.parse_names("no idea", "Olá João") == [] and cv.parse_names("{broken", "Olá João") == []
    assert cv.parse_names('{"names": [{"name": "João", "role": "friend"}]}', "Olá João") == []


def test_own_name_needs_a_self_introduction():
    for line, name in [("Eu sou o Pedro.", "Pedro"), ("Chamo-me Ana.", "Ana"), ("O meu nome é Rui.", "Rui"),
                       ("Olá, aqui é a Joana!", "Joana"), ("Muito prazer, João. Eu sou o Pedro.", "Pedro")]:
        assert cv.says_own_name(name, line), line
    assert not cv.says_own_name("Tedo", "O prazer é meu, Tedo.")             # a vocative, not a name of one's own
    assert not cv.says_own_name("João", "Muito prazer, João. Eu sou o Pedro.")
    misread = '{"names": [{"name": "Tedo", "role": "self"}, {"name": "Rita", "role": "mentioned"}]}'
    assert cv.parse_names(misread, "O prazer é meu, Tedo. A Rita também vem?") == [("Tedo", "replied"), ("Rita", "mentioned")]
    assert cv.parse_names('{"names": [{"name": "Joana", "role": "self"}]}', "Olá Joana, tudo bem contigo?") == \
        [("Joana", "addressed")]                                             # what gemma3:4b got wrong
    assert cv.parse_names('{"names": [{"name": "Ana", "role": "self"}]}', "A Ana chegou.") == []


def test_vocatives():
    assert cv.vocative("Pedro", "O prazer é meu, Pedro.") == "answer"         # Pedro spoke before
    assert cv.vocative("Maria", "Obrigado, Maria!") == "answer"
    for name, line in [("João", "Olá João! Há quanto tempo."), ("Joana", "Olá, Joana. Tudo bem?"),
                       ("Pedro", "Pedro, anda cá."), ("Pedro", "Anda cá, Pedro!"), ("João", "Muito prazer, João.")]:
        assert cv.vocative(name, line) == "call", line
    for name, line in [("Pedro", "Eu sou o Pedro."), ("Rita", "A Rita chega mais tarde."),
                       ("Pedro", "Apresento-te o Pedro, trabalha comigo.")]:
        assert cv.vocative(name, line) is None, line


def test_a_name_closing_a_reply_points_back():
    lines = [line(0, [("Pedro", "introduced")]), line(1, [("Pedro", "self")]),
             line(2, [("Pedro", "replied")]), line(0, [])]                  # "…, Pedro." then Joana speaks
    names = cv.resolve_names(lines, {})
    assert names[1].name == "Pedro" and len(names) == 1                    # Joana (voice 0) gets nothing
    assert names[1].evidence[-1] == 'line 3: called "Pedro" in the reply to line 2'


def test_rules_read_names_without_an_llm():
    read = lambda line: cv.parse_names(cv.RuleReader().ask([], line), line)
    assert read("Ó João, há quanto tempo não te via?") == [("João", "addressed")]
    assert read("Olá, Joana. Tudo bem contigo?") == [("Joana", "addressed")]
    assert read("Tudo ótimo. Olha, apresente que o Pedro trabalha comigo.") == [("Pedro", "introduced")]
    assert read("Apresento-te a Inês e o Rui.") == [("Inês", "introduced"), ("Rui", "introduced")]
    assert read("Muito prazer, João. Eu sou o Pedro.") == [("João", "addressed"), ("Pedro", "self")]
    assert read("O prazer é meu, Tedo. A Rita também vem?") == [("Tedo", "replied"), ("Rita", "mentioned")]
    assert read("Chamo-me Ana.") == [("Ana", "self")]
    assert read("Eu sou Pedro.") == [("Pedro", "self")]                # no article: fine for common first names…
    assert read("Oh João, há quanto tempo não te via?") == [("João", "addressed")]   # Whisper's spelling of "Ó"
    assert read("Oh, a Rita também vem?") == [("Rita", "mentioned")]
    for line in ("Então, vamos andando.", "Eu sou do Porto.", "Obrigado, Senhor.", "A camisola rosa é tua.",
                 "Fui a Lisboa ontem.", "Sou Benfica desde pequeno."):        # …not for any capitalised word
        assert read(line) == [], line


def test_rules_name_the_whisper_transcript():
    """The lines Whisper wrote on the user's PC, with the voices found there: rules alone find the same names."""
    said = [(0, "Ó João, há quanto tempo não te via?"), (1, "Olá, Joana. Tudo bem contigo? Que bom ver te."),
            (0, "Tudo ótimo. Olha, apresente que o Pedro trabalha comigo."), (2, "Muito prazer, João."),
            (2, "Eu sou o Pedro."), (1, "O prazer é meu, Tedo. A Rita também vem juntar conosco?"),
            (0, "A Rita Chega mais tarde ficou presa no trabalho."), (2, "Então, vamos andando, que estou cheio de fome.")]
    lines = [line(spk, cv.parse_names(cv.RuleReader().ask([], text), text), text) for spk, text in said]
    assert {s: n.name for s, n in cv.resolve_names(lines, {}).items()} == {0: "Joana", 1: "João", 2: "Pedro"}


# --------------------------------------------------------------------------- deciding who is who
def test_the_two_line_example():
    lines = [line(0, [("João", "addressed")]), line(1, None, None)]   # "Olá João" · someone else starts answering
    names = cv.resolve_names(lines, {})
    assert names[1].name == "João" and 0 not in names                 # named while his first line is transcribed
    lines[1].text, lines[1].names = "Olá Joana, tudo bem?", [("Joana", "addressed")]
    names = cv.resolve_names(lines, {})
    assert (names[0].name, names[1].name) == ("Joana", "João")
    assert names[1].evidence == ['line 2: spoke right after "João" was called (line 1)']
    assert names[0].evidence == ['line 2: called "Joana" in the reply to line 1']


def test_rules_mentions_self_and_enrolled():
    assert cv.resolve_names([line(0, [("Rita", "mentioned")]), line(1, [])], {}) == {}   # Rita is not here
    lines = [line(0, [("Pedro", "self")]), line(1, []), line(0, [("Pedro", "mentioned")])]
    assert cv.resolve_names(lines, {}) == {}                          # "Eu sou o Pedro" … "o Pedro": contradictory
    lines[1].names = [("Pedro", "addressed")]                         # …until someone else calls him Pedro
    names = cv.resolve_names(lines, {})
    assert names[0].name == "Pedro" and names[0].score == 3 + 2 + 2 - 3
    assert len(names[0].evidence) == 3                                # only the reasons for, not against
    names = cv.resolve_names([line(0, [("João", "addressed")]), line(1, [])], {7: "João"})
    assert names[7].name == "João" and 1 not in names                 # João is enrolled: his voice keeps the name
    names = cv.resolve_names([line(0, [("Joao", "addressed")]), line(1, [("Joana", "addressed")]),
                              line(0, [("João", "addressed")])], {})
    assert names[1].name == "João" and names[1].score == 4            # evidence adds up; accented spelling shown
    names = cv.resolve_names([line(0, []), line(1, [("Ana", "addressed")]), line(2, [])], {})
    assert names[0].name == "Ana" and 2 not in names                  # one voice per name: the earliest evidence wins


# --------------------------------------------------------------------------- the conversation
def test_conversation_names_people_from_what_they_say():
    reader = FakeReader()
    conv = make_conversation(reader)
    while_transcribing = []
    conv.transcriber = FakeTranscriber(lambda: while_transcribing.append(conv.label(conv.lines[-1].speaker)))
    process(conv, turns_of(dialogue(), SR))
    assert [ln.text for ln in conv.lines] == [text for *_, text in SCRIPT]
    assert labels(conv) == WHO and "Rita" not in labels(conv)
    assert while_transcribing[:2] == ["Speaker 1", "João"]            # João's name shows before his words do
    assert reader.calls[2] == ([SCRIPT[0][2], SCRIPT[1][2]], SCRIPT[2][2])   # previous lines as context
    assert conv.speaking is None and conv.reader_error is None


def test_quick_replies_and_background_workers_at_48k():
    conv = make_conversation(FakeReader())
    runner = cv.Runner(conv)
    for turn in turns_of(dialogue(48000, gap=0.3), 48000):          # answers 0.3 s after the other person
        runner.turns.put(turn)
    runner.finish()
    assert [ln.text for ln in conv.lines] == [text for *_, text in SCRIPT] and labels(conv) == WHO


def test_noise_is_dropped_and_forgotten():
    conv = make_conversation(FakeReader())
    assert conv.hear(cv.Turn(0, 1, tone(600, 1.0), 1.0)) is None       # a door closing: no words
    assert conv.lines == [] and conv.voices.clusters == {} and conv.voices.next_label == 1


def test_conversation_with_the_rule_reader():
    conv = make_conversation(cv.RuleReader())
    process(conv, turns_of(dialogue(), SR))
    assert labels(conv) == WHO


def test_enrolled_people_and_no_llm():
    conv = make_conversation(FakeReader(), enrolled={"João": np.eye(4)[1]})
    process(conv, turns_of(dialogue(), SR))
    assert labels(conv) == WHO and conv.names[0].evidence == ["recognised by voice (enrolled)"]
    conv = make_conversation(None)                                    # --llm none: voices without names
    process(conv, turns_of(dialogue(), SR))
    assert labels(conv) == ["Speaker 1", "Speaker 2", "Speaker 1", "Speaker 3", "Speaker 2", "Speaker 1", "Speaker 3"]


def test_llm_errors_do_not_stop_the_captions():
    class Broken(FakeReader):
        def ask(self, context, line):
            raise TimeoutError("timed out")
    conv = make_conversation(Broken())
    process(conv, turns_of(dialogue(), SR)[:2])
    assert [ln.text for ln in conv.lines] == [SCRIPT[0][2], SCRIPT[1][2]] and "timed out" in conv.reader_error
    assert labels(conv) == ["Speaker 1", "Speaker 2"]


def test_peek_shows_who_is_speaking_before_the_turn_ends():
    conv = make_conversation(FakeReader())
    process(conv, turns_of(dialogue(), SR)[:1])                       # Joana: "Olá João! …"
    conv.peek(tone(302, 1.5))                                         # a voice never heard before starts talking
    assert conv.speaking == (1, "João")                               # it is João, before his first line ends
    conv.peek(tone(450, 1.5))
    assert conv.speaking == (1, "João")                               # (whoever answers first)
    process(conv, turns_of(dialogue(), SR)[1:2])
    conv.peek(tone(205, 1.5))
    assert conv.speaking == (0, "Joana")
    conv.peek(tone(450, 1.5))
    assert conv.speaking == (2, "a new voice")                       # "Olá Joana" was an answer to Joana


# --------------------------------------------------------------------------- output
def test_plain_output_transcript_and_live_panel():
    pytest.importorskip("rich")
    from rich.console import Console
    conv = make_conversation(FakeReader())
    printed = []
    conv.on_change = cv.PlainPrinter(conv, printed.append)
    process(conv, turns_of(dialogue(), SR))
    assert printed[0] == "[00:00.7] Speaker 1: " + SCRIPT[0][2]
    assert printed[1].endswith("João: " + SCRIPT[1][2])
    assert printed[2] == '   ↳ Speaker 2 is João — line 2: spoke right after "João" was called (line 1)'
    assert printed[3].startswith("   ↳ Speaker 1 is Joana")
    text = cv.transcript(conv)
    assert "Joana: Olá João!" in text and "\nHow the names were found:" in text
    assert "Rita" not in text.split("How the names")[1]
    console = Console(width=60, height=12, record=True)
    console.print(cv.render(conv, "listening", 60, 12))
    out = console.export_text()
    assert "Então vamos andando." in out and "Olá João!" not in out and len(out.splitlines()) <= 12
    assert "voices: Joana, João, Pedro" in out


# --------------------------------------------------------------------------- calibration on simulated conversations
def test_room_simulation_and_score():
    rng = np.random.default_rng(1)
    x = tone(200, 2.0)
    y = room.apply_room(x, room.make_room(rng), rng)
    assert len(y) == len(x) and y.dtype == np.float32 and 0.01 < np.abs(y).max() <= 0.95
    truth = [(0.0, 2.0, "A"), (2.5, 4.0, "B")]
    assert room.score([(0.0, 2.0, 7), (2.5, 4.0, 9)], truth) == {"accuracy": 1.0, "speakers": 2, "labels": 2, "merged": 0}
    merged = room.score([(0.0, 2.0, 7), (2.5, 4.0, 7)], truth)
    assert merged["merged"] == 1 and merged["accuracy"] == pytest.approx(2.0 / 3.5)


def test_calibration_runs_the_live_rules():
    conv = {"truth": [], "turns": turns_of(dialogue(), SR)}
    t = 1.0
    for who, *_ in SCRIPT:
        conv["truth"].append((t, t + 1.6, who))
        t += 2.6
    embed = conv_eval.CachedEmbedder(FakeEmbedder())
    lines = conv_eval.run_rules(conv, embed, cv.VoiceRules(**RULES))
    assert len(lines) == 7 and room.score(lines, conv["truth"])["accuracy"] == pytest.approx(1.0, abs=0.01)
    n = len(embed.cache)
    conv_eval.run_rules(conv, embed, cv.VoiceRules(**RULES))
    assert len(embed.cache) == n                                            # the second pass uses the cache
    rules, result, base = conv_eval.calibrate([conv], embed, np.eye(4)[3:4], log=lambda *a: None)
    assert result["accuracy"] > 0.99 and result["merged"] == 0 and base["accuracy"] > 0.99


# --------------------------------------------------------------------------- Ollama
class FakeOllama(BaseHTTPRequestHandler):
    requests: list = []

    def do_GET(self):
        self._reply({"models": [{"name": "gemma3:12b"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.requests.append(body)
        self._reply({"message": {"role": "assistant", "content": as_json([("João", "addressed")]) if body["messages"] else ""}})

    def _reply(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def ollama():
    server = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_ollama_reader(ollama):
    reader = cv.OllamaReader("gemma3:12b", host=ollama)
    assert reader.check() is None
    assert "ollama pull gemma3:4b" in cv.OllamaReader("gemma3:4b", host=ollama).check()
    reader.warm()
    assert cv.parse_names(reader.ask(["Bom dia."], "Olá João"), "Olá João") == [("João", "addressed")]
    warm, ask = FakeOllama.requests[-2:]
    assert warm["messages"] == [] and ask["format"] == cv.SCHEMA and ask["options"]["temperature"] == 0
    assert ask["messages"][0]["content"] == cv.SYSTEM_PROMPT and "Bom dia." in ask["messages"][1]["content"]


def test_ollama_not_running():
    problem = cv.OllamaReader("gemma3:12b", host="http://127.0.0.1:9").check()
    assert "not running" in problem and "https://ollama.com" in problem


# --------------------------------------------------------------------------- command line
@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(cv.live, "Embedder", lambda ckpt, device=None: FakeEmbedder())
    monkeypatch.setattr(cv, "Transcriber", lambda *a, **k: FakeTranscriber())
    monkeypatch.setattr(cv, "make_reader", lambda spec: FakeReader())


def test_cli_over_a_recording(tmp_path, fakes, capsys):
    path, bank, run = tmp_path / "conversa.flac", tmp_path / "speakers.json", tmp_path / "run_x"
    sf.write(path, dialogue(44100, seconds=4.5), 44100)              # long enough to remember the voices
    run.mkdir()
    (run / cv.CALIBRATION).write_text(json.dumps({"model": "run_x/best.pt", **RULES, "room_directions": [],
                                                  "accuracy": 0.9}))
    cv.main(["--file", str(path), "--ckpt", str(run / "best.pt"), "--bank", str(bank),
             "--save", str(tmp_path / "conversa.txt"), "--remember"])
    out = capsys.readouterr().out
    assert "calibrated for one-microphone conversations" in out and "same speaker above 0.60" in out
    assert "↳ Speaker 2 is João" in out and "How the names were found:" in out
    assert "Pedro: Então vamos andando." in (tmp_path / "conversa.txt").read_text(encoding="utf-8")
    assert sorted(live.SpeakerBank(bank).speakers) == ["Joana", "João", "Pedro"]


def test_cli_without_a_microphone(fakes, monkeypatch):
    class NoMic:
        PortAudioError = RuntimeError

        def query_devices(self, device=None, kind=None):
            if kind == "input":
                raise RuntimeError("Error querying device -1")
            return []

        def query_hostapis(self):
            return ()
    monkeypatch.setattr(cv.live, "_sounddevice", lambda: NoMic())
    with pytest.raises(SystemExit) as err:
        cv.main([])
    assert "No microphone found" in str(err.value)


def test_cli_live_microphone_until_ctrl_c(fakes, monkeypatch, capsys):
    sr = 44100
    wav = dialogue(sr)[: int(5.5 * sr)]                               # two lines; Ctrl+C right after the second

    class FakeMic:
        PortAudioError = RuntimeError

        def query_devices(self, device=None, kind=None):
            return {"name": "Fake microphone", "default_samplerate": float(sr)}

        def InputStream(self, callback, blocksize, samplerate, **kw):
            assert samplerate == sr and blocksize == sr // 10

            class Stream:
                def __enter__(self):
                    def play():
                        for s in range(0, len(wav), blocksize):
                            callback(wav[s: s + blocksize, None], blocksize, None, None)
                        time.sleep(1.0)
                        _thread.interrupt_main()                      # Ctrl+C
                    threading.Thread(target=play, daemon=True).start()
                    return self

                def __exit__(self, *exc):
                    return False
            return Stream()
    monkeypatch.setattr(cv.live, "_sounddevice", lambda: FakeMic())
    cv.main(["--ckpt", "x.pt", "--cluster-threshold", "0.6"])
    out = capsys.readouterr().out
    assert "Joana: " + SCRIPT[0][2] in out and "João: " + SCRIPT[1][2] in out   # the last line was not lost


def test_parser_defaults():
    a = cv.build_parser().parse_args([])
    assert (a.llm, a.whisper, a.language, a.gap, a.file, a.realtime) == \
           ("ollama:gemma3:12b", "openai/whisper-large-v3-turbo", "pt", 0.6, None, False)
