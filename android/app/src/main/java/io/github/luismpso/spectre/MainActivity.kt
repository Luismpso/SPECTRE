package io.github.luismpso.spectre

import android.Manifest
import android.app.Application
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import io.github.luismpso.spectre.core.Caption
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch

class MainViewModel(app: Application) : AndroidViewModel(app) {
    val engine = Engine(app)

    private var starting: Job? = null

    fun start() {
        if (starting?.isActive == true) return                // a second tap while loading
        starting = viewModelScope.launch { if (engine.prepare()) engine.start() }
    }

    override fun onCleared() = engine.close()
}

class MainActivity : ComponentActivity() {
    private var onGranted: () -> Unit = {}
    private val askMicrophone = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) onGranted()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val bars = SystemBarStyle.dark(android.graphics.Color.TRANSPARENT)   // light icons on the dark screen
        enableEdgeToEdge(statusBarStyle = bars, navigationBarStyle = bars)
        setContent {
            val vm: MainViewModel = viewModel()
            val state by vm.engine.state.collectAsStateWithLifecycle()
            LaunchedEffect(state.listening) {                 // captions need the screen: keep it on meanwhile
                if (state.listening) window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                else window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            }
            Terminal(
                state,
                onStart = {
                    onGranted = { vm.start() }
                    if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
                        PackageManager.PERMISSION_GRANTED) onGranted()
                    else askMicrophone.launch(Manifest.permission.RECORD_AUDIO)
                },
                onStop = vm.engine::stop,
                onClear = vm.engine::clear,
                onShare = { share(vm.engine.transcript()) },
                onModel = vm.engine::choose,
            )
        }
    }

    private fun share(text: String) {
        if (text.isBlank()) return
        val send = Intent(Intent.ACTION_SEND).setType("text/plain").putExtra(Intent.EXTRA_TEXT, text)
        startActivity(Intent.createChooser(send, "Share the transcript"))
    }
}

// ------------------------------------------------------------------------------------------------ the terminal
private val Background = Color(0xFF0B0F14)
private val Panel = Color(0xFF11161D)
private val Frame = Color(0xFF2A3340)
private val Ink = Color(0xFFD7DEE7)
private val Dim = Color(0xFF6B7684)
private val Alert = Color(0xFFFF8A80)
private val Voices = listOf(0xFF4DD0E1, 0xFFF48FB1, 0xFF81C784, 0xFFFFD54F, 0xFF64B5F6, 0xFFE57373, 0xFFCE93D8,
    0xFFA5D6A7).map { Color(it) }

private fun voiceColor(speaker: Int) = Voices[Math.floorMod(speaker, Voices.size)]

private val Mono = TextStyle(fontFamily = FontFamily.Monospace, color = Ink, fontSize = 15.sp, lineHeight = 21.sp)

/** mm:ss of a caption, as in the PC version's live panel. */
private fun minutes(t: Double): String = "%02d:%02d".format(t.toInt() / 60, t.toInt() % 60)

@Composable
fun Terminal(
    state: Engine.State, onStart: () -> Unit, onStop: () -> Unit, onClear: () -> Unit, onShare: () -> Unit,
    onModel: (WhisperModel) -> Unit,
) {
    Column(Modifier.fillMaxSize().background(Background).safeDrawingPadding().padding(12.dp)) {
        Text("SPECTRE · live captions", style = Mono.copy(color = Dim, fontSize = 13.sp))
        Spacer(Modifier.height(6.dp))
        Box(Modifier.weight(1f).fillMaxWidth().clip(RoundedCornerShape(6.dp)).background(Panel)
            .border(1.dp, Frame, RoundedCornerShape(6.dp)).padding(10.dp)) {
            Captions(state)
        }
        Spacer(Modifier.height(6.dp))
        Status(state)
        Spacer(Modifier.height(8.dp))
        Controls(state, onStart, onStop, onClear, onShare, onModel)
    }
}

@Composable
private fun Captions(state: Engine.State) {
    val list = rememberLazyListState()
    var follow by remember { mutableStateOf(true) }     // stop following the end while the user reads back
    LaunchedEffect(list) {
        snapshotFlow { list.isScrollInProgress }.collect { scrolling -> if (!scrolling) follow = !list.canScrollForward }
    }
    val rows = state.captions.size + (if (state.speaking != null) 1 else 0)
    LaunchedEffect(rows, state.captions.lastOrNull()?.text, state.speaking) {
        if (follow && rows > 0) list.animateScrollToItem(rows - 1)
    }
    if (rows == 0) {
        Text(
            if (state.listening) "Waiting for someone to speak…"
            else "Tap Start and talk. Captions appear here with the names people say:\n" +
                "\"Olá João!\" — the next voice is João.\n\nEverything runs on this phone.",
            style = Mono.copy(color = Dim),
        )
        return
    }
    LazyColumn(state = list, verticalArrangement = Arrangement.spacedBy(6.dp)) {
        itemsIndexed(state.captions, key = { i, _ -> i }) { _, c -> CaptionRow(c) }
        state.speaking?.let { (voice, who) ->
            item(key = "speaking") {
                Text("▶ $who is speaking…", style = Mono.copy(color = voiceColor(voice), fontWeight = FontWeight.Bold))
            }
        }
    }
}

@Composable
private fun CaptionRow(c: Caption) {
    Text(buildAnnotatedString {
        withStyle(SpanStyle(color = Dim, fontSize = 12.sp)) { append(minutes(c.start) + "  ") }
        withStyle(SpanStyle(color = voiceColor(c.speaker), fontWeight = FontWeight.Bold)) { append(c.label + ": ") }
        if (c.text == null) withStyle(SpanStyle(color = Dim)) { append("…") } else append(c.text)
    }, style = Mono.copy(fontSize = 17.sp, lineHeight = 24.sp))
}

@Composable
private fun Status(state: Engine.State) {
    val voices = state.captions.distinctBy { it.speaker }.map { it.label }
    if (voices.isNotEmpty()) Text("voices: " + voices.joinToString(", "), style = Mono.copy(color = Dim, fontSize = 13.sp))
    val line = when {
        state.busy != null -> state.busy + (state.progress?.let { " ${(100 * it).toInt()} %" } ?: "")
        state.listening -> "listening · whisper ${state.model.label} · names: rules" +
            (if (state.waiting > 1) " · ${state.waiting - 1} waiting" else "")
        state.waiting > 0 -> "finishing · ${state.waiting} to transcribe"
        state.captions.isNotEmpty() -> "stopped · whisper ${state.model.label} · names: rules"
        else -> "ready" + (if (state.calibration.isNotEmpty()) " · voices: ${state.calibration}" else "")
    }
    Text(line, style = Mono.copy(color = Dim, fontSize = 13.sp))
    state.progress?.let {
        Spacer(Modifier.height(4.dp))
        LinearProgressIndicator(progress = { it }, modifier = Modifier.fillMaxWidth(), color = Voices[0],
            trackColor = Frame)
    }
    state.error?.let { Text("error: $it", style = Mono.copy(color = Alert, fontSize = 13.sp)) }
}

@Composable
private fun Controls(
    state: Engine.State, onStart: () -> Unit, onStop: () -> Unit, onClear: () -> Unit, onShare: () -> Unit,
    onModel: (WhisperModel) -> Unit,
) {
    val idle = !state.listening && state.busy == null
    if (idle) {
        Text("whisper model", style = Mono.copy(color = Dim, fontSize = 12.sp), modifier = Modifier.padding(bottom = 4.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            for (m in WhisperModel.entries) {
                val chosen = m == state.model
                Key(m.label, Modifier.weight(1f), if (chosen) Voices[0] else Dim, chosen,
                    sub = if (m in state.downloaded) "ready" else "↓ ${m.megabytes} MB") { onModel(m) }
            }
        }
        Text(state.model.note + (if (state.model in state.downloaded) "" else " · downloaded once, then offline"),
            style = Mono.copy(color = Dim, fontSize = 12.sp), modifier = Modifier.padding(top = 4.dp, bottom = 8.dp))
    }
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
        when {
            state.listening -> Key("■ Stop", Modifier.weight(1f), Alert, true, onClick = onStop)
            state.busy != null -> Key("…", Modifier.weight(1f), Dim, false) {}
            else -> Key(if (state.captions.isEmpty()) "● Start" else "● Continue", Modifier.weight(1f), Voices[2], true,
                onClick = onStart)
        }
        if (idle && state.captions.isNotEmpty()) {
            Key("Share", Modifier.weight(0.5f), Ink, false, onClick = onShare)
            Key("Clear", Modifier.weight(0.5f), Ink, false, onClick = onClear)
        }
    }
}

/** A terminal-style button: a framed label, with an optional second line. */
@Composable
private fun Key(label: String, modifier: Modifier, color: Color, strong: Boolean, sub: String? = null,
                onClick: () -> Unit) {
    Column(modifier.clip(RoundedCornerShape(6.dp)).border(1.dp, if (strong) color else Frame, RoundedCornerShape(6.dp))
        .clickable(onClick = onClick).padding(vertical = if (sub == null) 12.dp else 7.dp),
        horizontalAlignment = Alignment.CenterHorizontally) {
        Text(label, maxLines = 1, style = Mono.copy(color = color, fontWeight = if (strong) FontWeight.Bold else FontWeight.Normal))
        if (sub != null) Text(sub, maxLines = 1, style = Mono.copy(color = Dim, fontSize = 11.sp, lineHeight = 14.sp))
    }
}
