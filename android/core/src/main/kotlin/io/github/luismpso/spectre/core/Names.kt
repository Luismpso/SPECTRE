package io.github.luismpso.spectre.core

/**
 * Names in what is said, and who is who — the rules of spectre.conversation, without a language model: a list of
 * common first names, the words around each name (own name, called, introduced, mentioned) and turn-taking rules.
 */
object Roles {
    const val SELF = "self"
    const val ADDRESSED = "addressed"
    const val INTRODUCED = "introduced"
    const val MENTIONED = "mentioned"
    const val REPLIED = "replied"                 // addressed in an answer: points back to the person who spoke
    val ALL = listOf(SELF, ADDRESSED, INTRODUCED, MENTIONED)
    internal val PRIORITY = listOf(SELF, ADDRESSED, REPLIED, INTRODUCED, MENTIONED)
}

/** Whether a name really appears in the line (exactly, or with a small spelling difference in longer names). */
fun writtenIn(name: String, line: String): Boolean {
    val words = asciiWords(fold(line))
    val parts = asciiWords(fold(name)).filter { it.length > 1 }
    return parts.isNotEmpty() && words.any { w -> parts.any { p -> w == p || (p.length >= 4 && ratio(w, p) >= 0.85) } }
}

private val SELF_CUES = setOf("sou", "chamo", "chamam", "nome", "aqui", "fala", "am", "name")

private fun firstPart(name: String) = asciiWords(fold(name)).firstOrNull { it.length > 1 } ?: ""

/** A self-introduction word shortly before the name ("Eu sou o Pedro", "Chamo-me Ana", "O meu nome é Rui"). */
fun saysOwnName(name: String, line: String): Boolean {
    val first = firstPart(name)
    if (first.isEmpty()) return false
    val words = asciiWords(fold(line))
    return words.indices.any { k ->
        val w = words[k]
        (w == first || (first.length >= 4 && ratio(w, first) >= 0.85)) &&
            words.subList(maxOf(0, k - 3), k).any { it in SELF_CUES }
    }
}

private const val CALLING = "(?:ola|oi|bom dia|boa tarde|boa noite|adeus|tchau|ate logo|ate amanha|muito prazer|prazer|bem-vind[oa]|hello|hi)"
private const val ANSWERING = "(?:muito obrigad[oa]|obrigad[oa]|o prazer e meu|de nada|igualmente|desculp[ae]|thanks|thank you)"
private val PARTICLE = Regex("(?<![\\p{L}])(?:[óÓôÔ]|[Oo]h)(?=[\\s,])")   // "Ó João", "Oh João"

/**
 * How a name calls someone: "answer" right after thanks or a reply ("Obrigado, Maria") — that person spoke before;
 * "call" after a greeting, after "ó", or set off by a comma ("Olá, João", "Ó João", "Pedro, anda cá"); null otherwise.
 */
fun vocative(name: String, line: String): String? {
    val first = firstPart(name)
    if (first.isEmpty()) return null
    val t = fold(line)
    val n = Regex.escape(first)
    val marked = fold(PARTICLE.replace(line, "\u0001"))
    if (Regex("\\b$ANSWERING\\s*[,!]?\\s*$n\\b").containsMatchIn(t)) return "answer"
    if (Regex("\\b$CALLING\\s*[,!]?\\s*$n\\b").containsMatchIn(t) || Regex("\u0001[\\s,]+$n\\b").containsMatchIn(marked) ||
        Regex(",\\s*$n\\b\\s*(?:[.!?…]|$)").containsMatchIn(t) || Regex("(?:^|[.!?]\\s+)$n\\s*,").containsMatchIn(t)
    ) return "call"
    return null
}

/** Correct the usual misreadings with the words themselves (see check_role in spectre.conversation). */
fun checkRole(name: String, role: String, line: String): String? {
    val v = vocative(name, line)
    var r: String? = role
    if (role == Roles.SELF && !saysOwnName(name, line)) r = if (v != null) Roles.ADDRESSED else null
    else if (role == Roles.MENTIONED && v != null) r = Roles.ADDRESSED
    return if (r == Roles.ADDRESSED && v == "answer") Roles.REPLIED else r
}

/** Names read from a line (by rules or by a language model) → checked against the words, one role per name. */
fun checkedNames(found: List<Pair<String, String>>, line: String): List<Pair<String, String>> {
    val best = LinkedHashMap<String, Pair<String, String>>()
    for ((rawName, rawRole) in found) {
        val name = rawName.trim()
        val role = rawRole.trim().lowercase()
        if (role !in Roles.ALL || name.isEmpty() || !writtenIn(name, line)) continue
        val checked = checkRole(name, role, line) ?: continue
        val key = fold(name)
        val previous = best[key]
        if (previous == null || Roles.PRIORITY.indexOf(checked) < Roles.PRIORITY.indexOf(previous.second)) {
            best[key] = name to checked
        }
    }
    return best.values.toList()
}

private val FIRST_NAMES: Set<String> = """
Abel Abílio Adão Adelino Adriano Afonso Agostinho Albano Alberto Albino Alexandre Alfredo Álvaro Américo Amílcar André
Ângelo Aníbal António Antônio Armando Arnaldo Artur Arthur Augusto Aurélio Baltasar Benjamim Bento Bernardo Bruno
Caetano Caio Camilo Carlos Cauã Celso César Cristiano Cristóvão Custódio Daniel Dário David Davi Diego Diogo Dinis
Domingos Duarte Edgar Edson Eduardo Elias Emanuel Emílio Enzo Ernesto Estêvão Eugénio Fabiano Fábio Fausto Felipe
Félix Fernando Filipe Flávio Francisco Frederico Gabriel Gaspar Gil Gilberto Gonçalo Guilherme Gustavo Heitor Hélder
Hélio Henrique Horácio Hugo Humberto Igor Inácio Isaac Ivo Jaime João Joaquim Jonas Jorge José Josué Juliano Júlio
Kevin Leandro Leonardo Leonel Lorenzo Lourenço Lucas Luciano Lúcio Luís Luiz Manuel Marcelo Marcos Marco Mário Martim
Mateus Matheus Matias Maurício Miguel Moisés Murilo Natan Nélson Nicolau Noah Norberto Nuno Octávio Otávio Orlando
Óscar Osvaldo Patrício Paulo Pedro Pietro Rafael Raimundo Ramiro Raul Reinaldo Renan Renato Ricardo Roberto Rodrigo
Rogério Romeu Ronaldo Rúben Rui Salvador Samuel Sandro Santiago Sebastião Sérgio Silvano Silvestre Simão Tadeu Telmo
Teodoro Thiago Tiago Tomás Valentim Valter Vasco Vicente Victor Vítor Vinícius Wagner Wesley William Xavier Yuri Zé
Adelaide Adriana Alexandra Alice Amanda Amélia Ana Andreia Ângela Anabela Antónia Aurora Bárbara Beatriz Benedita
Bianca Bruna Camila Carla Carlota Carmen Carolina Catarina Cecília Célia Clara Cláudia Constança Cristiana Cristina
Daniela Débora Diana Elisa Elisabete Ema Emília Eva Fabiana Fátima Fernanda Filipa Flávia Francisca Gabriela Giovanna
Glória Graça Helena Heloísa Inês Irene Isabel Isabela Isadora Ivone Jéssica Joana Joaquina Júlia Juliana Laís Lara
Larissa Laura Leonor Letícia Lídia Lívia Lorena Lúcia Luana Luísa Luiza Madalena Mafalda Manuela Mara Márcia Margarida
Maria Mariana Marina Marisa Marta Matilde Melissa Mónica Natália Natacha Nádia Olívia Paula Patrícia Pietra Priscila
Raquel Rebeca Regina Renata Rita Rosa Rosana Rosário Sabrina Salomé Sandra Sara Sílvia Simone Sofia Sónia Sophia
Susana Tânia Tatiana Teresa Thaís Valentina Valéria Vanessa Vera Verónica Vitória Viviane Yara Yasmin Zélia
""".trim().split(Regex("\\s+")).map(::fold).toSet()

private val NOT_NAMES: Set<String> = """senhor senhora sr sra dona dom doutor doutora dr dra professor professora
engenheiro engenheira deus mãe pai mano mana filho filha amigo amiga querido querida pessoal malta gente chefe menino
menina rapaz rapariga tio tia avó avô primo prima""".trim().split(Regex("\\s+")).map(::fold).toSet()

private val SELF_INTRO = listOf("\\b(?:eu\\s+)?sou\\s+(?:o|a)\\s+%s\\b", "\\bchamo-me\\s+%s\\b", "\\bme\\s+chamo\\s+%s\\b",
    "\\bmeu\\s+nome\\s+e\\s+%s\\b", "\\baqui\\s+(?:e|fala)\\s+(?:o|a)\\s+%s\\b")
private const val SELF_INTRO_BARE = "\\b(?:eu\\s+)?sou\\s+%s\\b"   // "Eu sou Pedro": only for common first names
private val INTRODUCING = listOf("\\bapresent[\\w-]*\\s+(?:[\\w,]+\\s+){0,4}?(?:o|a)\\s+%s\\b",
    "\\b(?:este|esta)\\s+e\\s+(?:o|a)\\s+%s\\b", "\\bconhec[\\w-]*\\s+(?:o|a)\\s+%s\\b")

/** How a name is used, from the words around it: its own name, introduced, called (a vocative) or mentioned. */
fun nameRole(name: String, line: String): String {
    val t = fold(line)
    val n = Regex.escape(fold(name))
    fun any(patterns: List<String>) = patterns.any { Regex(it.replace("%s", n)).containsMatchIn(t) }
    return when {
        any(SELF_INTRO) || (fold(name) in FIRST_NAMES && any(listOf(SELF_INTRO_BARE))) -> Roles.SELF
        any(INTRODUCING) -> Roles.INTRODUCED
        vocative(name, line) != null -> Roles.ADDRESSED
        else -> Roles.MENTIONED
    }
}

private val TOKENS = Regex("\\p{L}+|[.!?…]")

/** Names in a line without a language model: capitalised common first names, and any other capitalised word used
 *  as a name (called, introduced, said as one's own); a sentence never starts with an unknown name. */
fun ruleNames(line: String): List<Pair<String, String>> {
    val out = mutableListOf<Pair<String, String>>()
    var start = true
    for (m in TOKENS.findAll(line)) {
        val word = m.value
        if (word in listOf(".", "!", "?", "…")) {
            start = true
            continue
        }
        val firstWord = start
        start = false
        val key = fold(word)
        if (!word[0].isUpperCase() || key.length < 2 || key in NOT_NAMES) continue
        val known = key in FIRST_NAMES
        if (firstWord && !known) continue
        val role = nameRole(word, line)
        if (known || role != Roles.MENTIONED) out += word to role
    }
    return out
}

/** One voice's name and why. */
data class Naming(val name: String, val score: Double, val evidence: List<String>)

/** What resolveNames needs from a line: whose voice, and the names read in it (null until read). */
interface NamedLine {
    val speaker: Int
    val names: List<Pair<String, String>>?
}

/**
 * The per-line cues → at most one name per voice, with turn-taking rules (resolve_names in spectre.conversation):
 * own name +3; the next other voice after a name is called or introduced +2; the previous other voice when a reply
 * uses a name +2 (a name in an answer only points back); saying a name to or about someone else −3 for that name.
 */
fun resolveNames(lines: List<NamedLine>, fixed: Map<Int, String>, minScore: Double = 2.0): Map<Int, Naming> {
    val score = HashMap<Pair<Int, String>, Double>()
    val evidence = HashMap<Pair<Int, String>, MutableList<String>>()
    val first = HashMap<Pair<Int, String>, Int>()
    val spelling = LinkedHashMap<String, LinkedHashMap<String, Int>>()
    val order = LinkedHashSet<Pair<Int, String>>()
    val sp = lines.map { it.speaker }

    fun other(i: Int, step: Int): Int? {
        var j = i + step
        while (j in lines.indices) {
            if (sp[j] != sp[i]) return j
            j += step
        }
        return null
    }

    fun add(spk: Int, key: String, w: Double, why: String, i: Int) {
        val k = spk to key
        order += k
        score[k] = (score[k] ?: 0.0) + w
        if (w > 0) {
            evidence.getOrPut(k) { mutableListOf() }.add(why)
            first.putIfAbsent(k, i)
        }
    }

    lines.forEachIndexed { i, ln ->
        for ((name, role) in ln.names.orEmpty()) {
            val key = fold(name)
            val counts = spelling.getOrPut(key) { LinkedHashMap() }
            counts[name] = (counts[name] ?: 0) + 1
            if (role == Roles.SELF) {
                add(sp[i], key, 3.0, "line ${i + 1}: said their name (\"$name\")", i)
                continue
            }
            add(sp[i], key, -3.0, "", i)
            if (role == Roles.ADDRESSED || role == Roles.INTRODUCED) {
                other(i, +1)?.let { j ->
                    val verb = if (role == Roles.ADDRESSED) "called" else "introduced"
                    add(sp[j], key, 2.0, "line ${j + 1}: spoke right after \"$name\" was $verb (line ${i + 1})", j)
                }
            }
            if (role == Roles.ADDRESSED || role == Roles.REPLIED) {
                other(i, -1)?.let { j ->
                    add(sp[j], key, 2.0, "line ${i + 1}: called \"$name\" in the reply to line ${j + 1}", i)
                }
            }
        }
    }

    val taken = fixed.values.map(::fold).toMutableSet()
    val result = LinkedHashMap<Int, Naming>()
    fixed.forEach { (spk, n) -> result[spk] = Naming(n, Double.POSITIVE_INFINITY, listOf("recognised by voice (enrolled)")) }
    val candidates = order.filter { (score[it] ?: 0.0) >= minScore }.sortedWith(
        compareByDescending<Pair<Int, String>> { score.getValue(it) }.thenBy { first.getValue(it) }
            .thenByDescending { it.first }.thenByDescending { it.second })
    for (k in candidates) {
        val (spk, key) = k
        if (spk in result || key in taken) continue
        val name = spelling.getValue(key).entries.maxWithOrNull(
            compareBy<Map.Entry<String, Int>> { it.value }.thenBy { e -> e.key.any { it.code >= 128 } }
                .thenBy { e -> e.key.firstOrNull()?.isUpperCase() == true })!!.key
        result[spk] = Naming(name, score.getValue(k), evidence[k].orEmpty())
        taken += key
    }
    return result
}
