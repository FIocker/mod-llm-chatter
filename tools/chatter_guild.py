"""Guild chatter event handlers.

Ambient guild-channel banter: online guild members occasionally
exchange short, in-character lines in guild chat. Driven by
``CheckGuildIdleChatter`` in LLMChatterWorld.cpp, gated by
``LLMChatter.GuildChatter.*`` config. Mirrors the structure of the
raid idle-morale and proximity handlers.
"""

import logging
import random
import re
from typing import Dict, List, Optional

from chatter_db import insert_chat_message
from chatter_llm import call_llm
from chatter_shared import (
    append_conversation_json_instruction,
    append_json_instruction,
    build_conversation_json_repair_prompt,
    calculate_dynamic_delay,
    get_chatter_mode,
    get_class_name,
    get_gender_label,
    get_race_name,
    get_zone_name,
    get_zone_flavor,
    parse_extra_data,
    parse_conversation_response,
    select_conversation_message_count,
)
from chatter_text import (
    cleanup_message,
    parse_single_response,
    strip_speaker_prefix,
)

logger = logging.getLogger(__name__)


def _mark_event(db, event_id: int, status: str) -> None:
    cursor = db.cursor()
    cursor.execute(
        "UPDATE llm_chatter_events SET status = %s "
        "WHERE id = %s",
        (status, event_id),
    )
    db.commit()


def _query_speaker(db, bot_guid: int) -> Dict[str, object]:
    """Load a speaker's class/race/level/gender plus their
    stored personality (traits/tone/backstory) from the
    bot-identity table. Returns {} if the bot is unknown."""
    if not bot_guid:
        return {}

    try:
        cursor = db.cursor(dictionary=True)
        cursor.execute(
            "SELECT class, race, gender, level "
            "FROM characters WHERE guid = %s",
            (bot_guid,),
        )
        base = cursor.fetchone()
        if not base:
            return {}

        cursor.execute(
            "SELECT trait1, trait2, trait3, tone, "
            "       backstory "
            "FROM llm_bot_identities "
            "WHERE bot_guid = %s LIMIT 1",
            (bot_guid,),
        )
        ident = cursor.fetchone() or {}

        traits = [
            trait for trait in (
                ident.get('trait1'),
                ident.get('trait2'),
                ident.get('trait3'),
            )
            if trait
        ]
        return {
            'class': get_class_name(
                int(base.get('class', 0) or 0)
            ),
            'race': get_race_name(
                int(base.get('race', 0) or 0)
            ),
            'gender': get_gender_label(
                int(base.get('gender', 0) or 0)
            ),
            'level': int(base.get('level', 0) or 0),
            'traits': traits,
            'tone': ident.get('tone') or '',
            'backstory': ident.get('backstory') or '',
        }
    except Exception:
        logger.error(
            "query guild speaker failed",
            exc_info=True,
        )
        return {}


from chatter_constants import GUILD_CHAT_TOPICS_RP
from chatter_general import _pick_length_hint
from chatter_prompts import (
    generate_conversation_length_sequence,
    generate_conversation_mood_sequence,
)


# Length control mirrors the General channel, which works well: we do NOT
# hard-cap or chop the model's output after the fact (the old per-bucket
# _truncate_to butchered lines mid-sentence). Instead we reuse General's
# _pick_length_hint(mode) — a char-range target plus a single generous
# "HARD LIMIT: Never exceed 150 characters total" stated in the prompt —
# and deliver the model's full, coherent sentence intact.


# Review #4: never insult your own faction. Derive Alliance/Horde from race so
# the prompt can pass the speaker's side and forbid self-faction jabs.
_ALLIANCE_RACES = {"Human", "Dwarf", "Night Elf", "Gnome", "Draenei"}
_HORDE_RACES = {"Orc", "Undead", "Scourge", "Tauren", "Troll", "Blood Elf"}


def _faction_of(race_name: str) -> str:
    if race_name in _ALLIANCE_RACES:
        return "Alliance"
    if race_name in _HORDE_RACES:
        return "Horde"
    return ""


def _speaker_faction(speaker: Dict) -> str:
    """Resolve the speaker's faction from its race (name or id), defensively."""
    race = speaker.get('race')
    candidates = []
    if isinstance(race, str):
        candidates.append(race)
    try:
        candidates.append(get_race_name(race))
    except Exception:
        pass
    for c in candidates:
        fac = _faction_of((c or '').strip())
        if fac:
            return fac
    return ""


# Review #2 (strict): guild lines are SPOKEN text only. The message-only schema +
# cleanup_message() handle the common cases, but the model can still embed marked
# RP artifacts INSIDE the message field. Deterministically strip them so we never
# rely on prompt pressure alone: /me|/e|/emote prefixes, *narrator* fragments
# (leading/trailing/inline), <emote>..</emote>-style tags, and stray backticks.
def _strip_rp_artifacts(message: str) -> str:
    if not message:
        return ""
    s = message
    # slash-emote command prefixes (/me, /e, /emote)
    s = re.sub(r'^\s*/(?:me|e|emote)\b[:,]?\s*', '', s, flags=re.IGNORECASE)
    # angle-bracket emote/action tags and any stray short html-ish tag
    s = re.sub(r'</?\s*(?:emote|action|i|em|rp|me)\s*>', ' ',
               s, flags=re.IGNORECASE)
    s = re.sub(r'<[^<>]{0,40}>', ' ', s)
    # *...* narrator fragments anywhere, then any leftover lone asterisks
    s = re.sub(r'\*[^*]{1,80}\*', ' ', s)
    s = s.replace('*', '')
    # explicit "emote:"/"action:" leads
    s = re.sub(r'^\s*(?:emote|action)\s*:\s*', '', s, flags=re.IGNORECASE)
    # stray fences/backticks
    s = s.replace('```', '').replace('`', '')
    # collapse whitespace opened up by removals
    s = re.sub(r'\s{2,}', ' ', s).strip(' ,;:-\t')
    return s.strip()


# Review #6: never put "level N" in the guild prompt — handing the model the exact
# forbidden mechanic ("levels") undermines the jargon ban. Translate level into
# non-mechanical flavor instead.
def _level_flavor(level) -> str:
    try:
        lvl = int(level)
    except (TypeError, ValueError):
        return ""
    if lvl <= 0:
        return ""
    if lvl < 20:
        return "a young adventurer"
    if lvl < 60:
        return "a seasoned traveler"
    if lvl < 80:
        return "a hardened campaigner"
    return "a battle-hardened veteran"


def _resolve_name(name_fn, val, default: str) -> str:
    """Resolve a race/class to a display name whether it's already a name or an id."""
    if isinstance(val, str) and val and not val.isdigit():
        return val
    try:
        r = name_fn(val)
        if r:
            return r
    except Exception:
        pass
    return default


def _guild_identity(speaker_name: str, speaker: Dict) -> str:
    """Non-mechanical identity line for guild chat (no 'level N')."""
    race = _resolve_name(get_race_name, speaker.get('race'), 'wanderer')
    klass = _resolve_name(get_class_name, speaker.get('class'), 'adventurer')
    flavor = _level_flavor(speaker.get('level'))
    base = f"You are {speaker_name}, a {race} {klass} of Azeroth"
    if flavor:
        base += f" — {flavor}"
    return base + "."


def _build_guild_prompt(
    speaker_name: str,
    speaker: Dict,
    guild_name: str,
    guildmates: str,
    config: Optional[Dict] = None,
    zone_id: int = 0,
    length_hint: str = "",
    topic: str = "",
    faction: str = "",
    name_zone: bool = False,
) -> str:
    lines = [_guild_identity(speaker_name, speaker)]
    lines.append(
        f"You are a member of the guild "
        f"\"{guild_name}\"."
    )

    traits = speaker.get('traits') or []
    if traits:
        lines.append(
            "Personality: " + ", ".join(traits) + "."
        )
    if speaker.get('tone'):
        lines.append(f"Tone: {speaker['tone']}.")
    if speaker.get('backstory'):
        lines.append(
            f"Background: {speaker['backstory']}"
        )
    if guildmates:
        lines.append(
            f"Guildmates currently online: "
            f"{guildmates}."
        )

    # Review #4: faction awareness — never insult your own side.
    # PR #30 follow-up #1: prefer the authoritative C++ GetTeamId() value
    # (extra_data "team"); fall back to the Python race-derived faction.
    if not faction:
        faction = _speaker_faction(speaker)
    if faction:
        lines.append(
            f"You fight for the {faction}. Never insult or mock "
            f"the {faction} — they are your own people. If you "
            "speak of rivalry, it is only toward the opposing "
            "faction."
        )

    if zone_id:
        zone = get_zone_name(zone_id)
        if zone:
            # Guild chat reaches guildmates scattered across other zones who
            # cannot see where the speaker stands, so deictic references ("this
            # swamp", "here") read as nonsense to them. Whether to name the
            # location is decided by an RNG roll in the handler (name_zone) —
            # the model cannot be trusted to self-pace it. We deliberately give
            # NO phrasing examples here: examples make the model echo them into
            # repetitive patterns.
            flavor = get_zone_flavor(zone_id)
            if name_zone:
                # Review #5: curated lore flavor as POSITIVE context so the
                # model draws on real local color rather than inventing it.
                if flavor:
                    lines.append(
                        f"You are currently in {zone}. Local color you may "
                        f"draw on: {flavor} Use only this for specifics; do "
                        "NOT invent other local NPCs, towns, factions, or "
                        "events."
                    )
                else:
                    lines.append(
                        f"You are currently in {zone}. You may react to the "
                        "land itself (its weather, danger, mood) but do NOT "
                        "invent specific local NPCs, towns, or events you "
                        "cannot be sure exist."
                    )
                lines.append(
                    f"Most of your guildmates are far away in other lands and "
                    f"cannot see where you are, so name {zone} somewhere in "
                    "your line, woven in naturally, so they know where you "
                    "speak from."
                )
            else:
                # RNG said no: forbid referencing the location at all this
                # round so we never get a deictic line with no place name.
                lines.append(
                    "Most of your guildmates are far away and cannot see "
                    "where you are. Do NOT name or describe your current "
                    "location or immediate surroundings this time — speak of "
                    "other matters instead."
                )
    if not topic:
        topic = random.choice(GUILD_CHAT_TOPICS_RP)
    lines.append(
        "Topic idea (optional - only use it if it fits "
        f"naturally, do not force it): {topic}."
    )
    # Review #3: keep content within the speaker's OWN class/race idiom — the
    # model otherwise borrows another class's fantasy (a warlock invoking
    # ancestors, a death knight using fel, etc.).
    lines.append(
        "Speak only in the idiom that fits your own race and class. Do NOT "
        "borrow another class's powers or imagery — do not invoke spirits, "
        "ancestors, the Light, the elements, nature, or fel unless that "
        "genuinely belongs to who you are."
    )
    lines.append(
        "Stay fully in character — you ARE this person in Azeroth, "
        "speaking to your guild. No fourth-wall breaks and no "
        "out-of-character or game-mechanic talk. NEVER use words "
        "like grinding, pulls, DPS, specs, talents, loot, mobs, "
        "XP, levels, rotations, addons, or any reference to the "
        "player behind the screen. Speak of foes, the road, your "
        "craft and your calling — not game systems."
    )
    lines.append(
        "Write ONE casual, in-character line for guild chat, the "
        "way this person would actually speak. No quotation marks, "
        "no name prefix, no roleplay asterisks, no emotes or "
        "actions — just the spoken line."
    )
    # Length control mirrors the General channel: a char-range target plus a
    # single generous HARD LIMIT, stated in the prompt. No post-parse cut.
    if length_hint:
        lines.append(length_hint)

    # Review #2: message-only JSON. Do not request emote/action
    # fields so they cannot leak into the displayed line.
    return append_json_instruction(
        "\n".join(lines) + "\n",
        allow_action=False,
        skip_emote=True,
        message_only=True,
    )


def _process_guild_statement_event(
    db,
    client,
    config,
    event,
    topic_override: str = "",
    name_zone_override=None,
    fallback: bool = False,
):
    """Handle guild_idle_chatter — one online guild member
    posts a short in-character line to guild chat."""
    event_id = event['id']
    extra = parse_extra_data(
        event.get('extra_data'),
        event_id, 'guild_idle_chatter')

    if not extra:
        _mark_event(db, event_id, 'skipped')
        return False

    speaker_guid = int(
        event.get('subject_guid', 0) or 0
    )
    speaker_name = (
        event.get('subject_name')
        or extra.get('speaker_name')
        or ''
    )
    guild_name = extra.get('guild_name') or 'the guild'
    guildmates = extra.get('guildmates') or ''

    speaker = _query_speaker(db, speaker_guid)
    if not speaker or not speaker_name:
        _mark_event(db, event_id, 'skipped')
        return False

    zone_id = int(extra.get('zone_id', 0) or 0)
    # Length control mirrors the General channel (which works well): reuse
    # its _pick_length_hint(mode) and let the prompt enforce length. No
    # post-parse truncation — the model's full sentence is delivered intact.
    length_hint = _pick_length_hint(get_chatter_mode(config))
    topic = (
        topic_override
        or random.choice(GUILD_CHAT_TOPICS_RP)
    )
    # PR #30 follow-up #1: prefer the C++ GetTeamId() faction (extra_data
    # "team"); fall back to the Python race-derived faction.
    faction = extra.get('team') or _speaker_faction(speaker)
    # Zone-naming is decided here by RNG, not left to the model (it cannot
    # self-pace "occasionally"). On a hit the prompt asks the bot to name its
    # zone so scattered guildmates have context; on a miss it forbids any
    # location reference. Tunable via LLMChatter.GuildChatter.ZoneNameChance.
    zone_name_chance = int(config.get(
        'LLMChatter.GuildChatter.ZoneNameChance', 20))
    name_zone = (
        name_zone_override
        if name_zone_override is not None
        else random.randint(1, 100) <= zone_name_chance
    )
    prompt = _build_guild_prompt(
        speaker_name, speaker, guild_name,
        guildmates, config, zone_id=zone_id,
        length_hint=length_hint, topic=topic, faction=faction,
        name_zone=name_zone,
    )

    max_tokens = int(config.get(
        'LLMChatter.GuildChatter.MaxTokens', 200
    ))
    # PR #30 follow-up #2: pass the generation controls as structured
    # metadata so they land as top-level fields in llm_requests.jsonl
    # (the monitoring pass can read them without parsing prompt text).
    metadata = {
        "guild_id": _safe_int(extra.get('guild_id')),
        "guild_mode": "statement",
        "guild_participant_count": 1,
        "guild_participants": speaker_name,
        "guild_requested_message_count": 1,
        "guild_statement_fallback": fallback,
        "guild_length_hint": length_hint.split("\n", 1)[0]
        .replace("Length: ", "").strip(),
        "guild_topic": topic,
        "guild_named_zone": name_zone,
        "guild_faction": faction,
        "guild_zone_id": zone_id,
        "guild_zone_name": get_zone_name(zone_id) or "",
        "guild_zone_flavor": get_zone_flavor(zone_id) or "",
    }
    response = call_llm(
        client, prompt, config,
        max_tokens_override=max_tokens,
        context=f"guild:{speaker_name}",
        label='guild_idle_chatter',
        metadata=metadata,
    )
    if not response:
        _mark_event(db, event_id, 'skipped')
        return False

    parsed = parse_single_response(response)
    message = strip_speaker_prefix(
        parsed.get('message', ''), speaker_name
    )
    # Review #2: message-only — drop any emote/action the model may have
    # leaked; never prepend a narrator action to guild lines.
    message = cleanup_message(message)
    # Review #2 (strict): deterministically strip marked RP artifacts that can
    # still ride inside the message field (/me, *action*, <emote>, fences).
    message = _strip_rp_artifacts(message)
    if not message:
        _mark_event(db, event_id, 'skipped')
        return False
    # No post-parse truncation (mirrors the General channel): the prompt's
    # length hint + HARD LIMIT control length, and the full coherent line is
    # delivered as-is so messages are never cut mid-sentence.
    logger.info(
        "guild_idle_chatter mode=statement speaker=%s "
        "faction=%s zone_id=%d topic=%r out_len=%d "
        "fallback=%s",
        speaker_name,
        faction or "-", zone_id, topic, len(message),
        fallback,
    )

    insert_chat_message(
        db,
        bot_guid=speaker_guid,
        bot_name=speaker_name,
        message=message,
        channel='guild',
        owner_subsystem='guild',
        event_id=event_id,
    )

    _mark_event(db, event_id, 'completed')
    return True


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def _normalize_guild_participants(
    event: Dict,
    extra: Dict,
) -> tuple:
    """Normalize structured and legacy Guild event payloads."""
    participants = []
    seen_guids = set()
    seen_names = set()
    raw_participants = extra.get('participants')

    if isinstance(raw_participants, list):
        for raw in raw_participants:
            if not isinstance(raw, dict):
                continue
            guid = _safe_int(raw.get('guid'))
            name = str(raw.get('name') or '').strip()
            name_key = name.casefold()
            if (
                not guid
                or not name
                or guid in seen_guids
                or name_key in seen_names
            ):
                continue
            seen_guids.add(guid)
            seen_names.add(name_key)
            participants.append({
                'guid': guid,
                'name': name,
                'zone_id': _safe_int(raw.get('zone_id')),
                'map_id': _safe_int(raw.get('map_id')),
            })

    if not participants:
        guid = _safe_int(event.get('subject_guid'))
        name = str(
            event.get('subject_name')
            or extra.get('speaker_name')
            or ''
        ).strip()
        if guid and name:
            participants.append({
                'guid': guid,
                'name': name,
                'zone_id': _safe_int(extra.get('zone_id')),
                'map_id': _safe_int(event.get('map_id')),
            })

    requested_mode = str(
        extra.get('mode') or 'statement'
    ).strip().lower()
    if requested_mode != 'conversation':
        return participants[:1], 'statement'
    if len(participants) < 2:
        return participants[:1], 'statement'
    return participants[:3], 'conversation'


def _participant_identity_lines(
    participant: Dict,
) -> List[str]:
    speaker = participant['speaker']
    name = participant['name']
    lines = [_guild_identity(name, speaker)]
    traits = speaker.get('traits') or []
    if traits:
        lines.append(
            f"{name} personality: {', '.join(traits)}."
        )
    if speaker.get('tone'):
        lines.append(
            f"{name} speaking tone: {speaker['tone']}."
        )
    if speaker.get('backstory'):
        background = str(speaker['backstory']).strip()
        lines.append(
            f"{name} background: {background[:400]}"
        )
    return lines


def _guild_location_lines(
    participants: List[Dict],
    name_zone: bool,
) -> List[str]:
    primary = participants[0]
    primary_zone = get_zone_name(
        primary.get('zone_id', 0)
    ) or ''
    locations = []
    location_keys = set()
    for participant in participants:
        zone_id = participant.get('zone_id', 0)
        map_id = participant.get('map_id', 0)
        zone_name = (
            get_zone_name(zone_id)
            or 'an unknown land'
        )
        locations.append(
            f"{participant['name']}: {zone_name}"
        )
        location_keys.add((zone_id, map_id))

    lines = [
        "Private live-location context: "
        + "; ".join(locations) + "."
    ]
    same_location = (
        len(location_keys) == 1
        and primary.get('zone_id', 0) != 0
    )
    if same_location:
        lines.append(
            "The speakers are in the same zone and map, "
            "so shared surroundings are possible, but do "
            "not invent a precise meeting place."
        )
    else:
        lines.append(
            "Guild chat reaches across Azeroth. The "
            "speakers are remote from one another. Never "
            "imply they can see, touch, or stand beside "
            "each other."
        )

    if name_zone and primary_zone:
        flavor = get_zone_flavor(
            primary.get('zone_id', 0)
        )
        lines.append(
            f"The exchange may name {primary_zone} as "
            f"{primary['name']}'s location. Other "
            "speakers must not present it as their own."
        )
        if flavor:
            lines.append(
                f"Curated {primary_zone} local color: "
                f"{flavor}"
            )
        lines.append(
            "Do not name any other speaker's current "
            "location."
        )
    else:
        lines.append(
            "Do not name or describe any speaker's "
            "current location or immediate surroundings "
            "this time."
        )
    return lines


def _build_guild_conversation_prompt(
    participants: List[Dict],
    guild_name: str,
    guildmates: str,
    topic: str,
    faction: str,
    name_zone: bool,
    message_count: int,
    reference_plans: Optional[List[Dict]] = None,
) -> str:
    bot_names = [
        participant['name']
        for participant in participants
    ]
    lines = [
        "Generate a short, coherent in-character Guild "
        "Chat exchange between "
        f"{', '.join(bot_names)}.",
        f"They are all members of \"{guild_name}\".",
    ]
    for participant in participants:
        lines.extend(
            _participant_identity_lines(participant)
        )
    if guildmates:
        lines.append(
            "Other guildmates currently online: "
            f"{guildmates}."
        )
    if faction:
        lines.append(
            f"They fight for the {faction}. Never "
            f"insult or mock the {faction}, their own "
            "people. Rivalry may only target the "
            "opposing faction."
        )
    lines.extend(
        _guild_location_lines(participants, name_zone)
    )
    lines.extend([
        f"Shared subject for the whole exchange: {topic}.",
        "Use that subject naturally and keep every line "
        "part of the same conversation.",
        "Every selected speaker MUST speak at least once.",
        "Each line is spoken text only: no quotation "
        "marks, name prefixes, narrator text, roleplay "
        "asterisks, slash commands, or stage directions.",
        "Stay fully in Azeroth and avoid game-mechanic "
        "terms such as DPS, specs, talents, loot, mobs, "
        "XP, levels, rotations, addons, or players "
        "behind screens.",
        "Each speaker must use the idiom of their own "
        "race and class, never another participant's "
        "powers or beliefs.",
        "HARD LIMIT: Never exceed 150 characters in any "
        "individual message.",
        "\nMOOD AND LENGTH SEQUENCE:",
    ])

    moods = generate_conversation_mood_sequence(
        message_count, 'roleplay'
    )
    lengths = generate_conversation_length_sequence(
        message_count
    )
    reference_by_index = {
        plan['message_index']: plan
        for plan in (reference_plans or [])
    }
    for index in range(message_count):
        speaker = bot_names[index % len(bot_names)]
        instruction = (
            f"  Message {index + 1} ({speaker}): "
            f"mood={moods[index]}, "
            f"length={lengths[index]}"
        )
        reference_plan = reference_by_index.get(index)
        if reference_plan:
            candidates = ", ".join(
                reference_plan['candidates']
            )
            target_count = reference_plan['target_count']
            if target_count == 1:
                instruction += (
                    "; naturally address one earlier "
                    "speaker by name while replying "
                    f"(choose from {candidates} based on "
                    "whose point this message answers); "
                    "use that name once"
                )
            else:
                instruction += (
                    f"; naturally address {target_count} "
                    "earlier speakers by name while "
                    f"replying (choose from {candidates} "
                    "based on whose points this message "
                    "answers); use each name once"
                )
        lines.append(instruction)

    return append_conversation_json_instruction(
        "\n".join(lines),
        bot_names,
        message_count,
        allow_action=False,
        message_only=True,
    )


def _select_participant_references(
    bot_names: List[str],
    message_count: int,
    chance: int,
    multi_chance: int,
    max_reference_lines: int,
) -> List[Dict]:
    """Randomly select reply lines that must name earlier speakers."""
    bounded_chance = max(0, min(100, int(chance)))
    bounded_multi_chance = max(
        0,
        min(100, int(multi_chance)),
    )
    max_lines = max(0, int(max_reference_lines))
    if (
        len(bot_names) < 2
        or message_count < 2
        or bounded_chance == 0
        or max_lines == 0
    ):
        return []

    plans = []
    for message_index in range(1, message_count):
        if random.randint(1, 100) > bounded_chance:
            continue
        speaker = bot_names[
            message_index % len(bot_names)
        ]
        candidates = []
        for index in range(message_index):
            candidate = bot_names[index % len(bot_names)]
            if (
                candidate != speaker
                and candidate not in candidates
            ):
                candidates.append(candidate)
        if not candidates:
            continue

        target_count = 1
        if (
            len(candidates) >= 2
            and random.randint(1, 100)
            <= bounded_multi_chance
        ):
            target_count = 2
        plans.append({
            'message_index': message_index,
            'speaker': speaker,
            'candidates': candidates,
            'target_count': target_count,
        })

    if len(plans) > max_lines:
        plans = random.sample(plans, max_lines)
    return sorted(
        plans,
        key=lambda plan: plan['message_index'],
    )


def _contains_speaker_name(
    text: str,
    speaker_name: str,
) -> bool:
    return bool(re.search(
        r'(?<![A-Za-z])'
        + re.escape(speaker_name)
        + r'(?![A-Za-z])',
        text,
        flags=re.IGNORECASE,
    ))


def _insert_reference_names(
    text: str,
    speaker_names: List[str],
) -> str:
    """Add natural vocatives without rewriting the generated line."""
    names = [
        name for name in speaker_names
        if not _contains_speaker_name(text, name)
    ]
    if not names:
        return text
    address = " and ".join(names)

    acknowledgement = re.match(
        r"^(you(?:'re| are) right|aye|yes|no|indeed|"
        r"exactly|agreed|true enough|fair enough)"
        r"(?P<punct>[.!?]+|$)",
        text,
        flags=re.IGNORECASE,
    )
    if acknowledgement:
        end = acknowledgement.end()
        lead = text[:acknowledgement.start('punct')]
        punct = acknowledgement.group('punct')
        return (
            f"{lead}, {address}{punct}"
            f"{text[end:]}"
        )

    terminal = re.search(r'(?P<punct>[.!?]+)$', text)
    if terminal:
        return (
            f"{text[:terminal.start()]}, {address}"
            f"{terminal.group('punct')}"
        )
    return f"{text.rstrip(',;: ')}, {address}"


def _apply_participant_references(
    messages: List[Dict],
    reference_plans: List[Dict],
) -> List[Dict]:
    """Guarantee selected lines name the required earlier speakers.

    The model may choose any contextually relevant earlier speaker
    combination. Missing names are selected randomly from the valid
    earlier speakers and inserted as deterministic vocatives.
    """
    results = []
    for plan in reference_plans:
        message_index = int(plan['message_index'])
        if (
            message_index < 1
            or message_index >= len(messages)
        ):
            continue

        current = messages[message_index]
        current_name = current.get('name', '')
        earlier_names = []
        for message in messages[:message_index]:
            earlier_name = message.get('name', '')
            if (
                earlier_name
                and earlier_name != current_name
                and earlier_name not in earlier_names
            ):
                earlier_names.append(earlier_name)
        if not earlier_names:
            continue

        required_count = min(
            int(plan['target_count']),
            len(earlier_names),
        )
        text = current.get('message', '')
        detected = [
            name for name in earlier_names
            if _contains_speaker_name(text, name)
        ]
        missing_count = max(
            0,
            required_count - len(detected),
        )
        remaining = [
            name for name in earlier_names
            if name not in detected
        ]
        fallback_names = (
            random.sample(remaining, missing_count)
            if missing_count else []
        )
        current['message'] = _insert_reference_names(
            text,
            fallback_names,
        )
        results.append({
            'message_index': message_index,
            'targets': (
                detected[:required_count]
                + fallback_names
            ),
            'fallback': bool(fallback_names),
        })
    return results


def _clean_guild_conversation(
    messages: List[Dict],
) -> List[Dict]:
    cleaned = []
    for message in messages:
        speaker_name = message.get('name', '')
        text = strip_speaker_prefix(
            message.get('message', ''),
            speaker_name,
        )
        text = cleanup_message(text)
        text = _strip_rp_artifacts(text)
        if text:
            cleaned.append({
                'name': speaker_name,
                'message': text,
            })
    return cleaned


def _valid_guild_conversation(
    messages: List[Dict],
    bot_names: List[str],
) -> bool:
    if len(messages) < 2:
        return False
    speakers = {
        message.get('name')
        for message in messages
        if message.get('name')
    }
    return all(
        name in speakers for name in bot_names
    )


def _guild_request_metadata(
    extra: Dict,
    participants: List[Dict],
    topic: str,
    faction: str,
    name_zone: bool,
) -> Dict:
    primary = participants[0]
    zone_id = primary.get('zone_id', 0)
    return {
        'guild_id': _safe_int(extra.get('guild_id')),
        'guild_mode': 'conversation',
        'guild_participant_count': len(participants),
        'guild_participants': ','.join(
            participant['name']
            for participant in participants
        ),
        'guild_topic': topic,
        'guild_named_zone': name_zone,
        'guild_faction': faction,
        'guild_zone_id': zone_id,
        'guild_zone_name': (
            get_zone_name(zone_id) or ''
        ),
        'guild_zone_flavor': (
            get_zone_flavor(zone_id) or ''
        ),
    }


def _generate_guild_conversation(
    db,
    client,
    config: Dict,
    event_id: int,
    extra: Dict,
    participants: List[Dict],
    guild_name: str,
    guildmates: str,
    topic: str,
    faction: str,
    name_zone: bool,
) -> bool:
    participant_count = len(participants)
    max_lines = int(config.get(
        'LLMChatter.GuildChatter.'
        'MaxConversationLines',
        4,
    ))
    message_count = select_conversation_message_count(
        participant_count,
        participant_count,
        max_lines,
    )
    bot_names = [
        participant['name']
        for participant in participants
    ]
    reference_chance = int(config.get(
        'LLMChatter.GuildChatter.'
        'ParticipantReferenceChance',
        25,
    ))
    multi_reference_chance = int(config.get(
        'LLMChatter.GuildChatter.'
        'MultiReferenceChance',
        15,
    ))
    max_reference_lines = int(config.get(
        'LLMChatter.GuildChatter.'
        'MaxReferenceLines',
        2,
    ))
    reference_plans = _select_participant_references(
        bot_names,
        message_count,
        reference_chance,
        multi_reference_chance,
        max_reference_lines,
    )
    prompt = _build_guild_conversation_prompt(
        participants,
        guild_name,
        guildmates,
        topic,
        faction,
        name_zone,
        message_count,
        reference_plans,
    )
    metadata = _guild_request_metadata(
        extra,
        participants,
        topic,
        faction,
        name_zone,
    )
    metadata['guild_requested_message_count'] = (
        message_count
    )
    metadata['guild_repair'] = False
    metadata['guild_reference_requested'] = bool(
        reference_plans
    )
    metadata['guild_reference_requested_count'] = len(
        reference_plans
    )
    metadata['guild_reference_message_indices'] = (
        ','.join(
            str(plan['message_index'] + 1)
            for plan in reference_plans
        )
    )
    metadata['guild_reference_speakers'] = ','.join(
        plan['speaker']
        for plan in reference_plans
    )
    metadata['guild_reference_target_counts'] = ','.join(
        str(plan['target_count'])
        for plan in reference_plans
    )
    metadata['guild_reference_candidate_sets'] = '|'.join(
        '/'.join(plan['candidates'])
        for plan in reference_plans
    )

    base_tokens = int(config.get(
        'LLMChatter.GuildChatter.MaxTokens', 200
    ))
    conversation_tokens = min(
        base_tokens * (1 + participant_count),
        1000,
    )
    response = call_llm(
        client,
        prompt,
        config,
        max_tokens_override=conversation_tokens,
        context=(
            "guild-conv:"
            + ",".join(bot_names)
        ),
        label='guild_idle_chatter',
        metadata=metadata,
    )
    messages = _clean_guild_conversation(
        parse_conversation_response(
            response or '',
            bot_names,
        )
    )[:message_count]
    repair_used = False

    if not _valid_guild_conversation(
        messages, bot_names
    ):
        repair_used = True
        repair_metadata = dict(metadata)
        repair_metadata['guild_repair'] = True
        repair_metadata[
            'guild_previous_accepted_message_count'
        ] = len(messages)
        repair_prompt = (
            build_conversation_json_repair_prompt(
                prompt,
                bot_names,
                message_only=True,
            )
        )
        response = call_llm(
            client,
            repair_prompt,
            config,
            max_tokens_override=conversation_tokens,
            context="guild-json-repair",
            label='guild_idle_chatter',
            metadata=repair_metadata,
        )
        messages = _clean_guild_conversation(
            parse_conversation_response(
                response or '',
                bot_names,
            )
        )[:message_count]

    if not _valid_guild_conversation(
        messages, bot_names
    ):
        return False

    reference_results = (
        _apply_participant_references(
            messages,
            reference_plans,
        )
    )
    guid_by_name = {
        participant['name']: participant['guid']
        for participant in participants
    }
    cumulative_delay = 2.0
    previous_length = 0
    for sequence, message in enumerate(messages):
        text = message['message']
        if sequence:
            cumulative_delay += calculate_dynamic_delay(
                len(text),
                config,
                prev_message_length=previous_length,
            )
        insert_chat_message(
            db,
            bot_guid=guid_by_name[message['name']],
            bot_name=message['name'],
            message=text,
            channel='guild',
            delay_seconds=cumulative_delay,
            event_id=event_id,
            sequence=sequence,
            owner_subsystem='guild',
        )
        previous_length = len(text)

    logger.info(
        "guild_idle_chatter mode=conversation "
        "participants=%s requested=%d accepted=%d "
        "repair=%s references=%s "
        "reference_fallbacks=%d topic=%r",
        ",".join(bot_names),
        message_count,
        len(messages),
        repair_used,
        ";".join(
            ",".join(result['targets'])
            for result in reference_results
        ) or 'none',
        sum(
            1 for result in reference_results
            if result['fallback']
        ),
        topic,
    )
    return True


def process_guild_idle_chatter_event(
    db, client, config, event
):
    """Handle a Guild statement or conversation event."""
    event_id = event['id']
    extra = parse_extra_data(
        event.get('extra_data'),
        event_id,
        'guild_idle_chatter',
    )
    if not extra:
        _mark_event(db, event_id, 'skipped')
        return False

    participants, mode = _normalize_guild_participants(
        event, extra
    )
    if not participants:
        _mark_event(db, event_id, 'skipped')
        return False

    loaded_participants = []
    for index, participant in enumerate(participants):
        speaker = _query_speaker(
            db, participant['guid']
        )
        if not speaker:
            if index == 0:
                _mark_event(db, event_id, 'skipped')
                return False
            continue
        loaded = dict(participant)
        loaded['speaker'] = speaker
        loaded_participants.append(loaded)

    requested_conversation = (mode == 'conversation')
    if len(loaded_participants) < 2:
        return _process_guild_statement_event(
            db,
            client,
            config,
            event,
            fallback=requested_conversation,
        )

    guild_name = extra.get('guild_name') or 'the guild'
    guildmates = extra.get('guildmates') or ''
    topic = random.choice(GUILD_CHAT_TOPICS_RP)
    primary = loaded_participants[0]
    faction = (
        extra.get('team')
        or _speaker_faction(primary['speaker'])
    )
    zone_name_chance = max(
        0,
        min(
            100,
            int(config.get(
                'LLMChatter.GuildChatter.'
                'ZoneNameChance',
                20,
            )),
        ),
    )
    name_zone = (
        random.randint(1, 100)
        <= zone_name_chance
    )

    completed = _generate_guild_conversation(
        db,
        client,
        config,
        event_id,
        extra,
        loaded_participants,
        guild_name,
        guildmates,
        topic,
        faction,
        name_zone,
    )
    if completed:
        _mark_event(db, event_id, 'completed')
        return True

    return _process_guild_statement_event(
        db,
        client,
        config,
        event,
        topic_override=topic,
        name_zone_override=name_zone,
        fallback=True,
    )
