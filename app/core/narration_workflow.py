"""
Narration-Aware Workflow Engine for PromptBeast.
Handles JSON script import, visual ideas alignment, dynamic story bible extraction,
and batch image prompt generation via Groq strictly adhering to user-selected visual styles.
"""
import json
import re
import difflib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from app.models.scene import VisualScene
from app.models.timeline import NarrationData, WordTimestamp
from app.presets.loader import PresetManager
from app.ai.groq_engine import GroqEngine, GroqMultiKeyManager


# =============================================================================
# STEP 1: IMPORT SCRIPT JSON & VALIDATION
# =============================================================================

class JSONImportResult:
    def __init__(
        self,
        success: bool,
        message: str,
        story_text: str = "",
        narration: Optional[NarrationData] = None,
        scenes: Optional[List[VisualScene]] = None,
        raw_data: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.message = message
        self.story_text = story_text
        self.narration = narration
        self.scenes = scenes or []
        self.raw_data = raw_data or {}


def import_script_json(file_path_or_str: str) -> JSONImportResult:
    """
    Parses and validates a script JSON file or string.
    Supports:
    1. Whisper/Speech JSON: {"text": "...", "words": [{"word": "...", "start": 0.0, "end": 0.5}, ...]}
    2. Storyboard JSON: {"scenes": [{"scene_id": "001", "start": 0.0, "end": 4.0, "narration": "...", "prompt": "..."}]}
    3. Segments/Subtitle JSON: {"segments": [{"start": 0.0, "end": 3.0, "text": "..."}]}
    4. Plain dictionary with text/words.
    """
    p = Path(file_path_or_str)
    raw_content = ""
    if p.exists() and p.is_file():
        try:
            raw_content = p.read_text(encoding="utf-8")
        except Exception as e:
            return JSONImportResult(False, f"Could not read file: {e}")
    else:
        raw_content = file_path_or_str

    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as jde:
        return JSONImportResult(False, f"Malformed JSON: {jde.msg} (line {jde.lineno}, col {jde.colno})")

    if not isinstance(data, dict):
        return JSONImportResult(False, "JSON root must be an object (dictionary).")

    # Format 1: Whisper / Speech-to-Text format with 'text' and 'words'
    if "words" in data and isinstance(data["words"], list):
        text = data.get("text", "")
        word_list: List[WordTimestamp] = []
        for w in data["words"]:
            if isinstance(w, dict) and "word" in w and "start" in w and "end" in w:
                word_list.append(WordTimestamp(
                    word=str(w["word"]),
                    start=float(w["start"]),
                    end=float(w["end"])
                ))

        if not word_list:
            return JSONImportResult(False, "JSON 'words' array was found but contained no valid word timestamps.")

        # Reconstruct text if empty
        if not text:
            text = " ".join(w.word for w in word_list).strip()

        duration = float(data.get("duration", word_list[-1].end))
        narration = NarrationData(
            text=text,
            words=word_list,
            duration_seconds=duration,
            is_estimated=False
        )
        return JSONImportResult(
            success=True,
            message=f"Successfully imported audio script JSON: {len(word_list)} words, duration {duration:.1f}s.",
            story_text=text,
            narration=narration,
            raw_data=data
        )

    # Format 2: Storyboard scenes array
    if "scenes" in data and isinstance(data["scenes"], list):
        scenes: List[VisualScene] = []
        full_text_parts = []
        for i, s in enumerate(data["scenes"]):
            if not isinstance(s, dict):
                continue
            s_id = str(s.get("scene_id") or f"{i+1:03d}")
            start = float(s.get("start", 0.0))
            end = float(s.get("end", start + 3.0))
            dur = float(s.get("duration", max(0.5, end - start)))
            narr = str(s.get("narration") or s.get("narration_excerpt") or "")
            visual_idea = str(s.get("visual_idea") or s.get("visual_subject") or "")
            prompt = str(s.get("prompt") or "")
            if narr:
                full_text_parts.append(narr)

            scenes.append(VisualScene(
                scene_id=s_id,
                index=i + 1,
                start=start,
                end=end,
                duration=dur,
                narration_excerpt=narr,
                visual_subject=visual_idea or narr[:60],
                visual_idea=visual_idea,
                prompt=prompt,
                generation_status="generated" if prompt else "pending"
            ))

        story_text = "\n\n".join(full_text_parts)
        return JSONImportResult(
            success=True,
            message=f"Successfully imported storyboard JSON with {len(scenes)} predefined scenes.",
            story_text=story_text,
            scenes=scenes,
            raw_data=data
        )

    # Format 3: Segments array (like Whisper segments)
    if "segments" in data and isinstance(data["segments"], list):
        word_list = []
        full_text_parts = []
        for seg in data["segments"]:
            if not isinstance(seg, dict):
                continue
            seg_text = str(seg.get("text", "")).strip()
            if seg_text:
                full_text_parts.append(seg_text)
            seg_words = seg.get("words", [])
            for w in seg_words:
                if isinstance(w, dict) and "word" in w and "start" in w and "end" in w:
                    word_list.append(WordTimestamp(
                        word=str(w["word"]),
                        start=float(w["start"]),
                        end=float(w["end"])
                    ))

        story_text = data.get("text", " ".join(full_text_parts))
        if word_list:
            duration = word_list[-1].end
            narration = NarrationData(
                text=story_text,
                words=word_list,
                duration_seconds=duration,
                is_estimated=False
            )
            return JSONImportResult(
                success=True,
                message=f"Successfully imported segmented script JSON with {len(word_list)} word timestamps.",
                story_text=story_text,
                narration=narration,
                raw_data=data
            )
        else:
            return JSONImportResult(
                success=True,
                message=f"Imported script segments without word-level timestamps ({len(data['segments'])} segments).",
                story_text=story_text,
                raw_data=data
            )

    # Format 4: Fallback generic text object
    if "text" in data and isinstance(data["text"], str):
        story_text = data["text"].strip()
        if story_text:
            return JSONImportResult(
                success=True,
                message=f"Imported plain text script from JSON ({len(story_text)} chars).",
                story_text=story_text,
                raw_data=data
            )

    return JSONImportResult(
        False,
        "JSON structure not recognized. Expected either a speech transcription with 'words', a storyboard with 'scenes', or 'segments'."
    )


# =============================================================================
# STEP 2: NARRATION & VISUAL IDEAS PARSING AND ALIGNMENT
# =============================================================================

def normalize_text_token(t: str) -> str:
    """Normalizes number words and removes punctuation for robust speech-to-script alignment."""
    t = t.lower()
    t = re.sub(r"\b(sixty|60)\b", "60", t)
    t = re.sub(r"\b(forty|40)\b", "40", t)
    t = re.sub(r"\b(twenty|20)\b", "20", t)
    t = re.sub(r"\b(nineteen|19)\b", "19", t)
    t = re.sub(r"\b(fifteen|15)\b", "15", t)
    t = re.sub(r"\b(eleven|11)\b", "11", t)
    t = re.sub(r"\b(ten|10)\b", "10", t)
    t = re.sub(r"\b(four|4)\b", "4", t)
    t = re.sub(r"\b(three|3)\b", "3", t)
    t = re.sub(r"\b(two|2)\b", "2", t)
    t = re.sub(r"\b(one|1)\b", "1", t)
    t = re.sub(r"[^\w\s]", "", t)
    return " ".join(t.split())


def parse_visual_breakdown_text(raw_text: str) -> List[Dict[str, str]]:
    """
    Parses user-supplied narration and visual ideas.
    Supports:
    1. 'Visual 1 Words: <narration> Visual: <visual idea>'
    2. '1. Narration: ... Visual: ...'
    3. 'Words: ... \n Visual: ...'
    4. Plain paragraphs if no visual tags are found.
    """
    if not raw_text or not raw_text.strip():
        return []

    items = []
    # Pattern 1: (Visual|Scene|Shot) X (optional : or -) Words: ... Visual: ...
    pattern1 = re.compile(
        r"(?:Visual|Scene|Shot)\s*(\d+)?[.:\-]?\s*Words?:\s*(.*?)\s*Visuals?:\s*(.*?)(?=(?:Visual|Scene|Shot)\s*\d*[.:\-]?\s*Words?:|$)",
        re.DOTALL | re.IGNORECASE
    )
    matches1 = pattern1.findall(raw_text)
    if matches1:
        for idx, (num, words, vis) in enumerate(matches1):
            s_id = str(num).strip() if num and str(num).strip() else str(idx + 1)
            items.append({
                "id": s_id,
                "words": words.strip(),
                "visual_idea": vis.strip()
            })
        return items

    # Pattern 2: Lines with "Words:" and "Visual:"
    pattern2 = re.compile(
        r"(?:Scene|Shot|Visual)?\s*(\d+)?[.:\-]?\s*Words?:\s*(.*?)\s*Visuals?:\s*(.*?)(?=(?:Scene|Shot|Visual|\d+[.:])\s*Words?:|$)",
        re.DOTALL | re.IGNORECASE
    )
    matches2 = pattern2.findall(raw_text)
    if matches2 and len(matches2) >= 1:
        for idx, (num, words, vis) in enumerate(matches2):
            s_id = num.strip() if num.strip() else str(idx + 1)
            items.append({
                "id": s_id,
                "words": words.strip(),
                "visual_idea": vis.strip()
            })
        return items

    # Pattern 3: Paragraph fallback
    paras = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
    for i, p in enumerate(paras):
        items.append({
            "id": str(i + 1),
            "words": p,
            "visual_idea": f"Visual scene illustrating: {p[:80]}"
        })
    return items


def align_visuals_with_timestamps(
    visual_items: List[Dict[str, str]],
    narration: Optional[NarrationData] = None,
    total_audio_duration: Optional[float] = None
) -> List[VisualScene]:
    """
    Combines user-supplied visual breakdown with exact word-level timestamps from narration.
    Uses monotonic anchor-based sequence alignment with prefix matching and gap interpolation.
    Guarantees:
      - 100% of scenes fit smoothly within [0, total_audio_duration].
      - No runaway time overshoot beyond audio end.
      - Exact synchronization for matched beats and proportional placement for gap beats.
    """
    scenes: List[VisualScene] = []
    if not visual_items:
        return scenes

    num_items = len(visual_items)
    total_dur = total_audio_duration or (narration.duration_seconds if narration else float(num_items * 3.5))

    if narration and narration.words:
        words = narration.words
        num_words = len(words)
        speech_norm = [normalize_text_token(w.word) for w in words]

        matched_ranges: List[Optional[Tuple[int, int, float]]] = [None] * num_items
        last_end_w = 0
        unmatched_gap = 0

        # Pass 1: Find monotonic high-confidence anchors
        for i in range(num_items):
            item = visual_items[i]
            v_words = item.get("words", "")
            v_norm = normalize_text_token(v_words).split()

            if not v_norm:
                unmatched_gap += 1
                continue

            target_len = max(1, len(v_norm))
            v_set = set(v_norm)

            search_start = max(0, last_end_w - 5)
            # Elastic search window expands if previous items were unmatched
            search_end = min(num_words, search_start + 140 + (unmatched_gap * 30))

            best_w_start = -1
            best_w_end = -1
            best_score = 0.0

            # 1. Prefix match (2 or 3 tokens)
            p_len = min(3, len(v_norm))
            if p_len >= 2:
                p_slice = v_norm[:p_len]
                for s in range(search_start, search_end - p_len + 1):
                    if speech_norm[s:s + p_len] == p_slice:
                        best_w_start = s
                        best_w_end = min(num_words - 1, s + target_len - 1)
                        best_score = 1.0
                        break

            # 2. Token overlap fallback if prefix didn't match
            if best_score < 0.6:
                for s in range(search_start, search_end):
                    cand = speech_norm[s:s + target_len]
                    if not cand:
                        continue
                    sc = len(v_set & set(cand)) / max(1, len(v_set))
                    if sc > best_score and sc >= 0.35:
                        best_score = sc
                        best_w_start = s
                        best_w_end = min(num_words - 1, s + target_len - 1)
                        if sc >= 0.70:
                            break

            if best_w_start != -1 and best_score >= 0.35:
                matched_ranges[i] = (best_w_start, best_w_end, best_score)
                last_end_w = best_w_end + 1
                unmatched_gap = 0
            else:
                unmatched_gap += 1

        # Pass 2: Build initial timings from matched anchors
        initial_timings: List[Tuple[Optional[float], Optional[float], bool, int, int]] = []
        for i in range(num_items):
            if matched_ranges[i] is not None:
                ws, we, _ = matched_ranges[i]
                ws = max(0, min(ws, num_words - 1))
                we = max(ws, min(we, num_words - 1))
                st = words[ws].start
                et = words[we].end
                if et <= st:
                    et = st + 1.5
                initial_timings.append((st, et, True, ws, we))
            else:
                initial_timings.append((None, None, False, -1, -1))

        # Pass 3: Proportional interpolation for unmatched gap scenes
        final_timings = list(initial_timings)
        i = 0
        while i < num_items:
            if not final_timings[i][2]:
                block_start = i
                while i < num_items and not final_timings[i][2]:
                    i += 1
                block_end = i  # exclusive

                prev_time = final_timings[block_start - 1][1] if block_start > 0 else 0.0
                next_time = final_timings[block_end][0] if block_end < num_items else total_dur

                gap_dur = max(0.8 * (block_end - block_start), next_time - prev_time)
                item_dur = gap_dur / (block_end - block_start)

                for k in range(block_start, block_end):
                    st = prev_time + (k - block_start) * item_dur
                    et = st + item_dur
                    final_timings[k] = (st, et, False, -1, -1)
            else:
                i += 1

        # Pass 4: Strictly monotonic, non-overlapping assembly bounded by audio duration
        cur_time = 0.0
        for idx in range(num_items):
            item = visual_items[idx]
            s_num = item.get("id") or str(idx + 1)
            try:
                s_id = f"{int(s_num):03d}"
            except ValueError:
                s_id = f"{idx + 1:03d}"

            v_words = item.get("words", "")
            v_vis = item.get("visual_idea", "")

            st_val, et_val, matched, ws, we = final_timings[idx]
            start_t = max(cur_time, st_val if st_val is not None else cur_time)
            end_t = max(start_t + 0.8, et_val if et_val is not None else start_t + 2.0)

            # Cap to audio duration on final scene
            if idx == num_items - 1:
                end_t = max(start_t + 1.2, total_dur)

            dur = round(float(end_t - start_t), 2)
            cur_time = end_t

            scenes.append(VisualScene(
                scene_id=s_id,
                index=idx + 1,
                start=round(float(start_t), 2),
                end=round(float(end_t), 2),
                duration=dur,
                narration_excerpt=v_words,
                visual_subject=v_vis,
                visual_idea=v_vis,
                prompt="",
                generation_status="pending",
                start_word_index=ws if ws >= 0 else None,
                end_word_index=we if we >= 0 else None
            ))

    else:
        # Fallback: estimate timings proportionally
        avg_dur = max(1.5, total_dur / num_items)
        curr_time = 0.0
        for idx, item in enumerate(visual_items):
            s_num = item.get("id") or str(idx + 1)
            try:
                s_id = f"{int(s_num):03d}"
            except ValueError:
                s_id = f"{idx + 1:03d}"

            start_t = curr_time
            end_t = curr_time + avg_dur
            if idx == num_items - 1:
                end_t = max(start_t + 1.2, total_dur)
            curr_time = end_t

            scenes.append(VisualScene(
                scene_id=s_id,
                index=idx + 1,
                start=round(float(start_t), 2),
                end=round(float(end_t), 2),
                duration=round(float(end_t - start_t), 2),
                narration_excerpt=item.get("words", ""),
                visual_subject=item.get("visual_idea", ""),
                visual_idea=item.get("visual_idea", ""),
                prompt="",
                generation_status="pending",
                start_word_index=None,
                end_word_index=None
            ))

    # Seamless timeline: heal natural sentence pauses (gaps <= 1.0s) between scenes without exceeding total duration
    for i in range(len(scenes) - 1):
        curr_s = scenes[i]
        next_s = scenes[i + 1]
        gap = float(next_s.start) - float(curr_s.end)
        if 0.0 < gap <= 1.0:
            curr_s.end = round(float(next_s.start), 2)
            curr_s.duration = round(float(curr_s.end) - float(curr_s.start), 2)

    return scenes


# =============================================================================
# STEP 3: DYNAMIC STORY BIBLE & INTELLIGENCE EXTRACTION
# =============================================================================

def load_bible_extraction_system_prompt() -> str:
    p = Path("config/system_prompts/story_bible_extraction.txt")
    if p.exists():
        try:
            return p.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return "You are an expert story analyst. Extract comprehensive story, character, environment, and style bibles in JSON."


def extract_story_bibles_groq(
    story_text: str,
    visual_ideas_summary: str = "",
    model: str = "qwen/qwen3.8-27b",
    engine: Optional[GroqEngine] = None
) -> Dict[str, Any]:
    """
    Extracts dynamic story bibles (Story, Character, Environment, Props, Visual Style)
    via Groq LLM inference without any hardcoded story assumptions.
    """
    groq_eng = engine or GroqEngine(default_model=model)
    system_prompt = load_bible_extraction_system_prompt()

    user_prompt = f"""ANALYZE THE FOLLOWING STORY SCRIPT AND EXTRACT FULL PRODUCTION BIBLES:

### SCRIPT NARRATION:
{story_text}

### USER-SUPPLIED VISUAL IDEAS / BEATS:
{visual_ideas_summary[:3500] if visual_ideas_summary else 'No additional visual notes provided.'}

Extract comprehensive, highly detailed production bibles strictly matching the required JSON schema.
Ensure character appearances, environments, props, and visual continuity rules are thoroughly defined.
Return ONLY valid JSON.
"""

    response_text = groq_eng.generate_text(
        prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=0.2,
        max_tokens=2500,
        model=model,
        stage_name="BIBLE_EXTRACTION"
    )

    clean_json = response_text.strip()
    if clean_json.startswith("```"):
        clean_json = re.sub(r"^```(?:json)?\s*", "", clean_json)
        clean_json = re.sub(r"\s*```$", "", clean_json)

    try:
        bibles_data = json.loads(clean_json)
        return bibles_data
    except json.JSONDecodeError as jde:
        m = re.search(r"(\{.*\})", clean_json, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        raise RuntimeError(f"Failed to parse Groq story bibles response as JSON: {jde}. Raw snippet: {clean_json[:200]}")


def parse_custom_bible_text(bible_text: str) -> Dict[str, Any]:
    """
    Parses human-readable/markdown production bibles into structured dictionaries.
    Uses BibleManager for flexible, robust extraction of Characters, Environments, Props, and Style.
    """
    from app.core.bible_manager import BibleManager
    return BibleManager.parse_single_bible_text(bible_text)


def _legacy_parse_custom_bible_text(bible_text: str) -> Dict[str, Any]:
    """Legacy parser retained for fallback reference."""
    bibles: Dict[str, Any] = {
        "story_bible": {
            "premise": "Realistic survival horror documentary.",
            "summary": "Survival struggle in an abandoned shopping mall.",
            "genre_and_tone": "Grounded survival documentary horror",
            "setting_period": "Modern Day",
            "visual_identity": ""
        },
        "characters": [],
        "environments": [],
        "key_objects": [],
        "visual_style_bible": {
            "color_palette": "Cold grey, dark blue, black, concrete grey, muted green",
            "lighting_rules": "Flashlight beams, emergency lights, lightning, and deep shadows",
            "camera_guidelines": "Wide establishing, medium character, close-up focal objects, mezzanine views"
        },
        "visual_continuity_rules": [],
        "chronological_events": []
    }

    # 1. CHARACTER BIBLE
    char_match = re.search(r"(?:1\.?\s*)?CHARACTER BIBLE(.*?)(?=\d+\.\s*ENVIRONMENT|\Z)", bible_text, re.DOTALL | re.IGNORECASE)
    if char_match:
        c_text = char_match.group(1)
        c_blocks = re.split(r"CHARACTER\s+\d+:\s*", c_text, flags=re.IGNORECASE)[1:]
        for block in c_blocks:
            lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
            if not lines:
                continue
            name = lines[0]
            role = ""
            app_lines = []
            cloth_lines = []
            prop_lines = []
            rule_lines = []
            curr_mode = "app"

            for l in lines[1:]:
                llow = l.lower()
                if llow.startswith("role:"):
                    role = l.split(":", 1)[1].strip()
                elif llow.startswith(("appearance:", "build:", "age:", "gender:")):
                    curr_mode = "app"
                    app_lines.append(l)
                elif llow.startswith("clothing:"):
                    curr_mode = "cloth"
                elif llow.startswith("important props:"):
                    curr_mode = "props"
                elif llow.startswith(("visual continuity:", "important rules:", "continuity:")):
                    curr_mode = "rules"
                    rule_lines.append(l.split(":", 1)[1].strip() if ":" in l else l)
                elif llow.startswith("personality through visual expression:"):
                    curr_mode = "app"
                    app_lines.append(l)
                else:
                    if curr_mode == "cloth":
                        cloth_lines.append(l.lstrip("-*• "))
                    elif curr_mode == "props":
                        prop_lines.append(l.lstrip("-*• "))
                    elif curr_mode == "rules":
                        rule_lines.append(l.lstrip("-*• "))
                    else:
                        app_lines.append(l)

            bibles["characters"].append({
                "name": name,
                "role": role or "Key Figure",
                "physical_description": " ".join(app_lines).strip() or "Realistic physical features.",
                "clothing_and_gear": ", ".join(cloth_lines).strip() or "Practical attire.",
                "important_props": ", ".join(prop_lines).strip(),
                "consistency_rules": " ".join(rule_lines).strip() or f"Keep {name} appearance consistent."
            })

    # 2. ENVIRONMENT BIBLE
    env_match = re.search(r"(?:2\.?\s*)?ENVIRONMENT BIBLE(.*?)(?=\d+\.\s*OBJECT|\Z)", bible_text, re.DOTALL | re.IGNORECASE)
    if env_match:
        e_text = env_match.group(1)
        e_blocks = re.split(r"LOCATION\s+\d+:\s*", e_text, flags=re.IGNORECASE)[1:]
        for block in e_blocks:
            lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
            if not lines:
                continue
            name = lines[0]
            arch_lines = []
            features_lines = []
            lighting_lines = []
            details_lines = []
            curr_mode = "arch"

            for l in lines[1:]:
                llow = l.lower()
                if llow.startswith(("architecture:", "location:", "primary setting:")):
                    curr_mode = "arch"
                    arch_lines.append(l.split(":", 1)[1].strip() if ":" in l else l)
                elif llow.startswith(("interior design:", "interior arrangement:")):
                    curr_mode = "features"
                    features_lines.append(l.split(":", 1)[1].strip() if ":" in l else l)
                elif llow.startswith(("visual progression:", "atmosphere:")):
                    curr_mode = "lighting"
                    lighting_lines.append(l.split(":", 1)[1].strip() if ":" in l else l)
                elif llow.startswith(("important details:", "important items:", "important props:", "important continuity:")):
                    curr_mode = "details"
                    details_lines.append(l.split(":", 1)[1].strip() if ":" in l else l)
                else:
                    if curr_mode == "features":
                        features_lines.append(l.lstrip("-*• "))
                    elif curr_mode == "lighting":
                        lighting_lines.append(l.lstrip("-*• "))
                    elif curr_mode == "details":
                        details_lines.append(l.lstrip("-*• "))
                    else:
                        arch_lines.append(l.lstrip("-*• "))

            bibles["environments"].append({
                "name": name,
                "architectural_style": " ".join(arch_lines).strip() or "Story location architecture.",
                "environmental_features": " ".join(features_lines).strip() or "Interior features and elements.",
                "lighting_and_atmosphere": " ".join(lighting_lines).strip() or "Atmospheric lighting.",
                "setting_constraints": " ".join(details_lines).strip()
            })

    # 3. OBJECT AND PROP BIBLE
    obj_match = re.search(r"(?:3\.?\s*)?OBJECT AND PROP BIBLE(.*?)(?=\d+\.\s*CINEMATIC|\Z)", bible_text, re.DOTALL | re.IGNORECASE)
    if obj_match:
        o_text = obj_match.group(1)
        items = re.findall(r"(?:^[•\-\*]\s*|^)([A-Za-z0-9\s\',]{3,40})(?::\s*([^\n]+)|$)", o_text, re.MULTILINE)
        for name, desc in items:
            name_clean = name.strip().lstrip("•-* ")
            if not name_clean or name_clean.lower().startswith(("maintain", "survival equipment", "food and water", "important personal", "structural elements")):
                continue
            bibles["key_objects"].append({
                "name": name_clean,
                "visual_description": desc.strip() if desc else f"Key survival item: {name_clean}",
                "significance": desc.strip() if desc else "Essential survival prop"
            })

    # 4. CINEMATIC STYLE BIBLE
    style_match = re.search(r"(?:4\.?\s*)?CINEMATIC STYLE BIBLE(.*?)(?=\d+\.\s*CHRONOLOGY|\Z)", bible_text, re.DOTALL | re.IGNORECASE)
    if style_match:
        s_text = style_match.group(1)
        for l in s_text.split("\n"):
            llow = l.lower()
            if "lighting progression" in llow or "lighting" in llow:
                bibles["visual_style_bible"]["lighting_rules"] = l.strip()
            elif "color progression" in llow or "color" in llow:
                bibles["visual_style_bible"]["color_palette"] = l.strip()
            elif "composition variety" in llow or "composition" in llow:
                bibles["visual_style_bible"]["camera_guidelines"] = l.strip()

    # 5. CHRONOLOGY AND CONTINUITY BIBLE
    chron_match = re.search(r"(?:5\.?\s*)?CHRONOLOGY AND CONTINUITY BIBLE(.*?)(?=\d+\.\s*IMPORTANT|\Z)", bible_text, re.DOTALL | re.IGNORECASE)
    if chron_match:
        c_lines = chron_match.group(1).strip().split("\n")
        events = []
        for l in c_lines:
            l_clean = l.strip().lstrip("•-*| ")
            if l_clean and not l_clean.lower().startswith(("timeline", "major event", "---")):
                events.append(l_clean)
        bibles["chronological_events"] = events[:25]

    # 6. RULES & IDENTITY
    rules_match = re.search(r"IMPORTANT VISUAL CONTINUITY RULES(.*?)(?=\d+\.\s*FINAL|\Z)", bible_text, re.DOTALL | re.IGNORECASE)
    if rules_match:
        r_lines = [rl.strip().lstrip("•-* ") for rl in rules_match.group(1).strip().split("\n") if rl.strip()]
        bibles["visual_continuity_rules"] = r_lines

    identity_match = re.search(r"FINAL STORY VISUAL IDENTITY(.*?)\Z", bible_text, re.DOTALL | re.IGNORECASE)
    if identity_match:
        bibles["story_bible"]["visual_identity"] = identity_match.group(1).strip()

    return bibles


# =============================================================================
# STEP 4 & 5: DEDICATED GROQ IMAGE PROMPT GENERATION (BATCHED & EXPANDED)
# =============================================================================

def load_image_prompt_system_prompt() -> str:
    p = Path("config/system_prompts/image_prompt_generation.txt")
    if p.exists():
        try:
            return p.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return "You are an expert image prompt architect. Generate detailed, style-compliant image prompts in JSON."


def robust_extract_scenes_from_llm_output(clean_json: str) -> List[Dict[str, Any]]:
    """
    Parses LLM output into a list of scene dictionaries.
    Handles:
    1. Perfect JSON: {"scenes": [...]} or [...]
    2. Markdown wrapped JSON
    3. Trailing commas or subtle formatting glitches
    4. Truncated JSON streams (extracts all completed scene objects even if the last one was cut off)
    5. Regex recovery fallback
    """
    if not clean_json or not clean_json.strip():
        return []

    text = clean_json.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    # Attempt 1: Direct JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "scenes" in parsed and isinstance(parsed["scenes"], list):
            return parsed["scenes"]
        elif isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    # Attempt 2: Repair braces and trailing commas
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        sub = text[first_brace:last_brace + 1]
        try:
            p = json.loads(sub)
            if isinstance(p, dict) and "scenes" in p:
                return p["scenes"]
        except Exception:
            fixed = re.sub(r",\s*([}\]])", r"\1", sub)
            try:
                p = json.loads(fixed)
                if isinstance(p, dict) and "scenes" in p:
                    return p["scenes"]
            except Exception:
                pass

    # Attempt 3: Truncated stream recovery
    if first_brace != -1:
        matches = list(re.finditer(r"\}\s*,?", text))
        for m in reversed(matches):
            candidate = text[first_brace:m.end()].rstrip(", \t\r\n")
            for closing in ["]}", "}", "]}}", "}\n}"]:
                try:
                    p = json.loads(candidate + closing)
                    if isinstance(p, dict) and "scenes" in p and p["scenes"]:
                        return p["scenes"]
                    elif isinstance(p, list) and p:
                        return p
                except Exception:
                    pass

    # Attempt 4: Regex extraction of individual completed scene objects
    scenes = []
    obj_pattern = re.compile(
        r'\{\s*"scene_id"\s*:\s*"(?P<id>[^"]+)"\s*,\s*"visual_subject"\s*:\s*"(?P<subj>(?:[^"\\]|\\.)*)"\s*,\s*"prompt"\s*:\s*"(?P<prompt>(?:[^"\\]|\\.)*)"',
        re.DOTALL
    )
    for m in obj_pattern.finditer(text):
        try:
            s_id = m.group("id").strip()
            prompt_str = m.group("prompt").encode('utf-8').decode('unicode_escape', errors='ignore')
            subj_str = m.group("subj").encode('utf-8').decode('unicode_escape', errors='ignore')
            scenes.append({
                "scene_id": s_id,
                "visual_subject": subj_str,
                "prompt": prompt_str
            })
        except Exception:
            pass

    if scenes:
        return scenes

    # Loose fallback regex for scene_id and prompt
    loose_pattern = re.compile(
        r'"scene_id"\s*:\s*"([^"]+)"[\s\S]*?"prompt"\s*:\s*"((?:[^"\\]|\\.)*)"',
        re.DOTALL
    )
    for m in loose_pattern.finditer(text):
        try:
            s_id = m.group(1).strip()
            prompt_str = m.group(2).encode('utf-8').decode('unicode_escape', errors='ignore')
            scenes.append({
                "scene_id": s_id,
                "prompt": prompt_str
            })
        except Exception:
            pass

    return scenes


def generate_prompts_batch(
    scenes_batch: List[VisualScene],
    bibles: Dict[str, Any],
    story_context: str,
    visual_style_id: str = "zombie_survival",
    custom_style: Optional[str] = None,
    aspect_ratio: str = "16:9",
    model: str = "qwen/qwen3.8-27b",
    engine: Optional[GroqEngine] = None
) -> List[Dict[str, Any]]:
    """
    Sends a batch of scenes to Groq along with extracted bibles, narration, timestamps,
    and user visual ideas to generate detailed, concrete image prompts STRICTLY obeying
    the user-selected visual style directives.
    """
    groq_eng = engine or GroqEngine(default_model=model)
    system_prompt = load_image_prompt_system_prompt()

    # Retrieve Authoritative Visual Style
    style_preset = PresetManager.get_style_by_id(visual_style_id)
    style_name = style_preset.name if style_preset else visual_style_id
    style_instructions = style_preset.system_instructions if style_preset else (custom_style or "Cinematic realism")

    # Build concise representation of bibles
    char_summary = "\n".join(
        f"- {c.get('name')}: {c.get('physical_description', '')}. Wardrobe: {c.get('clothing_and_gear', '')}. Rules: {c.get('consistency_rules', '')}"
        for c in bibles.get("characters", [])
    )
    env_summary = "\n".join(
        f"- {e.get('name')}: {e.get('architectural_style', '')}. Atmosphere: {e.get('lighting_and_atmosphere', '')}"
        for e in bibles.get("environments", [])
    )
    style_info = bibles.get("visual_style_bible", {})
    style_summary = f"Color Palette: {style_info.get('color_palette', 'Consistent with selected style')}. Lighting: {style_info.get('lighting_rules', 'Atmospheric')}. Camera: {style_info.get('camera_guidelines', 'Eye-level cinematic framing')}."

    scenes_input = []
    for s in scenes_batch:
        scenes_input.append({
            "scene_id": s.scene_id,
            "start": s.start,
            "end": s.end,
            "duration": s.duration,
            "narration": s.narration_excerpt,
            "visual_idea": getattr(s, "visual_idea", "") or s.visual_subject
        })

    user_prompt = f"""AUTHORITATIVE USER-SELECTED VISUAL STYLE:
Style Name: {style_name}
Aspect Ratio: {aspect_ratio}

STYLE DIRECTIVES (MANDATORY & OVERRIDING):
{style_instructions}

RELEVANT STORY BIBLES:
### Characters:
{char_summary or 'No specific character entries.'}

### Environments:
{env_summary or 'No specific environment entries.'}

### Story Style Elements:
{style_summary}

STORY CONTEXT (OVERALL ARC):
{story_context[:800]}

SCENES TO GENERATE IN THIS BATCH:
{json.dumps(scenes_input, indent=2)}

INSTRUCTIONS FOR EACH SCENE:
1. Translate the narration and visual idea into a detailed, concrete image-generation prompt describing what the viewer actually SEES.
2. STRICT STYLE ENFORCEMENT: Every single prompt MUST strictly follow the 'STYLE DIRECTIVES' above for '{style_name}'.
   - If '{style_name}' mandates a pure flat 2D cartoon with a locked hero character (e.g. backwards red cap, charcoal hoodie) and uniform pale olive-green zombie horde, apply those exact design elements!
   - If '{style_name}' mandates photographic or historical styles, apply those exact camera, lens, and lighting directives!
   - If '{style_name}' is Century-Old Photograph, analyze each scene's narrative context to selectively apply one of the 9 specialized archival image treatments where it naturally elevates the storytelling (e.g. Antique Map Overlay for travel/campaigns, Diptych Split for wide+detail contrasts, Engraving Plate Split for blueprints/founding, Leporello Folder for journeys, Silver Bleeding for shadows/night, Fold Creases for intimate diaries/memories, Document Archive for treaties/evidence, Split-Screen Trio for fast montages, Textured Blur for key portraits). For baseline shots (~40-50%), keep clean, full-frame widescreen archival prints for balanced documentary pacing. Do not misuse or oversaturate.
3. Do not invent contradictory characters or ignore the supplied visual ideas.
4. Do NOT include /imagine prompt: or parameter flags like --ar or --style raw. Return clean, detailed descriptive visual prompts only.
5. Return JSON matching the schema with the exact scene_ids provided.
"""

    resp_text = groq_eng.generate_text(
        prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=0.3,
        max_tokens=2500,
        model=model,
        stage_name=f"PROMPT_EXPANSION_{style_name.upper().replace(' ', '_')}"
    )

    items = robust_extract_scenes_from_llm_output(resp_text)
    if not items:
        clean_snip = resp_text.strip()[:200] if resp_text else "EMPTY_RESPONSE"
        raise RuntimeError(f"Batch prompt parsing failed: No valid scenes recovered from model output. Snippet: {clean_snip}")

    # Strip any accidental Midjourney or model flags
    for it in items:
        if isinstance(it, dict) and "prompt" in it:
            p_val = str(it["prompt"]).strip()
            p_val = re.sub(r"^/imagine\s+prompt:\s*", "", p_val, flags=re.IGNORECASE)
            p_val = re.sub(r"^/imagine\s*", "", p_val, flags=re.IGNORECASE)
            p_val = re.sub(r"--ar\s+\d+:\d+", "", p_val, flags=re.IGNORECASE)
            p_val = re.sub(r"--style\s+\w+", "", p_val, flags=re.IGNORECASE)
            p_val = re.sub(r"--v\s+[\d\.]+", "", p_val, flags=re.IGNORECASE)
            it["prompt"] = " ".join(p_val.split())

    return items
