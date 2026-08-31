"""
Ableton Push 3 — Mobile Beat Card Extractor (Streamlit Web App)
Run locally or deploy to Streamlit Community Cloud / Hugging Face Spaces.
Allows mobile devices (iPhone/iPad/Android) to upload drum clips, extract 16-step patterns,
and download 1080x1920 OLED Push 3 Beat Cards directly to Camera Roll / Files.
"""

import os
import io
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# Audio analysis imports
try:
    import librosa
    import soundfile as sf
    import scipy.signal
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


# ==============================================================================
# AUDIO EXTRACTION & MULTI-BAR TRANSIENT ENGINE (v2.0)
# ==============================================================================

def analyze_audio_clip(file_bytes_or_path, bar_choice="Bar 1", detail_level="Standard Groove (Medium)"):
    """
    Analyzes drum loops, detects tempo, trims silence, breaks into bars,
    and isolates Kick (<130Hz), Snare (180-1400Hz), and Hats (>4000Hz) onto 16 steps.
    """
    if not LIBROSA_AVAILABLE:
        raise RuntimeError("librosa, soundfile, and scipy are required. Run: pip install librosa soundfile scipy")

    # Load audio
    if isinstance(file_bytes_or_path, (str, os.PathLike)):
        y, sr = librosa.load(file_bytes_or_path, sr=44100, mono=True)
    else:
        # Streamlit UploadedFile (BytesIO)
        y, sr = librosa.load(file_bytes_or_path, sr=44100, mono=True)

    # 1. Trim leading silence to align Beat 1 with Step 1
    y_trimmed, _ = librosa.effects.trim(y, top_db=32)
    if len(y_trimmed) < sr * 0.5:
        y_trimmed = y
    y = y_trimmed

    # 2. Detect Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, np.ndarray):
        tempo = tempo[0]
    bpm = int(round(float(tempo)))
    if bpm < 65:
        bpm *= 2
    elif bpm > 190:
        bpm = int(round(bpm / 2))

    quarter_duration = 60.0 / bpm
    step_duration = quarter_duration / 4.0
    bar_duration = quarter_duration * 4.0

    total_duration = len(y) / sr
    total_bars = max(1, int(round(total_duration / bar_duration)))

    # 3. Sensitivity Thresholds based on Detail Level
    if "High Nuance" in detail_level:
        k_thresh, s_thresh, h_thresh = 0.16, 0.18, 0.14
    elif "Core Skeleton" in detail_level:
        k_thresh, s_thresh, h_thresh = 0.40, 0.45, 0.35
    else:  # Standard Groove
        k_thresh, s_thresh, h_thresh = 0.25, 0.28, 0.22

    # 4. Short-Time Fourier Transform (STFT) for Spectral Separation
    n_fft = 2048
    hop_length = 512
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    frame_times = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr, hop_length=hop_length)

    # Frequency Bins
    kick_mask = (freqs <= 130)
    snare_mask = (freqs >= 180) & (freqs <= 1400)
    hats_mask = (freqs >= 4000)

    kick_env = np.sum(D[kick_mask, :], axis=0)
    snare_env = np.sum(D[snare_mask, :], axis=0)
    hats_env = np.sum(D[hats_mask, :], axis=0)

    # Normalize envelopes
    if np.max(kick_env) > 0: kick_env /= np.max(kick_env)
    if np.max(snare_env) > 0: snare_env /= np.max(snare_env)
    if np.max(hats_env) > 0: hats_env /= np.max(hats_env)

    # 5. Extract Step Hits Bar by Bar
    bars_data = []
    for b in range(total_bars):
        b_start = b * bar_duration
        k_steps = [0] * 16
        s_steps = [0] * 16
        h_steps = [0] * 16

        for s in range(16):
            t_center = b_start + (s * step_duration)
            if t_center >= total_duration:
                break
            idx_start = np.searchsorted(frame_times, t_center - (step_duration * 0.4))
            idx_end = np.searchsorted(frame_times, t_center + (step_duration * 0.4))
            if idx_end > idx_start:
                k_val = np.max(kick_env[idx_start:idx_end])
                s_val = np.max(snare_env[idx_start:idx_end])
                h_val = np.max(hats_env[idx_start:idx_end])

                if k_val >= k_thresh:
                    k_steps[s] = 100 if k_val >= (k_thresh * 1.5) else 50
                if s_val >= s_thresh:
                    if k_val < (k_thresh * 1.3):
                        s_steps[s] = 100 if s_val >= (s_thresh * 1.4) else 50
                if h_val >= h_thresh:
                    h_steps[s] = 100 if h_val >= (h_thresh * 1.4) else 50

        bars_data.append({"kick": k_steps, "snare": s_steps, "hats": h_steps})

    # 6. Select Target Bar
    if bar_choice == "Dominant Groove":
        target_kick = [0] * 16
        target_snare = [0] * 16
        target_hats = [0] * 16
        for s in range(16):
            k_counts = sum(1 for b in bars_data if b["kick"][s] > 0)
            s_counts = sum(1 for b in bars_data if b["snare"][s] > 0)
            h_counts = sum(1 for b in bars_data if b["hats"][s] > 0)
            if k_counts >= (total_bars / 2.0):
                target_kick[s] = 100
            if s_counts >= (total_bars / 2.0):
                target_snare[s] = 100
            if h_counts >= (total_bars / 2.0):
                target_hats[s] = 100
    else:
        # Parse 'Bar 1', 'Bar 2', etc.
        try:
            bar_idx = int(bar_choice.split()[1]) - 1
            bar_idx = max(0, min(total_bars - 1, bar_idx))
        except Exception:
            bar_idx = 0
        target_kick = bars_data[bar_idx]["kick"]
        target_snare = bars_data[bar_idx]["snare"]
        target_hats = bars_data[bar_idx]["hats"]

    # 7. Kit & Swing Suggestions
    if bpm >= 135 and (target_kick[0] and (target_snare[8] or target_snare[4])):
        suggested_kit = "Kit-Core 808 (Trap / Halftime)"
    elif bpm in range(84, 98):
        suggested_kit = "Kit-Vinyl Chop (Boom Bap / Golden Era)"
    elif bpm in range(125, 136):
        suggested_kit = "Kit-Core 909 (UK Garage / House)"
    else:
        suggested_kit = "Kit-Session Drums (Acoustic / Hybrid)"

    if bpm in range(125, 138) and target_kick[4] == 0:
        swing_str = "MPC 16 Swing 58% (UKG Pocket)"
    elif bpm in range(80, 96):
        swing_str = "MPC60 16 Swing 56% (Boom Bap)"
    elif bpm in range(110, 124):
        swing_str = "Logic 16 Swing 54% (Rolling Bounce)"
    else:
        swing_str = "Straight 0% (Clean Grid)"

    return {
        "bpm": bpm,
        "total_bars": total_bars,
        "bar_used": bar_choice,
        "kit": suggested_kit,
        "swing": swing_str,
        "kick": target_kick,
        "snare": target_snare,
        "hats": target_hats
    }


# ==============================================================================
# CARD RENDERING ENGINE (1080x1920 VERTICAL BEAT CARD)
# ==============================================================================

def render_beat_card_image(metadata):
    """
    Renders an OLED Black Ableton Push 3 Beat Card at 1080x1920 PNG
    using PIL and returns an in-memory PIL Image object.
    """
    width, height = 1080, 1920
    im = Image.new("RGB", (width, height), (10, 11, 15))  # #0A0B0F Pure OLED Black
    draw = ImageDraw.Draw(im)

    # Color Palette
    BG_BLACK = (10, 11, 15)
    PANEL_BG = (18, 20, 28)
    PANEL_BORDER = (40, 46, 62)
    BLUE_ELECTRIC = (29, 99, 255)     # #1D63FF
    BLUE_GHOST = (22, 78, 170)        # #164EAA
    PAD_OFF_DOWNBEAT = (36, 40, 52)
    PAD_OFF_OFFBEAT = (44, 48, 64)
    PAD_OFF_BORDER = (70, 78, 102)
    TEXT_WHITE = (255, 255, 255)
    TEXT_MUTED = (160, 170, 195)
    TEXT_CYAN = (0, 210, 255)
    AMBER = (255, 179, 71)

    def get_font(size, bold=False):
        font_candidates = [
            "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "Arial Bold.ttf" if bold else "Arial.ttf",
            "Helvetica.ttc",
            "segoeui.ttf"
        ]
        for candidate in font_candidates:
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:
                pass
        return ImageFont.load_default()

    f_title = get_font(42, bold=True)
    f_sub = get_font(20, bold=False)
    f_badge = get_font(18, bold=True)
    f_inst = get_font(22, bold=True)
    f_inst_meta = get_font(17, bold=False)
    f_pad_num = get_font(16, bold=True)
    f_body_bold = get_font(19, bold=True)
    f_body = get_font(18, bold=False)
    f_tip = get_font(16, bold=False)

    # 1. Top Header
    title_text = metadata.get("title", "EXTRACTED AUDIO BEAT CARD").upper()
    draw.text((62, 54), title_text, fill=TEXT_WHITE, font=f_title)
    sub_label = f"ABLETON PUSH 3 SEQUENCER MAP  •  {metadata.get('bar_used', 'BAR 1').upper()}"
    draw.text((62, 108), sub_label, fill=TEXT_CYAN, font=f_sub)

    # Badges
    badges = [
        f"TEMPO: {metadata.get('bpm', 120)} BPM",
        f"KIT: {metadata.get('kit', 'Kit-Core 909')[:24]}",
        f"SWING: {metadata.get('swing', '0% Straight')[:24]}"
    ]
    cur_x = 62
    for b in badges:
        bbox = draw.textbbox((0, 0), b, font=f_badge)
        badge_w = (bbox[2] - bbox[0]) + 24
        draw.rounded_rectangle([cur_x, 142, cur_x + badge_w, 176], radius=6, fill=(24, 28, 40), outline=(50, 60, 84), width=1)
        draw.text((cur_x + 12, 149), b, fill=TEXT_CYAN, font=f_badge)
        cur_x += badge_w + 14

    draw.line([(62, 196), (width - 62, 196)], fill=PANEL_BORDER, width=2)

    # 2. Drum Lanes (Kick, Snare, Hats)
    tracks = [
        ("KICK DRUM", "Transients detected in sub/low-end band (< 130 Hz)", metadata.get("kick", [0]*16)),
        ("SNARE / CLAP", "Mid-frequency punch & acoustic crack (180 – 1400 Hz)", metadata.get("snare", [0]*16)),
        ("HI-HATS / PERCUSSION", "High-frequency sizzle & 16th shuffles (> 4000 Hz)", metadata.get("hats", [0]*16))
    ]

    pad_start_y = 226
    lane_spacing = 350
    pad_w = 104
    pad_h = 76
    pad_gap = 18
    row_gap = 14

    for idx, (inst_name, inst_desc, steps) in enumerate(tracks):
        y_base = pad_start_y + (idx * lane_spacing)
        draw.text((62, y_base), inst_name, fill=TEXT_WHITE, font=f_inst)
        draw.text((62, y_base + 32), inst_desc, fill=TEXT_MUTED, font=f_inst_meta)

        for step in range(16):
            row = 0 if step < 8 else 1
            col = step if step < 8 else step - 8

            px = 62 + col * (pad_w + pad_gap)
            py = y_base + 72 + row * (pad_h + row_gap)

            is_downbeat = (step % 4 == 0)
            val = steps[step]

            if val >= 80:
                fill_color = BLUE_ELECTRIC
                border_color = (120, 170, 255)
                text_col = TEXT_WHITE
            elif val > 0:
                fill_color = BLUE_GHOST
                border_color = (80, 130, 220)
                text_col = (200, 220, 255)
            else:
                fill_color = PAD_OFF_DOWNBEAT if is_downbeat else PAD_OFF_OFFBEAT
                border_color = PAD_OFF_BORDER
                text_col = (110, 125, 150)

            draw.rounded_rectangle([px, py, px + pad_w, py + pad_h], radius=8, fill=fill_color, outline=border_color, width=2)
            step_num = str(step + 1)
            t_bbox = draw.textbbox((0, 0), step_num, font=f_pad_num)
            tw = t_bbox[2] - t_bbox[0]
            draw.text((px + (pad_w - tw)//2, py + (pad_h - 18)//2), step_num, fill=text_col, font=f_pad_num)

    # 3. Footer Blueprint
    footer_y = 1290
    draw.rounded_rectangle([62, footer_y, width - 62, footer_y + 560], radius=14, fill=PANEL_BG, outline=PANEL_BORDER, width=2)
    draw.text((92, footer_y + 30), "PRODUCTION & GROOVE BLUEPRINT", fill=TEXT_CYAN, font=f_inst)

    bullets = [
        ("Multi-Bar Spectral Quantizer: ", f"Isolated {metadata.get('bar_used', 'Bar 1')} from {metadata.get('total_bars', 1)}-bar loop with downbeat alignment."),
        ("Quantized 16-Step Pocket: ", f"Patterns are locked to 16th-note boundaries at {metadata.get('bpm', 120)} BPM."),
        ("Ableton Hardware Mapping: ", f"Load {metadata.get('kit', 'Kit-Core 909')} and set Pad Repeat to 1/16."),
        ("Dynamic Velocity Tracking: ", "Solid cobalt blue marks primary accents (100-127), medium marks ghost notes.")
    ]

    b_y = footer_y + 80
    for b_title, b_desc in bullets:
        draw.text((92, b_y), "•  " + b_title, fill=BLUE_ELECTRIC, font=f_body_bold)
        title_w = draw.textbbox((0, 0), "•  " + b_title, font=f_body_bold)[2]
        draw.text((92 + title_w + 4, b_y), b_desc, fill=TEXT_WHITE, font=f_body)
        b_y += 58

    tip_box_y = footer_y + 360
    draw.rounded_rectangle([92, tip_box_y, width - 92, tip_box_y + 140], radius=10, fill=(14, 16, 22), outline=(50, 60, 80), width=1)
    draw.text((116, tip_box_y + 24), "💡 PUSH 3 HARDWARE PROGRAMMING TIP", fill=AMBER, font=f_badge)
    tip_str = "Press Push 3's 'Note' button to enter 16-step sequencer mode. Select Drum Rack pad (Kick/Snare/Hat)\nand tap the lit pads above. Turn the Swing encoder to match the detected groove template."
    draw.text((116, tip_box_y + 60), tip_str, fill=TEXT_MUTED, font=f_tip)

    return im


# ==============================================================================
# STREAMLIT MOBILE / RESPONSIVE UI
# ==============================================================================

def main():
    st.set_page_config(
        page_title="Push 3 Beat Card Extractor",
        page_icon="🥁",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # Custom Mobile OLED Styling
    st.markdown("""
        <style>
            .stApp {
                background-color: #0A0B0F;
                color: #FFFFFF;
            }
            .stButton>button {
                width: 100%;
                background-color: #1D63FF;
                color: white;
                font-weight: bold;
                border-radius: 8px;
                border: none;
                padding: 12px 20px;
                margin-top: 10px;
            }
            .stDownloadButton>button {
                width: 100%;
                background-color: #00D2FF;
                color: #0A0B0F;
                font-weight: bold;
                border-radius: 8px;
                border: none;
                padding: 14px 20px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("🥁 Push 3 Beat Card Extractor")
    st.caption("Drop any drum loop (.wav, .mp3, .m4a) → Extract 16-step patterns → Save 1080×1920 OLED Cards to your phone.")

    # 1. File Uploader
    uploaded_file = st.file_uploader(
        "Upload Audio Loop / Drum Break",
        type=["wav", "mp3", "m4a", "aiff", "flac", "ogg"],
        help="Upload an isolated drum clip or loop from your device."
    )

    if uploaded_file is not None:
        st.audio(uploaded_file, format=uploaded_file.type)

        # Quick Metadata Options
        col1, col2 = st.columns(2)
        with col1:
            bar_choice = st.selectbox(
                "Bar to Extract",
                ["Bar 1", "Dominant Groove", "Bar 2", "Bar 3", "Bar 4"],
                index=0,
                help="'Bar 1' extracts the first 16 steps. 'Dominant Groove' averages all bars to find the core repeating pattern."
            )
        with col2:
            detail_level = st.selectbox(
                "Level of Detail",
                ["Standard Groove (Medium)", "High Nuance (Ghosts & Fills)", "Core Skeleton (Low Detail)"],
                index=0
            )

        # Optional Custom Title
        default_title = os.path.splitext(uploaded_file.name)[0].replace("-", " ").replace("_", " ").upper()
        custom_title = st.text_input("Card Title", value=default_title)

        # Process Button
        if st.button("⚡ Extract Groove & Render Card"):
            with st.spinner("Analyzing audio transients and separating frequency bands..."):
                try:
                    # Run extraction
                    analysis = analyze_audio_clip(uploaded_file, bar_choice=bar_choice, detail_level=detail_level)
                    analysis["title"] = custom_title

                    # Display Quick Stats
                    st.success(f"Detected: **{analysis['bpm']} BPM** | Total Length: **{analysis['total_bars']} Bars** | Kit: **{analysis['kit']}**")

                    # Render Image in memory
                    card_image = render_beat_card_image(analysis)

                    # Save to BytesIO for download
                    buf = io.BytesIO()
                    card_image.save(buf, format="PNG", quality=95)
                    img_bytes = buf.getvalue()

                    st.markdown("---")
                    st.subheader("📱 Card Preview")
                    st.image(card_image, caption=f"Ableton Push 3 — {custom_title}", use_container_width=True)

                    # Mobile Download Button
                    slug = custom_title.lower().replace(" ", "-")
                    st.download_button(
                        label="⬇️ Save Beat Card to Device (PNG)",
                        data=img_bytes,
                        file_name=f"{slug}-beat-card.png",
                        mime="image/png"
                    )

                except Exception as e:
                    st.error(f"Error analyzing audio: {str(e)}")

    else:
        st.info("👆 Tap 'Browse files' above to choose a drum loop from your phone or computer.")


if __name__ == "__main__":
    main()
