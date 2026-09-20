"""
Procedural Synthetic Audio Engine & Grounded QA Dataset Generator (SoundContextQA)

Generates synthetic 16kHz multi-event audio samples with deterministic metadata, 
exact event timestamps, environmental background textures, and multi-category QA pairs:
- Apparent / Perceptual questions
- Counting questions
- Temporal ordering & duration questions
- Causal & Reasoning questions
"""

import os
import json
import random
import numpy as np
from scipy.io import wavfile
from typing import Dict, List, Tuple, Any

# Set deterministic seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

SAMPLE_RATE = 16000

# Available Ambient Background Types
ENVIRONMENTS = ["rain", "wind", "office", "quiet"]

# Sound Event Classes & Synthesis Parameters
EVENT_CLASSES = ["beep", "bark", "footstep", "bell", "siren", "drip", "engine"]


class AudioSynthesizer:
    """Procedural audio waveform synthesizer using numpy audio signal processing."""
    
    @staticmethod
    def generate_white_noise(duration: float, amplitude: float = 0.01) -> np.ndarray:
        num_samples = int(duration * SAMPLE_RATE)
        return np.random.uniform(-1.0, 1.0, num_samples).astype(np.float32) * amplitude

    @staticmethod
    def generate_pink_noise(duration: float, amplitude: float = 0.02) -> np.ndarray:
        num_samples = int(duration * SAMPLE_RATE)
        unequal = np.random.randn(num_samples)
        X = np.fft.rfft(unequal)
        S = np.sqrt(np.arange(len(X)) + 1.0)
        X = X / S
        pink = np.fft.irfft(X).astype(np.float32)
        if len(pink) < num_samples:
            pink = np.pad(pink, (0, num_samples - len(pink)))
        elif len(pink) > num_samples:
            pink = pink[:num_samples]
        max_val = np.max(np.abs(pink)) + 1e-8
        return (pink / max_val * amplitude).astype(np.float32)

    @staticmethod
    def generate_environment(env_type: str, duration: float) -> np.ndarray:
        """Synthesize continuous background environment ambient audio."""
        num_samples = int(duration * SAMPLE_RATE)
        t = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float32)
        
        if env_type == "rain":
            bg = AudioSynthesizer.generate_pink_noise(duration, amplitude=0.03)
            # Add random micro rain drops
            drops = np.zeros(num_samples, dtype=np.float32)
            num_micro_drops = int(duration * 25)
            for _ in range(num_micro_drops):
                idx = random.randint(0, num_samples - 200)
                freq = random.uniform(2500, 4500)
                t_drop = np.linspace(0, 0.01, int(0.01 * SAMPLE_RATE), dtype=np.float32)
                env = np.exp(-t_drop * 400)
                drops[idx:idx+len(t_drop)] += np.sin(2 * np.pi * freq * t_drop) * env * 0.02
            return bg + drops

        elif env_type == "wind":
            noise = AudioSynthesizer.generate_pink_noise(duration, amplitude=0.04)
            # Low pass sweep filter using sinusoidal modulation
            mod = 0.5 + 0.5 * np.sin(2 * np.pi * 0.2 * t)
            return (noise * mod).astype(np.float32)

        elif env_type == "office":
            hum = 0.01 * np.sin(2 * np.pi * 50 * t) + 0.005 * np.sin(2 * np.pi * 150 * t)
            white = AudioSynthesizer.generate_white_noise(duration, amplitude=0.008)
            return (hum + white).astype(np.float32)

        else:  # quiet
            return AudioSynthesizer.generate_white_noise(duration, amplitude=0.003)

    @staticmethod
    def generate_event(event_type: str) -> Tuple[np.ndarray, float]:
        """Synthesize a single sound event waveform and return (waveform, duration_sec)."""
        if event_type == "beep":
            duration = 0.25
            num_samples = int(duration * SAMPLE_RATE)
            t = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float32)
            freq = random.choice([1000, 1200, 1500])
            envelope = np.sin(np.pi * t / duration) ** 2
            wave = 0.3 * np.sin(2 * np.pi * freq * t) * envelope
            return wave.astype(np.float32), duration

        elif event_type == "bark":
            duration = 0.35
            num_samples = int(duration * SAMPLE_RATE)
            t = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float32)
            noise = np.random.uniform(-1, 1, num_samples).astype(np.float32)
            envelope = np.exp(-12 * t) * (1 - np.exp(-50 * t))
            wave = 0.4 * noise * envelope * np.sin(2 * np.pi * 400 * t)
            return wave.astype(np.float32), duration

        elif event_type == "footstep":
            duration = 0.3
            num_samples = int(duration * SAMPLE_RATE)
            t = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float32)
            # Damped low pitch impact sweep (160 Hz -> 50 Hz)
            freq_sweep = 160.0 * np.exp(-10 * t) + 40.0
            phase = 2 * np.pi * np.cumsum(freq_sweep) / SAMPLE_RATE
            envelope = np.exp(-15 * t)
            wave = 0.5 * np.sin(phase) * envelope
            return wave.astype(np.float32), duration

        elif event_type == "bell":
            duration = 0.7
            num_samples = int(duration * SAMPLE_RATE)
            t = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float32)
            base_freq = random.choice([523.25, 659.25, 783.99])  # C5, E5, G5
            wave = (
                1.0 * np.sin(2 * np.pi * base_freq * t) +
                0.5 * np.sin(2 * np.pi * base_freq * 2.0 * t) +
                0.25 * np.sin(2 * np.pi * base_freq * 3.01 * t)
            )
            envelope = np.exp(-5.0 * t)
            wave = 0.35 * wave * envelope
            return wave.astype(np.float32), duration

        elif event_type == "siren":
            duration = 1.2
            num_samples = int(duration * SAMPLE_RATE)
            t = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float32)
            sweep_freq = 700 + 400 * np.sin(2 * np.pi * 2.5 * t)
            phase = 2 * np.pi * np.cumsum(sweep_freq) / SAMPLE_RATE
            envelope = np.minimum(1.0, t * 10) * np.minimum(1.0, (duration - t) * 10)
            wave = 0.3 * np.sin(phase) * envelope
            return wave.astype(np.float32), duration

        elif event_type == "drip":
            duration = 0.18
            num_samples = int(duration * SAMPLE_RATE)
            t = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float32)
            freq_sweep = 400 + 1200 * (t / duration) ** 2
            phase = 2 * np.pi * np.cumsum(freq_sweep) / SAMPLE_RATE
            envelope = np.exp(-20 * t)
            wave = 0.4 * np.sin(phase) * envelope
            return wave.astype(np.float32), duration

        elif event_type == "engine":
            duration = random.uniform(3.5, 5.0)
            num_samples = int(duration * SAMPLE_RATE)
            t = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float32)
            noise = AudioSynthesizer.generate_pink_noise(duration, amplitude=0.25)
            rumble = 0.2 * np.sin(2 * np.pi * 60 * t) + 0.15 * np.sin(2 * np.pi * 120 * t)
            mod = 0.7 + 0.3 * np.sin(2 * np.pi * 12 * t)
            wave = (noise + rumble) * mod
            return wave.astype(np.float32), duration

        else:
            return np.zeros(int(0.2 * SAMPLE_RATE), dtype=np.float32), 0.2


def generate_sample_qa(
    sample_id: str,
    duration: float,
    env_type: str,
    events_placed: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Generates structured Question-Answer pairs grounded in audio clip metadata."""
    qa_list = []
    
    # Sort events chronologically by start time
    sorted_events = sorted(events_placed, key=lambda x: x["start_time"])
    event_counts = {}
    for ev in events_placed:
        cls = ev["class"]
        event_counts[cls] = event_counts.get(cls, 0) + 1
        
    unique_event_names = list(set([ev["class"] for ev in sorted_events]))

    # --- 1. APPARENT / PERCEPTUAL QUESTIONS ---
    qa_list.append({
        "question_id": f"{sample_id}_p1",
        "category": "perceptual",
        "question": "What background acoustic environment is present in this audio clip?",
        "answer": env_type,
        "answer_type": "categorical"
    })
    
    events_str = ", ".join(unique_event_names) if unique_event_names else "none"
    qa_list.append({
        "question_id": f"{sample_id}_p2",
        "category": "perceptual",
        "question": "Which sound events are present in the audio recording?",
        "answer": f"The audio contains {events_str}.",
        "answer_type": "text"
    })

    for env_check in ENVIRONMENTS:
        is_present = "yes" if env_check == env_type else "no"
        qa_list.append({
            "question_id": f"{sample_id}_p_check_{env_check}",
            "category": "perceptual",
            "question": f"Is the background environment characterized by {env_check}?",
            "answer": is_present,
            "answer_type": "boolean"
        })

    # --- 2. COUNTING QUESTIONS ---
    for ev_cls in EVENT_CLASSES:
        cnt = event_counts.get(ev_cls, 0)
        qa_list.append({
            "question_id": f"{sample_id}_c_{ev_cls}",
            "category": "counting",
            "question": f"How many times does the {ev_cls} sound occur in the audio?",
            "answer": str(cnt),
            "answer_type": "count"
        })

    total_events = len(sorted_events)
    qa_list.append({
        "question_id": f"{sample_id}_c_total",
        "category": "counting",
        "question": "What is the total number of sound events in the recording?",
        "answer": str(total_events),
        "answer_type": "count"
    })

    # --- 3. TEMPORAL QUESTIONS ---
    if total_events > 0:
        first_event = sorted_events[0]["class"]
        last_event = sorted_events[-1]["class"]
        
        qa_list.append({
            "question_id": f"{sample_id}_t_first",
            "category": "temporal",
            "question": "What is the first sound event that occurs in the audio clip?",
            "answer": first_event,
            "answer_type": "categorical"
        })
        
        qa_list.append({
            "question_id": f"{sample_id}_t_last",
            "category": "temporal",
            "question": "What is the last sound event that occurs in the audio clip?",
            "answer": last_event,
            "answer_type": "categorical"
        })

    if total_events >= 2:
        ev1 = sorted_events[0]
        ev2 = sorted_events[1]
        qa_list.append({
            "question_id": f"{sample_id}_t_after",
            "category": "temporal",
            "question": f"What sound event occurs immediately after the initial {ev1['class']}?",
            "answer": ev2["class"],
            "answer_type": "categorical"
        })

        # Relative order check
        if ev1["class"] != ev2["class"]:
            qa_list.append({
                "question_id": f"{sample_id}_t_order",
                "category": "temporal",
                "question": f"Does the {ev1['class']} sound occur before or after the {ev2['class']} sound?",
                "answer": "before",
                "answer_type": "categorical"
            })

    # Duration comparison
    if total_events > 0:
        longest_ev = max(sorted_events, key=lambda x: x["duration"])
        qa_list.append({
            "question_id": f"{sample_id}_t_longest",
            "category": "temporal",
            "question": "Which sound event has the longest continuous duration in the clip?",
            "answer": longest_ev["class"],
            "answer_type": "categorical"
        })

    # --- 4. CAUSAL / REASONING QUESTIONS ---
    # Rule 1: Drip overflow siren
    drip_count = event_counts.get("drip", 0)
    has_siren = event_counts.get("siren", 0) > 0
    if drip_count >= 3 and has_siren:
        qa_list.append({
            "question_id": f"{sample_id}_caus_1",
            "category": "causal",
            "question": "Why did the siren sound in the recording?",
            "answer": "The siren sounded because repeated water dripping triggered the overflow alarm system.",
            "answer_type": "text"
        })

    # Rule 2: Office intruder alert (footsteps -> bell/alarm in office)
    has_footstep = event_counts.get("footstep", 0) > 0
    has_bell = event_counts.get("bell", 0) > 0
    if env_type == "office" and has_footstep and has_bell:
        qa_list.append({
            "question_id": f"{sample_id}_caus_2",
            "category": "causal",
            "question": "Why did the bell alarm sound after the footsteps in the office?",
            "answer": "Footsteps inside the office environment triggered the security motion chime sensor.",
            "answer_type": "text"
        })

    # Rule 3: Engine startup sequence
    has_engine = event_counts.get("engine", 0) > 0
    has_beep = event_counts.get("beep", 0) > 0
    if has_engine and has_beep:
        qa_list.append({
            "question_id": f"{sample_id}_caus_3",
            "category": "causal",
            "question": "Why are warning beeps heard during the engine sound?",
            "answer": "The system generated safety warning beeps during the active engine operation sequence.",
            "answer_type": "text"
        })

    # General Causal Question if no specific rule matched
    if not any(q["category"] == "causal" for q in qa_list):
        if total_events > 0:
            qa_list.append({
                "question_id": f"{sample_id}_caus_gen",
                "category": "causal",
                "question": f"What is the contextual setting of this audio recording?",
                "answer": f"The recording is set in a {env_type} environment with acoustic events including {', '.join(unique_event_names)}.",
                "answer_type": "text"
            })
        else:
            qa_list.append({
                "question_id": f"{sample_id}_caus_gen",
                "category": "causal",
                "question": "Why is the audio sample quiet without loud events?",
                "answer": "The sample represents an ambient baseline recording without active transient sound triggers.",
                "answer_type": "text"
            })

    return qa_list


def generate_dataset(output_dir: str = "data/sound_context_qa", num_samples: int = 1000):
    """Synthesizes the complete SoundContextQA dataset and saves audio files + JSON splits."""
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    
    print(f"[Dataset Generator] Synthesizing {num_samples} audio samples into '{audio_dir}'...")

    samples_metadata = []

    for i in range(num_samples):
        sample_id = f"sample_{i:04d}"
        duration = random.uniform(6.0, 8.5)
        env_type = random.choice(ENVIRONMENTS)

        # Generate base ambient background
        audio_wave = AudioSynthesizer.generate_environment(env_type, duration)

        # Decide event placement
        num_event_types = random.randint(1, 3)
        chosen_types = random.sample(EVENT_CLASSES, num_event_types)

        # Causal Rule Trigger: If drip is chosen, randomly decide if it repeats >=3 times to trigger siren
        if "drip" in chosen_types and random.random() > 0.4:
            drip_count = random.randint(3, 4)
        else:
            drip_count = random.randint(1, 2) if "drip" in chosen_types else 0

        events_placed = []
        current_t = random.uniform(0.5, 1.2)

        for ev_cls in chosen_types:
            if ev_cls == "drip" and drip_count >= 3:
                count = drip_count
            elif ev_cls == "engine":
                count = 1  # Continuous engine
            else:
                count = random.randint(1, 3)

            for _ in range(count):
                wave_ev, ev_dur = AudioSynthesizer.generate_event(ev_cls)
                if current_t + ev_dur >= duration - 0.3:
                    break

                start_idx = int(current_t * SAMPLE_RATE)
                end_idx = start_idx + len(wave_ev)

                if end_idx <= len(audio_wave):
                    audio_wave[start_idx:end_idx] += wave_ev
                    events_placed.append({
                        "class": ev_cls,
                        "start_time": round(float(current_t), 3),
                        "end_time": round(float(current_t + ev_dur), 3),
                        "duration": round(float(ev_dur), 3)
                    })

                current_t += ev_dur + random.uniform(0.4, 1.1)

        # Check causal rule A: 3+ drips triggers a siren afterwards
        if drip_count >= 3 and current_t + 1.5 < duration:
            siren_wave, siren_dur = AudioSynthesizer.generate_event("siren")
            start_idx = int(current_t * SAMPLE_RATE)
            end_idx = start_idx + len(siren_wave)
            if end_idx <= len(audio_wave):
                audio_wave[start_idx:end_idx] += siren_wave
                events_placed.append({
                    "class": "siren",
                    "start_time": round(float(current_t), 3),
                    "end_time": round(float(current_t + siren_dur), 3),
                    "duration": round(float(siren_dur), 3)
                })

        # Normalize audio wave peak to avoid clipping
        max_val = np.max(np.abs(audio_wave))
        if max_val > 0.98:
            audio_wave = audio_wave / max_val * 0.95

        # Save WAV file
        audio_filename = f"{sample_id}.wav"
        audio_path = os.path.join(audio_dir, audio_filename)
        audio_int16 = (audio_wave * 32767).astype(np.int16)
        wavfile.write(audio_path, SAMPLE_RATE, audio_int16)

        # Generate grounded QA pairs
        qa_pairs = generate_sample_qa(sample_id, duration, env_type, events_placed)

        samples_metadata.append({
            "sample_id": sample_id,
            "audio_file": audio_filename,
            "duration": round(float(duration), 3),
            "environment": env_type,
            "events": events_placed,
            "qa_pairs": qa_pairs
        })

    # Shuffle & Split into 70% Train, 15% Val, 15% Test
    random.shuffle(samples_metadata)
    n_train = int(0.70 * num_samples)
    n_val = int(0.15 * num_samples)

    train_split = samples_metadata[:n_train]
    val_split = samples_metadata[n_train:n_train + n_val]
    test_split = samples_metadata[n_train + n_val:]

    # Save split JSONs
    splits = {
        "train.json": train_split,
        "val.json": val_split,
        "test.json": test_split
    }

    total_qa = 0
    category_counts = {}

    for split_filename, split_data in splits.items():
        split_path = os.path.join(output_dir, split_filename)
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=2)

        split_qa_count = sum(len(item["qa_pairs"]) for item in split_data)
        total_qa += split_qa_count
        print(f"[Dataset Generator] Saved {len(split_data)} audio samples with {split_qa_count} QA pairs to '{split_path}'.")

        for item in split_data:
            for qa in item["qa_pairs"]:
                cat = qa["category"]
                category_counts[cat] = category_counts.get(cat, 0) + 1

    summary_metadata = {
        "dataset_name": "SoundContextQA",
        "sample_rate": SAMPLE_RATE,
        "num_samples": num_samples,
        "total_qa_pairs": total_qa,
        "splits": {
            "train_samples": len(train_split),
            "val_samples": len(val_split),
            "test_samples": len(test_split)
        },
        "category_distribution": category_counts,
        "event_classes": EVENT_CLASSES,
        "environments": ENVIRONMENTS
    }

    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(summary_metadata, f, indent=2)

    print("[Dataset Generator] Dataset construction complete!")
    print(f"Summary: {summary_metadata}")


if __name__ == "__main__":
    generate_dataset(num_samples=1000)
