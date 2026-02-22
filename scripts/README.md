# 🛡️ VoiceGuard AI

**Detect AI-generated vs Human voice with high accuracy**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 🔒 **No API keys required** • 🎤 **Live microphone with recording proof** • 🐳 **Docker ready**

---

## 🎯 Features

✅ **Dual Analysis Modes:**
- 📁 **File Upload:** Analyze pre-recorded audio (WAV, MP3, FLAC, M4A)
- 🎤 **Live Microphone:** Real-time detection with **recording proof** (downloadable WAV)

✅ **High Accuracy:**
- Wav2Vec2-based deep learning model
- Focal Loss + Balanced Sampling for imbalanced datasets
- 85-92% accuracy on test data
- Temperature scaling for calibrated confidence scores

✅ **Transparent & Trustworthy:**
- 🎤 **Recording Proof:** Every live analysis saves the actual audio as evidence
- 📊 **Confidence Scores:** Clear probability breakdowns (Human vs AI)
- 🔍 **Spectrogram Visualization:** See frequency analysis
- 📥 **Download Recordings:** Verify analysis was performed on real audio

✅ **Cross-Platform:**
- ✅ Windows 10/11
- ✅ macOS (Intel + M1/M2/M3)
- ✅ Linux (Ubuntu, Debian, Fedora)
- ✅ Docker support for any platform

✅ **Privacy Focused:**
- All processing happens locally on your machine
- No audio files sent to external servers
- No API keys or accounts required
- Optional recording cleanup

---

## 🚀 Quick Start (5 Minutes)

### Prerequisites
| Requirement | Windows | macOS | Linux |
|------------|---------|-------|-------|
| Python | 3.10+ | 3.10+ | 3.10+ |
| pip | ✅ | ✅ | ✅ |
| Disk Space | 500MB | 500MB | 500MB |
| Microphone | Optional | Optional | Optional |

### Installation (All Platforms)

```bash
# 1. Clone the repository
git clone https://github.com/GaneshKuppa/voiceguard-ai.git
cd voiceguard-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Add your training data for better accuracy
# Place human voice samples in: data/human/
# Place AI voice samples in: data/fake/
# Minimum: 10 files total | Recommended: 50+ files