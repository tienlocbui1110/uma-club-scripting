# 🐴 Uma Club Helper Script

## Description
This tool assists in extracting and analyzing *Uma Musume Club* fan data from gameplay clips.  
It allows you to:
- Extract structured fan data from recorded clips.  
- Compare datasets to track fan count differences.  
- Highlight significant changes using a configurable threshold.  

---

## Setup

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/tienlocbui1110/uma-club-scripting.git
cd uma-club-scripting

# Create and activate venv
python -m venv venv
source venv/bin/activate        # On Linux / macOS
venv\Scripts\activate           # On Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Extract data
Extract fan data from a video clip and export to JSON:

```bash
python extract_club_fan.py <clip> -o <output.json>
```

Example:

```bash
python extract_club_fan.py input/uma.mp4 -o dist/data-10-15-2025.json
```

### Check differences
Compare two datasets and highlight differences above a threshold:

```bash
python check_diff.py -p <previous.json> -c <current.json> --threshold <threshold>
```

Example:

```bash
python check_diff.py -p dist/data-2025-10-14.json -c dist/data-2025-10-15.json --threshold 250000
```

## 🧠 Notes
- Output JSON files are versionable and easy to compare.  
- Recommended naming format: `data-YYYY-MM-DD.json`.  
- You can automate extraction daily via `cron` (Linux/macOS) or **Windows Task Scheduler**.

---

## 🪪 Credits
This project is based on [JohnDoeAntler/uma-club-helper-bot](https://github.com/JohnDoeAntler/uma-club-helper-bot).  
Original project licensed under [GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html).  
Modifications and maintenance by [Loc Bui](https://github.com/tienlocbui1110).
