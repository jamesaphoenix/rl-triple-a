#!/bin/bash
# Launch the HUD with the best available model.
# Usage: ./play.sh

cd "$(dirname "$0")"

# Clear old autosaves so HUD starts fresh
SAVE_DIR="$HOME/triplea/savedGames/autoSave"
if [ -d "$SAVE_DIR" ]; then
    rm -f "$SAVE_DIR"/*.tsvg 2>/dev/null
    echo "Cleared old autosaves"
fi

# Find the latest checkpoint
MODEL=""
for f in checkpoints_phase2/selfplay_final.pt \
         checkpoints_phase2/selfplay_*.pt \
         checkpoints_selfplay_v3/foundation_v1_282k_games_93pct.pt \
         checkpoints_selfplay_v3/selfplay_final.pt; do
    if [ -f "$f" ]; then
        MODEL="$f"
        break
    fi
done

if [ -z "$MODEL" ]; then
    echo "No model found!"
    exit 1
fi

echo ""
echo "=========================================="
echo "  TripleA Battle Advisor"
echo "  Model: $MODEL"
echo "  Open: http://localhost:8080"
echo "=========================================="
echo ""
echo "1. Open http://localhost:8080 in your browser"
echo "2. Start a game in TripleA"
echo "3. Recommendations update each phase automatically"
echo ""

conda run -n rl-triplea python hud/server.py --model "$MODEL"
