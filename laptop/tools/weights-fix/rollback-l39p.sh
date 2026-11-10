#!/bin/bash
# back to the previous 3-bit file
cp -p /home/hikkian/Desktop/AI/weights-fix/model-launch.json.bak-before-l39p /home/hikkian/shura/config/model-launch.json
systemctl --user restart ai-gateway; echo "rolled back to Ornith-1.5-35B-A3B-GSQ-RCO-3bit-MTP.gguf"
