#!/bin/bash

echo "generate_assets.sh: Starting figure and table generation"
echo "generate_assets.sh: ==============================================="
echo ""

echo "generate_assets.sh: [1/3] Generating Figures 1-3 and footnote 6..."
echo "generate_assets.sh:        Running assets_introduction.py"
python assets_introduction.py
echo "generate_assets.sh:        Introduction figures completed"
echo ""

echo "generate_assets.sh: [2/3] Generating Figures 4-5..."
echo "generate_assets.sh:        Running assets_empirical.py"
python assets_empirical.py
echo "generate_assets.sh:        Empirical figures completed"
echo ""

echo "generate_assets.sh: [3/3] Generating Table OA5.1 and Figures OA5.1-OA5.4..."
echo "generate_assets.sh:        Running assets_appendix.py"
python assets_appendix.py
echo "generate_assets.sh:        Appendix assets completed"
echo ""

echo "generate_assets.sh: All figures and tables generated successfully!"
echo "generate_assets.sh: Output location: ./assets/"
echo "generate_assets.sh: ==============================================="
