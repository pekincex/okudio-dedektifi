#!/bin/bash
echo ""
echo "══════════════════════════════════════════════════"
echo "  Okudio Okuma Dedektifi — Kurulum"
echo "══════════════════════════════════════════════════"
echo ""

if ! command -v python3 &> /dev/null; then
    echo "Python3 bulunamadi! python.org'dan indirin."
    exit 1
fi

echo "Python3: $(python3 --version)"
echo ""
echo "Kutuphaneler yukleniyor..."
echo ""

pip3 install flask openai anthropic praat-parselmouth librosa "numpy<2.0" soundfile reportlab

echo ""
echo "══════════════════════════════════════════════════"
echo "  Kurulum tamamlandi!"
echo ""
echo "  1. app.py icindeki API key'lerini degistir"
echo "  2. python3 app.py"
echo "  3. Tarayicida: http://localhost:5000"
echo "══════════════════════════════════════════════════"
echo ""
