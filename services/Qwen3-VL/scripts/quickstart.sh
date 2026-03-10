#!/bin/bash

# Quick Start Guide for Qwen3-VL Document Parser API
# =================================================

echo "======================================================================"
echo "  Qwen3-VL Document Parser API - Quick Start"
echo "======================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "api_vllm.py" ]; then
    echo -e "${RED}Error: Please run this script from the Qwen3-VL directory${NC}"
    echo "cd /root/tungn197/license_plate_recognition/services/Qwen3-VL"
    exit 1
fi

echo -e "${YELLOW}Step 1: Verify Installation${NC}"
echo "----------------------------------------------------------------------"
python verify_installation.py
if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}Installation verification failed!${NC}"
    echo "Please install missing dependencies:"
    echo "  pip install -r requirements_api.txt"
    echo "  sudo apt-get install -y poppler-utils"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Installation verified!${NC}"
echo ""

echo -e "${YELLOW}Step 2: Choose Your Option${NC}"
echo "----------------------------------------------------------------------"
echo "1. Start API server"
echo "2. Test with sample (requires API to be running)"
echo "3. View documentation"
echo "4. Exit"
echo ""
read -p "Enter your choice [1-4]: " choice

case $choice in
    1)
        echo ""
        echo -e "${YELLOW}Starting API Server...${NC}"
        echo "----------------------------------------------------------------------"
        echo "Model: ${MODEL_PATH:-Qwen/Qwen2-VL-7B-Instruct}"
        echo "Port: ${PORT:-8000}"
        echo ""
        echo "The server will start shortly. This may take a few minutes on first run."
        echo "Press Ctrl+C to stop the server."
        echo ""
        sleep 2
        ./start_api.sh
        ;;
    
    2)
        echo ""
        echo -e "${YELLOW}Testing API...${NC}"
        echo "----------------------------------------------------------------------"
        
        # Check if API is running
        if ! curl -s http://localhost:8000/ > /dev/null 2>&1; then
            echo -e "${RED}Error: API is not running!${NC}"
            echo "Please start the API first with: ./start_api.sh"
            exit 1
        fi
        
        echo "API is running. Health check:"
        curl -s http://localhost:8000/ | python -m json.tool
        
        echo ""
        echo "To test with your own files:"
        echo "  python test_client.py parse --file your_document.pdf"
        echo ""
        echo "Or use curl:"
        echo "  curl -X POST http://localhost:8000/parse -F 'file=@your_image.jpg'"
        ;;
    
    3)
        echo ""
        echo -e "${YELLOW}Documentation Files:${NC}"
        echo "----------------------------------------------------------------------"
        echo "1. SUMMARY.md          - Quick overview and getting started"
        echo "2. README_API.md       - Detailed API documentation"
        echo "3. USAGE_GUIDE.md      - Complete usage guide with examples"
        echo ""
        read -p "Which file would you like to view? [1-3]: " doc_choice
        
        case $doc_choice in
            1) less SUMMARY.md ;;
            2) less README_API.md ;;
            3) less USAGE_GUIDE.md ;;
            *) echo "Invalid choice" ;;
        esac
        ;;
    
    4)
        echo "Goodbye!"
        exit 0
        ;;
    
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo "======================================================================"
echo "  For more information, see:"
echo "    - SUMMARY.md (Quick overview)"
echo "    - README_API.md (API documentation)"
echo "    - USAGE_GUIDE.md (Usage examples)"
echo "======================================================================"
