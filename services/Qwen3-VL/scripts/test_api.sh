#!/bin/bash

# Test script for Qwen3-VL API

API_URL="http://localhost:8000"
TEST_IMAGE="test_image.jpg"

echo "================================================"
echo "Testing Qwen3-VL Document Parser API"
echo "================================================"

# Test 1: Health check
echo -e "\n[1] Testing health check endpoint..."
curl -s "$API_URL/" | jq '.'

# Test 2: Parse image
if [ -f "$TEST_IMAGE" ]; then
    echo -e "\n[2] Testing parse endpoint with image..."
    curl -s -X POST "$API_URL/parse" \
        -F "file=@$TEST_IMAGE" | jq '.'
else
    echo -e "\n[2] Skipping image test (test_image.jpg not found)"
fi

# Test 3: Parse PDF (if available)
if [ -f "test_document.pdf" ]; then
    echo -e "\n[3] Testing parse endpoint with PDF..."
    curl -s -X POST "$API_URL/parse" \
        -F "file=@test_document.pdf" | jq '.'
else
    echo -e "\n[3] Skipping PDF test (test_document.pdf not found)"
fi

echo -e "\n================================================"
echo "Testing completed!"
echo "================================================"
