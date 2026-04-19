#!/usr/bin/env python3
"""
🎭 Emotion Detection Comparison: DeepFace vs FER Model
This script compares the accuracy and performance of different approaches
"""

import time
import cv2
import numpy as np
import requests
from pathlib import Path

def test_emotion_detection(api_url, test_images):
    """Test emotion detection accuracy"""
    print(f"\n🧪 Testing {api_url}")

    results = []
    total_time = 0

    for img_path in test_images:
        try:
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            # Encode for API
            _, buffer = cv2.imencode('.jpg', image)
            files = {'frame': ('test.jpg', buffer.tobytes(), 'image/jpeg')}

            # Send request
            start_time = time.time()
            response = requests.post(f"{api_url}/detect-emotion", files=files, timeout=10)
            end_time = time.time()

            if response.status_code == 200:
                data = response.json()
                processing_time = end_time - start_time
                total_time += processing_time

                result = {
                    'image': img_path.name,
                    'emotion': data.get('emotion', 'unknown'),
                    'confidence': data.get('confidence', 0),
                    'time': processing_time,
                    'face_count': data.get('face_count', 0)
                }
                results.append(result)
                print(f"  ✅ {img_path.name}: {result['emotion']} ({result['confidence']:.1%}) - {processing_time:.2f}s")
            else:
                print(f"  ❌ {img_path.name}: API error {response.status_code}")

        except Exception as e:
            print(f"  ❌ {img_path.name}: {str(e)}")

    if results:
        avg_time = total_time / len(results)
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        print(f"\n📊 Summary: {len(results)} images, avg time: {avg_time:.2f}s, avg confidence: {avg_confidence:.1%}")

    return results

def main():
    print("🎭 EMOTION DETECTION MODEL COMPARISON")
    print("=" * 50)

    # Find test images
    uploads_dir = Path("uploads")
    test_images = list(uploads_dir.glob("*.jpg")) + list(uploads_dir.glob("*.png"))

    if not test_images:
        print("❌ No test images found in uploads/ directory")
        print("💡 Take some screenshots first using the web interface")
        return

    print(f"📸 Found {len(test_images)} test images")

    # Test both versions
    apis = [
        ("Original DeepFace", "http://localhost:5000"),
        ("Improved FER Model", "http://localhost:5000")  # Same port, different model
    ]

    all_results = {}

    for name, url in apis:
        try:
            # Check if API is running
            health = requests.get(f"{url}/health", timeout=5)
            if health.status_code == 200:
                results = test_emotion_detection(url, test_images)
                all_results[name] = results
            else:
                print(f"❌ {name} API not responding")
        except:
            print(f"❌ {name} API not available")

    # Compare results
    if len(all_results) >= 2:
        print("\n" + "=" * 50)
        print("🎯 COMPARISON RESULTS")
        print("=" * 50)

        names = list(all_results.keys())
        results1 = all_results[names[0]]
        results2 = all_results[names[1]]

        print(f"Comparing: {names[0]} vs {names[1]}")
        print()

        for i, (r1, r2) in enumerate(zip(results1, results2)):
            img_name = r1['image']
            emotion1 = r1['emotion']
            emotion2 = r2['emotion']
            conf1 = r1['confidence']
            conf2 = r2['confidence']

            match = "✅" if emotion1 == emotion2 else "❌"
            better_conf = ">" if conf1 > conf2 else "<" if conf2 > conf1 else "="

            print(f"{match} {img_name}:")
            print(f"   {names[0]}: {emotion1} ({conf1:.1%})")
            print(f"   {names[1]}: {emotion2} ({conf2:.1%}) - {better_conf}")
            print()

if __name__ == "__main__":
    main()