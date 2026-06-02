"""
Converts dscnn_int8.tflite → model_data.h (C array for Arduino/ESP-IDF).

Usage: python src/export_model_header.py
"""
from pathlib import Path

TFLITE_PATH = Path("models/student/dscnn_int8.tflite")
OUT_PATH = Path("firmware/esp32/model_data.h")

def tflite_to_header(tflite_path: Path, out_path: Path):
    data = tflite_path.read_bytes()
    lines = ["// Auto-generated — do not edit manually",
             f"// Source: {tflite_path.name} ({len(data)/1024:.1f} KB)",
             "#pragma once",
             "#include <stdint.h>",
             f"const unsigned int g_model_data_len = {len(data)};",
             "const uint8_t g_model_data[] = {"]
    for i in range(0, len(data), 12):
        chunk = data[i:i+12]
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
    lines.append("};")
    out_path.write_text("\n".join(lines))
    print(f"Written: {out_path} ({len(data)/1024:.1f} KB)")

if __name__ == "__main__":
    tflite_to_header(TFLITE_PATH, OUT_PATH)
