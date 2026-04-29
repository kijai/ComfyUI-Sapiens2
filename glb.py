import json
import struct
from io import BytesIO

import numpy as np
from PIL import Image as PILImage


def save_textured_glb(verts, uvs, faces, tex_rgb_uint8, out_path):
    """Write a textured GLB (single PBR material with embedded baseColorTexture).

    verts: (N, 3) float, uvs: (N, 2) float, faces: (M, 3) int, tex: (H, W, 3) uint8.
    """
    verts = np.ascontiguousarray(verts, dtype=np.float32)
    uvs = np.ascontiguousarray(uvs, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.uint32)

    png_buf = BytesIO()
    PILImage.fromarray(tex_rgb_uint8, mode="RGB").save(png_buf, format="PNG")
    png_bytes = png_buf.getvalue()

    def pad4(b, fill=b"\x00"):
        return b + fill * ((4 - (len(b) % 4)) % 4)

    v_bytes, uv_bytes, f_bytes = verts.tobytes(), uvs.tobytes(), faces.tobytes()
    v_off = 0
    uv_off = v_off + len(pad4(v_bytes))
    f_off = uv_off + len(pad4(uv_bytes))
    img_off = f_off + len(pad4(f_bytes))
    bin_blob = pad4(v_bytes) + pad4(uv_bytes) + pad4(f_bytes) + pad4(png_bytes)

    gltf = {
        "asset": {"version": "2.0", "generator": "ComfyUI-Sapiens2"},
        "buffers": [{"byteLength": len(bin_blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": v_off,   "byteLength": len(v_bytes),   "target": 34962},  # ARRAY_BUFFER
            {"buffer": 0, "byteOffset": uv_off,  "byteLength": len(uv_bytes),  "target": 34962},
            {"buffer": 0, "byteOffset": f_off,   "byteLength": len(f_bytes),   "target": 34963},  # ELEMENT_ARRAY_BUFFER
            {"buffer": 0, "byteOffset": img_off, "byteLength": len(png_bytes)},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": int(verts.shape[0]), "type": "VEC3",
             "min": verts.min(axis=0).tolist(), "max": verts.max(axis=0).tolist()},
            {"bufferView": 1, "componentType": 5126, "count": int(uvs.shape[0]),   "type": "VEC2"},
            {"bufferView": 2, "componentType": 5125, "count": int(faces.size),     "type": "SCALAR"},
        ],
        "images": [{"bufferView": 3, "mimeType": "image/png"}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
        "textures": [{"sampler": 0, "source": 0}],
        "materials": [{
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "metallicFactor": 0.5,
                "roughnessFactor": 1.0,
            },
            "doubleSided": True,
        }],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 0, "TEXCOORD_0": 1},
            "indices": 2,
            "material": 0,
            "mode": 4,  # TRIANGLES
        }]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }

    json_padded = pad4(json.dumps(gltf).encode("utf-8"), fill=b" ")
    total_len = 12 + 8 + len(json_padded) + 8 + len(bin_blob)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total_len))
        f.write(struct.pack("<II", len(json_padded), 0x4E4F534A))  # "JSON"
        f.write(json_padded)
        f.write(struct.pack("<II", len(bin_blob), 0x004E4942))     # "BIN\0"
        f.write(bin_blob)
