import pickle
import time
import numpy as np
from pathlib import Path
from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit
from tinygrad.device import Device

from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.selfdrive.modeld.compile_warp import (
  CAMERA_CONFIGS, MEDMODEL_INPUT_SIZE, make_frame_prepare, make_update_both_imgs,
  warp_pkl_path,
)
from openpilot.selfdrive.modeld.external_pickle import load_external_pickle

MODELS_DIR = Path(__file__).parent / 'models'
MODEL_W, MODEL_H = MEDMODEL_INPUT_SIZE
UPSTREAM_BUFFER_LENGTH = 5


def v2_warp_pkl_path(cam_w, cam_h, buffer_length):
  return MODELS_DIR / f'warp_{cam_w}x{cam_h}_b{buffer_length}_tinygrad.pkl'


def compile_v2_warp(cam_w, cam_h, buffer_length):
  _, _, _, yuv_size = get_nv12_info(cam_w, cam_h)
  img_buffer_shape = (buffer_length * 6, MODEL_H // 2, MODEL_W // 2)

  print(f"Compiling v2 warp for {cam_w}x{cam_h} buffer_length={buffer_length}...")

  frame_prepare = make_frame_prepare(cam_w, cam_h, MODEL_W, MODEL_H)
  update_both_imgs = make_update_both_imgs(frame_prepare, MODEL_W, MODEL_H)
  update_img_jit = TinyJit(update_both_imgs, prune=True)

  full_buffer = Tensor.zeros(img_buffer_shape, dtype='uint8').contiguous().realize()
  big_full_buffer = Tensor.zeros(img_buffer_shape, dtype='uint8').contiguous().realize()
  full_buffer_np = np.zeros(img_buffer_shape, dtype=np.uint8)
  big_full_buffer_np = np.zeros(img_buffer_shape, dtype=np.uint8)

  for i in range(10):
    new_frame_np = (32 * np.random.randn(yuv_size).astype(np.float32) + 128).clip(0, 255).astype(np.uint8)
    img_inputs = [full_buffer,
                  Tensor.from_blob(new_frame_np.ctypes.data, (yuv_size,), dtype='uint8').realize(),
                  Tensor(Tensor.randn(3, 3).mul(8).realize().numpy(), device='NPY')]
    new_big_frame_np = (32 * np.random.randn(yuv_size).astype(np.float32) + 128).clip(0, 255).astype(np.uint8)
    big_img_inputs = [big_full_buffer,
                      Tensor.from_blob(new_big_frame_np.ctypes.data, (yuv_size,), dtype='uint8').realize(),
                      Tensor(Tensor.randn(3, 3).mul(8).realize().numpy(), device='NPY')]
    inputs = img_inputs + big_img_inputs
    Device.default.synchronize()

    inputs_np = [x.numpy() for x in inputs]
    inputs_np[0] = full_buffer_np
    inputs_np[3] = big_full_buffer_np

    st = time.perf_counter()
    out = update_img_jit(*inputs)
    full_buffer = out[0].contiguous().realize().clone()
    big_full_buffer = out[2].contiguous().realize().clone()
    mt = time.perf_counter()
    Device.default.synchronize()
    et = time.perf_counter()
    print(f"  [{i+1}/10] enqueue {(mt-st)*1e3:6.2f} ms -- total {(et-st)*1e3:6.2f} ms")

  pkl_path = v2_warp_pkl_path(cam_w, cam_h, buffer_length)
  with open(pkl_path, "wb") as f:
    pickle.dump(update_img_jit, f)
  print(f"  Saved to {pkl_path}")

  jit = pickle.load(open(pkl_path, "rb"))
  jit(*inputs)


class Warp:
  def __init__(self, buffer_length=2):
    self.buffer_length = buffer_length
    self.img_buffer_shape = (buffer_length * 6, MODEL_H // 2, MODEL_W // 2)

    self.jit_cache = {}
    self.full_buffers = {k: Tensor.zeros(self.img_buffer_shape, dtype='uint8').contiguous().realize() for k in ['img', 'big_img']}

  def process(self, bufs, transforms):
    if not bufs:
      return {}
    cam_w, cam_h = bufs['img'].width, bufs['img'].height
    key = (cam_w, cam_h)

    if key not in self.jit_cache:
      v2_pkl = v2_warp_pkl_path(cam_w, cam_h, self.buffer_length)
      if v2_pkl.exists():
        with open(v2_pkl, 'rb') as f:
          self.jit_cache[key] = pickle.load(f)
      elif Path(str(v2_pkl) + ".parts").exists():
        self.jit_cache[key] = load_external_pickle(v2_pkl)
      elif self.buffer_length == UPSTREAM_BUFFER_LENGTH:
        upstream_pkl = warp_pkl_path(cam_w, cam_h)
        if upstream_pkl.exists():
          with open(upstream_pkl, 'rb') as f:
            self.jit_cache[key] = pickle.load(f)
        elif Path(str(upstream_pkl) + ".parts").exists():
          self.jit_cache[key] = load_external_pickle(upstream_pkl)
      if key not in self.jit_cache:
        frame_prepare = make_frame_prepare(cam_w, cam_h, MODEL_W, MODEL_H)
        update_both_imgs = make_update_both_imgs(frame_prepare, MODEL_W, MODEL_H)
        self.jit_cache[key] = TinyJit(update_both_imgs, prune=True)

    jit = self.jit_cache[key]
    _, _, _, yuv_size = get_nv12_info(cam_w, cam_h)

    inputs = []
    for k in ['img', 'big_img']:
      inputs.append(self.full_buffers[k])
      inputs.append(Tensor.from_blob(bufs[k].data.ctypes.data, (yuv_size,), dtype='uint8').realize())
      inputs.append(Tensor(transforms[k].reshape(3, 3), device='NPY').realize())

    Device.default.synchronize()
    res = jit(*inputs)
    self.full_buffers['img'], out_road = res[0].realize(), res[1].realize()
    self.full_buffers['big_img'], out_wide = res[2].realize(), res[3].realize()

    return {'img': out_road, 'big_img': out_wide}


if __name__ == "__main__":
  for cam_w, cam_h in CAMERA_CONFIGS:
    for bl in [2, 5]:
      compile_v2_warp(cam_w, cam_h, bl)
