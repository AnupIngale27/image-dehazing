# ===== Add this as a NEW CELL at the end of Image_Dehazing.ipynb and run it once (after training) =====
# It converts best.pth into clearview_512.onnx, checks it gives the same output as PyTorch,
# and downloads it so you can upload it to a Hugging Face model repo.

!pip -q install onnx onnxruntime
import os, torch, numpy as np, onnxruntime as ort
from google.colab import files
from restormer_arch import Restormer

TILE = 512                                    # the web page processes images in 512x512 tiles
CKPT = '/content/fog_checkpoints/best.pth'
OUT  = '/content/clearview_512.onnx'

ck = torch.load(CKPT, map_location='cpu')
net = Restormer(inp_channels=3, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
                heads=[1, 2, 4, 8], ffn_expansion_factor=2.66, bias=False,
                LayerNorm_type=ck.get('norm', 'WithBias'), dual_pixel_task=False)
net.load_state_dict(ck['model']); net.eval()

x = torch.rand(1, 3, TILE, TILE)
kw = dict(input_names=['input'], output_names=['output'], opset_version=17, do_constant_folding=True)
with torch.no_grad():
    try:
        torch.onnx.export(net, x, OUT, dynamo=False, **kw)   # newer PyTorch
    except TypeError:
        torch.onnx.export(net, x, OUT, **kw)                 # older PyTorch
    ref = net(x).numpy()

sess = ort.InferenceSession(OUT, providers=['CPUExecutionProvider'])
out = sess.run(None, {'input': x.numpy()})[0]
diff = float(np.abs(out - ref).max())
print(f'Saved {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)')
print(f'Max difference PyTorch vs ONNX: {diff:.6f}  ->', 'OK' if diff < 1e-3 else 'CHECK THIS (should be < 0.001)')

files.download(OUT)
