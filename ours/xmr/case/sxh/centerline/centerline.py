"""Export the complete SXH MRCP bile-duct skeleton tree and a standalone viewer."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion
from skimage.morphology import skeletonize


PROJECT_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_MASK_PATH = PROJECT_ROOT / "mrct" / "data-duet" / "bile_duct" / "mrcp_006.nii.gz"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


@dataclass(frozen=True)
class CenterlineTree:
    """A complete voxel-skeleton graph represented in MRCP physical coordinates."""

    skeleton_mask: np.ndarray
    vertices_mm: np.ndarray
    edges: np.ndarray


def voxel_to_world(voxels: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Map NIfTI voxel coordinates to their full affine world coordinates in mm."""
    voxels = np.asarray(voxels, dtype=np.float64)
    if not len(voxels):
        return np.empty((0, 3), dtype=np.float64)
    return (np.c_[voxels, np.ones(len(voxels))] @ np.asarray(affine, dtype=np.float64).T)[:, :3]


def _positive_neighbor_offsets() -> tuple[tuple[int, int, int], ...]:
    """Return one half of the 26-neighborhood so undirected edges are unique."""
    return tuple(offset for offset in product((-1, 0, 1), repeat=3) if offset > (0, 0, 0))


def extract_centerline_tree(mask: np.ndarray, affine: np.ndarray) -> CenterlineTree:
    """Skeletonize a binary bile-duct mask and retain every 26-neighbor tree edge."""
    binary_mask = np.asarray(mask) > 0
    if binary_mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {binary_mask.shape}")
    if not binary_mask.any():
        raise ValueError("Cannot extract a centerline from an empty mask")

    skeleton_mask = skeletonize(binary_mask).astype(np.uint8)
    voxels = np.argwhere(skeleton_mask > 0)
    index_by_voxel = {tuple(int(v) for v in voxel): index for index, voxel in enumerate(voxels)}
    edges: list[tuple[int, int]] = []
    for source_index, voxel in enumerate(voxels):
        for offset in _positive_neighbor_offsets():
            neighbor = tuple(int(voxel[axis] + offset[axis]) for axis in range(3))
            target_index = index_by_voxel.get(neighbor)
            if target_index is not None:
                edges.append((source_index, target_index))

    return CenterlineTree(
        skeleton_mask=skeleton_mask,
        vertices_mm=voxel_to_world(voxels, affine),
        edges=np.asarray(edges, dtype=np.int32).reshape(-1, 2),
    )


def extract_surface_points(mask: np.ndarray, affine: np.ndarray, max_points: int, seed: int = 17) -> np.ndarray:
    """Return a deterministic subsample of binary-mask surface points in mm."""
    binary_mask = np.asarray(mask) > 0
    surface = binary_mask & ~binary_erosion(binary_mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    voxels = np.argwhere(surface)
    if len(voxels) > max_points:
        rng = np.random.default_rng(seed)
        voxels = voxels[np.sort(rng.choice(len(voxels), size=max_points, replace=False))]
    return voxel_to_world(voxels, affine)


def _rounded(points: np.ndarray) -> list[list[float]]:
    return np.round(np.asarray(points, dtype=np.float64), 3).tolist()


def build_viewer_html(mask_surface_mm: np.ndarray, tree: CenterlineTree) -> str:
    """Build a dependency-free WebGL viewer for the mask surface and skeleton tree."""
    all_points = np.vstack((mask_surface_mm, tree.vertices_mm))
    center = all_points.mean(axis=0)
    radius = float(np.linalg.norm(all_points - center, axis=1).max())
    payload = json.dumps(
        {
            "mask_surface_mm": _rounded(mask_surface_mm),
            "centerline_vertices_mm": _rounded(tree.vertices_mm),
            "centerline_edges": tree.edges.tolist(),
            "center": _rounded(center.reshape(1, 3))[0],
            "radius": radius,
        },
        separators=(",", ":"),
    )
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SXH MRCP Bile-Duct Centerline Tree</title>
<style>
:root {{ color-scheme: dark; --bg:#101414; --panel:rgba(19,27,27,.93); --line:rgba(235,248,244,.16); --text:#edf5f2; --muted:#aab9b5; --mask:#5fc5b2; --centerline:#ff7659; }}
* {{ box-sizing:border-box; }} html,body,#app {{ width:100%; height:100%; margin:0; overflow:hidden; background:var(--bg); color:var(--text); font-family:"Noto Sans SC","Microsoft YaHei",sans-serif; }}
canvas {{ width:100%; height:100%; display:block; cursor:grab; }} canvas:active {{ cursor:grabbing; }}
.panel {{ position:fixed; top:16px; left:16px; width:min(350px,calc(100vw - 32px)); padding:15px; background:var(--panel); border:1px solid var(--line); border-radius:8px; box-shadow:0 18px 48px rgba(0,0,0,.35); }}
h1 {{ margin:0 0 9px; font-size:17px; }} p {{ margin:0 0 13px; color:var(--muted); font-size:12px; line-height:1.5; }} label {{ display:flex; align-items:center; gap:8px; padding:7px 0; font-size:13px; }} input {{ width:16px; height:16px; accent-color:#e8f1ee; }} .swatch {{ width:11px; height:11px; border-radius:50%; }} button {{ margin-top:10px; padding:7px 10px; color:var(--text); background:#27302f; border:1px solid var(--line); border-radius:5px; cursor:pointer; }}
.stats {{ margin-top:11px; padding-top:10px; border-top:1px solid var(--line); color:var(--muted); font-size:12px; line-height:1.6; }} .hint {{ position:fixed; right:16px; bottom:12px; color:var(--muted); font-size:12px; }}
</style></head><body><div id="app"><canvas id="viewer"></canvas></div><section class="panel"><h1>SXH 胆管中心线树</h1><p>半透明绿色为 MRCP 胆管分割表面；橙红色为完整 3D skeleton tree。左键拖拽旋转，滚轮缩放。</p><label><input id="showMask" type="checkbox" checked><span class="swatch" style="background:var(--mask)"></span>胆管分割</label><label><input id="showCenterline" type="checkbox" checked><span class="swatch" style="background:var(--centerline)"></span>中心线树</label><button id="reset">重置视角</button><div id="stats" class="stats"></div></section><div class="hint">MRCP physical space (mm)</div>
<script>
const DATA={payload}; const canvas=document.getElementById("viewer"),gl=canvas.getContext("webgl",{{antialias:true}}); if(!gl)throw new Error("WebGL unavailable");
const vs=`attribute vec3 position;uniform mat4 matrix;uniform float pointSize;void main(){{gl_Position=matrix*vec4(position,1.0);gl_PointSize=pointSize;}}`,fs=`precision mediump float;uniform vec3 color;uniform float alpha;void main(){{vec2 c=gl_PointCoord-vec2(.5);if(dot(c,c)>.25)discard;gl_FragColor=vec4(color,alpha);}}`;
function shader(t,s){{const x=gl.createShader(t);gl.shaderSource(x,s);gl.compileShader(x);if(!gl.getShaderParameter(x,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(x));return x;}} const program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,vs));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,fs));gl.linkProgram(program);gl.useProgram(program);
const pos=gl.getAttribLocation(program,"position"),matrix=gl.getUniformLocation(program,"matrix"),color=gl.getUniformLocation(program,"color"),alpha=gl.getUniformLocation(program,"alpha"),pointSize=gl.getUniformLocation(program,"pointSize");
function normalized(points){{const out=new Float32Array(points.length*3),s=1/Math.max(DATA.radius,1),c=DATA.center;for(let i=0;i<points.length;i++)for(let a=0;a<3;a++)out[i*3+a]=(points[i][a]-c[a])*s;return out;}} function buffer(data){{const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,data,gl.STATIC_DRAW);return b;}}
const mask={{buffer:buffer(normalized(DATA.mask_surface_mm)),count:DATA.mask_surface_mm.length}}; const linePoints=[]; for(const e of DATA.centerline_edges){{linePoints.push(DATA.centerline_vertices_mm[e[0]],DATA.centerline_vertices_mm[e[1]]);}} const centerline={{buffer:buffer(normalized(linePoints)),count:linePoints.length}};
let rx=-.58,ry=.82,zoom=1.55,drag=false,lastX=0,lastY=0; canvas.onpointerdown=e=>{{drag=true;lastX=e.clientX;lastY=e.clientY;canvas.setPointerCapture(e.pointerId)}};canvas.onpointerup=()=>drag=false;canvas.onpointermove=e=>{{if(!drag)return;ry+=(e.clientX-lastX)*.008;rx+=(e.clientY-lastY)*.008;lastX=e.clientX;lastY=e.clientY;draw()}};canvas.onwheel=e=>{{e.preventDefault();zoom=Math.min(5,Math.max(.42,zoom*Math.exp(e.deltaY*.001)));draw()}};for(const id of ["showMask","showCenterline"])document.getElementById(id).onchange=draw;document.getElementById("reset").onclick=()=>{{rx=-.58;ry=.82;zoom=1.55;draw()}};
function mul(a,b){{const o=new Float32Array(16);for(let r=0;r<4;r++)for(let c=0;c<4;c++)o[c*4+r]=a[r]*b[c*4]+a[4+r]*b[c*4+1]+a[8+r]*b[c*4+2]+a[12+r]*b[c*4+3];return o}} function perspective(f,a,n,z){{const q=1/Math.tan(f/2),d=1/(n-z);return new Float32Array([q/a,0,0,0,0,q,0,0,0,0,(z+n)*d,-1,0,0,2*z*n*d,0])}} function trans(z){{return new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,z,1])}} function rotX(a){{const c=Math.cos(a),s=Math.sin(a);return new Float32Array([1,0,0,0,0,c,s,0,0,-s,c,0,0,0,0,1])}} function rotY(a){{const c=Math.cos(a),s=Math.sin(a);return new Float32Array([c,0,-s,0,0,1,0,0,s,0,c,0,0,0,0,1])}}
function resize(){{const d=devicePixelRatio||1,w=Math.floor(canvas.clientWidth*d),h=Math.floor(canvas.clientHeight*d);if(canvas.width!==w||canvas.height!==h){{canvas.width=w;canvas.height=h}}gl.viewport(0,0,w,h)}} function bind(item){{gl.bindBuffer(gl.ARRAY_BUFFER,item.buffer);gl.enableVertexAttribArray(pos);gl.vertexAttribPointer(pos,3,gl.FLOAT,false,0,0)}}
function drawMask(){{bind(mask);gl.uniform3f(color,.373,.773,.698);gl.uniform1f(alpha,.19);gl.uniform1f(pointSize,Math.max(1.4,3.2/zoom));gl.drawArrays(gl.POINTS,0,mask.count)}} function drawCenterline(){{bind(centerline);gl.uniform3f(color,1,.463,.349);gl.uniform1f(alpha,1);gl.uniform1f(pointSize,1);gl.drawArrays(gl.LINES,0,centerline.count)}}
function draw(){{resize();gl.clearColor(.063,.078,.078,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.uniformMatrix4fv(matrix,false,mul(perspective(Math.PI/4,canvas.width/canvas.height,.01,100),mul(mul(trans(-3.15*zoom),rotX(rx)),rotY(ry))));if(document.getElementById("showMask").checked)drawMask();if(document.getElementById("showCenterline").checked)drawCenterline();}} window.onresize=draw;document.getElementById("stats").textContent=`${{DATA.mask_surface_mm.length.toLocaleString()}} surface points | ${{DATA.centerline_vertices_mm.length.toLocaleString()}} skeleton voxels | ${{DATA.centerline_edges.length.toLocaleString()}} edges`;draw();
</script></body></html>'''


def export_centerline(mask_path: Path, output_dir: Path, max_surface_points: int = 12000) -> tuple[Path, Path]:
    """Write a NIfTI skeleton mask and its standalone HTML visualizer."""
    nii = nib.load(str(mask_path))
    mask = np.asanyarray(nii.dataobj) > 0
    tree = extract_centerline_tree(mask, nii.affine)
    surface_mm = extract_surface_points(mask, nii.affine, max_surface_points)
    output_dir.mkdir(parents=True, exist_ok=True)
    skeleton_path = output_dir / "sxh_mrcp_006_centerline_tree.nii.gz"
    html_path = output_dir / "sxh_mrcp_006_centerline_tree_viewer.html"
    skeleton_nii = nib.Nifti1Image(tree.skeleton_mask, nii.affine, nii.header.copy())
    skeleton_nii.set_data_dtype(np.uint8)
    nib.save(skeleton_nii, str(skeleton_path))
    html_path.write_text(build_viewer_html(surface_mm, tree), encoding="utf-8")
    return skeleton_path, html_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and view the complete SXH MRCP bile-duct skeleton tree.")
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-surface-points", type=int, default=12000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    skeleton_path, html_path = export_centerline(args.mask, args.output_dir, args.max_surface_points)
    print(f"Wrote skeleton: {skeleton_path}")
    print(f"Wrote viewer:   {html_path}")


if __name__ == "__main__":
    main()
