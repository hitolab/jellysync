#!/usr/bin/env python3
"""Generate Fig.1C from fig1.mp4 and its DeepLabCut CSV.

Requires Python 3, numpy, pandas, matplotlib, Helvetica, ffmpeg and ffprobe.
Run: python3 fig1.py [--output-dir /path/to/output]

CSV row/frame numbers are mapped to MP4 presentation timestamps, sorted in
presentation order. The counts and consecutive CSV frame numbers are checked.
Waveform: marker2_l x, original sign, centered three-point moving average.
Display: MP4 85–90 s, with 85 s labelled 0 s; subtract 665 px for y display.
Three major local maxima in MP4 87–88 s are selected with 0.25 s separation.
Only the first-to-second peak interval is divided into six equal time intervals. Each target is
mapped to the nearest actual MP4 frame. Dots, image labels and peak lines use
the selected frame timestamps; their heights use the CSV values at those frames.
All snapshots use the same linear display contrast: clip(RGB*1.6-0.02, 0, 1).
Images are rotated 10 degrees counterclockwise before a common 900x900 crop
at (280, 110). The figure is 151x65 mm. Times are Helvetica 6 pt, two decimals.
"""
from pathlib import Path
import argparse
import shutil
import subprocess
import json
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib import font_manager

base = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, default=base)
args = parser.parse_args()
out = args.output_dir.resolve()
out.mkdir(parents=True, exist_ok=True)
frames = out / 'Fig1C_assets_mp4_7frames'
frames.mkdir(exist_ok=True)
ffmpeg = shutil.which('ffmpeg')
ffprobe = shutil.which('ffprobe')
if not ffmpeg or not ffprobe:
    raise RuntimeError('ffmpeg and ffprobe must be available on PATH.')
for weight in ['normal', 'bold']:
    font_manager.findfont(font_manager.FontProperties(family='Helvetica', weight=weight), fallback_to_default=False)
video = base / 'fig1.mp4'
raw = pd.read_csv(base / 'fig1.csv', skiprows=2)
probe = json.loads(subprocess.check_output([
    ffprobe, '-v', 'error', '-select_streams', 'v:0',
    '-show_entries', 'packet=pts_time', '-of', 'json', str(video)
], text=True))
video_times = np.array(sorted(float(packet['pts_time']) for packet in probe['packets']))
if len(video_times) != len(raw):
    raise RuntimeError('CSV row count does not match MP4 frame count.')
if not np.array_equal(raw['coords'].to_numpy(), np.arange(len(raw))):
    raise RuntimeError('CSV frame indices must be consecutive and start at zero.')
if not np.all(np.diff(video_times) > 0):
    raise RuntimeError('MP4 presentation timestamps must be strictly increasing.')
wave = pd.DataFrame({
    'frame': raw['coords'],
    'time_s': video_times,
    'marker2_l_x_original_px': raw['x.2'],
    'sm_marker2_l_x_original_sign_px': raw['x.2'].rolling(3, center=True).mean(),
})
start, end = 85.0, 90.0
baseline = 665.0
df = wave.loc[wave.time_s.between(start, end)].copy()
t = df.time_s.to_numpy() - start
y = df.sm_marker2_l_x_original_sign_px.to_numpy() - baseline
all_y = wave.sm_marker2_l_x_original_sign_px.to_numpy()
candidates = np.where((all_y[1:-1] > all_y[:-2]) & (all_y[1:-1] >= all_y[2:]))[0] + 1
candidates = candidates[(video_times[candidates] >= 87) & (video_times[candidates] <= 88)]
selected = []
for k in sorted(candidates, key=lambda k: all_y[k], reverse=True):
    if all(abs(video_times[k] - video_times[j]) >= 0.25 for j in selected):
        selected.append(k)
peaks = np.array(sorted(selected))
if len(peaks) != 3:
    raise RuntimeError(f'Expected three major peaks, found {len(peaks)}.')
peak_times = video_times[peaks]
target_times = np.linspace(peak_times[0], peak_times[1], 7)
indices = np.array([np.argmin(abs(video_times - value)) for value in target_times])
if len(np.unique(indices)) != 7:
    raise RuntimeError('Expected 7 distinct selected frames.')
actual_times = video_times[indices]
dots = pd.DataFrame({
    'target_mp4_time_s': target_times,
    'actual_mp4_time_s': actual_times,
    'relative_time_s': actual_times - start,
    'relative_position_px': all_y[indices] - baseline,
    'video_frame_index_0based': indices,
    'selection_error_s': actual_times - target_times,
    'snapshot_file': [f'snapshot_{i:02d}.png' for i in range(1, 8)],
})
pt = dots.relative_time_s.iloc[[0, 6]].to_numpy()
expr = '+'.join(f'eq(n,{n})' for n in indices)
extraction = subprocess.run([
    ffmpeg, '-hide_banner', '-loglevel', 'info', '-i', str(video),
    '-vf', f"select='{expr}',rotate=-10*PI/180:fillcolor=black,crop=900:900:280:110,showinfo",
    '-fps_mode', 'vfr', '-frames:v', '7', '-y', str(frames / 'snapshot_%02d.png'),
], check=True, capture_output=True, text=True)
extracted_pts = np.array([float(value) for value in re.findall(
    r'\bn:\s*\d+.*?\bpts_time:([0-9.]+)', extraction.stderr
)])
if len(extracted_pts) != 7 or not np.allclose(extracted_pts, actual_times, rtol=0, atol=0.00006):
    raise RuntimeError('Extracted frame timestamps do not match selected MP4 timestamps.')
if not np.all((y >= 0) & (y <= 35)):
    raise RuntimeError('Waveform would be clipped by the fixed y-axis limits.')
dots.to_csv(frames / 'snapshot_times.csv', index=False)
df.to_csv(frames / 'waveform_85-90s.csv', index=False)
pd.DataFrame({'frame': peaks, 'mp4_time_s': peak_times,
              'relative_time_s': peak_times-start, 'position_px': all_y[peaks]}).to_csv(frames/'selected_peaks.csv', index=False)
print('Selected peak MP4 times:', peak_times)
print('Maximum nearest-frame error (s):', np.max(abs(actual_times-target_times)))

pink='#FF1493'
plt.rcParams.update({'font.family':'Helvetica','font.size':7,'axes.linewidth':1.,'axes.labelweight':'normal','svg.fonttype':'path','pdf.fonttype':42})
fig=plt.figure(figsize=(151 / 25.4, 65 / 25.4))
for i,row in dots.iterrows():
    x=.09+i*(.89/7)
    imgax=fig.add_axes([x,.694,.89/7-.003,.23])
    # Uniform linear display adjustment for every snapshot; source PNGs are unchanged.
    image = plt.imread(frames/row.snapshot_file)
    image = np.clip(image * 1.6 - 0.02, 0, 1)
    imgax.imshow(image)
    imgax.set_xticks([])
    imgax.set_yticks([])
    for spine in imgax.spines.values():
        spine.set_visible(i in [0,6])
        spine.set_color(pink)
        spine.set_linewidth(2)
    imgax.text(.96,.035,f'{row.relative_time_s:.2f} s',transform=imgax.transAxes,
               ha='right',va='bottom',color='white',fontfamily='Helvetica',fontsize=6)
ax=fig.add_axes([.09,.19,.89,.47])
ax.plot(t,y,color='black',lw=0.8)
for peak_t in pt:
    ax.axvline(peak_t,color=pink,linestyle='--',linewidth=1,zorder=2)
colors=[pink if i in [0,6] else 'black' for i in range(len(dots))]
ax.scatter(dots.relative_time_s,dots.relative_position_px,s=12,c=colors,zorder=4)
ax.set(xlim=(0,5),ylim=(0,35),xlabel='Time (s)',ylabel='Position (px)')
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1)
ax.xaxis.label.set_size(10)
ax.xaxis.label.set_weight('normal')
ax.yaxis.label.set_size(10)
ax.yaxis.label.set_weight('normal')
ax.tick_params(axis='both', which='major', length=0.8 * 72 / 25.4, width=0.3, labelsize=8)
ax.grid(False)
for ext in ['png','svg','pdf']:
    fig.savefig(out/f'Fig.1C.{ext}',dpi=220)

plt.close(fig)
print(f"Saved: {out / 'Fig.1C.pdf'}")
