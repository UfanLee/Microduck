from pathlib import Path
import sys

base = Path(__file__).resolve()
# Can be run from repository source directory or absolute location.
folder = Path.cwd()
source = folder / '../training/evaluate_sequence.py'
entry = folder / '../training/run.py'
if not source.is_file() or not entry.is_file():
    sys.exit('ERROR: First cd to experiments/desk-climb/source')
text = source.read_text()
anchor = "raw=ManagerBasedRlEnv(cfg,device='cuda:0');env=RslRlVecEnvWrapper(raw,clip_actions=agent.get('clip_actions'));runner="
replacement = """from mjlab.utils.wrappers import VideoRecorder
from mjlab_microduck.video_effects import configure_video_cfg
configure_video_cfg(cfg)
raw=ManagerBasedRlEnv(cfg,device='cuda:0',render_mode='rgb_array')
video_env=VideoRecorder(raw,video_folder=out/'video',step_trigger=lambda step: step==0,video_length=int(round(a.seconds/raw.step_dt)),disable_logger=True)
env=RslRlVecEnvWrapper(video_env,clip_actions=agent.get('clip_actions'));runner="""
if text.count(anchor)!=1:
    sys.exit('ERROR: Could not locate expected env constructor in original file')
text = text.replace(anchor,replacement,1)
end = "(out/'eval.json').write_text(json.dumps(result));assert not nonfinite;raw.close()"
if text.count(end)!=1:
    sys.exit('ERROR: Could not locate expected final line')
text = text.replace(end,"(out/'eval.json').write_text(json.dumps(result));assert not nonfinite;env.close()",1)
run = entry.read_text()
needle="script='evaluate_sequence.py'"
if run.count(needle)!=1:
    sys.exit('ERROR: Could not locate evaluate entry point')
run = run.replace(needle,"script='evaluate_video.py'",1)
video_py = folder / '../training/evaluate_video.py'
run_py = folder / '../training/run_video.py'
compile(text,str(video_py),'exec')
compile(run,str(run_py),'exec')
video_py.write_text(text)
run_py.write_text(run)
print('SUCCESS: created',video_py,run_py)
print('Original evaluation files were not changed.')
