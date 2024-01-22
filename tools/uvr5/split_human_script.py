import os
import sys
current_working_directory = os.getcwd()

sys.path.append(current_working_directory)
sys.path.append(current_working_directory + '/tools')
sys.path.append(current_working_directory + '/GPT_SoVITS')
import logging
from tools.i18n.i18n import I18nAuto
i18n = I18nAuto()
import traceback

logger = logging.getLogger(__name__)
import ffmpeg
import torch
import sys
from mdxnet import MDXNetDereverb
from vr import AudioPre, AudioPreDeEcho
from config import python_exec,infer_device,is_half,exp_root,webui_port_main,webui_port_infer_tts,webui_port_uvr5,webui_port_subfix


weight_uvr5_root = "tools/uvr5/uvr5_weights"
uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", ""))

device=infer_device
is_half=True


def uvr(model_name, paths, save_root_vocal, save_root_ins, agg, format0):
    infos = []
    try:
        save_root_vocal = (
            save_root_vocal.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        save_root_ins = (
            save_root_ins.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        if model_name == "onnx_dereverb_By_FoxJoy":
            pre_fun = MDXNetDereverb(15, device)
        else:
            func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
            pre_fun = func(
                agg=int(agg),
                model_path=os.path.join(
                    weight_uvr5_root, model_name + ".pth"
                ),
                device=device,
                is_half=is_half,
            )
        is_hp3 = "HP3" in model_name
        for path in paths:
            inp_path = path
            need_reformat = 1
            done = 0
            try:
                info = ffmpeg.probe(inp_path, cmd="ffprobe")
                if (
                    info["streams"][0]["channels"] == 2
                    and info["streams"][0]["sample_rate"] == "44100"
                ):
                    need_reformat = 0
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0, is_hp3=is_hp3
                    )
                    done = 1
            except:
                need_reformat = 1
                traceback.print_exc()
            if need_reformat == 1:
                tmp_path = "%s/%s.reformatted.wav" % (
                    os.path.join(os.environ["TEMP"]),
                    os.path.basename(inp_path),
                )
                os.system(
                    "ffmpeg -i %s -vn -acodec pcm_s16le -ac 2 -ar 44100 %s -y"
                    % (inp_path, tmp_path)
                )
                inp_path = tmp_path
            try:
                if done == 0:
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0
                    )
                infos.append("%s->Success" % (os.path.basename(inp_path)))
                yield "\n".join(infos)
            except:
                try:
                    if done == 0:
                        pre_fun._path_audio_(
                            inp_path, save_root_ins, save_root_vocal, format0
                        )
                    infos.append("%s->Success" % (os.path.basename(inp_path)))
                    yield "\n".join(infos)
                except:
                    infos.append(
                        "%s->%s" % (os.path.basename(inp_path), traceback.format_exc())
                    )
                    yield "\n".join(infos)
    except:
        infos.append(traceback.format_exc())
        yield "\n".join(infos)
    finally:
        try:
            if model_name == "onnx_dereverb_By_FoxJoy":
                del pre_fun.pred.model
                del pre_fun.pred.model_
            else:
                del pre_fun.model
                del pre_fun
        except:
            traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Executed torch.cuda.empty_cache()")
    yield "\n".join(infos)

def split_human_audio(model_name, paths, save_root_vocal, save_root_ins, agg, format0):
    uvrGenerator = uvr(model_name, paths, save_root_vocal, save_root_ins, agg, format0)
    for value in uvrGenerator:
        print(value)


model_name = "VR-DeEchoAggressive"
current_working_directory = os.getcwd()
inp_root = ""
save_root_vocal = current_working_directory + "/" + "data_handle"
paths = [current_working_directory + "/" + "data/ready.wav"]
save_root_ins = current_working_directory + "/" + "data_handle"
agg = 10
formate0 = "wav"
split_human_audio(model_name=model_name,
                  paths=paths, save_root_vocal=save_root_vocal , save_root_ins=save_root_ins, agg=agg, format0=formate0)


