import traceback
import sys
sys.path.append('/content/GPT-SoVITS')
sys.path.append('/content/GPT-SoVITS/tools')
sys.path.append('/content/GPT-SoVITS/GPT_SoVITS')
from collections import OrderedDict

import torch
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()


def savee(ckpt, name, epoch, steps, hps):
    try:
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = ckpt[key].half()
        opt["config"] = hps
        opt["info"] = "%sepoch_%siteration" % (epoch, steps)
        torch.save(opt, "%s/%s.pth" % (hps.save_weight_dir, name))
        return "Success."
    except:
        return traceback.format_exc()
