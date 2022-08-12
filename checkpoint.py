#%%

import torch
import re

backbone = torch.load("pretrained/model_best.pth")

key_list = list(backbone['state_dict'].keys())
new_backbone ={}
new_backbone['model'] = {}
for older_val in key_list:
    val = re.sub('module.', '', older_val)
    new_backbone['model'][val] = backbone['state_dict'].pop(older_val)


#%%
origin = torch.load("pretrained/mae_pretrain_vit_base_full.pth")
# print(origin)
patt = "decoder."
pattern = re.compile(patt)
key_list = list(origin['model'].keys())
for older_val in key_list:
    if len(pattern.findall(older_val)) != 0:
        new_backbone['model'][older_val] = origin['model'].pop(older_val)
    

print(new_backbone['model'].keys())
torch.save(new_backbone, "pretrained/model_converted.pth")

#%%
import torch
model = torch.load("pretrained/model_converted.pth")
key = model['model'].keys()
print(key)
# %%
origin = torch.load("pretrained/mae_pretrain_vit_base_full.pth")
print(origin['model'].keys())
# %%
