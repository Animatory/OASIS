import config
import dataloaders.dataloaders as dataloaders
import models.models as models
import utils.utils as utils

# --- read options ---#
opt = config.read_arguments(train=False)

# --- create dataloader ---#
_, dataloader_val = dataloaders.get_dataloaders(opt)

# --- create utils ---#
image_saver = utils.ResultsSaver(opt)

# --- create models ---#
model = models.OASIS(opt)
model = models.put_on_multi_gpus(model, opt)
model.eval()

# --- iterate over validation set ---#
for i, data_i in enumerate(dataloader_val):
    label = models.preprocess_input(opt, data_i)['label']
    generated = model(None, label, "generate", None)
    image_saver(label, generated, data_i["name"])
