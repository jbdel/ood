import torch
import glob
import os
path = "retina_baseline"
ckpt = glob.glob(os.path.join(path,'*.pth'))[0]
print(ckpt)

d = torch.load(ckpt)
for k,v in d.items():
    if 'ood_metrics' in k:
        for l in v:
            print(l['OOD Name'], l)

    if 'ind_metrics' in k:
        print(k,v)
    if 'best_early_stop_value' in k:
        print(k,v)

# utils.save_checkpoint(checkpoints_folder,
#                                       {
#                                           "init_epoch": epoch + 1,
#                                           "net": net.state_dict(),
#                                           "optimizer": model_config.optimizer.state_dict(),
#                                           "scheduler": model_config.scheduler.state_dict() if args.use_scheduler else None,
#                                           "ood_metrics": ood_metric_dicts,
#                                           "ind_metrics": ind_metrics,
#                                           "best_early_stop_value": best_early_stop_value,
#                                           "args": args,
#                                       },
#