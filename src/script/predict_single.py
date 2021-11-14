import conf.model_006 as config

import torch
import net.util as net_util
import post.make_submission as make_submission


def main():
    cfg = config.get_default_conf()

    stage = 3
    epoch = 1
    cfg.fold = 0
    saved_model = '{}_{}_{}_{}'.format(cfg.model.name, cfg.fold, stage, epoch)
    model_path = '{}model/{}/{}.pth'.format(cfg.output_path, cfg.model.name, saved_model)
    model = torch.load(model_path)

    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    model.to(cfg.model.device)

    cfg = config.get_default_conf()
    # Set seeds
    net_util.set_seed(cfg.seed)
    make_submission.predict_with_tta(cfg, model)


if __name__ == "__main__":
    main()
