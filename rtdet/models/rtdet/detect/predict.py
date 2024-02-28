from ....nn.autobackend import AutoBackend

from ultralytics.engine.predictor import BasePredictor
from ultralytics.utils.torch_utils import select_device
from ultralytics.engine.results import Results
from ultralytics.utils import ops
from ultralytics.utils.files import increment_path
from pathlib import Path
import pdb


class DetectionPredictor(BasePredictor):
    
    def __call__(self, source=None, model=None, stream=False, *args, **kwargs):

        self.stream = stream
        if stream:
            return self.stream_inference(source, model, *args, **kwargs)
        else:
            res = self.stream_inference(source, model, *args, **kwargs)
            return list(res)

    def stream_inference(self, source=None, model=None, *args, **kwargs):
        # pdb.set_trace()
        return super().stream_inference(source, model, *args, **kwargs)
    
    def setup_model(self, model, verbose=True):
        self.model = AutoBackend(
            model or self.args.model,
            device=select_device(self.args.device, verbose=verbose),
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.half,
            fuse=True,
            verbose=verbose,
        )

        self.device = self.model.device  # update device
        self.args.half = self.model.fp16  # update half
        self.model.eval()

    def preprocess(self, im):
        # tbreak /root/envs/py38/lib/python3.8/site-packages/ultralytics/engine/predictor.py:278
        return super().preprocess(im)
    
    def inference(self, im, *args, **kwargs):
        visualize = (
            increment_path(self.save_dir / Path(self.batch[0][0]).stem, mkdir=True)
            if self.args.visualize and (not self.source_type.tensor)
            else False
        )
        return self.model(im, augment=self.args.augment, visualize=visualize, embed=self.args.embed, *args, **kwargs)
    
    def pre_transform(self, im):
        return super().pre_transform(im)

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results
