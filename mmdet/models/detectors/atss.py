from collections import defaultdict
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from .base import BaseDetector


@DETECTORS.register_module()
class ATSS(SingleStageDetector):
    """Implementation of `ATSS <https://arxiv.org/abs/1912.02424>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(ATSS, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)


    def train_step(self, data, optimizer, compression_ctrl=None):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        losses, loss_dynamics = self(**data)
        loss, log_vars = self._parse_losses(losses)

        if compression_ctrl is not None:
            compression_loss = compression_ctrl.loss()
            loss += compression_loss

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']), loss_dynamics=loss_dynamics)

        return outputs


    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None, **kwargs):
        BaseDetector.forward_train(self, img, img_metas)
        x = self.extract_feat(img)
        anno_ids = kwargs["anno_ids"]
        losses, loss_dynamics = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore, anno_ids=anno_ids)

        return losses, loss_dynamics
