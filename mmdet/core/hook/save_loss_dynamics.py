from collections import defaultdict
from mmcv.runner.hooks import HOOKS, Hook

@HOOKS.register_module()
class SaveLossDynamicsHook(Hook):
    def before_run(self, runner):
        runner.meta["loss_dynamics"] = {}

    def after_train_iter(self, runner):
        outputs = getattr(runner, "outputs", None)
        loss_dynamics = outputs["loss_dynamics"]

        for k, dyns in loss_dynamics.items():
            if k not in runner.meta["loss_dynamics"]:
                runner.meta["loss_dynamics"][k] = defaultdict(list)

            save_to = runner.meta["loss_dynamics"][k]
            for anno_id, loss_dyn in dyns.items():
                save_to[anno_id] += [loss_dyn]
