import torch

# based on 
# https://github.com/educating-dip/educated_deep_image_prior/blob/103c52dfb53e98e381ae5c7cd775795be63abb21/src/dataset/walnuts.py#L532

class _objectiveFunctionModule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, image_template, sirf_obj):
        ctx.sirf_obj = sirf_obj
        ctx.x = x
        ctx.image_template = image_template

        x_np = x.detach().cpu().numpy()
        x_np = ctx.image_template.fill(x_np)
        value_np = ctx.sirf_obj.get_value(x_np)
        value = torch.tensor(
            value_np).to(x.device)

        return value

    @staticmethod
    def backward(ctx, in_grad):

        grads_np = ctx.sirf_obj.get_gradient(
            ctx.image_template.fill(
                ctx.x.detach().cpu().numpy()
                )
            ).as_array()

        grads = torch.from_numpy(
            grads_np).to(in_grad.device
            ) * in_grad

        return grads, None, None, None

class ObjectiveFunctionModule(torch.nn.Module):
    def __init__(self, image_template, obj_fun):
        super().__init__()
        self.image_template = image_template.clone()
        self.obj_fun = obj_fun

    def forward(self, out):

        obj_fun_value_batch = torch.zeros(1, device=out.device)
        for out_i in out:
            obj_fun_value = _objectiveFunctionModule.apply(
                out_i, self.image_template, self.obj_fun
                )
            obj_fun_value_batch = obj_fun_value_batch + obj_fun_value
        return obj_fun_value