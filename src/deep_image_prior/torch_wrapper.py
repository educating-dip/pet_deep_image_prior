import torch
import numpy as np

class _objectiveFunctionalModule(torch.autograd.Function):
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

class ObjectiveFunctionalModule(torch.nn.Module):
    def __init__(self, image_template, obj_fun):
        super().__init__()
        self.image_template = image_template.clone()
        self.obj_fun = obj_fun

    def forward(self, out):

        obj_fun_value_batch = torch.zeros(1, device=out.device)
        for out_i in out:
            obj_fun_value = _objectiveFunctionalModule.apply(
                out_i, self.image_template, self.obj_fun
                )
            obj_fun_value_batch = obj_fun_value_batch + obj_fun_value
        return obj_fun_value

# class _acquisitionModelNumpyFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, image_template, data_template, sirf_obj):

#         ctx.sirf_obj = sirf_obj
#         ctx.image_template = image_template
#         ctx.data_template = data_template

#         x_np = x.detach().cpu().numpy()
#         x_np = ctx.image_template.fill(x_np[None])
#         proj_data_np = ctx.sirf_obj.forward(x_np).as_array()
#         proj_data = torch.from_numpy(proj_data_np).to(x.device)
#         return proj_data

#     @staticmethod
#     def backward(ctx, data):

#         data_np = data.detach().cpu().numpy()
#         data_np = ctx.data_template.fill(data_np)
#         grads_np = ctx.sirf_obj.backward(data_np).as_array()
#         grads = torch.from_numpy(grads_np).to(data.device)
#         return grads, None, None, None, None

# class AcquisitionModelModule(torch.nn.Module):
#     def __init__(self, image_template, data_template, acq_model):
#         super().__init__()

#         self.image_template = image_template.clone()
#         self.data_template = data_template.clone()
#         self.acq_model = acq_model

#     def forward(self, image):
#         # x.shape: (N, C, H, W) or (N, C, D, H, W)
#         image_nc_flat = image.view(-1, *image.shape[2:])
#         acquired_data_nc_flat = []
#         for x_i in image_nc_flat:
#             sym_data_i = _acquisitionModelNumpyFunction.apply(
#                 x_i, 
#                 self.image_template, 
#                 self.data_template, 
#                 self.acq_model
#             )
#             acquired_data_nc_flat.append(sym_data_i)
#         acquired_data = torch.cat(acquired_data_nc_flat).unsqueeze_(dim=0)
#         return acquired_data