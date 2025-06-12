from _kan import lib, ffi
import torch
from kan import *
import torch
from kan.spline import coef2curve

width = [200, 32, 32, 32, 16]
grid = 10
k = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# This function modifies the width variable for some reason >:(. So we redeclare right after I guess
model = KAN(width=width, grid = grid, k=k, seed=42, device=device, symbolic_enabled=False)
width = [200, 32, 32, 32, 16]

model.load_state_dict(torch.load('temp_model.pt', map_location=device))
        
# For now I just define this here. In the future, it would be better to sync this with a C code through a global var
# Eg make a macro for it, define a global var to be equal to the macro, expose the global var in this code with cffi
TBL_SIZE = lib.TABLE_SIZE

c_model = ffi.new("struct model *")
c_model = lib.init_model(width, len(width))
# print(c_model.len)
# for i in range(c_model.len):
#     layer = ffi.new("struct layer *", c_model.layers[i])
#     print("layer idx:" + str(layer.idx))
#     print("layer len:" + str(layer.len))
    # for i in range(layer.len):
        
# print(len(model.width) - 1)
# print(model.act_fun)
# layer = model.act_fun[0]
# print(layer.grid)
# print(layer.grid.shape)

# print(layer.coef)
# print(layer.coef.shape)

# Accepts something like model.act_fun[i], a layer of input
def compute_layer_input(layer):
    grid = layer.grid

    x = []
    # For each input dim (aka for each node)
    for i in range(len(grid)):
        minimum = grid[i, 0]
        maximum = grid[i, -1]
        eval_pts = torch.linspace(start=minimum, end=maximum, steps=TBL_SIZE)
        x.append(eval_pts)
    x = torch.stack(x)
    return (x.T).to(device)

# Note that most of the output values are small, some of them are entirely zero. Is this the desired behavior?
def compute_layer_output(layer):
    x = compute_layer_input(layer)
    with torch.no_grad():
        y = coef2curve(x, layer.grid, layer.coef, k)
    return y.to(device)

# y = compute_layer_output(layer)
# print(y)
# print(y.shape)
# print(len(model.act_fun))

for i, func in enumerate(model.act_fun):
    x = compute_layer_input(func)
    print(x)
    print(x.shape)
    y = compute_layer_output(func)
    y_flat = torch.flatten(y)
    
    c_val_shape = ffi.new("struct Shape *")
    c_val_shape.len = 3  # also: = len(list(y.shape))
    c_val_shape.dim = ffi.new("int[]", list(y.shape))

    c_val_tens = ffi.new("struct Tensor *")
    c_val_tens[0] = lib.construct_tensor(c_val_shape[0])

    c_val_tens.data = ffi.new("float []", y_flat.tolist())

    c_meta_shape = ffi.new("struct Shape *")
    c_meta_shape.len = 2
    c_meta_shape.dim = ffi.new("int[]", [len(y), 4])

    c_meta_tens = ffi.new("struct Tensor *")
    c_meta_tens[0] = lib.construct_tensor(c_val_shape[0])

    # Need to do something to fill these fields for c_meta_tens
    # Do this with the x var, use shape to make sure everything matches

    # TODO: Just finish this loop, test inference
    # note the concerning fact that I failed a malloc

    print(c_val_tens.shape.len)

    
# print(x)
# model(x)

# Note to self: I am struggling to find in the documentation how I can access, get the range, and run the forward pass on a single activation layer. This is kind of being a pain in my butt

# def make_buf(arr, type):
#     length = len(arr)
#     x = ffi.new(f"{type}[{length}]")
#     print(x)
#     for i, el in enumerate(arr):
#         print(i)
#         print(arr[i])
#         x[i] = int(el)
#     return x
