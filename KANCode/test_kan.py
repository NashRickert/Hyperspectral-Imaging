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
model = KAN(width=width, grid = grid, k=k, seed=42, device=device, symbolic_enabled=False, auto_save=False)
width = [200, 32, 32, 32, 16]

model.load_state_dict(torch.load('temp_model.pt', map_location=device))
        
# Done to sync C macro def with this code (Macro became global var accessible here)
TBL_SIZE = lib.TABLE_SIZE

ex_layer = model.act_fun[0]

x = torch.rand((1,200)).to(device)

# print(x)

y, _, _, _ = ex_layer.forward(x)

y_diff, _, _, _ = ex_layer.act_forward(x)

y_coef = coef2curve(x, ex_layer.grid, ex_layer.coef, ex_layer.k)

print(y.shape)
print(y)

print(y_diff.shape)
print(y_diff)


print(y_coef.shape)
print(y_coef)

grid = ex_layer.grid

