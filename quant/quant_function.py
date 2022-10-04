import torch
import torch.nn.functional as F
#from qtorch.quant import float_quantize
import numpy as np
from .number import BlockMinifloat, Number
from .block_design import block_design


__all__ = ['block_minifloat_quantize', "quantizer"]


def logr2(data):
    const = torch.zeros_like(data) + 2**(-0.5)
    return torch.log(data)/torch.log(const)

def r2(data):
    return (2**(-0.5))**(data)

def add_r_(data):
    r = torch.rand_like(data)
    data.add_(r)

# Developing a function to round to a multiple - https://datagy.io/python-round-to-multiple/
def round_to_multiple(number, multiple):
    return multiple * torch.round(number / multiple)

# print(round_to_multiple(-10,2))

def block_minifloat_quantize(x, number, rounding="stochastic", tensor_type="x", k_exp = 1):

    assert isinstance(x, torch.Tensor), "x is not a single precision Floating Point Tensor"

    # print("quant input: ", x)
    # print(f"k value being used for quantisation is {k_exp}")
    # shared exponent
    mean_func = lambda x, dim: torch.mean(x, dim)
    max_func = lambda x, dim: torch.max(x, dim)[0]

    # compute max exponent
    max_exponent = block_design(x, number.tile, tensor_type, max_func) 

    # log
    if number.man == 0:
        i = x * 2**(-max_exponent + number.bias)
        sgn = torch.sign(i)
        #i = torch.log2(torch.abs(i)+1e-60)
        i = logr2(torch.abs(i)+1e-60)
        add_r_(i)
        i.floor_()
        #i = 2**(i)
        i = r2(i)
        i = torch.where(i<=2**(-2**(number.exp+1-1) + 1), torch.zeros_like(i), i)
        i.clamp_(0, 1)
        i = i * sgn
        out = i * 2**(max_exponent-number.bias)
        return out
    
    # fixed
    elif number.exp == 0:
        bits = number.man + 1
        i = x * 2**(-max_exponent + number.bias + bits - 1)
        #i = fixed_point_quantize(i, number.man+1, number.man, rounding=rounding)
        if rounding == "stochastic":
            r = torch.rand_like(i)
            i.add_(r).floor_().clamp_(-2**(bits-1), 2**(bits-1)-1)
        else:
            i.round_().clamp_(-2**(bits-1), 2**(bits-1)-1)
        out = i * 2**(max_exponent-number.bias -bits + 1)
        return out
    
    # minifloat
    else:

        ### is scaling needed here? k x offset  ----- no need to multiply max exponent
        offset = max_exponent - number.emax
        # print("max exp: ", max_exponent)

        # print(f"exp to calc emax: {number.exp}")
        # print(f"k to calc emax: {number.k_exp}")
        # print(f"emax: {number.emax}")
        # print("offset: ", offset)

        rem = torch.remainder(offset, k_exp)
        # print(f"{rem}")
        rounded = round_to_multiple(offset, k_exp)
        # print("rounded: ", rounded)
        offset = torch.where(rem!=0, rounded, offset)
        # print("offset post rounding: ", offset)

        # if (offset % k_exp != 0):
        #     offset = round_to_multiple(offset, k_exp)
            # print("offset post rounding: ", offset)

        # shared exponent shifting
        shift = 2**(-offset)
        # print("shift: ", shift)
        i = x * shift #this multiplication is done in order to left or right shift the data to centre it around 0 -- anything that cannot be represented will be 0
        # print("i = x*shift: ", i)

        # clamping at zero (uses QPyTorch float_quantizer - qtorch doesn't have a zero bit?)
        if (number.flush_to_zero):
            raise NotImplementedError
            #k = float_quantize(i, number.exp, number.man, rounding=rounding)
            #k = torch.where(torch.abs(i)<(2**(number.emin+1)), torch.zeros_like(i), k) # flush to zero
            #out = k * 2**(offset) 
            #return out
        

        # handle subnormal and normal quantization
        emin = number.emin 
        emax = number.emax # number.of_emax
        esbn = 2**(emin+1)
        lsbn = 2**(number.emax)
        mval = 2**(number.man)
        rlim = number.max_number

        sgn = torch.sign(i)
        i = torch.abs(i)
        e = torch.floor(torch.log2(i+1e-60)) #exponent of every point in the data
        # print("exp of every point: ", e)
        # clamp the exponent
        e.clamp_(emin+1, emax) # emin+1 for subnormal region
        # print(f"exp being clamped to the range: {emin+1} to {emax}")
        # print("exp of every point post clamp: ", e)
        # unpack frac for subnormal and normal region
        ie = i*2**(-e)
        # print("ie: ",ie)
        me = 2**(e)
        #something that has its exponent clamped out in the e.clamp stage enters into the subnorm region
        f = torch.where(i<esbn, ie, ie-1)  #for the subnormal region, just use ie, else for normal region, subtact 1 to remove the 1.xy part 
        #f stands for just the fractional part of the number to be represented  --- between 1 and 0
        #f is an integer number
        # print("frac part: ", f)

        # rounding on frac
        if rounding == "stochastic":
            r = torch.rand_like(f)  #generates decimal number between 0 and 1 
            # print("r: ", r)
            f.mul_(mval).add_(r).floor_()
            clipped = f.clamp_(0, mval) #it works dont touch it
            clipped.div_(mval).mul_(me)
            # print("f post clip: ",clipped)
        else:
            f.mul_(mval).round_()
            clipped.div_(mval).mul_(me)
        # sign magnitude multiplication for subnormal and normal
        k = torch.where(i<esbn, clipped, me+clipped)
        k.clamp_(-rlim, rlim)
        out = sgn * k * 2**(offset)

        # print(f"quantised entry: {k} x 2^{offset}")
        # print("")
        # print("Error: ", x-out)
        # print("")
        # print("")
        # print("")
        # print("")
        return out


def quantizer(forward_number=None, backward_number=None,
              forward_rounding="stochastic", backward_rounding="stochastic",
              clamping_grad_zero=False, backward_hooks=[]):
    """
    Creates a quantization function to support quantizing forward and backward process differently.

    Args:
        - :param: forward_number (qtorch.Number, optional) : the number format used for forward quantization.
                  if is None, the quantization would be a identity mapping.
        - :param: backward_number (qtorch.Number, optional) : the number format used for backward quantization.
                  if is None, the quantization would be a identity mapping.
        - :param: forward_rounding (string) : rounding mode, \"stochastic\" or \"nearest\" (default: \"stochastic\")
        - :param: backward_rounding (string) : rounding mode, \"stochastic\" or \"nearest\" (default: \"stochastic\")
        - :param: clamping_grad_zero (bool) : zero out the gradient of numbers that are being clamped during forward propagation.
                  currently requires forward_number to be a fixed point number.
        - :param: backward_hooks (iterable) : iterable of functions that will be applied to gradients before backward quantization.
                  For example, this can be used to support custom scaling.

    Returns:
        A quantization function as specified (torch.Tensor -> torch.Tensor)
    """
    if forward_number is not None:
        # import pdb; pdb.set_trace()
        print(f"forward number k val inside quantiser: {forward_number.k_exp}")
        if forward_number.exp == -1 or forward_number.man == -1:
            forward_number = None
    if backward_number is not None:
        if backward_number.exp == -1 or backward_number.man == -1:
            backward_number = None


    for rounding in [forward_rounding, backward_rounding]:
        assert rounding in ["stochastic", "nearest"], "invalid rounding type {:s}".format(rounding)
    for num in [forward_number, backward_number]:
        if num != None: assert isinstance(num, Number)

   
    # forward and backward quantisation functions
    tensor_type = "w" if backward_number is None else "x"
    forward_quant = lambda x, num, rd, tt, k_exp: block_minifloat_quantize(x, num, rd, tt, k_exp)
    backward_quant = lambda x, num, rd, tt, k_exp: block_minifloat_quantize(x, num, rd, tt, k_exp)  


    class Rounding(torch.autograd.Function):
        @staticmethod
        def forward(self, x):
            if forward_number==None: return x

            k_val = forward_number.k_exp
            out = forward_quant(x.contiguous(), forward_number, forward_rounding, tensor_type, k_val)

            return out.clone()

        @staticmethod
        def backward(self, grad_output):
            if self.needs_input_grad[0]:
                if backward_number == None:
                    grad_input = grad_output
                else:
                    k_val = backward_number.k_exp
                    grad_input = backward_quant(grad_output.contiguous(), backward_number, 
                        backward_rounding, tensor_type, k_val)
            else:
                grad_input = None

            return grad_input.clone()

    return Rounding.apply


