"""Fused SwiGLU/GeGLU vector kernels (not zero-cost GEMM epilogues)."""
from .layers import Layer


class GatedActivation(Layer):
  def __init__(self, name, sys, elements, kind='swiglu',
               needs_recompute=False):
    if kind not in ('swiglu', 'geglu'):
      raise ValueError('Expected swiglu or geglu')
    self.elements = elements
    # Activation plus multiply in forward; chain rule for both inputs.
    fw, bw = (5, 9) if kind == 'swiglu' else (9, 16)
    super().__init__(name, sys, fw_flops=fw*elements,
                     agrad_flops=bw*elements, inputs_size=2*elements,
                     output_size=elements, activation_space=2*elements,
                     activation_grads=2*elements,
                     needs_recompute=needs_recompute)

  def get_fw_mem_accessed(self):
    return 3*self.elements*self.bytes_per_element

  def get_agrad_mem_accessed(self):
    return 5*self.elements*self.bytes_per_element
