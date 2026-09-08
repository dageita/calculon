"""Two-resource framework scheduler for eager PyTorch/Transformer Engine graphs."""

class RuntimeEventDAG:
  """Schedule host submission and one ordered GPU stream without double counting."""

  def __init__(self, system):
    self.system = system
    self.enabled = bool(system.framework_runtime_enabled)
    self.host_tail = 0.0
    self.gpu_tail = 0.0
    self.host_work = 0.0
    self.gpu_work = 0.0
    self.sync_wait = 0.0
    self.sync_overhead = 0.0
    self.events = []

  def enqueue(self, label, device_s, kernels=1, autograd_nodes=0, host_s=None):
    """Submit an asynchronous operator and append its work to the GPU stream."""
    device_s = max(0.0, float(device_s))
    kernels = max(0, int(kernels))
    autograd_nodes = max(0, int(autograd_nodes))
    if not self.enabled:
      self.gpu_tail += device_s
      self.gpu_work += device_s
      return
    host_before = self.host_tail
    host_cost = float(host_s) if host_s is not None else (
      kernels * self.system.framework_kernel_dispatch_s +
      autograd_nodes * self.system.framework_autograd_node_s)
    self.host_work += host_cost
    if kernels:
      chunk = device_s / kernels
      host_chunk = host_cost / kernels
      for _ in range(kernels):
        self.host_tail += host_chunk
        self.gpu_tail = max(self.gpu_tail, self.host_tail) + chunk
    else:
      self.host_tail += host_cost
      if device_s:
        self.gpu_tail = max(self.gpu_tail, self.host_tail) + device_s
    self.gpu_work += device_s
    self.events.append({
      'type': 'enqueue', 'label': label, 'host_start': host_before,
      'host_end': self.host_tail, 'gpu_end': self.gpu_tail})

  def host_compute(self, label, duration_s, wait_for_gpu=False):
    """Run synchronous host/CPU work, optionally after its GPU dependency."""
    duration_s = max(0.0, float(duration_s))
    before = self.host_tail
    if wait_for_gpu:
      wait = max(0.0, self.gpu_tail - self.host_tail)
      self.host_tail = max(self.host_tail, self.gpu_tail)
      self.sync_wait += wait
    self.host_tail += duration_s
    self.host_work += duration_s
    self.events.append({
      'type': 'host', 'label': label, 'host_start': before,
      'host_end': self.host_tail, 'gpu_end': self.gpu_tail})

  def synchronize(self, label, intrinsic_s):
    """Block the host on the current GPU tail, then charge only API overhead."""
    intrinsic_s = (max(0.0, float(intrinsic_s)) if self.enabled else 0.0)
    before = self.host_tail
    wait = max(0.0, self.gpu_tail - self.host_tail)
    self.host_tail = max(self.host_tail, self.gpu_tail) + intrinsic_s
    self.host_work += intrinsic_s
    self.sync_wait += wait
    self.sync_overhead += intrinsic_s
    self.events.append({
      'type': 'sync', 'label': label, 'host_start': before,
      'host_end': self.host_tail, 'gpu_end': self.gpu_tail, 'wait': wait})

  @property
  def elapsed(self):
    return max(self.host_tail, self.gpu_tail)

  def summary(self):
    return {
      'elapsed': self.elapsed,
      'host_tail': self.host_tail,
      'gpu_tail': self.gpu_tail,
      'host_work': self.host_work,
      'gpu_work': self.gpu_work,
      'sync_wait': self.sync_wait,
      'sync_overhead': self.sync_overhead,
      'event_count': len(self.events),
      'sync_count': sum(1 for event in self.events if event['type'] == 'sync'),
    }
