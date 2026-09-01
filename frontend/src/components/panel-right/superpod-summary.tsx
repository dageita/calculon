import { Descriptions, Divider } from 'antd';

const value = (input: any, digits = 6) => {
  if (input === null || input === undefined) return '-';
  const numeric = Number(input);
  return Number.isFinite(numeric) ? Number(numeric.toFixed(digits)) : String(input);
};

const metrics = (data: any) => <>
  <Descriptions.Item label="Batch Total Time (s)">
    {value(data?.batch_total_time)}
  </Descriptions.Item>
  <Descriptions.Item label="Throughput (samples/s)">
    {value(data?.linear_scaling_throughput)}
  </Descriptions.Item>
  <Descriptions.Item label="Total Efficiency (≈MFU)">
    {value(data?.total_efficiency)}
  </Descriptions.Item>
  <Descriptions.Item label="Total Comm Time (s)">
    {value(data?.total_comm_time)}
  </Descriptions.Item>
  {data?.time_to_train_seconds != null && (
    <Descriptions.Item label="Time-to-Train (s)">
      {value(data.time_to_train_seconds)}
    </Descriptions.Item>
  )}
</>;

const SuperpodSummary = ({ summary }: { summary: any }) => {
  const conclusions = summary?.design_conclusions;
  if (!conclusions) return null;
  const hardwareConclusion = conclusions.hardware_design || {};
  const gpuConclusion = conclusions.gpu_design || {};
  const fixedGpu = hardwareConclusion.fixed_gpu || {};
  const hardware = hardwareConclusion.optimal_hardware || {};
  const software = hardwareConclusion.optimal_software || {};
  const baselineGpu = gpuConclusion.baseline_gpu || {};
  const optimalGpu = gpuConclusion.optimal_gpu || {};
  const fixedHardware = gpuConclusion.fixed_hardware || {};
  const mapping = Object.entries(hardware.parallel_dimension_mapping || {})
    .map(([dimension, tier]) => `${dimension}: ${tier}`)
    .join('; ');

  return <>
    <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 12 }}>
      {hardwareConclusion.title || 'Conclusion 1: Optimal Superpod Design'}
    </div>
    <Descriptions colon={false} className="customize-des"
      column={{ xxl: 3, xl: 2, lg: 2, md: 2, sm: 2, xs: 1 }}>
      <Descriptions.Item label="Fixed GPU Model">{fixedGpu.name || '-'}</Descriptions.Item>
      <Descriptions.Item label={`Fixed ${fixedGpu.matrix_dtype || 'Matrix'} Peak (TFLOPS)`}>
        {value(fixedGpu.compute_tflops)}
      </Descriptions.Item>
      <Descriptions.Item label="Fixed GPU Memory (GiB)">
        {value(fixedGpu.memory_capacity_gib)}
      </Descriptions.Item>
      <Descriptions.Item label="Fixed HBM Bandwidth (GB/s)">
        {value(fixedGpu.memory_bandwidth_gbps)}
      </Descriptions.Item>
      <Descriptions.Item label="Comparison Global Batch Size">
        {hardwareConclusion.comparison_batch_size}
      </Descriptions.Item>
      <Descriptions.Item label="Search Objectives">
        {(hardwareConclusion.objectives || []).join(', ')}
      </Descriptions.Item>
      <Descriptions.Item label="GPU Numbers">
        {hardware.gpu_numbers ?? hardware.selected_gpu_numbers}
      </Descriptions.Item>
      <Descriptions.Item label="Optimal Max Scale-up Size">
        {hardware.scale_up_size}
      </Descriptions.Item>
      <Descriptions.Item label="Scale-out Size">
        {hardware.scale_out_size}
      </Descriptions.Item>
      <Descriptions.Item label="Optimal Intra-node Bandwidth (GB/s)">
        {value(hardware.intra_bandwidth)}
      </Descriptions.Item>
      <Descriptions.Item label="Optimal Inter-node Aggregate Bandwidth (GB/s/pod)">
        {value(hardware.inter_pod_bandwidth)}
      </Descriptions.Item>
      <Descriptions.Item label="Optimal Intra-node Latency (s)">
        {value(hardware.intra_latency, 12)}
      </Descriptions.Item>
      <Descriptions.Item label="Optimal Inter-node Latency (s)">
        {value(hardware.inter_latency, 12)}
      </Descriptions.Item>
      <Descriptions.Item label="Placement Policy">{hardware.placement_policy}</Descriptions.Item>
      <Descriptions.Item span={3} label="Parallel → Fabric Mapping">
        {mapping || '-'}
      </Descriptions.Item>
      <Descriptions.Item label="TP / PP / DP / EP / CP">
        {`${software.tensor_parallel}/${software.pipeline_parallel}/${software.data_parallel}/${software.expert_parallel}/${software.context_parallel}`}
      </Descriptions.Item>
      <Descriptions.Item label="Batch / Microbatch">
        {`${software.batch_size}/${software.microbatch_size}`}
      </Descriptions.Item>
      <Descriptions.Item label="Global Balanced Score">
        {value(hardwareConclusion.balanced_score)}
      </Descriptions.Item>
      {metrics(hardwareConclusion.metrics)}
      <Descriptions.Item span={3} label="Selection Method">
        {hardwareConclusion.selection_method}
      </Descriptions.Item>
    </Descriptions>

    <Divider />
    <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 12 }}>
      {gpuConclusion.title || 'Conclusion 2: Optimal GPU Design'}
    </div>
    {gpuConclusion.status === 'error' ? (
      <div>{gpuConclusion.error}</div>
    ) : (
      <Descriptions colon={false} className="customize-des"
        column={{ xxl: 3, xl: 2, lg: 2, md: 2, sm: 2, xs: 1 }}>
        <Descriptions.Item label="Fixed Max Scale-up Size">
          {fixedHardware.scale_up_size}
        </Descriptions.Item>
        <Descriptions.Item label="Fixed Intra / Inter Bandwidth">
          {`${value(fixedHardware.intra_bandwidth)} GB/s / ${value(fixedHardware.inter_pod_bandwidth)} GB/s/pod`}
        </Descriptions.Item>
        <Descriptions.Item label="Fixed Intra / Inter Latency">
          {`${value(fixedHardware.intra_latency, 12)}s / ${value(fixedHardware.inter_latency, 12)}s`}
        </Descriptions.Item>
        <Descriptions.Item label={`Baseline ${baselineGpu.matrix_dtype || 'Matrix'} Peak (TFLOPS)`}>
          {value(baselineGpu.compute_tflops)}
        </Descriptions.Item>
        <Descriptions.Item label="Baseline GPU Memory (GiB)">
          {value(baselineGpu.memory_capacity_gib)}
        </Descriptions.Item>
        <Descriptions.Item label="Baseline HBM Bandwidth (GB/s)">
          {value(baselineGpu.memory_bandwidth_gbps)}
        </Descriptions.Item>
        <Descriptions.Item label={`Optimal ${optimalGpu.matrix_dtype || 'Matrix'} Peak (TFLOPS)`}>
          {value(optimalGpu.compute_tflops)}
        </Descriptions.Item>
        <Descriptions.Item label="Optimal GPU Memory (GiB)">
          {value(optimalGpu.memory_capacity_gib)}
        </Descriptions.Item>
        <Descriptions.Item label="Optimal HBM Bandwidth (GB/s)">
          {value(optimalGpu.memory_bandwidth_gbps)}
        </Descriptions.Item>
        <Descriptions.Item label="Compute / Memory / HBM Factors">
          {`${value(optimalGpu.compute_factor)}× / ${value(optimalGpu.memory_capacity_factor)}× / ${value(optimalGpu.memory_bandwidth_factor)}×`}
        </Descriptions.Item>
        <Descriptions.Item label="Throughput Change vs Baseline">
          {`${value(gpuConclusion.throughput_change_vs_baseline_percent)}%`}
        </Descriptions.Item>
        <Descriptions.Item label="GPU Designs Evaluated">
          {gpuConclusion.evaluated_gpu_designs}
        </Descriptions.Item>
        {metrics(gpuConclusion.metrics)}
        <Descriptions.Item span={3} label="Selection Method">
          {gpuConclusion.selection_method}
        </Descriptions.Item>
      </Descriptions>
    )}
  </>;
};

export default SuperpodSummary;
