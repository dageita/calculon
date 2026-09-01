import { Empty, Select } from 'antd';
import * as echarts from 'echarts';
import { useEffect, useMemo, useRef, useState } from 'react';

const X_OPTIONS = [
  { value: 'scale_up_size', label: 'Max Scale-up Size (GPUs)' },
  { value: 'intra_bandwidth', label: 'Intra-node Bandwidths (GB/s)' },
  { value: 'inter_pod_bandwidth', label: 'Inter-node Aggregate Bandwidth (GB/s per Scale-up Pod)' },
  { value: 'intra_latency', label: 'Intra-node Latencies (s)' },
  { value: 'inter_latency', label: 'Inter-node Latencies (s)' },
];

const Y_OPTIONS = [
  { value: 'batch_total_time', label: 'Batch Total Time (s)' },
  { value: 'linear_scaling_throughput', label: 'Linear Scaling Throughput (samples/s)' },
  { value: 'total_efficiency', label: 'Total Efficiency (≈MFU)' },
  { value: 'total_comm_time', label: 'Total Comm Time (s)' },
  { value: 'time_to_train_seconds', label: 'Time-to-Train (s)' },
];

const SYMBOLS = ['circle', 'rect', 'roundRect', 'triangle', 'diamond', 'pin', 'arrow'];

const HARDWARE_LABELS: Record<string, string> = {
  scale_up_size: 'Scale-up',
  intra_bandwidth: 'Intra BW',
  inter_pod_bandwidth: 'Inter BW/pod',
  intra_latency: 'Intra latency',
  inter_latency: 'Inter latency',
  network_topology: 'Topology',
};

const hardwareValue = (key: string, value: any) => {
  if (key.includes('latency')) return Number(value).toExponential(1);
  if (key.includes('bandwidth')) return `${Number(value).toFixed(1)} GB/s`;
  if (key === 'scale_up_size') return `${value} GPUs`;
  return String(value);
};

const hardwareDescriptor = (candidate: any, xKey: string) => {
  const hardware = candidate?.hardware || {};
  const supplied = hardware.chart_series_context || {};
  const context = Object.keys(supplied).length ? supplied : Object.fromEntries(
    [...X_OPTIONS.map((item) => item.value), 'network_topology']
      .filter((key) => key !== xKey)
      .map((key) => [key, hardware[key]]),
  );
  const stableContext = Object.entries(context)
    .filter(([, value]) => value !== undefined && value !== null)
    .sort(([left], [right]) => left.localeCompare(right));
  return {
    key: JSON.stringify(stableContext),
    label: stableContext.map(([key, value]) => (
      `${HARDWARE_LABELS[key] || key}: ${hardwareValue(key, value)}`
    )).join(' · '),
  };
};

const strategyColor = (index: number) => (
  `hsl(${Math.round((index * 137.508) % 360)}, 68%, 46%)`
);

const CandidateSearchChart = ({ candidates = [] }: { candidates?: any[] }) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const instanceRef = useRef<echarts.ECharts | null>(null);
  const [xKey, setXKey] = useState('intra_bandwidth');
  const [yKey, setYKey] = useState('batch_total_time');
  const [selectedStrategyKeys, setSelectedStrategyKeys] = useState<string[]>([]);
  const xLabel = X_OPTIONS.find((item) => item.value === xKey)?.label || xKey;

  useEffect(() => {
    setSelectedStrategyKeys([]);
  }, [candidates, xKey]);
  const yLabel = Y_OPTIONS.find((item) => item.value === yKey)?.label || yKey;

  const points = useMemo(() => {
    const uniquePoints = new Map<string, any>();
    candidates.forEach((candidate) => {
      const sweepDimension = candidate?.hardware?.sweep_dimension;
      if (
        sweepDimension
        && !['baseline', xKey].includes(sweepDimension)
      ) return;
      const x = Number(candidate?.hardware?.[xKey]);
      const y = Number(candidate?.metrics?.[yKey]);
      if (!Number.isFinite(x) || !Number.isFinite(y)) return;
      const hardwareSeries = hardwareDescriptor(candidate, xKey);
      const point = {
        candidate,
        x,
        y,
        strategyKey: hardwareSeries.key,
        strategyLabel: hardwareSeries.label,
      };
      // Baseline and the selected axis sweep overlap at their baseline value.
      // Render that configuration once so every series has one y per x.
      const identity = JSON.stringify([hardwareSeries.key, x]);
      if (!uniquePoints.has(identity)) uniquePoints.set(identity, point);
    });
    return Array.from(uniquePoints.values())
      .sort((left, right) => left.x - right.x);
  }, [candidates, xKey, yKey]);

  const strategyGroups = useMemo(() => {
    const grouped = new Map<string, { key: string; name: string; points: typeof points }>();
    points.forEach((point) => {
      const current = grouped.get(point.strategyKey);
      if (current) current.points.push(point);
      else grouped.set(point.strategyKey, { key: point.strategyKey, name: point.strategyLabel, points: [point] });
    });
    return Array.from(grouped.values())
      .sort((left, right) => left.name.localeCompare(right.name))
      .map((group, styleIndex) => ({ ...group, styleIndex }));
  }, [points]);

  const selectedStrategySet = useMemo(
    () => new Set(selectedStrategyKeys), [selectedStrategyKeys],
  );
  const visibleStrategyGroups = useMemo(
    () => strategyGroups.filter((group) => selectedStrategySet.has(group.key)),
    [strategyGroups, selectedStrategySet],
  );
  const visiblePointCount = visibleStrategyGroups.reduce(
    (total, group) => total + group.points.length, 0,

  );
  useEffect(() => {
    if (!chartRef.current) return;
    instanceRef.current = echarts.init(chartRef.current);
    const resize = () => instanceRef.current?.resize();
    window.addEventListener('resize', resize);
    return () => {
      window.removeEventListener('resize', resize);
      instanceRef.current?.dispose();
      instanceRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!instanceRef.current) return;
    instanceRef.current.clear();
    instanceRef.current.setOption({
      animation: false,
      grid: { left: 80, right: 32, top: 30, bottom: visiblePointCount > 50 ? 90 : 65 },
      tooltip: {
        trigger: 'item',
        formatter: (params: any) => {
          const candidate = params.data.candidate;
          const hardware = candidate.hardware || {};
          return [
            `Candidate ${candidate.candidate_id}`,
            `${xLabel}: ${params.value[0]}`,
            `${yLabel}: ${params.value[1]}`,
            `Scale-up/Scale-out: ${hardware.scale_up_size}/${hardware.scale_out_size}`,
            `Intra BW: ${hardware.intra_bandwidth} GB/s`,
            `Inter BW: ${hardware.inter_pod_bandwidth} GB/s per pod`,
            `Intra/Inter latency: ${hardware.intra_latency}/${hardware.inter_latency} s`,
            `Topology: ${hardware.network_topology || '-'}`,
          ].join('<br/>');
        },
      },
      xAxis: { type: 'value', name: xLabel, nameLocation: 'middle', nameGap: 38, scale: true },
      yAxis: { type: 'value', name: yLabel, nameLocation: 'middle', nameGap: 58, scale: true },
      dataZoom: visiblePointCount > 50 ? [
        { type: 'slider', xAxisIndex: 0, bottom: 20 },
        { type: 'inside', xAxisIndex: 0 },
      ] : [],
      series: visibleStrategyGroups.map((group) => ({
        id: group.key,
        name: group.name,
        type: 'scatter',
        symbol: SYMBOLS[group.styleIndex % SYMBOLS.length],
        symbolSize: 9,
        itemStyle: { color: strategyColor(group.styleIndex) },
        data: group.points.map((point) => ({
          value: [point.x, point.y],
          candidate: point.candidate,
        })),
      })),
    }, true);
  }, [visibleStrategyGroups, visiblePointCount, xLabel, yLabel]);

  if (!candidates.length) return null;
  return (
    <div style={{ marginTop: 16 }}>
      <div style={{ display: 'flex', gap: 16, marginBottom: 12 }}>
        <span style={{ alignSelf: 'center' }}>X Axis</span>
        <Select style={{ minWidth: 280 }} value={xKey} options={X_OPTIONS} onChange={setXKey} />
        <span style={{ alignSelf: 'center' }}>Y Axis</span>
        <Select style={{ minWidth: 300 }} value={yKey} options={Y_OPTIONS} onChange={setYKey} />
      </div>
      <div style={{ marginBottom: 12, maxWidth: 650 }}>
        <div style={{ marginBottom: 6 }}>Legend: Hardware values held fixed while X changes</div>
        <Select
          mode="multiple"
          style={{ width: '100%' }}
          value={selectedStrategyKeys}
          options={strategyGroups.map((group) => ({ value: group.key, label: group.name }))}
          onChange={setSelectedStrategyKeys}
          placeholder="Select a fixed hardware design to compare"
          maxTagCount={1}
          listHeight={180}
          allowClear
        />
      </div>

      <div style={{ color: '#666', marginBottom: 8 }}>
        Software strategy, global batch, and placement are fixed to the final choice shown in
        Optimal Result. Within each series, only X changes; the legend lists every other fixed
        Superpod parameter. Different series compare alternative high-quality hardware contexts.
      </div>
      <div ref={chartRef} style={{ width: '100%', height: 420, display: points.length ? 'block' : 'none' }} />
      {!points.length && <Empty description="No candidate has both selected metrics" />}
    </div>
  );
};

export default CandidateSearchChart;

