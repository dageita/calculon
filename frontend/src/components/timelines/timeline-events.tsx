import React, { useEffect, useRef } from 'react';
import * as echarts from 'echarts';

const BASE_EVENT_TYPES = [
  { name: 'COMPUTE_FWD', color: '#507399' },
  { name: 'COMPUTE_BWD', color: '#f28e2b' },
  { name: 'TP_COMM_FWD', color: '#59a14f' },
  { name: 'TP_COMM_BWD', color: '#8bd17c' },
  { name: 'PP_COMM_FWD', color: '#e15759' },
  { name: 'PP_COMM_BWD', color: '#ffa19f' },
  { name: 'DP_COMM_EVENT', color: '#b07aa1' },
];

const EXTRA_COLORS = ['#94a3b8', '#bc80bd', '#8c564b', '#e377c2', '#7f7f7f', '#17becf'];

export interface IBaseTLProps {
  result: any;
  /** 嵌入深色页面时使用更小高度与紧凑边距 */
  embedded?: boolean;
}

function buildMergedTypes(data: any[]) {
  const palette = [...BASE_EVENT_TYPES];
  const names = new Set(palette.map((t) => t.name));
  for (const item of data) {
    const raw = item?.event_type != null ? String(item.event_type).trim() : '';
    const key = raw || 'unknown';
    if (!names.has(key)) {
      names.add(key);
      palette.push({
        name: key,
        color: EXTRA_COLORS[(palette.length - BASE_EVENT_TYPES.length) % EXTRA_COLORS.length],
      });
    }
  }
  if (!names.has('unknown')) {
    palette.push({ name: 'unknown', color: '#94a3b8' });
  }
  return palette;
}

function updateChartDataWithEvents(
  chart: echarts.ECharts,
  raw: any[],
  eventTypes: { name: string; color: string }[],
) {
  const data = raw.map((e) => ({ ...e }));
  const ranks = [...new Set(data.map((item) => item.rank))].sort((a, b) => b - a);

  const dataByRank: Record<string, any[]> = {};
  ranks.forEach((rank) => {
    dataByRank[rank] = data.filter((item) => item.rank === rank);
  });

  function assignLayers(events: any[]) {
    events.sort((a, b) => a.start_time - b.start_time);
    const layers: any[][] = [];
    events.forEach((event) => {
      let placed = false;
      for (let i = 0; i < layers.length; i++) {
        const layer = layers[i];
        const canPlace = layer.every(
          (existingEvent) =>
            event.end_time <= existingEvent.start_time || event.start_time >= existingEvent.end_time,
        );
        if (canPlace) {
          layer.push(event);
          event.visualLayer = i;
          placed = true;
          break;
        }
      }
      if (!placed) {
        layers.push([event]);
        event.visualLayer = layers.length - 1;
      }
    });
    return layers.length;
  }

  let maxLayers = 1;
  Object.values(dataByRank).forEach((rankEvents) => {
    const layerCount = assignLayers(rankEvents as any[]);
    maxLayers = Math.max(maxLayers, layerCount);
  });

  const extendedRanks: string[] = [];
  ranks.forEach((rank, baseIndex) => {
    for (let i = 0; i < maxLayers; i++) {
      const virtualIndex = baseIndex * maxLayers + i;
      extendedRanks.push(`${rank}${i > 0 ? ` L${i + 1}` : ''}`);
    }
  });

  const dataByType: Record<string, any[]> = {};
  eventTypes.forEach((type) => {
    dataByType[type.name] = [];
  });

  const typeSet = new Set(eventTypes.map((t) => t.name));

  data.forEach((item) => {
    const raw = item.event_type != null ? String(item.event_type).trim() : '';
    let type = raw || 'unknown';
    if (!typeSet.has(type)) type = 'unknown';
    const rank = item.rank;
    const baseIndex = ranks.indexOf(rank);
    const visualLayer = item.visualLayer || 0;
    const yPosition = baseIndex * maxLayers + (maxLayers - visualLayer - 1);
    const start = item.start_time * 1000;
    const end = item.end_time * 1000;
    const duration = end - start;
    if (!dataByType[type]) dataByType[type] = [];
    dataByType[type].push({
      name: type,
      value: [yPosition, start, end, duration],
      itemStyle: {
        color: eventTypes.find((t) => t.name === type)?.color || '#ccc',
      },
      layer: item.layer,
      rank: item.rank,
      event_type: item.event_type,
      microbatch: item.microbatch,
      visualLayer,
    });
  });

  function renderItem(params: any, api: any) {
    const yPosition = api.value(0);
    const start = api.coord([api.value(1), yPosition]);
    const end = api.coord([api.value(2), yPosition]);
    const height = api.size([0, 1])[1] * 0.8;
    const rectShape = echarts.graphic.clipRectByRect(
      {
        x: start[0],
        y: start[1] - height / 2,
        width: end[0] - start[0],
        height,
      },
      {
        x: params.coordSys.x,
        y: params.coordSys.y,
        width: params.coordSys.width,
        height: params.coordSys.height,
      },
    );
    return (
      rectShape && {
        type: 'rect',
        transition: ['shape'],
        shape: rectShape,
        style: api.style(),
      }
    );
  }

  const series = eventTypes.map((type) => ({
    name: type.name,
    yAxisIndex: 1,
    type: 'custom' as const,
    label: {
      show: true,
      position: 'inside' as const,
      formatter(params: any) {
        return `{a|${params.data.event_type}}\n{b|MB${params.data.microbatch}}`;
      },
      rich: {
        a: { color: '#000000', lineHeight: 20, fontSize: 8, align: 'center' },
        b: { color: '#000000', fontSize: 8, lineHeight: 20, align: 'center' },
      },
    },
    renderItem,
    itemStyle: {
      opacity: 0.8,
      color: type.color,
      borderWidth: 1,
      borderColor: '#333',
    },
    encode: { x: [1, 2], y: 0 },
    data: dataByType[type.name] || [],
  }));

  const option = {
    backgroundColor: '#fafafa',
    tooltip: {
      trigger: 'item',
      formatter(params: any) {
        if (!params.data) return '';
        const rank = params.data.rank;
        const start = (params.value[1] / 1000).toFixed(3);
        const end = (params.value[2] / 1000).toFixed(3);
        const duration = (params.value[3] / 1000).toFixed(3);
        const microbatch = params.data.microbatch;
        const visualLayer = params.data.visualLayer + 1;
        return `
                        <div style="margin-bottom: 5px;"><b>${params.name}</b></div>
                        <div>Rank: ${rank}</div>
                        <div>Microbatch: ${microbatch}</div>
                        <div>StartTime: ${start} s</div>
                        <div>EndTime: ${end} s</div>
                        <div>Duration: ${duration} s</div>
                        <div>Layer: ${visualLayer}</div>
                    `;
      },
    },
    legend: {
      data: eventTypes.map((type) => type.name),
      top: 10,
      textStyle: { fontSize: 11 },
    },
    grid: { left: 120, right: 30, top: 56, bottom: 72 },
    xAxis: {
      type: 'value',
      scale: true,
      axisLabel: {
        formatter(val: number) {
          return (val / 1000).toFixed(3) + ' s';
        },
      },
      name: '',
      nameLocation: 'middle',
      nameGap: 30,
    },
    yAxis: [
      {
        type: 'category',
        data: ranks,
        name: 'GPU Cluster Timeline with Event Layering',
        nameLocation: 'middle',
        nameGap: 70,
        axisLabel: {
          fontSize: 11,
          formatter(value: string) {
            if (value.includes(' L')) return '';
            return `Rank ${value}`;
          },
        },
        axisTick: { show: true, alignWithLabel: true, length: 4 },
        splitLine: { show: true, lineStyle: { type: 'dashed', color: '#eee' } },
      },
      {
        type: 'category',
        data: extendedRanks,
        name: '设备ID与层级',
        axisLabel: {
          fontSize: 11,
          formatter(value: string) {
            if (value.includes(' L')) return '';
            return value;
          },
        },
        axisTick: { show: true, alignWithLabel: true, length: 4 },
        splitLine: { show: true, lineStyle: { type: 'dashed', color: '#eee' } },
        show: false,
      },
    ],
    dataZoom: [
      {
        type: 'slider',
        xAxisIndex: 0,
        filterMode: 'filter',
        bottom: 22,
        height: 18,
        labelFormatter(value: number) {
          return (value / 1000).toFixed(3) + ' s';
        },
      },
      { type: 'inside', xAxisIndex: 0, filterMode: 'filter' },
    ],
    series,
  };

  chart.setOption(option, true);
}

const TimeLineCharts: React.FC<IBaseTLProps> = (props) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);
  const { result, embedded } = props;

  useEffect(() => {
    const events = Array.isArray(result) ? result : [];
    const handleResize = () => chartInstance.current?.resize();
    window.addEventListener('resize', handleResize);

    if (!chartRef.current) {
      return () => window.removeEventListener('resize', handleResize);
    }

    if (events.length === 0) {
      chartInstance.current?.dispose();
      chartInstance.current = null;
      return () => window.removeEventListener('resize', handleResize);
    }

    if (!chartInstance.current) {
      chartInstance.current = echarts.init(chartRef.current);
    }
    const merged = buildMergedTypes(events);
    updateChartDataWithEvents(chartInstance.current, events, merged);
    chartInstance.current.resize();

    return () => {
      window.removeEventListener('resize', handleResize);
      chartInstance.current?.dispose();
      chartInstance.current = null;
    };
  }, [result]);

  const outerH = embedded ? 380 : 600;
  const chartH = embedded ? '78%' : '85%';

  return (
    <div style={{ width: '100%', height: outerH, padding: embedded ? '8px 4px 12px' : '20px' }}>
      <div ref={chartRef} style={{ width: '100%', height: chartH, minHeight: embedded ? 260 : 400 }} />
    </div>
  );
};

export default TimeLineCharts;
