import React, { useEffect, useRef } from 'react';
import * as echarts from 'echarts';

export interface IBaseTLProps {
  result: any,
}

/** Known palette; unknown types get a stable hashed color. */
const KNOWN_EVENT_COLORS: Record<string, string> = {
  COMPUTE_FWD: '#507399',
  COMPUTE_FFN_FWD: '#6b8fad',
  COMPUTE_BWD: '#f28e2b',
  COMPUTE_MLA_BWD: '#f5a85a',
  TP_COMM_FWD: '#59a14f',
  TP_COMM_BWD: '#8bd17c',
  PP_COMM_FWD: '#e15759',
  PP_COMM_BWD: '#ffa19f',
  DP_COMM_EVENT: '#b07aa1',
  EP_COMM_FWD: '#4e79a7',
  EP_COMM_BWD: '#a0cbe8',
  EP_DISPATCH_FWD: '#76b7b2',
  EP_DISPATCH_BWD: '#b2dfdb',
  EP_COMBINE_FWD: '#edc948',
  EP_COMBINE_BWD: '#f1d96b',
  CP_COMM_FWD: '#9c755f',
  CP_COMM_BWD: '#c9a99a',
};

const FALLBACK_PALETTE = [
  '#86bcb6', '#d37295', '#bab0ac', '#79706e', '#d4a6c8',
  '#9d7660', '#b07aa1', '#ff9d9a', '#fabfd2', '#c6c8ce',
];

function colorForType(name: string): string {
  if (KNOWN_EVENT_COLORS[name]) return KNOWN_EVENT_COLORS[name];
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = (hash * 31 + name.charCodeAt(i)) >>> 0;
  }
  return FALLBACK_PALETTE[hash % FALLBACK_PALETTE.length];
}

function assignLayers(events: any[]): number {
  events.sort((a, b) => a.start_time - b.start_time);
  const layers: any[][] = [];

  events.forEach((event) => {
    let placed = false;
    for (let i = 0; i < layers.length; i++) {
      const layer = layers[i];
      const canPlace = layer.every(
        (existingEvent) =>
          event.end_time <= existingEvent.start_time ||
          event.start_time >= existingEvent.end_time,
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

const TimeLineCharts: React.FC<IBaseTLProps> = (props) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);
  const { result } = props;
  const data = Array.isArray(result) ? result : [];

  const updateChartData = () => {
    if (!chartInstance.current) return;

    if (data.length === 0) {
      chartInstance.current.clear();
      chartInstance.current.setOption({
        title: {
          text: 'No timeline events',
          left: 'center',
          top: 'middle',
          textStyle: { color: '#999', fontSize: 14, fontWeight: 'normal' },
        },
      });
      return;
    }

    const ranks = [...new Set(data.map((item) => item.rank))].sort(
      (a: number, b: number) => b - a,
    );

    const dataByRank: Record<string, any[]> = {};
    ranks.forEach((rank) => {
      dataByRank[rank] = data.filter((item) => item.rank === rank);
    });

    let maxLayers = 1;
    Object.values(dataByRank).forEach((rankEvents) => {
      maxLayers = Math.max(maxLayers, assignLayers(rankEvents));
    });

    const extendedRanks: string[] = [];
    ranks.forEach((rank) => {
      for (let i = 0; i < maxLayers; i++) {
        extendedRanks.push(`${rank}${i > 0 ? ` L${i + 1}` : ''}`);
      }
    });

    // Discover types from payload (covers layered MLA/FFN, EP, CP, etc.)
    const discoveredTypes = [
      ...new Set(
        data
          .map((item) => item.event_type || 'unknown')
          .filter((name: string) => !!name),
      ),
    ] as string[];
    const eventTypes = discoveredTypes.map((name) => ({
      name,
      color: colorForType(name),
    }));

    const dataByType: Record<string, any[]> = {};
    eventTypes.forEach((type) => {
      dataByType[type.name] = [];
    });

    data.forEach((item) => {
      const type = item.event_type || 'unknown';
      if (!dataByType[type]) {
        dataByType[type] = [];
      }
      const baseIndex = ranks.indexOf(item.rank);
      const visualLayer = item.visualLayer || 0;
      const yPosition = baseIndex * maxLayers + (maxLayers - visualLayer - 1);
      const start = item.start_time * 1000;
      const end = item.end_time * 1000;
      const duration = end - start;

      dataByType[type].push({
        name: type,
        value: [yPosition, start, end, duration],
        itemStyle: { color: colorForType(type) },
        layer: item.layer,
        rank: item.rank,
        event_type: type,
        microbatch: item.microbatch,
        visualLayer,
      });
    });

    // Labels on every bar become unreadable / expensive with many events.
    const showLabels = data.length <= 120;

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
      type: 'custom',
      label: {
        show: showLabels,
        position: 'inside',
        formatter: function (params: any) {
          return `{a|${params.data.event_type}}\n{b|MB${params.data.microbatch}}`;
        },
        rich: {
          a: {
            color: '#000000',
            lineHeight: 20,
            fontSize: 8,
            align: 'center',
          },
          b: {
            color: '#000000',
            fontSize: 8,
            lineHeight: 20,
            align: 'center',
          },
        },
      },
      renderItem,
      itemStyle: {
        opacity: 0.8,
        color: type.color,
        borderWidth: 1,
        borderColor: '#333',
      },
      encode: {
        x: [1, 2],
        y: 0,
      },
      data: dataByType[type.name],
    }));

    const chartHeight = Math.max(480, ranks.length * maxLayers * 28 + 140);
    if (chartRef.current) {
      chartRef.current.style.height = `${Math.min(chartHeight, 1200)}px`;
      chartInstance.current.resize();
    }

    const option = {
      tooltip: {
        trigger: 'item',
        formatter: function (params: any) {
          if (!params.data) return '';
          const rank = params.data.rank;
          const start = (params.value[1] / 1000).toFixed(3);
          const end = (params.value[2] / 1000).toFixed(3);
          const duration = (params.value[3] / 1000).toFixed(3);
          const microbatch = params.data.microbatch;
          return `
                        <div style="margin-bottom: 5px;"><b>${params.name}</b></div>
                        <div>Rank: ${rank}</div>
                        <div>Microbatch: ${microbatch}</div>
                        <div>StartTime: ${start} s</div>
                        <div>EndTime: ${end} s</div>
                        <div>Duration: ${duration} s</div>
                    `;
        },
      },
      legend: {
        data: eventTypes.map((type) => type.name),
        top: 10,
        type: 'scroll',
        textStyle: { fontSize: 12 },
      },
      grid: {
        left: 120,
        right: 30,
        top: 60,
        bottom: 80,
      },
      xAxis: {
        type: 'value',
        scale: true,
        axisLabel: {
          formatter: function (val: number) {
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
            fontSize: 12,
            formatter: function (value: string | number) {
              const label = String(value);
              if (label.includes(' L')) {
                return '';
              }
              return `Rank ${label}`;
            },
          },
          axisTick: {
            show: true,
            alignWithLabel: true,
            length: 4,
          },
          splitLine: {
            show: true,
            lineStyle: {
              type: 'dashed',
              color: '#eee',
            },
          },
        },
        {
          type: 'category',
          data: extendedRanks,
          show: false,
        },
      ],
      dataZoom: [
        {
          type: 'slider',
          xAxisIndex: 0,
          filterMode: 'filter',
          bottom: 30,
          height: 20,
          labelFormatter: function (value: number) {
            return (value / 1000).toFixed(3) + ' s';
          },
        },
        {
          type: 'inside',
          xAxisIndex: 0,
          filterMode: 'filter',
        },
      ],
      series,
    };

    chartInstance.current.clear();
    chartInstance.current.setOption(option, true);
  };

  const handleResize = () => {
    chartInstance.current?.resize();
  };

  useEffect(() => {
    if (chartRef.current && !chartInstance.current) {
      chartInstance.current = echarts.init(chartRef.current);
    }
    updateChartData();
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result]);

  useEffect(() => {
    return () => {
      chartInstance.current?.dispose();
      chartInstance.current = null;
    };
  }, []);

  return (
    <div style={{ width: '100%', minHeight: '480px', padding: '20px' }}>
      <div ref={chartRef} style={{ width: '100%', height: '480px' }} />
    </div>
  );
};

export default TimeLineCharts;
