import React, { useEffect, useRef } from 'react';
import * as echarts from 'echarts';

export interface IBaseTLProps {
  result: any,
}

const TimeLineCharts: React.FC<IBaseTLProps> = (props) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);
  const { result } = props;
  // 操作类型与颜色映射
  const eventTypes = [
    { name: 'COMPUTE_FWD', color: '#507399' },
    { name: 'COMPUTE_BWD', color: '#f28e2b' },
    { name: 'TP_COMM_FWD', color: '#59a14f' },
    { name: 'TP_COMM_BWD', color: '#8bd17c' },
    { name: 'PP_COMM_FWD', color: '#e15759' },
    { name: 'PP_COMM_BWD', color: '#ffa19f' },
    { name: 'DP_COMM_EVENT', color: '#b07aa1' }
  ];
  const data = result;

  // 初始化图表
  const initChart = () => {
    if (chartRef.current && !chartInstance.current) {
      chartInstance.current = echarts.init(chartRef.current);
      updateChartData();
    }
  };


  // 更新图表数据
  const updateChartData = () => {
    if (!chartInstance.current) return;
    // 获取唯一的设备ID
    const ranks = [...new Set(data.map(item => item.rank))].sort((a, b) => b - a);

    // 按设备ID分组数据
    const dataByRank = {};
    ranks.forEach(rank => {
      dataByRank[rank] = data.filter(item => item.rank === rank);
    });

    // 检测时间重叠并分配层级
    function assignLayers(events) {
      // 按开始时间排序
      events.sort((a, b) => a.start_time - b.start_time);

      const layers = [];

      events.forEach(event => {
        let placed = false;

        // 尝试将事件放入现有层级
        for (let i = 0; i < layers.length; i++) {
          const layer = layers[i];
          const canPlace = layer.every(existingEvent =>
            event.end_time <= existingEvent.start_time || event.start_time >= existingEvent.end_time
          );

          if (canPlace) {
            layer.push(event);
            event.visualLayer = i;
            placed = true;
            break;
          }
        }

        // 如果没有合适的层级，创建新层级
        if (!placed) {
          layers.push([event]);
          event.visualLayer = layers.length - 1;
        }
      });

      return layers.length; // 返回总层数
    }

    // 为每个设备的数据分配层级
    let maxLayers = 1;
    Object.values(dataByRank).forEach(rankEvents => {
      const layerCount = assignLayers(rankEvents);
      maxLayers = Math.max(maxLayers, layerCount);
    });

    // 创建扩展的Y轴数据，为每个设备ID创建多个虚拟行
    const extendedRanks: any[] = [];
    const yAxisPositions = {};

    ranks.forEach((rank, baseIndex) => {
      yAxisPositions[rank] = [];
      for (let i = 0; i < maxLayers; i++) {
        const virtualIndex = baseIndex * maxLayers + i;
        extendedRanks.push(`${rank}${i > 0 ? ` L${i + 1}` : ''}`);
        yAxisPositions[rank].push(virtualIndex);
      }
    });

    // 按操作类型分类数据
    const dataByType = {};
    eventTypes.forEach(type => {
      dataByType[type.name] = [];
    });

    // 处理数据格式
    data.forEach(item => {
      const type = item.event_type;
      const rank = item.rank;
      const baseIndex = ranks.indexOf(rank);
      const visualLayer = item.visualLayer || 0;

      // 计算Y轴位置
      const yPosition = baseIndex * maxLayers + (maxLayers - visualLayer - 1);
      const start = item.start_time * 1000; // 转换为毫秒
      const end = item.end_time * 1000; // 转换为毫秒
      const duration = end - start;

      dataByType[type].push({
        name: type,
        value: [yPosition, start, end, duration],
        itemStyle: {
          color: eventTypes.find(t => t.name === type)?.color || '#ccc'
        },
        layer: item.layer,
        rank: item.rank,
        event_type: item.event_type,
        microbatch: item.microbatch,
        visualLayer: visualLayer
      });
    });

    // 渲染函数
    function renderItem(params, api) {
      const yPosition = api.value(0);
      const start = api.coord([api.value(1), yPosition]);
      const end = api.coord([api.value(2), yPosition]);
      const height = api.size([0, 1])[1] * 0.8; // 增加高度以填充可用空间

      const rectShape = echarts.graphic.clipRectByRect(
        {
          x: start[0],
          y: start[1] - height / 2,
          width: end[0] - start[0],
          height: height
        },
        {
          x: params.coordSys.x,
          y: params.coordSys.y,
          width: params.coordSys.width,
          height: params.coordSys.height
        }
      );

      return rectShape && {
        type: 'rect',
        transition: ['shape'],
        shape: rectShape,
        style: api.style()
      };
    }

    // 创建系列
    const series = eventTypes.map(type => ({
      name: type.name,
      yAxisIndex: 1,
      type: 'custom',
            label: {
                show: true,
                position: 'inside', // 可以调整为 'insideTop' 或其他位置            
                formatter: function (params) {
                    // 这里使用富文本格式化，允许我们显示多行文本                
                    return `{a|${params.data.event_type}}\n{b|MB${params.data.microbatch}}`;
                },
                rich: {
                    a: {
                        color: '#000000',
                        lineHeight: 20, // 控制行高   
                        fontSize: 8,                 
                        align: 'center'
                    },
                    b: {
                        color: '#000000',
                        fontSize: 8,
                        lineHeight: 20, // 控制行高                    
                        align: 'center'
                    }
                }
            },
      renderItem: renderItem,
      itemStyle: {
        opacity: 0.8,
        color: type.color,
        borderWidth: 1,
        borderColor: '#333'
      },
      encode: {
        x: [1, 2],
        y: 0
      },
      data: dataByType[type.name]
    }));

    // 配置选项
    const option = {
      tooltip: {
        trigger: 'item',
        formatter: function (params) {
          if (!params.data) return
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
                    `;
        }
      },
      legend: {
        data: eventTypes.map(type => type.name),
        top: 10,
        textStyle: {
          fontSize: 12
        }
      },
      grid: {
        left: 120,
        right: 30,
        top: 60,
        bottom: 80
      },
      xAxis: {
        type: 'value',
        scale: true,
        axisLabel: {
          formatter: function (val) {
            return (val / 1000).toFixed(3) + ' s';
          }
        },
        name: '',
        nameLocation: 'middle',
        nameGap: 30
      },
      yAxis: [{
        type: 'category',
        data: ranks,
        name: 'GPU Cluster Timeline with Event Layering',
        nameLocation: 'middle',
        nameGap: 70,
        axisLabel: {
          fontSize: 12,
          formatter: function (value) {
            // 只显示主要设备ID，隐藏层级标签
            if (value.includes(' L')) {
              return '';
            }
            return `Rank ${value}`;
          }
        },
        axisTick: {
          show: true,
          alignWithLabel: true,
          length: 4
        },
        splitLine: {
          show: true,
          lineStyle: {
            type: 'dashed',
            color: '#eee'
          }
        }
      }, {
        type: 'category',
        data: extendedRanks,
        name: '设备ID与层级',
        axisLabel: {
          fontSize: 12,
          formatter: function (value) {
            // 简化显示，只显示主要设备ID
            if (value.includes(' L')) {
              return '';
            }
            return value;
          },

        },
        axisTick: {
          show: true,
          alignWithLabel: true,
          length: 4
        },
        splitLine: {
          show: true,
          lineStyle: {
            type: 'dashed',
            color: '#eee'
          }
        },
        show: false
      }],
      dataZoom: [
        {
          type: 'slider',
          xAxisIndex: 0,
          filterMode: 'filter',
          bottom: 30,
          height: 20,
          labelFormatter: function (value) {
            return (value / 1000).toFixed(3) + ' s';
          }
        },
        {
          type: 'inside',
          xAxisIndex: 0,
          filterMode: 'filter'
        }
      ],
      series: series
    };


    chartInstance.current.setOption(option);
  };

  // 窗口 resize 处理
  const handleResize = () => {
    chartInstance.current?.resize();
  };

  useEffect(() => {
    initChart();
    window.addEventListener('resize', handleResize);
    return () => {
      chartInstance.current?.dispose();
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  return (
    <div style={{ width: '100%', height: '600px', padding: '20px' }}>
      <div ref={chartRef} style={{ width: '100%', height: '85%' }} />
      {/* <button
        onClick={updateChartData}
        style={{
          marginTop: '10px',
          padding: '8px 16px',
          backgroundColor: '#1890ff',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: 'pointer'
        }}
      >
        更新图表数据
      </button> */}
    </div>
  );
};

export default TimeLineCharts;