import { FC, Fragment } from 'react';
import { Divider, Popover } from 'antd'
import styles from './index.less';
import PopPanel from './pops'
import { keys, sum } from 'lodash';
import { useTranslation } from 'react-i18next';
const COLOR_MAPPING: any = {
  warmup: {
    label: 'Warmup time',
    color: '#3793FF',
    key: 'warmup_time'
  },
  forward: {
    label: 'Forward time',
    color: '#92CC76',
    key: 'forward_time'
  },
  backward: {
    label: 'Backward time',
    color: '#AAE7FF',
    key: 'backward_time'
  },
  cooldown: {
    label: 'Cooldown time',
    color: '#FAC858',
    key: 'cooldown_time'
  },
  allReduce: {
    label: 'All Reduce time',
    color: '#EF6666',
    key: 'allreduce_time'
  },
  microbatch_forward_computation_time: {
    label: 'Forward comp time',
    color: '#aae7ff',
    key: 'microbatch_forward_computation_time'
  },
  microbatch_backward_computation_time: {
    label: 'Backward comp time',
    color: '#72d3af',
    key: 'microbatch_backward_computation_time'
  },
  microbatch_tp_fw_comm_time: {
    label: 'TP FW comm time',
    color: '#bbd372',
    key: 'microbatch_tp_fw_comm_time'
  },
  microbatch_tp_bw_comm_time: {
    label: 'TP BW comm time',
    color: '#948cd9',
    key: 'microbatch_tp_bw_comm_time'
  },
  microbatch_pp_fw_comm_time: {
    label: 'PP FW comm time',
    color: '#d1539e',
    key: 'microbatch_pp_fw_comm_time'
  },
  microbatch_pp_bw_comm_time: {
    label: 'PP BW comm time',
    color: '#5d0feeff',
    key: 'microbatch_pp_bw_comm_time'
  },
  batch_dp_comm_time: {
    label: 'DP comm time',
    color: '#e60b71ff',
    key: 'batch_dp_comm_time'
  }
}

export interface IBaseTLProps {
  result: any,
  latest_result?: any,
  widthScale?: string,
  curMode: string
}
const BaseTL: FC<IBaseTLProps> = (props) => {
  const { result, latest_result, curMode } = props;
  const { t } = useTranslation();
  const dataParse = (d: number, toGB?: boolean) => {
    if (!d) return d
    if (toGB) {
      d = d / (1024 * 1024 * 1024)
    }
    // 整数
    if (d.toString() === d.toFixed(0)) {
      return d
    }
    // 大于1的浮点数，保留2位
    if (d > 1) {
      return d.toFixed(2)
    }
    // 小于1的浮点数，保留6位
    return d.toFixed(6)
  }
  const { warmup_time, forward_time, backward_time, cooldown_time, allreduce_time, num_microbatches, batch_total_time } = result?.timeline || {}

  const totalTime = sum([warmup_time, batch_total_time, cooldown_time, allreduce_time])
  let period = 0;
  const colors = ['#aae7ff', '#72d3af', '#bbd372', '#948cd9', '#d1539e', '#e6b20bff', '#5d0feeff', '#e60b71ff']
  const microBatchPeriods = result?.timeline ? Object.keys(result.timeline).filter(o => o.startsWith('microbatch_')).map((o, n) => {
    period += result.timeline[o]
    return { key: o, v: result.timeline[o], color: colors[n] }
  }) : []


  let total =0
  const calcLength = (time: number, isMulti?: boolean) => {
    if (isMulti) {
      total+=time
      console.log(total)
      return `${(time / batch_total_time) * (100 - Math.ceil(num_microbatches / 10))}%`
    }
    return `${(time / totalTime) * 98}%`
  }
  const checkChanged = (val: any, preVal: any) => {
    if (curMode !== 'guide') {
      return ''
    }
    if (preVal && val !== preVal) {
      return styles.changed
    }
    return ''
  }
  const renderLoopTime = (index: number) => {
    return <Fragment key={index}>
      {microBatchPeriods.map((rowItem: any, idx: number) => {
        return <div className={styles.light} style={{
          width: calcLength(rowItem.v, true),
          backgroundColor: COLOR_MAPPING[rowItem.key].color
        }}>
        </div>
      })}
    </Fragment>
  }
  const renderMultiLoopTime = () => {
    const numsArray = []
    for (let i = 0; i < num_microbatches; i++) {
      numsArray.push(i)
    }
    return numsArray.map((_, index) =>
      renderLoopTime(index)
    )
  }
  const renderTip = (time: number, title: string) => {
    return <div className={styles.pop_tip}>
      <div>{title}</div>
      {/* <div>{dataParse(time)} ({((time / totalTime) * 100).toFixed(2)}%)</div> */}
      <div>{dataParse(time)}</div>
    </div>
  }
  const renderDetail = () => {
    return <PopPanel result={result} otherConfig={result.other_config} />
  }

  return (
    <div>
      <div className={styles.timeline_group_total} style={{ width: props.widthScale || '100%' }}>
        {/* {dataParse(totalTime)}s */}
        <span className={styles.timeline_total_label}>{t('number of microbatches')}</span>
        <span >
          {result.timeline?.num_microbatches}
        </span>
        <Divider type="vertical" />
        <span className={styles.timeline_total_label}>{t('global time')}</span>
        <span >
          {result.summary?.batch_total_time.toFixed(6)}
        </span>
      </div>

      <div className={styles.timeline_group} style={{ width: props.widthScale || '100%' }}>
        <Popover content={renderTip(warmup_time, COLOR_MAPPING['warmup'].label)} title="" trigger="hover">
          <div className={styles.timeline_block} style={{
            width: calcLength(warmup_time),
            backgroundColor: COLOR_MAPPING['warmup'].color
          }}>
          </div>
        </Popover>

        <div className={styles.timeline_block_loop} style={{ width: calcLength(batch_total_time) }}>
          {renderMultiLoopTime()}
        </div>

        <Popover content={renderTip(cooldown_time, COLOR_MAPPING['cooldown'].label)} title="" trigger="hover">
          <div className={styles.timeline_block} style={{
            width: calcLength(cooldown_time),
            backgroundColor: COLOR_MAPPING['cooldown'].color
          }}>
          </div>
        </Popover>
        <Popover content={renderTip(allreduce_time, COLOR_MAPPING['allReduce'].label)} style={styles.timeline_block_pop} title="" trigger="hover"
          placement="left">
          <div className={styles.timeline_block} style={{
            width: calcLength(allreduce_time),
            backgroundColor: COLOR_MAPPING['allReduce'].color
          }}>
          </div>
        </Popover>
      </div>
      <div className={styles.timeline_group_legend}>
        {keys(COLOR_MAPPING).map((key: string) => {
          const item: any = COLOR_MAPPING[key]
          if (!result.timeline[item.key]) {
            return
          }
          return <Popover content={['forward', 'backward'].indexOf(key) > -1 ? renderDetail() : renderTip(result.timeline[item.key], item.label)
          } title="" trigger="hover" key={key}>
            <div key={key}>
              <div className={styles.timeline_legend_item} style={!item.border ? { backgroundColor: item.color } : { border: `1px ${item.border} ${item.color}` }}></div>
              <span>{item.label}</span>
            </div>
          </Popover>
        })}
      </div>
    </div>
  );
};

export default BaseTL;
