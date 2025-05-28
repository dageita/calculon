import {
  Select,
  InputNumber,
  Slider,
  Popover
} from 'antd';
import useModel from 'flooks';
import styles from './index.less';
import ProjectModel from '@/models/projectModel';
import { InfoCircleOutlined } from '@ant-design/icons';
import LogModel from '@/models/logModel';
import { getStrategies } from '@/services';
import { useImmer } from 'use-immer';
import { useEffect } from 'react';
import { useTranslation } from 'react-i18next';

const DEFAULT_SRATEGY_LIST: any[] = [
  // {
  //   label: 'No recomputation',
  //   value: 'No recomputation'
  // },
  // {
  //   label: 'Selective recomputation',
  //   value: 'Selective recomputation'
  // },
  // {
  //   label: 'Full recomputation',
  //   value: 'Full recomputation'
  // },
];

const PARAMS_LIST = [
  {
    title: 'Tensor parallel degree',
    key: 'tensor_parallel_degree',
    min: 1,
    max: 8,
    precision: 0,
    step: 1
  },
  {
    title: 'Pipeline parallel degree',
    key: 'pipeline_parallel_degree',
    min: 1,
    max: 10000,
    precision: 0,
    step: 1

  }
]
const OtherPanel = (props: any) => {
  const { setProject, setOtherConfig, otherConfig, recommendConfig, curModel, curGpu,
    checkSize, checkPipeline } = useModel(ProjectModel);
  const { t } = useTranslation();
  const { setChangeLog } = useModel(LogModel);
  const [state, setState] = useImmer({
    SRATEGY_LIST: DEFAULT_SRATEGY_LIST
  })
  const setParamValue = (key: string, val: any, title?: string) => {
    setChangeLog(title, val, otherConfig?.[key])
    setOtherConfig({
      [key]: val
    });
  };
  const calcMin = (cf: any) => {
    // if (cf.key === 'pipeline_parallel_degree') {
    //   return recommendConfig.recomended_pipeline_parallel_degree
    // }
    return cf.min
  }
  const calcMax = (cf: any) => {
    if (cf.key === 'pipeline_parallel_degree') {
      return curModel?.num_layers
    }
    if (cf.key === 'tensor_parallel_degree') {
      // return recommendConfig.recomended_tensor_parallel_degree
      return 8
    }
    return cf.max
  }
  const loadStrategies = async () => {
    const strategyRes: any = await getStrategies()
    const strList = strategyRes.map((item: any) => {
      return {
        key: item,
        label: item,
        value: item,
      }
    })
    setState({
      SRATEGY_LIST: strList
    })
  }
  const closeErrorMsg = () => {
    setProject({
      errorMsg: null,
      showError: false
    })
  }
  useEffect(() => {
    loadStrategies()
  }, []);
  return (
    <div className={styles.nest}>
      <p className={styles.section_title}>
        {/* Optimization Strategy */}
        {t('optimization strategy')}
      </p>
      <div className={styles['group-content']}>
        {/* {SRATEGY_LIST.map((m: any) => {
          return (
            <Button
              key={m.value}
              size="small"
              value="line"
              className={`${styles['mode-btn']} ${otherConfig.optimization_strategy === m.value ? styles['active'] : ''
                }`}
              onClick={(e) => {
                setParamValue('optimization_strategy', m.value, 'Optimization Strategy')
              }}
            >
              {m.label}
            </Button>
          );
        })} */}
        <Select
          options={state.SRATEGY_LIST}
          value={otherConfig.optimization_strategy}
          onChange={val => {
            setParamValue('optimization_strategy', val, 'Optimization Strategy')
          }}
        ></Select>
      </div>
      <div className={styles['group_slider']}>
        {PARAMS_LIST.map((cf: any) => {
          return (
            <div className={styles['group-list-item']} key={cf.key}>
              <div className={styles['item-wrapper']}>
                <span>
                  {cf.title}
                  {/* <Tooltip placement="top" title={cf.description}>
                      <QuestionCircleFilled
                        style={{ paddingLeft: 10, cursor: 'pointer' }}
                      />
                    </Tooltip> */}
                </span>
                <InputNumber
                  precision={cf.precision || 0}
                  width={100}
                  min={calcMin(cf)}
                  max={calcMax(cf)}
                  value={otherConfig[cf.key]}
                  onChange={(val) => {
                    setParamValue(cf.key, val, cf.title)
                  }}
                />
              </div>
              {cf.key === 'tensor_parallel_degree' &&
                <div className={styles.slider_tip}>
                  {/* No larger than recommended value (<b>{recommendConfig?.recomended_tensor_parallel_degree}</b>) to balance GPU communication/computation time. */}
                  {t('tensor recommend', { value: recommendConfig?.recomended_tensor_parallel_degree })}
                </div>}
              {cf.key === 'pipeline_parallel_degree' && otherConfig.tensor_parallel_degree &&
                <div className={styles.slider_tip}>
                  {recommendConfig.recomended_pipeline_parallel_degree > 0 ?
                    <span>
                      {/* No smaller than  recommended value (<b>{recommendConfig.recomended_pipeline_parallel_degree}</b>) to avoid OOM */}
                      {t('pipeline recommend', { value: recommendConfig.recomended_pipeline_parallel_degree })}
                    </span>
                    :
                    <span style={{ color: '#ff4d4f' }}>
                      {t('pipeline tips')}
                    </span>
                  }</div>}
              <Slider
                min={cf.min}
                max={cf.key === 'pipeline_parallel_degree' ? curModel?.num_layers : cf.max}
                onChange={(val) => {
                  setParamValue(cf.key, val, cf.title)
                }}
                value={otherConfig[cf.key]}
                step={cf.step}
              />
              {cf.key === 'pipeline_parallel_degree' && !checkPipeline() && curModel?.minibatch_size && <div className={styles.error_tip}>
                {t('pipeline divide tips')}({curModel?.num_layers}).
              </div>}
            </div>
          );
        })}
      </div>
      <p className={styles.section_title}>
        {/* Microbatch size */}
        {t('microbatch')}
      </p>
      <div className={styles.section_content}>
        {recommendConfig.recomended_microbatch && <div className={styles.slider_tip}>
          {/* No larger than  recommended value (<b>{recommendConfig.recomended_microbatch}</b>) to reduce pipeline bubble time. */}
          {t('microbatch recommend', { value: recommendConfig.recomended_microbatch })}
        </div>}
        <InputNumber
          className={styles.number_item}
          precision={0}
          min={0}
          // max={recommendConfig.recomended_microbatch || curModel?.minibatch_size}
          max={curModel?.minibatch_size}
          value={otherConfig?.microbatch_size}
          onChange={(val) => setParamValue('microbatch_size', val, 'Microbatch size')}
          addonAfter={<Popover
            content={<div>
              Need to be able to divide minibatch size.
            </div>}>
            <InfoCircleOutlined style={{ cursor: 'pointer' }} />
          </Popover>}>
        </InputNumber>
        {!checkSize() && curModel?.minibatch_size && <div className={styles.error_tip}>Need to be able to divide minibatch size({curModel?.minibatch_size}).</div>}
      </div>
    </div>
  );
};

export default OtherPanel;
