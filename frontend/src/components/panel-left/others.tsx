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

// 策略列表 - 优化命名
const DEFAULT_STRATEGY_LIST = [];

// 数据类型列表 - 移除不必要的key属性
const DATATYPE_LIST = [
  { label: 'float32', value: 'float32' },
  { label: 'float16', value: 'float16' }
];

// 参数配置列表
const PARAMS_LIST = [
  {
    title: 'Tensor parallel degree',
    key: 'tensor_par',
    min: 0,
    max: 8,
    precision: 0,
    step: 1
  },
  {
    title: 'Pipeline parallel degree',
    key: 'pipeline_par',
    min: 0,
    max: 10000,
    precision: 0,
    step: 1
  },
  {
    title: 'Data parallel degree',
    key: 'data_par',
    min: 0,
    max: 10000,
    precision: 0,
    step: 1
  }
];

const OtherPanel = (props) => {
  const { setProject, setOtherConfig, otherConfig, recommendConfig, curModel, curGpu,
    checkSize, checkPipeline } = useModel(ProjectModel);
  const { t } = useTranslation();
  const { setChangeLog } = useModel(LogModel);

  const [state, setState] = useImmer({
    strategyList: DEFAULT_STRATEGY_LIST,
    dataTypeList: DATATYPE_LIST
  });

  // 设置参数值并记录变更日志
  const setParamValue = (key, val, title) => {
    setChangeLog(title, val, otherConfig?.[key]);
    setOtherConfig({ [key]: val });
  };

  // 计算最小值
  const calcMin = (cf) => cf.min;

  // 计算最大值
  const calcMax = () => {
    return curGpu.num_procs;
  };

  // 加载优化策略列表
  const loadStrategies = async () => {
    const strategyRes = await getStrategies();
    const strList = strategyRes.map(item => ({
      key: item,
      label: item,
      value: item,
    }));
    setState(prev => ({
      ...prev,
      strategyList: strList
    }));
  };

  // 关闭错误消息
  const closeErrorMsg = () => {
    setProject({
      errorMsg: null,
      showError: false
    });
  };

  // 组件加载时获取策略列表
  useEffect(() => {
    loadStrategies();
  }, []);

  // 渲染微批次大小设置组件
  const renderMicrobatchSize = () => (
    <div>
      {recommendConfig.recomended_microbatch && (
        <div className={styles.slider_tip}>
          {t('microbatch recommend', { value: recommendConfig.recomended_microbatch })}
        </div>
      )}
      <InputNumber
        className={styles.number_item}
        precision={0}
        min={1}
        max={curModel?.minibatch_size}
        value={otherConfig?.microbatch_size}
        onChange={(val) => setParamValue('microbatch_size', val, 'Microbatch size')}
        addonAfter={(
          <Popover content={<div>Need to be able to divide minibatch size.</div>}>
            <InfoCircleOutlined style={{ cursor: 'pointer' }} />
          </Popover>
        )}
      />
      {!checkSize() && curModel?.minibatch_size && (
        <div className={styles.error_tip}>
          Need to be able to divide minibatch size({curModel?.minibatch_size}).
        </div>
      )}
    </div>
  );
  // 渲染微批次大小设置组件
  const renderBatchSize = () => (
    <div>
      {recommendConfig.recomended_microbatch && (
        <div className={styles.slider_tip}>
          {t('microbatch recommend', { value: recommendConfig.recomended_microbatch })}
        </div>
      )}
      <InputNumber
        className={styles.number_item}
        precision={0}
        min={1}
        max={curModel?.batch_size}
        value={otherConfig?.batch_size}
        onChange={(val) => setParamValue('batch_size', val, 'Batch size')}
        addonAfter={(
          <Popover content={<div>Need to be able to divide data parallel degree.</div>}>
            <InfoCircleOutlined style={{ cursor: 'pointer' }} />
          </Popover>
        )}
      />
      {!checkSize() && curModel?.batch_size && (
        <div className={styles.error_tip}>
          Need to be able to divide data parallel degree ({curModel?.batch_size}).
        </div>
      )}
    </div>
  );

  return (
    <div className={styles.nest}>
      <p className={styles.section_title}>{t('optimization strategy')}</p>
      <div className={styles['group-content']}>
        <Select
          options={state.strategyList}
          placeholder={t('Please select')}
          value={otherConfig['optimization_strategy']}
          onChange={(val) => setParamValue('optimization_strategy', val, 'Optimization Strategy')}
        />
      </div>

      <div className={styles.slider_tip}>
        <span style={{color:otherConfig['tensor_par'] * otherConfig['pipeline_par'] *otherConfig['data_par'] ==curGpu.num_procs? '': '#ff4d4f'}} >{t('pp_dp_tp_recommend', {value: curGpu.num_procs})}</span>
      </div>

      <div className={styles['group_slider']}>
        {PARAMS_LIST.map((cf) => (
          <div className={styles['group-list-item']} key={cf.key}>
            <div className={styles['item-wrapper']}>
              <span>{cf.title}</span>
              <InputNumber
                precision={cf.precision || 0}
                width={100}
                min={calcMin(cf)}
                max={calcMax()}
                value={otherConfig[cf.key]}
                onChange={(val) => setParamValue(cf.key, val, cf.title)}
              />
            </div>

            {/* {cf.key === 'tensor_par' && (
              <div className={styles.slider_tip}>
                {t('tensor recommend', { value: recommendConfig?.recomended_tensor_par })}
              </div>
            )} */}

            {/* {cf.key === 'pipeline_par' && otherConfig.tensor_par && (
              <div className={styles.slider_tip}>
                {recommendConfig.recomended_pipeline_par > 0 ? (
                  <span>{t('pipeline recommend', { value: recommendConfig.recomended_pipeline_par })}</span>
                ) : (
                  <span style={{ color: '#ff4d4f' }}>{t('pipeline tips')}</span>
                )}
              </div>
            )} */}

            {/* {cf.key === 'data_par' && otherConfig.tensor_par && (
              <div className={styles.slider_tip}>
                {t('data parallel tips', {
                  value: recommendConfig.recomended_data_par || 1
                })}
              </div>
            )} */}

            <Slider
              min={cf.min}
              max={cf.key === 'pipeline_par' ? curModel?.num_layers : cf.max}
              onChange={(val) => setParamValue(cf.key, val, cf.title)}
              value={otherConfig[cf.key]}
              step={cf.step}
            />

            {cf.key === 'pipeline_par' && !checkPipeline() && curModel?.minibatch_size && (
              <div className={styles.error_tip}>
                {t('pipeline divide tips')}({curModel?.num_layers}).
              </div>
            )}
          </div>
        ))}
      </div>

      <p className={styles.section_title}>{t('batch size')}</p>
      <div className={styles.section_content}>
        {renderBatchSize()}
      </div>

      <p className={styles.section_title}>{t('microbatch')}</p>
      <div className={styles.section_content}>
        {renderMicrobatchSize()}
      </div>

      <p className={styles.section_title}>{t('datatype')}</p>
      <div className={styles['group-content']}>
        <Select
          options={state.dataTypeList}
          placeholder={t('Select one datatype')}
          value={otherConfig['datatype']}
          onChange={(val) => setParamValue('datatype', val, 'Data Type')}
        />
      </div>
    </div>
  );
};

export default OtherPanel;